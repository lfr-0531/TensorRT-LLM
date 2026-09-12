# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native ratio-2 compression and existing ratio-4/128 regressions."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(kv_score, ratio, dim):
    overlap = ratio == 4
    state_dim = dim * (2 if overlap else 1)
    values, gates = kv_score.split(state_dim, dim=-1)
    outputs = []
    for end in range(ratio, kv_score.shape[0] + 1, ratio):
        if overlap:
            current_values = values[end - ratio : end, dim:]
            current_gates = gates[end - ratio : end, dim:]
            if end > ratio:
                current_values = torch.cat(
                    (values[end - 2 * ratio : end - ratio, :dim], current_values)
                )
                current_gates = torch.cat(
                    (gates[end - 2 * ratio : end - ratio, :dim], current_gates)
                )
        else:
            current_values = values[end - ratio : end]
            current_gates = gates[end - ratio : end]
        outputs.append((current_values * current_gates.softmax(0)).sum(0).bfloat16())
    return torch.stack(outputs)


@pytest.mark.parametrize("ratio", [2, 4, 128])
@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("decode_tail", [False, True])
def test_native_chunked_compression(ratio, dim, decode_tail):
    torch.manual_seed(732)
    length = 2 * ratio + 3
    state_dim = dim * (2 if ratio == 4 else 1)
    data = torch.randn(length, state_dim * 2, dtype=torch.float32, device="cuda")
    page_size = 32
    pages = (length + page_size - 1) // page_size
    kv = torch.full((pages, page_size, state_dim), float("nan"), device="cuda")
    score = torch.full_like(kv, float("nan"))
    kv_table = torch.randperm(pages, device="cuda").int().unsqueeze(0)
    score_table = torch.randperm(pages, device="cuda").int().unsqueeze(0)
    ape = torch.zeros(ratio, state_dim, device="cuda")
    outputs = []
    start = 0
    for chunk_idx, count in enumerate((ratio - 1, 2, ratio + 2)):
        end = start + count
        num_outputs = end // ratio - start // ratio
        output = torch.empty(num_outputs, dim, dtype=torch.bfloat16, device="cuda")
        lengths = torch.tensor([end], dtype=torch.int32, device="cuda")
        cu_seq = torch.tensor([0, count], dtype=torch.int32, device="cuda")
        cu_comp = torch.tensor([0, num_outputs], dtype=torch.int32, device="cuda")
        chunk = data[start:end].contiguous()
        if decode_tail and chunk_idx > 0:
            torch.ops.trtllm.compressor_paged_kv_compress(
                chunk,
                ape,
                kv,
                score,
                kv_table,
                score_table,
                output,
                lengths,
                cu_seq,
                cu_comp,
                1,
                page_size,
                dim,
                ratio,
                count,
            )
        else:
            starts = torch.tensor([start], dtype=torch.int32, device="cuda")
            torch.ops.trtllm.compressor_prefill_reduction(
                chunk,
                ape,
                kv,
                score,
                kv_table,
                score_table,
                output,
                lengths,
                starts,
                cu_seq,
                cu_comp,
                1,
                page_size,
                dim,
                ratio,
                max(1, num_outputs),
            )
        outputs.append(output)
        start = end
    torch.testing.assert_close(
        torch.cat(outputs), _reference(data, ratio, dim), atol=0.016, rtol=0.008
    )


def test_ratio2_native_cuda_graph_changed_partial_state():
    dim, page_size = 512, 32
    kv = torch.zeros(1, page_size, dim, device="cuda")
    score = torch.zeros_like(kv)
    table = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    ape = torch.zeros(2, dim, device="cuda")
    data = torch.zeros(1, dim * 2, device="cuda")
    output = torch.empty(1, dim, dtype=torch.bfloat16, device="cuda")
    lengths = torch.tensor([2], dtype=torch.int32, device="cuda")
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")

    def run():
        torch.ops.trtllm.compressor_paged_kv_compress(
            data, ape, kv, score, table, table, output, lengths, cu, cu, 1, page_size, dim, 2, 1
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for previous, current in ((2.0, 4.0), (-2.0, 6.0)):
        kv[:, 0].fill_(previous)
        score.zero_()
        data[:, :dim].fill_(current)
        data[:, dim:].zero_()
        graph.replay()
        torch.testing.assert_close(
            output, torch.full_like(output, (previous + current) / 2), atol=0, rtol=0
        )


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_module_pre_rope_latent(ratio):
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v41.compressor import (
        CSA2CompressionBatch,
        CSA2Compressor,
    )

    torch.manual_seed(927)
    model = CSA2Compressor(16, 128, ratio, 1e-20).cuda()
    x = torch.randn(6, 16, dtype=torch.bfloat16, device="cuda")
    with torch.no_grad():
        projected = model.wkv(x.float() if ratio == 2 else x)
        if ratio == 2:
            gates = model.wgate(x.float()).reshape(3, 2, 128)
            projected = (projected.reshape(3, 2, 128) * gates.softmax(1)).sum(1).bfloat16()
        v = projected.float()
        expected = (v * torch.rsqrt(v.square().mean(-1, keepdim=True) + 1e-20)).bfloat16()
        if ratio == 1:
            actual = model(x)
        else:
            state = torch.zeros(1, 32, 128, device="cuda")
            score = torch.zeros_like(state)
            table = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
            chunks = []
            start = 0
            for count in (1, 3, 2):
                end = start + count
                outputs = end // 2 - start // 2
                batch = CSA2CompressionBatch(
                    state,
                    score,
                    table,
                    table,
                    torch.tensor([end], dtype=torch.int32, device="cuda"),
                    torch.tensor([start], dtype=torch.int32, device="cuda"),
                    torch.tensor([0, count], dtype=torch.int32, device="cuda"),
                    torch.tensor([0, outputs], dtype=torch.int32, device="cuda"),
                    outputs,
                    32,
                    max(outputs, 1),
                )
                chunks.append(model(x[start:end], batch))
                start = end
            actual = torch.cat(chunks)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
    if ratio == 2:
        assert "zero_ape" not in dict(model.named_parameters())
        assert model.wkv.weight.dtype == model.wgate.weight.dtype == torch.float32
