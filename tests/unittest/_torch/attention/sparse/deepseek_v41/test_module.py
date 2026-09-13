# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual CSA2 module regression across incomplete/completed ratio-two groups."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rotate(x, positions, cos_sin, inverse=False):
    values = x.float().clone()
    rope_dim = cos_sin.shape[-1] * 2
    cos = cos_sin[positions.long(), 0].unsqueeze(1)
    sin = cos_sin[positions.long(), 1].unsqueeze(1)
    if inverse:
        sin = -sin
    even = values[..., -rope_dim::2].clone()
    odd = values[..., -rope_dim + 1 :: 2].clone()
    values[..., -rope_dim::2] = even * cos - odd * sin
    values[..., -rope_dim + 1 :: 2] = odd * cos + even * sin
    return values.to(x.dtype)


def _norm(x, weight, eps):
    return F.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps).to(x.dtype)


@torch.inference_mode()
def test_ratio2_module_partial_groups(monkeypatch):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v41.backend import (
        CSA2Batch,
        CSA2Cache,
        CSA2GlobalPages,
        CSA2Routing,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v41.compressor import (
        CSA2CompressionBatch,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v41.module import (
        DeepseekV41Attention,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v41.params import CSA2Layout
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v41.quantization import (
        pack_rows,
        unpack_rows,
    )
    from tensorrt_llm.functional import PositionEmbeddingType

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    torch.manual_seed(818)
    layout = CSA2Layout((2,), (0,), (0,), index_topk=2, window_size=4)
    with torch.device("cuda"):
        model = DeepseekV41Attention(
            layout,
            0,
            PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=RopeParams(dim=64, theta=160000, max_positions=8),
                is_neox=False,
            ),
            hidden_size=32,
            num_heads=8,
            head_dim=512,
            q_lora_rank=32,
            o_lora_rank=16,
            num_groups=2,
            index_heads=2,
        )
        for name, parameter in model.named_parameters():
            if "norm_weight" in name:
                parameter.fill_(1)
            else:
                parameter.normal_(std=0.1)
        hidden = torch.randn(4, 32, dtype=torch.bfloat16)
        cache = CSA2Cache.allocate(layout, 4, {0: 2}, torch.device("cuda"))
        for pool in (*cache.swa.values(), *cache.main.values(), *cache.index.values()):
            pool.zero_()
        state = torch.zeros(1, 32, 512)
        gate_state = torch.zeros_like(state)
        table = torch.zeros(1, 1, dtype=torch.int32)
        page_map = CSA2GlobalPages(
            torch.tensor([[0, 1]], dtype=torch.int32), torch.tensor([0], dtype=torch.int32), 1, 2
        )
        cu_seq = torch.tensor([0, 1], dtype=torch.int32)

    native_rope = torch.ops.trtllm.mla_rope_inplace
    rope_rows = []

    def record_rope_rows(x, *args):
        # Observe the dispatch contract without replacing any CUDA math.
        rope_rows.append(x.shape[0])
        return native_rope(x, *args)

    monkeypatch.setattr(torch.ops.trtllm, "mla_rope_inplace", record_rope_rows)
    compressor = model.compressor
    cos_sin = model.rotary_emb.rotary_cos_sin
    for token in range(4):
        rope_rows.clear()
        completed = (token + 1) // 2
        new_rows = (token + 1) % 2 == 0
        positions = torch.tensor([token], device="cuda", dtype=torch.int32)
        compressed_positions = torch.tensor(
            [token - 1] if new_rows else [], device="cuda", dtype=torch.int32
        )
        batch = CSA2Batch(
            torch.arange(token + 1, device="cuda").unsqueeze(0),
            positions,
            page_map,
            torch.tensor([completed], device="cuda", dtype=torch.int32),
            torch.tensor([completed - 1] if new_rows else [], device="cuda", dtype=torch.int32),
        )
        compression = CSA2CompressionBatch(
            state,
            gate_state,
            table,
            table,
            torch.tensor([token + 1], device="cuda", dtype=torch.int32),
            positions,
            cu_seq,
            torch.tensor([0, int(new_rows)], device="cuda", dtype=torch.int32),
            int(new_rows),
            32,
            1,
        )
        old_main, old_index = cache.main[0].clone(), cache.index[0].clone()
        actual = model(
            hidden[token : token + 1],
            positions,
            cache,
            batch,
            CSA2Routing(),
            compression=compression,
            compressed_positions=compressed_positions,
        )
        torch.cuda.synchronize()
        # Numerics alone can miss an empty native dispatch if a runtime/kernel
        # handles it as a no-op or clears its launch error in a later call.
        assert all(rows > 0 for rows in rope_rows), "Empty rows reached native RoPE"
        assert torch.isfinite(actual).all()
        if not new_rows:
            torch.testing.assert_close(cache.main[0], old_main, atol=0, rtol=0)
            torch.testing.assert_close(cache.index[0], old_index, atol=0, rtol=0)

        prefix_positions = torch.arange(token + 1, device="cuda", dtype=torch.int32)
        qr = _norm(
            F.linear(hidden[token : token + 1], model.wq_a.weight), model.q_norm_weight, model.eps
        )
        q = _rotate(F.linear(qr, model.wq_b.weight).reshape(1, 8, 512), positions, cos_sin)
        swa = _norm(
            F.linear(hidden[: token + 1], model.wkv.weight), model.kv_norm_weight, model.eps
        )
        swa = _rotate(swa.unsqueeze(1), prefix_positions, cos_sin).squeeze(1)
        selected = unpack_rows(pack_rows(swa, "swa"), 512, "swa")
        if completed:
            source = hidden[: completed * 2].float()
            values = F.linear(source, compressor.wkv.weight).reshape(completed, 2, 512)
            gates = F.linear(source, compressor.wgate.weight).reshape(completed, 2, 512)
            latent = (values * gates.softmax(1)).sum(1).bfloat16()
            latent = _norm(latent, compressor.norm_weight, model.eps)
            group_positions = torch.arange(completed, device="cuda", dtype=torch.int32) * 2
            main = _rotate(latent.unsqueeze(1), group_positions, cos_sin).squeeze(1)
            main = unpack_rows(pack_rows(main, "main"), 512, "main")
            selected = torch.cat((selected, main))
        # All visible global positions fit in top-k, so this reference needs
        # no indexer implementation. Quantization is covered independently.
        scores = torch.einsum("qhd,kd->qhk", q.float(), selected.float()) * 512**-0.5
        probs = torch.cat((scores, model.attn_sink[None, :, None]), -1).softmax(-1)[..., :-1]
        output = torch.einsum("qhk,kd->qhd", probs, selected.float()).bfloat16()
        output = _rotate(output, positions, cos_sin, inverse=True).reshape(1, 2, -1)
        projected = torch.einsum("qgd,grd->qgr", output, model.o_a_proj).flatten(1)
        expected = F.linear(projected, model.o_b_proj.weight)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
