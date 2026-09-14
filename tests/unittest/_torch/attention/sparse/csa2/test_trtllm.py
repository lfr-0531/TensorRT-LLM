# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 hardware policy and reuse of the native DSV4 dynamic sparse path."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
    CSA2Params,
    select_csa2_backend,
)


@pytest.mark.parametrize(
    "sm,backend",
    [
        (90, "flash_mla"),
        (100, "trtllm"),
        (103, "trtllm"),
        (107, "trtllm"),
        (120, "flashinfer"),
        (121, "flashinfer"),
    ],
)
def test_hardware_policy(sm, backend):
    assert select_csa2_backend(sm) == backend


@pytest.mark.parametrize("sm", [80, 89, 110, 130])
def test_unsupported_hardware(sm):
    with pytest.raises(ValueError, match="unsupported"):
        select_csa2_backend(sm)


def _reference(q, swa, extra, swa_valid, extra_valid, sink):
    kv, valid = swa, swa_valid
    if extra is not None:
        kv = torch.cat((kv, extra), dim=1)
        valid = torch.cat((valid, extra_valid), dim=1)
    kv = torch.where(valid[..., None], kv, 0)
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * q.shape[-1] ** -0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    probs = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, -1)), -1).softmax(-1)[
        ..., :-1
    ]
    return torch.einsum("qhk,qkd->qhd", probs, kv.float()).to(q.dtype)


def _backend(heads):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.trtllm import CSA2TrtllmAttention
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.backend import (
        DeepseekV4TrtllmAttention,
    )
    from tensorrt_llm._torch.attention.backends.utils import create_attention, get_attention_backend
    from tensorrt_llm._utils import is_sm_100f

    if not is_sm_100f():
        pytest.skip("TRTLLM dynamic sparse MLA requires SM100-family")
    params = CSA2Params(max_query_tokens=16)
    assert get_attention_backend("TRTLLM", params) is CSA2TrtllmAttention
    attn = create_attention(
        "TRTLLM",
        20,
        heads,
        512,
        num_kv_heads=1,
        is_mla_enable=True,
        q_lora_rank=1280,
        kv_lora_rank=448,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        v_head_dim=512,
        rope_append=False,
        sparse_params=params,
    )
    assert isinstance(attn, DeepseekV4TrtllmAttention)
    return attn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("extra_width", [0, 17, 512])
@torch.inference_mode()
def test_native_dual_pool(heads, extra_width, monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

    calls = []
    original = FallbackFmha.forward

    def record(attn, *args, **kwargs):
        calls.append(attn.attn.sparse_params.algorithm)
        return original(attn, *args, **kwargs)

    monkeypatch.setattr(FallbackFmha, "forward", record)
    torch.manual_seed(451)
    attn = _backend(heads)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(3, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(3, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 2:] = False
    swa_valid[1] = False
    extra = (
        torch.randn(3, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(3, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[0, 1::2] = False  # holes before later valid entries
        extra_valid[1] = False
    sink = torch.randn(heads, device="cuda")
    out = attn.forward_selected(q, swa, extra, swa_valid, extra_valid, sink)
    torch.cuda.synchronize()
    assert calls and set(calls) == {"csa2"}
    torch.testing.assert_close(
        out, _reference(q, swa, extra, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_trtllm_graph_replay_resets_sparse_state():
    attn = _backend(64)
    q = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    extra = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    extra_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    sink = torch.zeros(64, device="cuda")

    def run():
        return attn.forward_selected(q, swa, extra, swa_valid, extra_valid, sink)

    for _ in range(3):
        run()
    frame = attn._prepared[128]
    pointers = frame.pool_pointers.clone()
    workspace_ptr = frame.workspace.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    # An eager larger batch after capture must not invalidate captured storage.
    attn.forward_selected(
        q.repeat(8, 1, 1),
        swa.repeat(8, 1, 1),
        extra.repeat(8, 1, 1),
        swa_valid.repeat(8, 1),
        extra_valid.repeat(8, 1),
        sink,
    )
    for width in (17, 0, 3, 17):
        extra_valid.zero_()
        extra_valid[:, :width] = True
        swa_valid[0, 1:] = width != 0
        q.mul_(-1)
        extra.mul_(-1)
        graph.replay()
        torch.testing.assert_close(
            output, _reference(q, swa, extra, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
        )
        torch.testing.assert_close(frame.pool_pointers, pointers, atol=0, rtol=0)
        assert frame.workspace.data_ptr() == workspace_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_controller_multiple_tiles_and_reuse():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (
        CSA2Batch,
        CSA2Cache,
        CSA2Routing,
        DeepseekV41SparseAttention,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._utils import is_sm_100f

    if not is_sm_100f():
        pytest.skip("TRTLLM dynamic sparse MLA requires SM100-family")
    torch.manual_seed(419)
    layout = CSA2Layout((1, 1, 1), (0,), (0, 2), index_topk=4, window_size=4)
    count, heads = 19, 8  # one full tile followed by a partial tile
    cache = CSA2Cache.allocate(layout, count, {0: 6}, torch.device("cuda"))
    reference_cache = CSA2Cache.allocate(layout, count, {0: 6}, torch.device("cuda"))
    positions = torch.arange(count, device="cuda")
    swa_indices = positions[:, None] - torch.arange(4, device="cuda")[None, :]
    batch = CSA2Batch(
        swa_indices.clamp_min(-1),
        positions,
        torch.arange(6, device="cuda").expand(count, -1),
        positions.remainder(6) + 1,
        torch.arange(6, device="cuda"),
    )
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
    index_q = torch.randn(count, 2, 128, device="cuda", dtype=torch.bfloat16)
    index_weights = torch.ones(count, 2, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda")
    routing, reference_routing = CSA2Routing(), CSA2Routing()
    for layer in range(3):
        kwargs = {}
        if layer != 1:
            kwargs.update(index_q=index_q * (-1 if layer else 1), index_weights=index_weights)
        if layer == 0:
            kwargs.update(main_kv=main, index_k=index_k)
        actual = DeepseekV41SparseAttention(layout, layer, compute_backend="auto")
        expected = DeepseekV41SparseAttention(layout, layer, compute_backend="torch")
        assert actual.compute_backend == "trtllm"
        output = actual.forward(q, swa, sink, cache, batch, routing, **kwargs)
        reference = expected.forward(
            q, swa, sink, reference_cache, batch, reference_routing, **kwargs
        )
        torch.testing.assert_close(output, reference, atol=0.03, rtol=0.03)
        frame = actual.trtllm_backend._prepared[128]
        assert frame.host_total_kv_lens.tolist() == [0, 3 * 256]
        assert frame.warmed_query_counts == {16, 3}
    torch.testing.assert_close(routing.indices[2], reference_routing.indices[2])
