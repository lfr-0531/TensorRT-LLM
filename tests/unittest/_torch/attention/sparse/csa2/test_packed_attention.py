# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct packed-cache attention parity and replay with changing selections."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import packed_sparse_attention
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows, unpack_rows


def _inputs(main_count=73, dtype=torch.int64):
    torch.manual_seed(7401)
    q = torch.randn(2, 32, 512, device="cuda", dtype=torch.bfloat16)
    swa = pack_rows(torch.randn(141, 512, device="cuda", dtype=torch.bfloat16), "swa")
    owner = torch.empty(103, 356, device="cuda", dtype=torch.uint8)
    main = owner[:, :288]
    main.copy_(pack_rows(torch.randn(103, 512, device="cuda", dtype=torch.bfloat16), "main"))
    si = torch.randint(0, 141, (2, 77), device="cuda", dtype=dtype)
    mi = torch.randint(0, 103, (2, main_count), device="cuda", dtype=dtype)
    si[0, 3:] = -1
    if main_count:
        mi[0] = -1
        mi[1, 0] = 103
        if dtype == torch.int64:
            mi[1, 1] = 1 << 40
    sink = torch.randn(32, device="cuda")
    return q, swa, main, si, mi, sink, 512**-0.5


def _reference(q, swa, main, si, mi, sink, scale):
    parts, masks = [], []
    for pool, indices, fmt in ((swa, si, "swa"), (main, mi, "main")):
        valid = (indices >= 0) & (indices < pool.shape[0])
        values = unpack_rows(pool[indices.clamp(0, pool.shape[0] - 1).long()], 512, fmt)
        parts.append(torch.where(valid[..., None], values, 0))
        masks.append(valid)
    kv = torch.cat(parts, 1).float()
    logits = torch.einsum("qhd,qkd->qhk", q.float(), kv) * scale
    logits.masked_fill_(~torch.cat(masks, 1)[:, None, :], float("-inf"))
    probabilities = torch.softmax(
        torch.cat((logits, sink[None, :, None].expand(q.shape[0], -1, -1)), -1), -1
    )[..., :-1]
    return torch.einsum("qhk,qkd->qhd", probabilities, kv).bfloat16()


@pytest.mark.parametrize("main_count", [0, 73])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_packed_attention_parity(main_count, dtype):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args = _inputs(main_count, dtype)
    actual = packed_sparse_attention(*args)
    torch.testing.assert_close(actual, _reference(*args), atol=0.016, rtol=0.016)


def test_packed_attention_graph_refresh():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args = _inputs()
    output = torch.empty_like(args[0])
    packed_sparse_attention(*args, output=output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed_sparse_attention(*args, output=output)
    args[3][0].fill_(-1)
    args[4][0].fill_(-1)
    args[4][1].fill_(0)
    graph.replay()
    torch.testing.assert_close(output, _reference(*args), atol=0.016, rtol=0.016)
    assert torch.count_nonzero(output[0]) == 0


def test_packed_attention_multiple_tiles_per_split():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    q, swa, main, si, mi, sink, scale = _inputs(512)
    q = q.repeat(8, 2, 1)
    si = si.repeat(8, 1)
    mi = mi.repeat(8, 1)
    sink = sink.repeat(2)
    args = (q, swa, main, si, mi, sink, scale)
    torch.testing.assert_close(
        packed_sparse_attention(*args), _reference(*args), atol=0.016, rtol=0.016
    )


def _rope_inputs(heads):
    q, swa, main, si, mi, sink, scale = _inputs()
    if heads == 16:
        q, sink = q[:, :16].contiguous(), sink[:16].contiguous()
    else:
        q, sink = q.repeat(1, 2, 1), sink.repeat(2)
    angles = torch.randn(37, 32, device="cuda")
    cos_sin = torch.stack((angles.cos(), angles.sin()), dim=1).contiguous()
    positions = torch.tensor([3, 19], dtype=torch.int64, device="cuda")
    return (q, swa, main, si, mi, sink, scale), positions, cos_sin


@pytest.mark.parametrize("heads", [16, 64])
def test_packed_inverse_rope_native_parity(heads):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args, positions, cos_sin = _rope_inputs(heads)
    attention = packed_sparse_attention(*args)
    expected = attention.clone()
    torch.ops.trtllm.mla_rope_inplace(
        expected, positions.int(), cos_sin, heads, 448, 64, True, False
    )
    actual = packed_sparse_attention(*args, position_ids=positions, rotary_cos_sin=cos_sin)
    torch.testing.assert_close(actual[..., :448], attention[..., :448], atol=0, rtol=0)
    torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.008)


def test_packed_inverse_rope_graph_positions_and_bounds():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args, positions, cos_sin = _rope_inputs(16)
    attention = packed_sparse_attention(*args)
    output = torch.empty_like(attention)
    packed_sparse_attention(*args, output=output, position_ids=positions, rotary_cos_sin=cos_sin)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed_sparse_attention(
            *args, output=output, position_ids=positions, rotary_cos_sin=cos_sin
        )
    positions.copy_(torch.tensor([17, 2], device="cuda"))
    graph.replay()
    expected = attention.clone()
    torch.ops.trtllm.mla_rope_inplace(expected, positions.int(), cos_sin, 16, 448, 64, True, False)
    torch.testing.assert_close(output, expected, atol=0.002, rtol=0.008)
    # Standalone callers receive a nonfinite tail, never an out-of-bounds read.
    positions.copy_(torch.tensor([-1, cos_sin.shape[0]], device="cuda"))
    graph.replay()
    assert torch.isnan(output[..., 448:]).all()
    torch.testing.assert_close(output[..., :448], attention[..., :448], atol=0, rtol=0)


def test_packed_inverse_rope_rejects_invalid_geometry():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import supports_packed_attention

    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")
    args, positions, cos_sin = _rope_inputs(16)
    assert not supports_packed_attention(*args, position_ids=positions)
    assert not supports_packed_attention(*args, rotary_cos_sin=cos_sin)
    assert not supports_packed_attention(
        *args, position_ids=positions, rotary_cos_sin=cos_sin.bfloat16()
    )
    assert not supports_packed_attention(
        *args, position_ids=positions, rotary_cos_sin=cos_sin[:, :, :16]
    )
