# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact packed-byte parity and graph publication for fused CSA2 cache writes."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import quantize_scatter_rows
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows, row_bytes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _sm100():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Fused CSA2 store is enabled only after SM100-family validation")


def _values(rows, dim, cache_format):
    storage = torch.randn(rows, dim + 17, dtype=torch.bfloat16, device="cuda")
    x = storage[:, :dim]
    pattern = torch.tensor(
        [6, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5],
        dtype=torch.bfloat16,
        device="cuda",
    )
    x[0].copy_(pattern.repeat((dim + 15) // 16)[:dim])
    x[2].zero_()
    x[2, 1::2] = -0.0
    x[4].copy_(pattern.repeat((dim + 15) // 16)[:dim] * 2.0**-12)
    x[6].copy_(pattern.repeat((dim + 15) // 16)[:dim] * 2.0**-126)
    # Include BF16 subnormal input channels and positive/negative zero.
    x[8].fill_(2.0**-133)
    x[8, 1::2].neg_()
    x[10].fill_(6 * 1.0625)
    x[12].fill_(6 * 1.1875)
    x[14].fill_(6 * 1.5 * 2.0**-9)
    x[16].fill_(6 * 464)
    x[18].fill_(6 * 468)
    if cache_format == "swa":
        x[20].fill_(2.0**-30)
    x[22, 0] = float("nan")
    x[24, 0] = float("inf")
    x[26, 0] = -float("inf")
    x[28, 0] = -float("nan")
    return x


def _storage(rows, dim, cache_format):
    width = row_bytes(dim, cache_format)
    stride = 356 if cache_format in ("main", "index") else width + 13
    offset = 288 if cache_format == "index" else 0
    storage = torch.full((rows + 8, stride), 77, dtype=torch.uint8, device="cuda")
    return storage, storage[:, offset : offset + width]


def _expected(storage, pool, slots, x, cache_format):
    expected = storage.clone()
    offset = pool.storage_offset() - storage.storage_offset()
    target = expected[:, offset : offset + pool.shape[1]]
    valid = (slots >= 0) & (slots < pool.shape[0])
    target.index_copy_(0, slots[valid].long(), pack_rows(x, cache_format)[valid])
    return expected


@pytest.mark.parametrize(
    "cache_format,dim",
    [
        ("main", 512),
        ("index", 128),
        ("swa", 512),
        ("main", 48),
        ("index", 96),
        ("swa", 96),
    ],
)
@torch.inference_mode()
def test_fused_store_exact_bytes(cache_format, dim):
    _sm100()
    torch.manual_seed(781)
    rows = 33
    x = _values(rows, dim, cache_format)
    storage, pool = _storage(rows, dim, cache_format)
    slot_storage = torch.empty(rows * 2, dtype=torch.int64, device="cuda")
    slots = slot_storage[::2]
    slots.copy_((torch.arange(rows, device="cuda") * 7) % pool.shape[0])
    slots[1], slots[3], slots[5], slots[7] = 2**40, -1, pool.shape[0], -(2**50)
    expected = _expected(storage, pool, slots, x, cache_format)
    quantize_scatter_rows(pool, slots, x, cache_format)
    torch.testing.assert_close(storage, expected, atol=0, rtol=0)


@pytest.mark.parametrize("cache_format,dim", [("main", 512), ("index", 128), ("swa", 512)])
@torch.inference_mode()
def test_fused_store_graph_changes_padding(cache_format, dim):
    _sm100()
    torch.manual_seed(913)
    x = _values(33, dim, cache_format)
    storage, pool = _storage(33, dim, cache_format)
    slots = torch.arange(33, device="cuda", dtype=torch.int64)
    for _ in range(3):
        quantize_scatter_rows(pool, slots, x, cache_format)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantize_scatter_rows(pool, slots, x, cache_format)
    for valid_count in (33, 0, 1, 17, 33):
        slots.copy_(torch.arange(33, device="cuda"))
        slots[valid_count:] = 2**40
        x.neg_()
        storage.fill_(77)
        expected = _expected(storage, pool, slots, x, cache_format)
        graph.replay()
        torch.testing.assert_close(storage, expected, atol=0, rtol=0)


def _gather_reference(pool, slots, dim, cache_format):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    valid = (slots >= 0) & (slots < pool.shape[0])
    if pool.shape[0] == 0:
        return torch.zeros((*slots.shape, dim), dtype=torch.bfloat16, device=pool.device)
    selected = pool[torch.where(valid, slots, 0).long()]
    decoded = unpack_rows(selected, dim, cache_format)
    return torch.where(valid[..., None], decoded, 0)


def _assert_bf16_decode(actual, expected):
    torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        actual.view(torch.int16)[finite], expected.view(torch.int16)[finite], atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "cache_format,dim,slot_dtype,rank",
    [
        ("main", 512, torch.int64, 2),
        ("index", 128, torch.int32, 1),
        ("swa", 512, torch.int64, 2),
        ("main", 48, torch.int32, 1),
        ("index", 96, torch.int64, 2),
        ("swa", 96, torch.int32, 1),
    ],
)
def test_fused_gather_strided_exact(cache_format, dim, slot_dtype, rank):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows

    _sm100()
    torch.manual_seed(791)
    storage, pool = _storage(33, dim, cache_format)
    pool[:33].copy_(pack_rows(_values(33, dim, cache_format), cache_format))
    # Additional allocated rows are initialized, including padding ownership bytes.
    before = storage.clone()
    backing = torch.empty(66, dtype=slot_dtype, device="cuda")
    flat = backing[::2]
    flat.copy_(torch.arange(33, device="cuda") % 33)
    flat[1], flat[3] = -1, pool.shape[0]
    flat[5] = 2**40 if slot_dtype == torch.int64 else 2**30
    flat[7] = -(2**50) if slot_dtype == torch.int64 else -100
    slots = flat.reshape(3, 11) if rank == 2 else flat
    expected = _gather_reference(pool, slots, dim, cache_format)
    actual = gather_rows(pool, slots, dim, cache_format)
    _assert_bf16_decode(actual, expected)
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


def _fp8_python(code):
    import math

    sign = -1.0 if code & 128 else 1.0
    exponent, mantissa = (code >> 3) & 15, code & 7
    if exponent == 15 and mantissa == 7:
        return math.copysign(float("nan"), sign)
    if exponent == 0:
        return sign * mantissa * 2.0**-9
    return sign * (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


@pytest.mark.parametrize("cache_format,dim", [("main", 512), ("index", 128), ("swa", 512)])
def test_fused_gather_all_scale_bytes(cache_format, dim):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows

    _sm100()
    _, pool = _storage(256, dim, cache_format)
    data_bytes = dim if cache_format == "swa" else dim // 2
    payload = torch.arange(data_bytes, dtype=torch.int64, device="cuda").to(torch.uint8)
    pool[:256, :data_bytes].copy_(payload)
    pool[:256, data_bytes:].copy_(torch.arange(256, device="cuda", dtype=torch.uint8)[:, None])
    slots = torch.arange(256, device="cuda", dtype=torch.int64)
    actual = gather_rows(pool, slots, dim, cache_format)
    _assert_bf16_decode(actual, _gather_reference(pool, slots, dim, cache_format))
    levels = (
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    )
    expected_rows = []
    for scale_byte in range(256):
        scale = _fp8_python(scale_byte) if cache_format == "main" else 2.0 ** (scale_byte - 127)
        # Mirror the FP32 intermediate range before rounding into BF16.
        if cache_format != "main" and scale_byte == 255:
            scale = float("inf")
        values = []
        for channel in range(dim):
            if cache_format == "swa":
                value = _fp8_python(channel % 256)
            else:
                byte = (channel // 2) % 256
                value = levels[(byte >> (channel % 2 * 4)) & 15]
            values.append(value * scale)
        expected_rows.append(values)
    independent = torch.tensor(expected_rows, dtype=torch.float64).float().bfloat16().cuda()
    _assert_bf16_decode(actual, independent)


@pytest.mark.parametrize("cache_format,dim", [("main", 512), ("index", 128), ("swa", 512)])
def test_fused_gather_graph_refresh(cache_format, dim):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows

    _sm100()
    torch.manual_seed(792)
    _, pool = _storage(33, dim, cache_format)
    pool[:33].copy_(pack_rows(_values(33, dim, cache_format), cache_format))
    slots = torch.arange(33, device="cuda", dtype=torch.int64).reshape(3, 11)
    gather_rows(pool, slots, dim, cache_format)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gather_rows(pool, slots, dim, cache_format)
    for active in (0, 17, 33):
        flat = torch.arange(33, device="cuda", dtype=torch.int64)
        slots.copy_(torch.where(flat < active, flat, 2**40).reshape_as(slots))
        pool[:33, 0].bitwise_xor_(8)
        graph.replay()
        _assert_bf16_decode(actual, _gather_reference(pool, slots, dim, cache_format))


def test_fused_gather_empty_and_fallback(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2 import quantization

    _sm100()
    pool = torch.empty((0, 288), dtype=torch.uint8, device="cuda")
    slots = torch.tensor([[-1, 0, 2**40]], dtype=torch.int64, device="cuda")
    assert torch.count_nonzero(quantization.gather_rows(pool, slots, 512, "main")) == 0
    empty = slots[:, :0]
    assert quantization.gather_rows(pool, empty, 512, "main").shape == (1, 0, 512)
    _, pool = _storage(33, 512, "main")
    expected = _gather_reference(pool, slots, 512, "main")
    monkeypatch.setattr(quantization, "_fused_gather_supported", lambda _: False)
    _assert_bf16_decode(quantization.gather_rows(pool, slots, 512, "main"), expected)


def test_staging_excludes_out_of_capacity_rows():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs

    _sm100()
    q = torch.empty((2, 32, 512), dtype=torch.bfloat16, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17)
    swa = pack_rows(torch.randn(3, 512, dtype=torch.bfloat16, device="cuda"), "swa")
    main = pack_rows(torch.randn(3, 512, dtype=torch.bfloat16, device="cuda"), "main")
    swa_slots = torch.tensor([[0, 3, 2**40, -1], [3, -1, -1, -1]], device="cuda")
    main_slots = torch.tensor([[2, 3], [-1, 3]], device="cuda")
    inputs = CSA2BackendForwardArgs(
        swa_pool=swa, swa_indices=swa_slots, main_pool=main, topk_indices=main_slots
    )
    metadata.stage_selected(inputs)
    assert metadata.prepared_lens.tolist() == [2, 1]
    expected = torch.cat(
        (
            _gather_reference(swa, swa_slots[:, :1], 512, "swa"),
            _gather_reference(main, main_slots[:, :1], 512, "main"),
        ),
        dim=1,
    )
    _assert_bf16_decode(metadata.swa_pool[0, :2], expected[0])
    assert torch.count_nonzero(metadata.swa_pool[1, 0]) == 0
