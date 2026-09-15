# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 cache encodings, including the quantized RoPE channels.

Rows contain packed values followed by scale bytes. Main KV uses E2M1/E4M3
with groups of 16; index Q/K use E2M1/UE8M0 with groups of 32. Neither format
uses the tensor-wide scaling factor of the generic NVFP4 weight quantizer.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import torch

CacheFormat = Literal["main", "index", "swa"]


def row_bytes(head_dim: int, cache_format: CacheFormat) -> int:
    group = 16 if cache_format == "main" else 32
    if cache_format not in ("main", "index", "swa") or head_dim % group:
        raise ValueError("Invalid CSA2 cache format or head dimension")
    return (head_dim if cache_format == "swa" else head_dim // 2) + head_dim // group


def _fp4_levels(device: torch.device) -> torch.Tensor:
    codes = torch.arange(8, device=device)
    return torch.where(
        codes < 4,
        codes.float() * 0.5,
        torch.exp2(((codes - 2) // 2).float()) * (1 + 0.5 * (codes % 2)),
    )


def pack_rows(x: torch.Tensor, cache_format: CacheFormat) -> torch.Tensor:
    """Encode floating rows [..., head_dim] as uint8 [..., row_bytes]."""
    dim = x.shape[-1]
    row_bytes(dim, cache_format)
    group = 16 if cache_format == "main" else 32
    blocks = x.float().reshape(*x.shape[:-1], dim // group, group)
    maximum = blocks.abs().amax(-1)
    if cache_format == "main":
        scales = (maximum.clamp(min=6 * 2**-9) / 6).to(torch.float8_e4m3fn)
        scale_bytes = scales.view(torch.uint8)
        scales = scales.float()
    else:
        divisor = 448 if cache_format == "swa" else 6
        floor = 1e-4 if cache_format == "swa" else 6 * 2**-126
        exponents = torch.ceil(torch.log2(maximum.clamp(min=floor) / divisor))
        scale_bytes = (exponents + 127).to(torch.uint8)
        scales = torch.exp2(exponents)
    scaled = blocks / scales.unsqueeze(-1)
    if cache_format == "swa":
        values = scaled.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        values = values.reshape(*x.shape[:-1], dim)
    else:
        # E2M1 round-to-nearest-even, including midpoint ties. Preserve the
        # sign bit of zero, matching the native packed representation.
        levels = _fp4_levels(x.device)
        midpoints = (levels[:-1] + levels[1:]) * 0.5
        magnitude = scaled.abs().clamp(max=6).contiguous()
        codes = torch.bucketize(magnitude, midpoints)
        tie = magnitude == midpoints[codes.clamp(max=6)]
        codes += (tie & (codes % 2 == 1)).long()
        codes = (codes | (torch.signbit(scaled).long() << 3)).to(torch.uint8)
        codes = codes.reshape(*x.shape[:-1], dim)
        values = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return torch.cat((values, scale_bytes), dim=-1)


def unpack_rows(
    rows: torch.Tensor,
    head_dim: int,
    cache_format: CacheFormat,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode uint8 rows after gathering from a paged or contiguous pool."""
    if rows.dtype != torch.uint8 or rows.shape[-1] != row_bytes(head_dim, cache_format):
        raise ValueError("CSA2 packed cache has the wrong row shape or dtype")
    data_bytes = head_dim if cache_format == "swa" else head_dim // 2
    data = rows[..., :data_bytes].contiguous()
    scale_bytes = rows[..., data_bytes:].contiguous()
    group = 16 if cache_format == "main" else 32
    if cache_format == "main":
        scales = scale_bytes.view(torch.float8_e4m3fn).float()
    else:
        scales = torch.exp2(scale_bytes.float() - 127)
    if cache_format == "swa":
        values = data.view(torch.float8_e4m3fn).float()
    else:
        codes = torch.stack((data & 15, data >> 4), dim=-1).flatten(-2).long()
        levels = _fp4_levels(rows.device)
        values = levels[codes & 7] * torch.where(codes & 8 != 0, -1, 1)
    values = values.reshape(*rows.shape[:-1], head_dim // group, group)
    return (values * scales.unsqueeze(-1)).reshape(*rows.shape[:-1], head_dim).to(dtype)


def gather_rows(
    pool: torch.Tensor, slots: torch.Tensor, dim: int, cache_format: CacheFormat
) -> torch.Tensor:
    """Gather/dequantize rows; invalid physical slots yield zero BF16 rows."""
    if pool.ndim != 2 or pool.dtype != torch.uint8 or pool.shape[1] != row_bytes(dim, cache_format):
        raise ValueError("CSA2 gathered cache has the wrong row shape or dtype")
    if slots.dtype not in (torch.int32, torch.int64) or slots.device != pool.device:
        raise ValueError("CSA2 gather slots must be integer tensors on the pool device")
    if (
        pool.is_cuda
        and pool.stride(1) == 1
        and slots.ndim in (1, 2)
        and dim > 0
        and _fused_gather_supported(pool.device.index)
    ):
        from .kernel import gather_dequant_rows

        return gather_dequant_rows(pool, slots, dim, cache_format)
    valid = (slots >= 0) & (slots < pool.shape[0])
    if pool.shape[0] == 0:
        return torch.zeros((*slots.shape, dim), dtype=torch.bfloat16, device=pool.device)
    rows = pool[torch.where(valid, slots, 0).long()]
    values = unpack_rows(rows, dim, cache_format)
    return torch.where(valid.unsqueeze(-1), values, 0)


@lru_cache(maxsize=None)
def _fused_gather_supported(device_index: int) -> bool:
    # Retain the reference route on other architectures until validated there.
    return torch.cuda.get_device_capability(device_index)[0] == 10


@lru_cache(maxsize=None)
def _fused_store_supported(device_index: int) -> bool:
    major, _ = torch.cuda.get_device_capability(device_index)
    # Exact-byte parity (including nonfinite values) is validated on SM100.
    return major == 10


def store_rows(
    pool: torch.Tensor, slots: torch.Tensor, values: torch.Tensor, cache_format: CacheFormat
) -> None:
    """Publish quantized rows into a packed, possibly strided cache view."""
    if values.ndim != 2 or slots.ndim != 1 or pool.ndim != 2:
        raise ValueError("CSA2 publication requires row matrices and one-dimensional slots")
    if slots.numel() != values.shape[0] or pool.shape[1] != row_bytes(
        values.shape[1], cache_format
    ):
        raise ValueError("CSA2 publication requires one slot per packed row")
    if (
        values.is_cuda
        and values.dtype == torch.bfloat16
        and pool.device == values.device
        and slots.device == values.device
        and pool.dtype == torch.uint8
        and pool.stride(1) == 1
        and values.stride(1) == 1
        and _fused_store_supported(values.device.index)
    ):
        from .kernel import quantize_scatter_rows

        quantize_scatter_rows(pool, slots, values, cache_format)
        return
    packed = pack_rows(values, cache_format)
    if slots.is_cuda:
        from .kernel import scatter_packed_rows

        scatter_packed_rows(pool, slots, packed)
    else:
        valid = (slots >= 0) & (slots < pool.shape[0])
        pool.index_copy_(0, slots[valid].long(), packed[valid])
