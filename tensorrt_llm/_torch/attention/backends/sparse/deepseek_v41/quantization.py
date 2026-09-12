# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 cache encodings, including the quantized RoPE channels.

Rows contain packed values followed by scale bytes. Main KV uses E2M1/E4M3
with groups of 16; index Q/K use E2M1/UE8M0 with groups of 32. Neither format
uses the tensor-wide scaling factor of the generic NVFP4 weight quantizer.
"""

from __future__ import annotations

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
