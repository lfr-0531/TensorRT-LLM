# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Masked publication into the manager's strided packed cache views."""

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_packed_rows(
    pool,
    slots,
    rows,
    pool_stride: tl.constexpr,
    row_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    capacity: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row * slot_stride).to(tl.int64)
    if (slot >= 0) & (slot < capacity):
        offsets = tl.arange(0, block)
        values = tl.load(rows + row * row_stride + offsets, offsets < width, 0)
        tl.store(pool + slot * pool_stride + offsets, values, offsets < width)


def scatter_packed_rows(pool: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """Ignore padding slots before any store, including under graph replay."""
    if rows.shape[0] == 0:
        return
    if pool.stride(1) != 1 or rows.stride(1) != 1:
        raise ValueError("CSA2 packed cache columns must be contiguous")
    _scatter_packed_rows[(rows.shape[0],)](
        pool,
        slots,
        rows,
        pool.stride(0),
        rows.stride(0),
        slots.stride(0),
        pool.shape[0],
        rows.shape[1],
        triton.next_power_of_2(rows.shape[1]),
    )
