# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer BF16 selected-row attention for SM120/SM121.

DSV4's packed footer path would requantize CSA2 FP4/per-32 FP8 values. FA2's
512D ragged path consumes their decoded BF16 values without that extra loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper


def pack_query_masks(valid: torch.Tensor) -> torch.Tensor:
    """Pack each query's mask separately in FlashInfer's little-endian order."""
    padded = F.pad(valid, (0, -valid.shape[1] % 8))
    bits = padded.reshape(valid.shape[0], padded.shape[1] // 8, 8).to(torch.int32)
    shifts = torch.arange(8, device=valid.device, dtype=torch.int32)
    return (bits << shifts).sum(-1).to(torch.uint8).flatten()


@dataclass
class _Plan:
    wrapper: BatchPrefillWithRaggedKVCacheWrapper
    packed_mask: torch.Tensor


class FlashInferCSA2:
    """Persistent FA2 plans; planning is always outside CUDA Graph capture."""

    def __init__(self) -> None:
        self._plans: dict[tuple[int, int, int, torch.device, float], _Plan] = {}
        self._workspaces: dict[torch.device, torch.Tensor] = {}

    def __call__(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        valid: torch.Tensor,
        sink: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        with torch.cuda.device(q.device):
            return self._forward(q, kv, valid, sink, scale)

    def _forward(self, q, kv, valid, sink, scale):
        from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

        count, heads, dim = q.shape
        width = kv.shape[1]
        if dim != 512 or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
            raise ValueError("CSA2 FlashInfer requires BF16 Q/KV with head dimension 512")
        if count == 0 or width == 0:
            return torch.zeros_like(q)
        key = (count, heads, width, q.device, scale)
        plan = self._plans.get(key)
        if plan is None:
            with torch.cuda.device(q.device):
                capturing = torch.cuda.is_current_stream_capturing()
            if capturing:
                raise RuntimeError("Warm up the CSA2 FlashInfer shape before CUDA Graph capture")
            if q.device not in self._workspaces:
                # Selected-row queries are tiled and split-KV is disabled.
                self._workspaces[q.device] = torch.empty(
                    8 * 1024 * 1024, dtype=torch.uint8, device=q.device
                )
            qo_host = torch.arange(count + 1, dtype=torch.int32, device="cpu")
            kv_host = qo_host * width
            packed_mask = torch.empty(
                count * ((width + 7) // 8), dtype=torch.uint8, device=q.device
            )
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self._workspaces[q.device],
                kv_layout="NHD",
                backend="fa2",
                use_cuda_graph=True,
                qo_indptr_buf=torch.empty(count + 1, dtype=torch.int32, device=q.device),
                kv_indptr_buf=torch.empty(count + 1, dtype=torch.int32, device=q.device),
                custom_mask_buf=packed_mask,
                mask_indptr_buf=torch.empty(count + 1, dtype=torch.int32, device=q.device),
            )
            wrapper.plan(
                qo_host,
                kv_host,
                heads,
                1,
                dim,
                head_dim_vo=dim,
                custom_mask=torch.ones(count * width, dtype=torch.bool, device=q.device),
                causal=False,
                pos_encoding_mode="NONE",
                sm_scale=scale,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
                disable_split_kv=True,
            )
            plan = _Plan(wrapper, packed_mask)
            self._plans[key] = plan
        # Updating the original bool mask would not update the planned wrapper.
        plan.packed_mask.copy_(pack_query_masks(valid))
        rows = torch.where(valid[..., None], kv, 0).reshape(-1, 1, dim).contiguous()
        output, lse = plan.wrapper.run(q.contiguous(), rows, rows, return_lse=True)
        # FA2 returns base-2 LSE. Restore the shared zero-valued sink without
        # requantizing the input cache. Empty selections have zero output.
        correction = torch.sigmoid(lse.float() * math.log(2.0) - sink.float()[None, :])
        output = output.float() * correction[..., None]
        return torch.where(valid.any(-1)[:, None, None], output, 0).to(q.dtype)
