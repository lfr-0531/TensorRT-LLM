# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashMLA and FlashInfer compute libraries for the CSA2 sparse contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import PredefinedAttentionMask

from .interface import Fmha

if TYPE_CHECKING:
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

_HEAD_DIM = 512
_SWA_TILE = 128


class CSA2Fmha(Fmha):
    @classmethod
    def _is_available(cls, attn) -> bool:
        return getattr(
            attn.sparse_params, "algorithm", None
        ) == "csa2" and attn.compute_backend in ("flash_mla", "flashinfer")

    def __init__(self, attn):
        super().__init__(attn)
        self._flashinfer = FlashInferCSA2() if attn.compute_backend == "flashinfer" else None

    def _is_supported(self, q, k, v, metadata, forward_args, *, phase=None) -> bool:
        # The selected-row providers emit unquantized BF16 and consume the
        # sparsity mask from prediction. Other mask/output formats need a
        # provider that explicitly implements those contracts.
        return (
            forward_args.attention_mask
            in (PredefinedAttentionMask.CAUSAL, PredefinedAttentionMask.FULL)
            and forward_args.attention_mask_data is None
            and forward_args.out_scale is None
            and forward_args.out_scale_sf is None
            and forward_args.output_sf is None
            and (forward_args.output is None or forward_args.output.dtype == q.dtype)
        )

    def forward(self, q, k, v, metadata, forward_args) -> None:
        """Compute non-native providers after the inherited forward ran prediction."""
        # FmhaManager can reuse a selection when scale presence changes with
        # the same output dtype, so validate this contract on cache hits too.
        if not self._is_supported(q, k, v, metadata, forward_args):
            raise RuntimeError(
                "CSA2 FlashMLA/FlashInfer do not support the requested mask/output format"
            )
        query = q.view(metadata.num_tokens, self.attn.num_heads, _HEAD_DIM)
        kv = metadata.swa_pool
        if metadata.num_sparse_topk > _SWA_TILE:
            kv = torch.cat((kv, metadata.extra_pool), dim=1)
        valid = (
            torch.arange(metadata.num_sparse_topk, device=q.device)[None, :]
            < metadata.prepared_lens[:, None]
        )
        sink = forward_args.attention_sinks
        if sink is None:
            raise ValueError("CSA2 requires an attention sink")
        scale = 1.0 / (self.attn.q_scaling * math.sqrt(_HEAD_DIM))
        if self.attn.compute_backend == "flash_mla":
            result = run_flash_mla(query, kv, valid, sink, scale)
        elif self.attn.compute_backend == "flashinfer":
            result = self._flashinfer(query, kv, valid, sink, scale)
        else:
            raise RuntimeError("Native CSA2 computation belongs to the TRTLLM fallback provider")
        forward_args.output.copy_(result.flatten(1))


def run_flash_mla(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run one combined SWA/global selection with a single attention sink.

    Gathered KV is [queries, selected, 512]. Indices point into a temporary
    flattened pool, as in the V4 BF16 FlashMLA path; FP4 cache bytes are never
    passed to an FP8-only kernel. This uses the repository-pinned FlashMLA API,
    which accepts attention sinks directly.
    """
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd

    tokens, heads, dim = q.shape
    if dim != 512 or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("CSA2 FlashMLA requires BF16 Q/KV with head dimension 512")
    if heads <= 0 or heads > 128:
        raise ValueError("CSA2 FlashMLA supports up to 128 query heads")
    if tokens == 0:
        return torch.empty_like(q)
    padded_heads = 64 if heads <= 64 else 128
    q_padded = F.pad(q, (0, 0, 0, padded_heads - heads)).contiguous()
    sink_padded = F.pad(sink.float(), (0, padded_heads - heads), value=torch.inf)
    indices = torch.arange(valid.numel(), device=q.device, dtype=torch.int32).reshape_as(valid)
    indices = torch.where(valid, indices, -1)
    indices = F.pad(indices, (0, -indices.shape[1] % 128), value=-1).unsqueeze(1)
    output, _, _ = flash_mla_sparse_fwd(
        q_padded,
        kv.reshape(-1, 1, dim).contiguous(),
        indices,
        scale,
        d_v=dim,
        attn_sink=sink_padded,
    )
    return output[:, :heads]


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
            # Geometry is fixed per cached plan. FA2's graph-mode scheduler
            # pads CTAs but does not mask that padding with disable_split_kv
            # in FlashInfer 0.6.18. Use the fixed (non-padded) scheduler; run()
            # remains capturable because these plans and buffers never resize.
            wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self._workspaces[q.device],
                kv_layout="NHD",
                backend="fa2",
                use_cuda_graph=False,
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
            # plan() packs the bool mask into its own persistent byte buffer.
            # Update that exact buffer on replay, not the original bool mask.
            plan = _Plan(wrapper, wrapper._custom_mask_buf)
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
