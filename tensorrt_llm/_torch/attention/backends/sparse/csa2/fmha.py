# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reuse the V4 FlashMLA sparse BF16 computation after CSA2 cache gathering."""

import torch
import torch.nn.functional as F


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
