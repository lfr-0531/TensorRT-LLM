# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 logical selections, independent of physical cache page addresses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def select_candidate_positions(
    scores: torch.Tensor,
    visible_lengths: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Return bounded candidate positions [queries, blocks * block_size], int64.

    Scores are float32 [queries, positions]. Unreachable positions are masked
    before block reduction. The newest visible block is always retained, as
    required by the trained hierarchical indexer. Invalid slots contain -1.
    """
    if scores.ndim != 2 or visible_lengths.shape != scores.shape[:1]:
        raise ValueError("Candidate scores and visible lengths must have matching query rows")
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("Candidate block count and size must be positive")
    width = scores.shape[-1]
    positions = torch.arange(width, device=scores.device)
    reachable = positions[None, :] < visible_lengths[:, None]
    scores = scores.masked_fill(~reachable, -torch.inf)
    padded = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    blocks = padded.reshape(scores.shape[0], padded.shape[-1] // block_size, block_size).amax(-1)
    block_ids = torch.arange(blocks.shape[-1], device=scores.device)
    latest = (visible_lengths - 1) // block_size
    blocks = blocks.masked_fill(block_ids[None, :] == latest[:, None], torch.inf)
    values, selected = blocks.topk(min(topk_blocks, blocks.shape[-1]), dim=-1, sorted=False)
    candidates = selected[:, :, None] * block_size + torch.arange(block_size, device=scores.device)
    valid = (values[:, :, None] > -torch.inf) & (candidates < visible_lengths[:, None, None])
    valid &= candidates < width
    return torch.where(valid, candidates, -1).flatten(1)


def select_topk_positions(
    scores: torch.Tensor,
    positions: torch.Tensor,
    visible_lengths: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select sorted logical positions; pad unreachable selections with -1.

    ``positions`` is either [positions] or [queries, candidates]. Selection
    never publishes candidate-array offsets or cache addresses to Reuse layers.
    """
    if scores.ndim != 2 or visible_lengths.shape != scores.shape[:1]:
        raise ValueError("Top-k scores and visible lengths must have matching query rows")
    if topk <= 0 or positions.shape[-1] != scores.shape[-1]:
        raise ValueError("Invalid top-k size or candidate positions")
    positions = positions.long().expand_as(scores)
    valid = (positions >= 0) & (positions < visible_lengths[:, None])
    scores = scores.masked_fill(~valid, -torch.inf)
    values, offsets = scores.topk(min(topk, scores.shape[-1]), dim=-1, sorted=False)
    selected = positions.gather(1, offsets)
    # Sort invalid entries last. In particular, an unreachable candidate with a
    # low logical index must not become a valid selection after the gather.
    sentinel = torch.iinfo(torch.int64).max
    selected = torch.where(values > -torch.inf, selected, sentinel).sort(dim=-1).values
    selected = torch.where(selected == sentinel, -1, selected).to(torch.int32)
    return F.pad(selected, (0, topk - selected.shape[-1]), value=-1)


def index_scores(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Rectified indexer scores with already-scaled per-head weights.

    Q is [queries, heads, dim], K is [positions, dim] or
    [queries, candidates, dim], weights is [queries, heads]. The gathered-K
    form bounds Reindex work by candidate count, independently of context size.
    """
    if k.ndim == 2:
        dots = torch.einsum("qhd,kd->qhk", q, k)
    elif k.ndim == 3:
        dots = torch.einsum("qhd,qkd->qhk", q, k)
    else:
        raise ValueError("Indexer keys must be shared or gathered per query")
    return (dots.relu() * weights.unsqueeze(-1)).sum(1).float()
