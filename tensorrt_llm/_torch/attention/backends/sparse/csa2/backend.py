# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 sparse computation over externally allocated cache pools.

The executor owns page allocation and supplies pool-relative indices. Shared
routing is scoped to one packed forward, never to a process-global variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .params import CSA2Layer, CSA2Layout, CSA2Mode
from .quantization import CacheFormat, pack_rows, row_bytes, unpack_rows
from .selection import index_scores, select_candidate_positions, select_topk_positions


@dataclass
class CSA2Cache:
    """Physical uint8 pools. Allocate global/index pools only for KV sources."""

    swa: dict[int, torch.Tensor]
    main: dict[int, torch.Tensor]
    index: dict[int, torch.Tensor]

    @classmethod
    def allocate(
        cls,
        layout: CSA2Layout,
        swa_rows: int,
        global_rows: dict[int, int],
        device: torch.device,
        head_dim: int = 512,
        index_head_dim: int = 128,
    ) -> CSA2Cache:
        """Allocate owner pools from the executor's physical row capacities.

        SWA capacity includes staging space for the current packed prefill,
        so writing a long chunk cannot overwrite the history of earlier queries.
        The executor supplies distinct write slots, then retires staging rows.
        """
        if swa_rows <= 0 or set(global_rows) != set(layout.kv_source_layer_ids):
            raise ValueError("CSA2 cache capacities must name exactly the configured owners")
        if any(rows <= 0 for rows in global_rows.values()):
            raise ValueError("CSA2 pools require at least one padding row")
        cache = cls(
            {
                i: torch.empty(
                    swa_rows, row_bytes(head_dim, "swa"), dtype=torch.uint8, device=device
                )
                for i in range(len(layout.compress_ratios))
            },
            {
                i: torch.empty(rows, row_bytes(head_dim, "main"), dtype=torch.uint8, device=device)
                for i, rows in global_rows.items()
            },
            {
                i: torch.empty(
                    rows, row_bytes(index_head_dim, "index"), dtype=torch.uint8, device=device
                )
                for i, rows in global_rows.items()
            },
        )
        cache.validate(layout)
        return cache

    @staticmethod
    def global_bytes_per_token(
        layout: CSA2Layout, head_dim: int = 512, index_head_dim: int = 128
    ) -> int:
        """Persistent cache budget; excludes bounded SWA and compressor state."""
        owner_bytes = row_bytes(head_dim, "main") + row_bytes(index_head_dim, "index")
        return sum(owner_bytes // layout.compress_ratios[i] for i in layout.kv_source_layer_ids)

    def validate(self, layout: CSA2Layout) -> None:
        if set(self.swa) != set(range(len(layout.compress_ratios))):
            raise ValueError("Every CSA2 layer requires a private SWA pool")
        owners = set(layout.kv_source_layer_ids)
        if set(self.main) != owners or set(self.index) != owners:
            raise ValueError("Global and index pools must be allocated exactly once per KV source")


@dataclass
class CSA2Routing:
    """Per-forward logical indices; construct afresh for each packed batch.

    CUDA Graph capture records producers and consumers on the same stream.
    Each replay recomputes the captured tensors; Python dictionaries are only
    traversed during capture. Eager forwards must not recycle this object.
    """

    indices: dict[int, torch.Tensor] = field(default_factory=dict)
    candidates: dict[int, torch.Tensor] = field(default_factory=dict)
    _last_layer: int = -1

    def enter(self, layer: CSA2Layer) -> None:
        if layer.layer_idx <= self._last_layer:
            raise ValueError("CSA2 routing cannot be reused across forwards or reordered layers")
        self._last_layer = layer.layer_idx


@dataclass(frozen=True)
class CSA2GlobalPages:
    """Source pool page table without a tokens-by-context mapping allocation.

    page_table is [requests, logical_pages], request_ids is [query_tokens].
    Page IDs are relative to this owner's pool; -1 denotes an absent page.
    ``tokens_per_page`` counts global entries, after ratio-two compression.
    """

    page_table: torch.Tensor
    request_ids: torch.Tensor
    tokens_per_page: int
    max_positions: int

    def __post_init__(self) -> None:
        if self.page_table.ndim != 2 or self.request_ids.ndim != 1:
            raise ValueError("CSA2 page table/request IDs have invalid ranks")
        if self.tokens_per_page <= 0 or self.max_positions < 0:
            raise ValueError("CSA2 page capacity must be positive and context bound nonnegative")

    def resolve(self, start: int, end: int, logical: torch.Tensor | None = None) -> torch.Tensor:
        requests = self.request_ids[start:end].long()
        if logical is None:
            logical = torch.arange(self.max_positions, device=self.page_table.device)
            logical = logical.expand(requests.shape[0], -1)
        logical = logical.long()
        if self.page_table.shape[0] == 0 or self.page_table.shape[1] == 0:
            return torch.full_like(logical, -1)
        pages = logical.clamp_min(0) // self.tokens_per_page
        valid = (logical >= 0) & (logical < self.max_positions) & (pages < self.page_table.shape[1])
        valid &= ((requests >= 0) & (requests < self.page_table.shape[0]))[:, None]
        physical = self.page_table[
            requests.clamp(0, self.page_table.shape[0] - 1)[:, None],
            pages.clamp(max=self.page_table.shape[1] - 1),
        ]
        slots = physical.long() * self.tokens_per_page + logical % self.tokens_per_page
        return torch.where(valid & (physical >= 0), slots, -1)


@dataclass(frozen=True)
class CSA2Batch:
    """Device metadata for one packed layer execution.

    swa_indices: [tokens, window] pool-relative row indices, -1 for padding.
    swa_write_slots: [tokens], distinct valid pool rows (including dummy rows).
    global_slots: [tokens, max_global_positions], maps logical global positions
        to the owner's pool rows. Missing entries are -1.
    visible_lengths: [tokens], floor((absolute_position + 1) / ratio).
    main_write_slots: [new_latents], distinct valid rows for completed groups.

    A consumer resolves its logical routing against its own view of the source
    pool. Prefix reuse and request reordering therefore cannot reuse stale
    physical indices. Main/index pools use the same physical slot numbering.
    """

    swa_indices: torch.Tensor
    swa_write_slots: torch.Tensor
    global_slots: torch.Tensor | CSA2GlobalPages
    visible_lengths: torch.Tensor
    main_write_slots: torch.Tensor

    def global_slot_tile(
        self, start: int, end: int, logical: torch.Tensor | None = None
    ) -> torch.Tensor:
        if isinstance(self.global_slots, CSA2GlobalPages):
            return self.global_slots.resolve(start, end, logical)
        slots = self.global_slots[start:end]
        return slots if logical is None else _resolve_slots(slots, logical)


def _resolve_slots(global_slots: torch.Tensor, logical: torch.Tensor) -> torch.Tensor:
    if global_slots.shape[1] == 0:
        return torch.full_like(logical, -1, dtype=torch.int64)
    valid = (logical >= 0) & (logical < global_slots.shape[1])
    slots = global_slots.gather(1, logical.long().clamp(0, global_slots.shape[1] - 1))
    return torch.where(valid, slots, -1).long()


def _gather_rows(
    pool: torch.Tensor, slots: torch.Tensor, dim: int, cache_format: CacheFormat
) -> torch.Tensor:
    # Every pool includes at least one allocated padding row. Mask invalid
    # values after gathering so uninitialized padding never enters a matmul.
    rows = pool[slots.clamp_min(0).long()]
    values = unpack_rows(rows, dim, cache_format)
    return torch.where((slots >= 0).unsqueeze(-1), values, 0)


def sparse_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    valid: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """One softmax over selected SWA/global rows and the zero-valued sink.

    Q [tokens, heads, dim], gathered KV [tokens, selected, dim], valid
    [tokens, selected]. BF16 inputs accumulate scores and softmax in FP32.
    Query tiling is owned by the caller to bound gathered-KV workspace.
    """
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * scale
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
    sink_scores = sink.float().view(1, -1, 1).expand(q.shape[0], -1, -1)
    scores = torch.cat((scores, sink_scores), dim=-1)
    probs = torch.softmax(scores, dim=-1)[..., :-1]
    return torch.einsum("qhk,qkd->qhd", probs, kv.float()).to(q.dtype)


class DeepseekV41SparseAttention:
    """CSA2 computation with Full, Reindex and Reuse execution.

    Inputs have already undergone projection/RoPE, without V4's Q-B head norm.
    The module adapter must publish index K from the pre-RoPE compressed latent
    before rotating/quantizing main KV. This backend owns quantization and cache
    writes. Bounded gathered workspaces also support cached/chunked prefill.
    """

    def __init__(
        self, layout: CSA2Layout, layer_idx: int, query_tile: int = 16, use_flash_mla: bool = False
    ) -> None:
        if query_tile <= 0:
            raise ValueError("CSA2 query tile must be positive")
        self.layout = layout
        self.layer = layout.layer(layer_idx)
        self.query_tile = query_tile
        if use_flash_mla:
            from .fmha import run_flash_mla

            self.attention = run_flash_mla
        else:
            self.attention = sparse_attention

    def forward(
        self,
        q: torch.Tensor,
        swa_kv: torch.Tensor,
        sink: torch.Tensor,
        cache: CSA2Cache,
        batch: CSA2Batch,
        routing: CSA2Routing,
        *,
        index_q: torch.Tensor | None = None,
        index_weights: torch.Tensor | None = None,
        main_kv: torch.Tensor | None = None,
        index_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer, layout = self.layer, self.layout
        routing.enter(layer)
        cache.swa[layer.layer_idx].index_copy_(
            0, batch.swa_write_slots.long(), pack_rows(swa_kv, "swa")
        )
        if layer.mode == CSA2Mode.FULL:
            if main_kv is None or index_k is None:
                raise ValueError(
                    "Full mode must supply new main and index rows, including empty rows"
                )
            cache.main[layer.kv_source].index_copy_(
                0, batch.main_write_slots.long(), pack_rows(main_kv, "main")
            )
            cache.index[layer.kv_source].index_copy_(
                0, batch.main_write_slots.long(), pack_rows(index_k, "index")
            )
        elif main_kv is not None or index_k is not None:
            raise ValueError("Only Full mode may write shared main/index caches")

        if layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            if index_q is None or index_weights is None:
                raise ValueError("Full/Reindex mode requires index queries and weights")
            index_q = unpack_rows(pack_rows(index_q, "index"), index_q.shape[-1], "index")
            selections = []
            candidates = []
            for start in range(0, q.shape[0], self.query_tile):
                end = start + self.query_tile
                visible = batch.visible_lengths[start:end]
                if layer.candidate_source is not None and layer.layer_idx != layer.candidate_source:
                    if layer.candidate_source not in routing.candidates:
                        raise ValueError("CSA2 candidate source did not run in this forward")
                    positions = routing.candidates[layer.candidate_source][start:end]
                    slots = batch.global_slot_tile(start, end, positions)
                else:
                    slots = batch.global_slot_tile(start, end)
                    positions = torch.arange(slots.shape[-1], device=q.device)
                keys = _gather_rows(cache.index[layer.kv_source], slots, index_q.shape[-1], "index")
                scores = index_scores(index_q[start:end], keys, index_weights[start:end])
                scores = scores.masked_fill(slots < 0, -torch.inf)
                if layer.candidate_source == layer.layer_idx:
                    candidates.append(
                        select_candidate_positions(
                            scores,
                            visible,
                            layout.candidate_topk_blocks,
                            layout.candidate_block_size,
                        )
                    )
                selections.append(
                    select_topk_positions(scores, positions, visible, layout.index_topk)
                )
            indices = (
                torch.cat(selections)
                if selections
                else torch.empty((0, layout.index_topk), dtype=torch.int32, device=q.device)
            )
            routing.indices[layer.layer_idx] = indices
            if candidates:
                routing.candidates[layer.layer_idx] = torch.cat(candidates)
        elif layer.mode == CSA2Mode.REUSE:
            if layer.index_source not in routing.indices:
                raise ValueError("CSA2 index source did not run in this forward")
            indices = routing.indices[layer.index_source]
            if indices.shape[0] != q.shape[0]:
                raise ValueError("CSA2 Reuse rows must match their source's packed query order")
        else:
            indices = None

        output = torch.empty_like(q)
        for start in range(0, q.shape[0], self.query_tile):
            end = start + self.query_tile
            swa_slots = batch.swa_indices[start:end]
            selected = _gather_rows(cache.swa[layer.layer_idx], swa_slots, q.shape[-1], "swa")
            valid = swa_slots >= 0
            if indices is not None:
                logical = indices[start:end]
                slots = batch.global_slot_tile(start, end, logical)
                global_kv = _gather_rows(cache.main[layer.kv_source], slots, q.shape[-1], "main")
                selected = torch.cat((selected, global_kv), dim=1)
                valid = torch.cat(
                    (valid, (slots >= 0) & (logical < batch.visible_lengths[start:end, None])),
                    dim=1,
                )
            output[start:end] = self.attention(
                q[start:end], selected, valid, sink, q.shape[-1] ** -0.5
            )
        return output
