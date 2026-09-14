# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 candidate adaptation of the shared DSA/DeepSeek-V4 indexer workflow."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..dsa.indexer import Indexer
from ..dsa.params import DSAParams
from .params import CSA2ForwardState, CSA2Layout, CSA2Mode
from .quantization import pack_rows


class CSA2Indexer(Indexer):
    """Prepare owner-cache candidates and publish CSA2 logical selections.

    The inherited Indexer owns quantized MQA dispatch, exact Top-K, masking
    and logical output mapping. This subclass supplies the model's shared
    cache inputs and its hierarchical blockmax/latest-block candidate hook.
    Cache writes and Full/Reindex/Reuse layer ordering belong to the backend.
    """

    def __init__(self, layout: CSA2Layout, layer_idx: int, heads: int, head_dim: int) -> None:
        layer = layout.layer(layer_idx)
        if layer.mode not in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            raise ValueError("Only CSA2 Full/Reindex layers construct an indexer")
        super().__init__(
            quant_config=None,
            pos_embd_params=None,
            mla_params=None,
            skip_create_weights_in_init=True,
            sparse_params=DSAParams(
                index_n_heads=heads,
                index_head_dim=head_dim,
                index_topk=layout.index_topk,
                indexer_k_dtype="fp4",
            ),
            dtype=torch.bfloat16,
            layer_idx=layer_idx,
            projection_free=True,
        )
        self.layout = layout

    def _publish_tile(
        self,
        results: dict[int, torch.Tensor],
        value: torch.Tensor,
        start: int,
        total_queries: int,
    ) -> None:
        if start == 0:
            results[self.layer_idx] = torch.empty(
                (total_queries, value.shape[1]), dtype=value.dtype, device=value.device
            )
        target = results.get(self.layer_idx)
        if target is None or target.shape != (total_queries, value.shape[1]):
            raise ValueError("CSA2 selection tiles must retain their full-query shape and order")
        target[start : start + value.shape[0]].copy_(value)

    def _publish_candidates(
        self,
        scores: torch.Tensor,
        visible: torch.Tensor,
        results: dict[int, torch.Tensor],
        start: int,
        total_queries: int,
    ) -> None:
        """CSA2-only block hierarchy; native selection stays in the base class."""
        count, width = scores.shape
        block_size = self.layout.candidate_block_size
        padded = F.pad(scores, (0, -width % block_size), value=-torch.inf)
        blocks = padded.reshape(count, padded.shape[1] // block_size, block_size).amax(-1)
        latest = (visible - 1) // block_size
        block_ids = torch.arange(blocks.shape[1], device=scores.device)
        blocks = blocks.masked_fill(block_ids[None, :] == latest[:, None], torch.inf)
        selected = torch.empty(
            (count, min(self.layout.candidate_topk_blocks, blocks.shape[1])),
            dtype=torch.int32,
            device=scores.device,
        )
        starts = torch.zeros(count, dtype=torch.int32, device=scores.device)
        self.select_prepared_scores(
            blocks, selected, starts, torch.full_like(starts, blocks.shape[1])
        )
        candidates = selected.long()[..., None] * block_size + torch.arange(
            block_size, device=scores.device
        )
        valid = (selected >= 0)[..., None]
        valid = valid & (blocks.gather(1, selected.long().clamp_min(0)) > -torch.inf)[..., None]
        valid = valid & (candidates < visible[:, None, None]) & (candidates < width)
        self._publish_tile(
            results, torch.where(valid, candidates, -1).flatten(1), start, total_queries
        )

    def forward(self, state: CSA2ForwardState, query_start: int, count: int) -> torch.Tensor:
        """Run inherited prediction on a Full/Reindex tile and return logical IDs.

        Projected index Q is [total_queries, heads, head_dim]; per-head weights
        are [total_queries, heads]. Metadata supplies packed-query visibility,
        owner page mappings and full-query candidate/selection storage.
        """
        metadata = state.metadata
        manager = metadata.kv_cache_manager
        if manager is None or state.index_q is None or state.index_weights is None:
            raise ValueError("CSA2 indexer requires an owner cache, queries and head weights")
        total_queries = state.swa_kv.shape[0]
        end = query_start + count
        if query_start < 0 or count < 0 or end > total_queries:
            raise ValueError("CSA2 indexer tile is outside the packed query batch")
        if state.index_q.shape != (total_queries, self.n_heads, self.head_dim):
            raise ValueError("CSA2 index queries must match the layer's full-query geometry")
        if state.index_weights.shape != (total_queries, self.n_heads):
            raise ValueError("CSA2 index weights must match the layer's query rows and heads")
        layer = self.layout.layer(self.layer_idx)
        iq = state.index_q[query_start:end]
        packed_q = pack_rows(iq, "index")
        visible = metadata.csa2_visible_lengths[self.layer_idx][query_start:end]
        if layer.candidate_source is not None and layer.candidate_source != self.layer_idx:
            source = metadata.csa2_candidates.get(layer.candidate_source)
            if source is None or source.ndim != 2 or source.shape[0] != total_queries:
                raise ValueError(
                    "CSA2 candidate source must publish matching full-query rows first"
                )
            positions = source[query_start:end]
            slots = metadata.global_slot_tile(self.layer_idx, query_start, end, positions)
        else:
            slots = metadata.global_slot_tile(self.layer_idx, query_start, end)
            positions = torch.arange(slots.shape[1], device=iq.device).expand(count, -1)
        keys = manager.get_index_buffer(layer.kv_source)[slots.clamp_min(0).long()]
        keys = torch.where((slots >= 0)[..., None], keys, 0)
        logical_positions = torch.where(slots >= 0, positions.long(), -1)
        width = keys.shape[1]
        starts = torch.arange(count, dtype=torch.int32, device=iq.device) * width
        keys = keys.reshape(-1, keys.shape[-1])
        data_width = self.head_dim // 2
        logical = torch.empty((count, self.index_topk), dtype=torch.int32, device=iq.device)

        def publish_candidates(scores: torch.Tensor) -> None:
            self._publish_candidates(
                scores, visible, metadata.csa2_candidates, query_start, total_queries
            )

        self.forward_prepared(
            packed_q[..., :data_width].contiguous().view(torch.int8),
            keys[:, :data_width].contiguous().view(torch.int8),
            keys[:, data_width:].contiguous(),
            state.index_weights[query_start:end].float(),
            starts,
            starts + width,
            logical,
            q_scale=packed_q[..., data_width:].contiguous(),
            logical_positions=logical_positions,
            visible_lengths=visible,
            score_hook=publish_candidates if layer.candidate_source == self.layer_idx else None,
        )
        self._publish_tile(metadata.csa2_indices, logical, query_start, total_queries)
        return logical
