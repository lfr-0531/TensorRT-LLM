# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 candidate adaptation of the shared DSA/DeepSeek-V4 indexer workflow."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm._utils import get_sm_version

from ..dsa.indexer import Indexer, IndexerChunkInputs, IndexerQueryChunk
from ..dsa.params import DSAParams
from .metadata import CSA2TrtllmMetadata
from .params import CSA2ForwardState, CSA2Layout, CSA2Mode
from .quantization import pack_rows


class CSA2Indexer(Indexer):
    """Prepare owner-cache candidates and publish CSA2 logical selections.

    The inherited Indexer owns quantized MQA dispatch and the shared chunk,
    query-splitting and collective workflow. This subclass adapts owner caches,
    logical selection and hierarchical block candidates using the shared TopK
    module. Cache writes and layer ordering belong to the backend.
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
        self._prepared_topk: dict[tuple[int, bool], TopK] = {}

    def select_prepared_scores(
        self,
        scores: torch.Tensor,
        output_indices: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        *,
        is_prefill: bool = True,
        radix_aux_indices: Optional[torch.Tensor] = None,
        radix_aux_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write row-local Top-K offsets through the shared selection module.

        Auxiliary selection widths support hierarchical indexers; they use
        exact radix selection and never participate in temporal GVR state.
        """
        if scores.shape[0] == 0 or scores.shape[1] == 0:
            return output_indices.fill_(-1)
        topk = output_indices.shape[1]
        if scores.is_cuda and topk == self.index_topk:
            selector = self.top_k
        else:
            key = (topk, scores.is_cuda)
            if key not in self._prepared_topk:
                self._prepared_topk[key] = TopK(
                    topk,
                    prefill_implementation=(
                        TopKImplementation.CUDA_RADIX
                        if scores.is_cuda
                        else TopKImplementation.TORCH
                    ),
                    decode_implementation=(
                        TopKImplementation.CUDA_RADIX
                        if scores.is_cuda
                        else TopKImplementation.TORCH
                    ),
                )
            selector = self._prepared_topk[key]
        if not is_prefill:
            return selector(
                scores,
                output_indices,
                is_prefill=False,
                # The prepared bounds are already in score-column units.
                sequence_lengths=row_ends * selector.compress_ratio,
                scan_lengths=row_ends,
                next_n=1,
                max_seq_len=scores.shape[1],
                radix_aux_indices=radix_aux_indices,
                radix_aux_logits=radix_aux_logits,
            )
        return selector(
            scores, output_indices, is_prefill=True, row_starts=row_starts, row_ends=row_ends
        )

    def forward_prepared(
        self,
        q_data: torch.Tensor,
        k_data: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        q_scale: Optional[torch.Tensor] = None,
        *,
        logical_positions: Optional[torch.Tensor] = None,
        visible_lengths: Optional[torch.Tensor] = None,
        score_hook: Optional[Callable[[torch.Tensor], None]] = None,
        is_prefill: bool = True,
        radix_aux_indices: Optional[torch.Tensor] = None,
        radix_aux_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """CSA2 logical selection over the shared quantized MQA kernels.

        Q is [queries, heads, data_dim], K is [keys, data_dim]. FP4 data
        contains packed E2M1 bytes; Q/K scales contain one int32 word of
        four UE8M0 bytes per 128D vector. FP8 retains the ordinary DSA
        scale contract. Row bounds are int32 [queries] in the shared K axis.
        Output is caller-owned int32 [queries, top_k].

        Without a logical map the output preserves DSA's row-local, unsorted
        offsets and its no-clean logits fast path. With a [queries, candidates]
        logical map, each map starts at its row's K start. Invalid/future
        candidates are masked before the optional hook. The output contains
        sorted logical positions followed by -1 padding. The hook can publish
        hierarchical candidate information, but cannot change request/cache
        ownership or bypass this instance's QK/TopK dispatch.
        """
        count = q_data.shape[0]
        if not is_prefill and logical_positions is None:
            logical_positions = torch.arange(k_data.shape[0], device=q_data.device).expand(
                count, -1
            )
            visible_lengths = row_ends - row_starts
        if output_indices.shape != (count, self.index_topk):
            raise ValueError("Prepared indexer output must match query rows and configured Top-K")
        if score_hook is not None and logical_positions is None:
            raise ValueError("A score hook requires explicit logical candidate positions")
        if logical_positions is not None:
            if logical_positions.ndim != 2 or logical_positions.shape[0] != count:
                raise ValueError("Logical candidates must have one row per query")
            if visible_lengths is None or visible_lengths.shape != (count,):
                raise ValueError("Logical candidates require per-query visibility lengths")
        if count == 0 or k_data.shape[0] == 0:
            if score_hook is not None:
                score_hook(
                    torch.full(
                        logical_positions.shape,
                        -torch.inf,
                        dtype=torch.float32,
                        device=q_data.device,
                    )
                )
            return output_indices.fill_(-1)
        logits = self._call_mqa_logits(
            q_data,
            k_data,
            k_scale,
            weights,
            row_starts,
            row_ends,
            q_scale,
            clean_logits=logical_positions is not None,
        )
        if logical_positions is None:
            return self.select_prepared_scores(logits, output_indices, row_starts, row_ends)
        return self._select_mapped_logits(
            logits,
            row_starts,
            row_ends,
            logical_positions,
            visible_lengths,
            output_indices,
            score_hook,
            is_prefill,
            radix_aux_indices,
            radix_aux_logits,
        )

    def _select_mapped_logits(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        logical_positions: torch.Tensor,
        visible_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        score_hook: Optional[Callable[[torch.Tensor], None]],
        is_prefill: bool = True,
        radix_aux_indices: Optional[torch.Tensor] = None,
        radix_aux_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        width = logical_positions.shape[1]
        offsets = torch.arange(width, device=logits.device)
        columns = row_starts.long()[:, None] + offsets
        in_bounds = (columns >= 0) & (columns < row_ends[:, None]) & (columns < logits.shape[1])
        scores = logits.gather(1, columns.clamp(0, logits.shape[1] - 1))
        valid = in_bounds & (logical_positions >= 0)
        valid &= logical_positions < visible_lengths[:, None]
        scores = scores.masked_fill(~valid, -torch.inf)
        if score_hook is not None:
            score_hook(scores)
        local_starts = torch.zeros_like(row_starts)
        local_ends = (row_ends - row_starts).clamp(min=0, max=width)
        self.select_prepared_scores(
            scores,
            output_indices,
            local_starts,
            local_ends,
            is_prefill=is_prefill,
            radix_aux_indices=radix_aux_indices,
            radix_aux_logits=radix_aux_logits,
        )
        if width == 0:
            return output_indices
        selected_offsets = output_indices.long()
        safe_offsets = selected_offsets.clamp(0, width - 1)
        selected_valid = (selected_offsets >= 0) & (selected_offsets < width)
        selected_valid &= scores.gather(1, safe_offsets) > -torch.inf
        selected = logical_positions.gather(1, safe_offsets).long()
        sentinel = torch.iinfo(torch.int64).max
        selected = torch.where(selected_valid, selected, sentinel).sort(-1).values
        output_indices.copy_(torch.where(selected == sentinel, -1, selected).to(torch.int32))
        return output_indices

    def _forward_indexer_tile(
        self,
        inputs: IndexerChunkInputs,
        q_data: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor | None,
        output: torch.Tensor,
        *,
        is_prefill: bool = True,
        score_hook: Callable[[torch.Tensor], None] | None = None,
        radix_aux_indices: torch.Tensor | None = None,
        radix_aux_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_prepared(
            q_data,
            inputs.k_data,
            inputs.k_scale,
            weights,
            inputs.row_starts,
            inputs.row_ends,
            output,
            q_scale,
            logical_positions=inputs.logical_positions,
            visible_lengths=inputs.visible_lengths,
            score_hook=score_hook,
            is_prefill=is_prefill,
            radix_aux_indices=radix_aux_indices,
            radix_aux_logits=radix_aux_logits,
        )

    def _publish_tile(
        self,
        results: dict[int, torch.Tensor],
        value: torch.Tensor,
        start: int,
        total_queries: int,
    ) -> None:
        if start == 0 and self.layer_idx not in results:
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
        candidate_width: int | None = None,
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
        candidates = torch.where(valid, candidates, -1).flatten(1)
        if candidate_width is not None:
            candidates = F.pad(candidates, (0, candidate_width - candidates.shape[1]), value=-1)
        self._publish_tile(results, candidates, start, total_queries)

    def forward(self, state: CSA2ForwardState, query_start: int, count: int) -> torch.Tensor:
        """Predict the complete model batch through the shared phase dispatcher.

        Attention tiles consume metadata.csa2_indices after this call. The
        request ranges describe the real context/decode batch, independently of
        the synthetic one-query decode requests used by the attention kernel.
        """
        metadata = state.metadata
        manager = metadata.kv_cache_manager
        if manager is None or state.index_q is None or state.index_weights is None:
            raise ValueError("CSA2 indexer requires an owner cache, queries and head weights")
        total_queries = state.swa_kv.shape[0]
        if query_start != 0 or count != total_queries:
            raise ValueError("CSA2 indexer must run once for the complete model query batch")
        if state.index_q.shape != (count, self.n_heads, self.head_dim):
            raise ValueError("CSA2 index queries must match the layer's full-query geometry")
        if state.index_weights.shape != (count, self.n_heads):
            raise ValueError("CSA2 index weights must match the layer's query rows and heads")
        packed_q = pack_rows(state.index_q, "index")
        data_width = self.head_dim // 2
        return self.sparse_attn_indexer(
            metadata,
            state.swa_kv,
            packed_q[..., :data_width].contiguous().view(torch.int8),
            None,
            None,
            state.index_weights.float(),
            q_scale=packed_q[..., data_width:].contiguous(),
        )

    def sparse_attn_indexer(
        self,
        metadata: CSA2TrtllmMetadata,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k_fp8: torch.Tensor | None,
        k_scale: torch.Tensor | None,
        weights: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        is_generation: bool | None = None,
    ) -> torch.Tensor:
        """Adapt CSA2 phases to the shared chunk executor and native kernels.

        Inputs span the complete model batch. K is read from the owner cache;
        the base signature is retained for the backend's indexer contract.
        """
        if is_generation is not None:
            raise ValueError("CSA2 prediction requires the complete model query batch")
        manager = metadata.kv_cache_manager
        count = hidden_states.shape[0]
        layer = self.layout.layer(self.layer_idx)
        visible = metadata.csa2_visible_lengths[self.layer_idx]
        max_positions = metadata.csa2_global_max_positions[layer.kv_source]
        logical = torch.empty((count, self.index_topk), dtype=torch.int32, device=q_fp8.device)
        metadata.csa2_indices[self.layer_idx] = logical
        candidates = None
        auxiliary = ()
        score_hook = None
        if layer.candidate_source is not None and layer.candidate_source != self.layer_idx:
            candidates = metadata.csa2_candidates.get(layer.candidate_source)
            if candidates is None or candidates.ndim != 2 or candidates.shape[0] != count:
                raise ValueError(
                    "CSA2 candidate source must publish matching full-query rows first"
                )
        elif layer.candidate_source == self.layer_idx:
            candidate_width = (
                min(
                    self.layout.candidate_topk_blocks,
                    (max_positions + self.layout.candidate_block_size - 1)
                    // self.layout.candidate_block_size,
                )
                * self.layout.candidate_block_size
            )
            published = torch.full(
                (count, candidate_width), -1, dtype=torch.int64, device=q_fp8.device
            )
            metadata.csa2_candidates[self.layer_idx] = published
            auxiliary = (published,)

            def score_hook(scores: torch.Tensor, start: int, end: int) -> None:
                self._publish_candidates(
                    scores,
                    visible[start:end],
                    metadata.csa2_candidates,
                    start,
                    count,
                    candidate_width,
                )

        def candidate_tile(start: int, end: int) -> IndexerChunkInputs:
            positions = candidates[start:end]
            slots = metadata.global_slot_tile(self.layer_idx, start, end, positions)
            keys, scales = manager.gather_indexer_keys(layer.kv_source, slots.flatten())
            width = positions.shape[1]
            starts = torch.arange(end - start, dtype=torch.int32, device=q_fp8.device) * width
            return IndexerChunkInputs(
                keys,
                scales,
                starts,
                starts + width,
                torch.where(slots >= 0, positions.long(), -1),
                visible[start:end],
            )

        def shared_keys(start: int, end: int, width: int) -> IndexerChunkInputs:
            # A request's compressed prefix is gathered once and shared by all
            # query tiles. Cached and newly published keys have the same owner.
            positions = torch.arange(width, device=q_fp8.device)
            slots = metadata.global_slot_tile(self.layer_idx, start, start + 1, positions[None, :])[
                0
            ]
            keys, scales = manager.gather_indexer_keys(layer.kv_source, slots)
            starts = torch.zeros(end - start, dtype=torch.int32, device=q_fp8.device)
            logical_positions = torch.where(slots >= 0, positions, -1).expand(end - start, -1)
            return IndexerChunkInputs(
                keys,
                scales,
                starts,
                torch.full_like(starts, width),
                logical_positions,
                visible[start:end],
            )

        ranges = metadata.csa2_request_query_ranges
        num_contexts = metadata.csa2_num_context_requests
        decode_start = ranges[num_contexts][0] if num_contexts < len(ranges) else count
        native_decode = (
            candidates is None
            and decode_start < count
            and q_fp8.is_cuda
            and 100 <= get_sm_version() < 110
            and self.n_heads in (32, 64)
            and self.head_dim == 128
        )
        paged = metadata.prepare_indexer(self.layer_idx) if native_decode else None
        chunks = []
        for request, (begin, end) in enumerate(ranges):
            is_prefill = request < num_contexts
            if not is_prefill and paged is not None:
                continue
            if candidates is not None:
                # Split across ranks before gathering candidate rows. The
                # shared driver then bounds the Q-by-(Q*candidates) transient.
                chunks.append(
                    IndexerQueryChunk(
                        begin,
                        end,
                        candidates.shape[1],
                        load_tile=candidate_tile,
                        keys_per_query=candidates.shape[1],
                        max_query_tokens=16,
                        is_prefill=is_prefill,
                    )
                )
                continue
            chunk_size = metadata.indexer_max_chunk_size
            for start in range(begin, end, chunk_size):
                stop = min(start + chunk_size, end)
                # Decode graphs must continue seeing newly published keys as
                # visibility grows. Warmup uses this same admitted bound.
                width = (
                    max_positions
                    if not is_prefill or metadata.is_cuda_graph
                    else min(
                        max_positions,
                        (metadata.csa2_request_start_positions[request] + stop - begin)
                        // layer.compress_ratio,
                    )
                )
                chunks.append(
                    IndexerQueryChunk(
                        start,
                        stop,
                        width,
                        load=lambda start=start, stop=stop, width=width: shared_keys(
                            start, stop, width
                        ),
                        is_prefill=is_prefill,
                    )
                )
        # Exact radix decode scratch is also needed by non-native candidate
        # paths. Its rows correspond to model queries, not FMHA staging tiles.
        radix_indices = radix_logits = None
        if any(not chunk.is_prefill for chunk in chunks):
            radix_indices = torch.empty(
                (count, 10, self.index_topk), dtype=torch.int32, device=q_fp8.device
            )
            radix_logits = torch.empty_like(radix_indices, dtype=torch.float32)
        self._run_query_chunks(
            chunks,
            q_fp8,
            weights,
            q_scale,
            logical,
            metadata.mapping,
            metadata.indexer_q_split_threshold,
            score_hook,
            auxiliary,
            radix_indices,
            radix_logits,
        )
        if paged is not None:
            decode_count = count - decode_start
            decode_scale = (
                q_scale[decode_start:]
                .contiguous()
                .view(torch.int32)
                .reshape(decode_count, 1, self.n_heads)
            )
            logits = self._call_paged_mqa_logits(
                q_fp8[decode_start:].unsqueeze(1),
                metadata.csa2_indexer_k_cache,
                weights[decode_start:],
                metadata.csa2_indexer_context_lengths,
                metadata.csa2_indexer_block_table,
                metadata.csa2_indexer_scheduler_metadata,
                metadata.csa2_indexer_max_seq_len,
                decode_scale,
            )
            hook = None
            if score_hook is not None:

                def hook(scores: torch.Tensor) -> None:
                    score_hook(scores, decode_start, count)

            lengths = metadata.csa2_indexer_context_lengths.flatten()
            self._select_mapped_logits(
                logits,
                torch.zeros_like(lengths),
                lengths,
                metadata.csa2_indexer_logical_positions,
                metadata.csa2_indexer_visible_lengths,
                logical[decode_start:],
                hook,
                is_prefill=False,
                radix_aux_indices=metadata.csa2_indexer_radix_aux_indices,
                radix_aux_logits=metadata.csa2_indexer_radix_aux_logits,
            )
        return logical
