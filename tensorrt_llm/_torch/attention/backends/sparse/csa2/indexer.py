# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 candidate adaptation of the shared DSA/DeepSeek-V4 indexer workflow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.distributed.ops import allgather
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

from ..dsa.indexer import _INDEXER_MQA_LOGITS_ELEM_BUDGET, Indexer, _split_prefill_queries
from ..dsa.params import DSAParams, use_self_sampling_gvr
from .metadata import CSA2TrtllmMetadata
from .params import CSA2ForwardState, CSA2Layout, CSA2Mode, CSA2Params
from .quantization import pack_rows


@dataclass
class _ChunkInputs:
    """Packed owner keys and logical bounds for one CSA2 query tile."""

    k_data: torch.Tensor
    k_scale: torch.Tensor
    row_starts: torch.Tensor
    row_ends: torch.Tensor
    logical_positions: Optional[torch.Tensor] = None
    visible_lengths: Optional[torch.Tensor] = None


@dataclass
class _QueryChunk:
    """Lazy K preparation, after request-level TP splitting for candidate K."""

    token_start: int
    token_end: int
    k_token_count: int
    load: Optional[Callable[[], _ChunkInputs]] = None
    load_tile: Optional[Callable[[int, int], _ChunkInputs]] = None
    keys_per_query: int = 0
    max_query_tokens: Optional[int] = None
    is_prefill: bool = True


class CSA2Indexer(Indexer):
    """Prepare owner-cache candidates and publish CSA2 logical selections.

    This subclass owns chunk loading, candidate workspace limits and routing
    publication. It reuses the base indexer's MQA dispatch, TP query partition
    calculation and TopK module. Cache writes and layer ordering belong to the
    backend.
    """

    def __init__(
        self,
        layout: CSA2Layout,
        layer_idx: int,
        heads: int,
        head_dim: int,
        options: CSA2Params | None = None,
    ) -> None:
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
        self.options = options or CSA2Params(layout=layout)
        sm = get_sm_version() if torch.cuda.is_available() else 0
        compatible = IS_CUTLASS_DSL_AVAILABLE and 100 <= sm < 110
        exact = (
            TopKImplementation.CUTE_DSL_RADIX
            if compatible and self.options.use_cute_dsl_topk
            else TopKImplementation.CUDA_RADIX
        )
        if not torch.cuda.is_available():
            exact = TopKImplementation.TORCH
        prefill, decode = exact, exact
        if compatible and self.options.enable_heuristic_topk:
            if self.options.use_self_sampling_topk and use_self_sampling_gvr(
                enable_heuristic_topk=True,
                use_self_sampling_topk=True,
                index_topk=self.index_topk,
                compress_ratio=1,
                is_cute_dsl_available=True,
                sm_version=sm,
            ):
                prefill = decode = TopKImplementation.CUTE_DSL_GVR
            elif (
                not self.options.use_self_sampling_topk
                and layer.mode == CSA2Mode.FULL
                and head_dim == 128
                and self.index_topk in (512, 1024, 2048)
            ):
                decode = TopKImplementation.CUTE_DSL_GVR
        self.top_k.prefill_implementation = prefill
        self.top_k.decode_implementation = decode
        self.top_k.gvr_self_sampling = self.options.use_self_sampling_topk
        self.candidate_top_k: TopK | None = None

    def _configure_candidate_topk(self, max_positions: int) -> None:
        count = min(
            self.layout.candidate_topk_blocks,
            (max_positions + self.layout.candidate_block_size - 1)
            // self.layout.candidate_block_size,
        )
        if self.candidate_top_k is None:
            self.candidate_top_k = TopK(
                max(1, count),
                prefill_implementation=(
                    TopKImplementation.TORCH
                    if self.top_k.prefill_implementation == TopKImplementation.TORCH
                    else TopKImplementation.CUDA_RADIX
                ),
            )
        elif self.candidate_top_k.top_k != max(1, count):
            raise ValueError("CSA2 admitted candidate geometry must remain fixed for an indexer")

    def _temporal_enabled(self, q_data: torch.Tensor) -> bool:
        return q_data.is_cuda and self.top_k.needs_gvr_prior

    def _reset_emission(self) -> None:
        self.top_k.reset_gvr_emission_rows(slice(None))

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
        candidate: bool = False,
        gvr_prior_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Invoke the configured main TopK or the candidate-source block TopK."""
        if scores.shape[0] == 0 or scores.shape[1] == 0:
            return output_indices.fill_(-1)
        selector = self.candidate_top_k if candidate else self.top_k
        if selector is None or output_indices.shape[1] != selector.top_k:
            raise ValueError("CSA2 TopK output must match its configured selection domain")
        implementation = (
            selector.prefill_implementation if is_prefill else selector.decode_implementation
        )
        if implementation in (TopKImplementation.CUTE_DSL_RADIX, TopKImplementation.CUTE_DSL_GVR):
            # Exact DSL uses wide vector copies; pad storage without extending
            # row bounds. Self-sampling also benefits from this native layout.
            if (
                scores.stride(1) != 1
                or scores.stride(0) % 8
                or scores.data_ptr() % 32
                or scores.shape[1] % 8
            ):
                scores = F.pad(scores.contiguous(), (0, -scores.shape[1] % 8), value=-torch.inf)
        if not is_prefill:
            return selector(
                scores,
                output_indices,
                is_prefill=False,
                sequence_lengths=row_ends * selector.compress_ratio,
                scan_lengths=row_ends,
                next_n=1,
                max_seq_len=scores.shape[1],
                radix_aux_indices=radix_aux_indices,
                radix_aux_logits=radix_aux_logits,
                gvr_ext_kwargs={"gvr_prior_indices": gvr_prior_indices}
                if selector.needs_gvr_prior
                else None,
            )
        return selector(
            scores, output_indices, is_prefill=True, row_starts=row_starts, row_ends=row_ends
        )

    def _select_all_visible(
        self,
        positions: torch.Tensor,
        visible: torch.Tensor,
        row_limits: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Select every valid logical candidate once its cardinality is bounded by K."""
        offsets = torch.arange(positions.shape[1], device=positions.device)[None, :]
        valid = (positions >= 0) & (positions < visible[:, None])
        valid &= offsets < row_limits[:, None]
        sentinel = torch.iinfo(torch.int64).max
        selected = torch.where(valid, positions.long(), sentinel).sort(-1).values
        selected = torch.where(selected == sentinel, -1, selected).to(torch.int32)
        selected = selected[:, : self.index_topk]
        output.copy_(F.pad(selected, (0, self.index_topk - selected.shape[1]), value=-1))
        return output

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
        gvr_prior_indices: Optional[torch.Tensor] = None,
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
        if (
            self.options.skip_indexer_for_short_seqs
            and logical_positions is not None
            and logical_positions.shape[1] <= self.index_topk
            and score_hook is None
        ):
            return self._select_all_visible(
                logical_positions,
                visible_lengths,
                row_ends - row_starts,
                output_indices,
            )
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
            gvr_prior_indices,
        )

    def _invalidate_emission_holes(self, bad_rows: torch.Tensor) -> None:
        """Force stock masked-score GVR for rows whose raw emission saw page holes.

        The native emitter runs before logical masking. Invalidating just its
        prior or seeds is insufficient: candidate lists have independent
        admission controls. All side channels are reset on the device so a
        captured graph can change page validity without a host-side branch.
        """
        state = self.top_k._gvr_emission_state
        if state is None:
            return
        count = bad_rows.shape[0]
        mask = bad_rows[:, None]
        state.seed_row[:count, :3].masked_fill_(mask, torch.inf)
        state.seed_row[:count, 3:].masked_fill_(mask, 0)
        state.seed_rungs[:count].masked_fill_(mask, torch.inf)
        state.xstate[:count].masked_fill_(mask, 0)
        if state.cand_ctl is not None:
            rows = min(count, state.cand_ctl.shape[0])
            state.cand_ctl[:rows].masked_fill_(bad_rows[:rows, None], 0)
            state.cand_ctl[:rows, 1].masked_fill_(bad_rows[:rows], 1)
        if state.block_max is not None:
            state.block_max[:count].masked_fill_(mask, torch.inf)

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
        gvr_prior_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        width = logical_positions.shape[1]
        offsets = torch.arange(width, device=logits.device)
        columns = row_starts.long()[:, None] + offsets
        in_bounds = (columns >= 0) & (columns < row_ends[:, None]) & (columns < logits.shape[1])
        scores = logits.gather(1, columns.clamp(0, logits.shape[1] - 1))
        valid = in_bounds & (logical_positions >= 0)
        valid &= logical_positions < visible_lengths[:, None]
        scores = scores.masked_fill(~valid, -torch.inf)
        if gvr_prior_indices is not None:
            if width:
                prior = gvr_prior_indices.long()
                prior_valid = (prior >= 0) & (prior < width)
                prior_valid &= valid.gather(1, prior.clamp(0, width - 1))
                gvr_prior_indices = torch.where(prior_valid, gvr_prior_indices, -1)
            else:
                gvr_prior_indices = torch.full_like(gvr_prior_indices, -1)
        if not is_prefill and self.options.use_gvr_emission and self.top_k._gvr_emission_armed:
            bad_rows = (
                (offsets[None, :] < visible_lengths[:, None]) & (logical_positions < 0)
            ).any(dim=1)
            self._invalidate_emission_holes(bad_rows)
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
            gvr_prior_indices=gvr_prior_indices,
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

    def _run_csa2_chunks(
        self,
        chunks: list[_QueryChunk],
        q_data: torch.Tensor,
        weights: torch.Tensor,
        q_scale: Optional[torch.Tensor],
        output: torch.Tensor,
        mapping: Optional[Mapping],
        q_split_threshold: int,
        score_hook: Optional[Callable[[torch.Tensor, int, int], None]] = None,
        auxiliary_outputs: tuple[torch.Tensor, ...] = (),
        radix_aux_indices: Optional[torch.Tensor] = None,
        radix_aux_logits: Optional[torch.Tensor] = None,
        gvr_prior_indices: Optional[torch.Tensor] = None,
        prior_query_start: int = 0,
    ) -> None:
        """Execute CSA2 owner-prefix or candidate chunks and publish all routing outputs."""
        for chunk in chunks:
            count = chunk.token_end - chunk.token_start
            if count < 0 or chunk.token_start < 0 or chunk.token_end > q_data.shape[0]:
                raise ValueError("Indexer chunk query bounds are outside the projected batch")
            if count == 0 or chunk.k_token_count <= 0:
                output[chunk.token_start : chunk.token_end].fill_(-1)
                for auxiliary in auxiliary_outputs:
                    auxiliary[chunk.token_start : chunk.token_end].fill_(-1)
                continue
            if (chunk.load is None) == (chunk.load_tile is None):
                raise ValueError("Indexer chunks require exactly one shared or tiled K loader")
            local_start, local_end, sizes = _split_prefill_queries(
                count, mapping, q_split_threshold if chunk.is_prefill else -1
            )
            shared = chunk.load() if chunk.load is not None else None
            budget = _INDEXER_MQA_LOGITS_ELEM_BUDGET
            if self.use_fp4 and (
                self.head_dim != 128 or not q_data.is_cuda or get_sm_version() < 100
            ):
                # The decoded fallback also materializes per-head dot products.
                budget = max(1, budget // self.n_heads)
            if chunk.load_tile is not None:
                if chunk.keys_per_query <= 0:
                    raise ValueError("Tiled candidate K loaders require a per-query key bound")
                tile_size = max(1, math.isqrt(budget // chunk.keys_per_query))
            else:
                tile_size = max(1, budget // max(1, chunk.k_token_count))
            if chunk.max_query_tokens is not None:
                if chunk.max_query_tokens <= 0:
                    raise ValueError("Indexer query tile capacity must be positive")
                tile_size = min(tile_size, chunk.max_query_tokens)
            for offset in range(local_start, local_end, tile_size):
                stop = min(offset + tile_size, local_end)
                first, last = chunk.token_start + offset, chunk.token_start + stop
                if shared is None:
                    inputs = chunk.load_tile(first, last)
                    row_slice = slice(None)
                else:
                    inputs = shared
                    row_slice = slice(offset, stop)
                logical = (
                    inputs.logical_positions[row_slice]
                    if inputs.logical_positions is not None
                    else None
                )
                visible = (
                    inputs.visible_lengths[row_slice]
                    if inputs.visible_lengths is not None
                    else None
                )
                hook = None
                if score_hook is not None:

                    def hook(scores, first=first, last=last):
                        score_hook(scores, first, last)

                self.forward_prepared(
                    q_data[first:last],
                    inputs.k_data,
                    inputs.k_scale,
                    weights[first:last],
                    inputs.row_starts[row_slice],
                    inputs.row_ends[row_slice],
                    output[first:last],
                    q_scale[first:last] if q_scale is not None else None,
                    logical_positions=logical,
                    visible_lengths=visible,
                    is_prefill=chunk.is_prefill,
                    score_hook=hook,
                    radix_aux_indices=radix_aux_indices[first:last]
                    if radix_aux_indices is not None
                    else None,
                    radix_aux_logits=radix_aux_logits[first:last]
                    if radix_aux_logits is not None
                    else None,
                    gvr_prior_indices=(
                        gvr_prior_indices[first - prior_query_start : last - prior_query_start]
                        if gvr_prior_indices is not None and not chunk.is_prefill
                        else None
                    ),
                )
            if sizes is not None:
                first, last = chunk.token_start + local_start, chunk.token_start + local_end
                for tensor in (output, *auxiliary_outputs):
                    tensor[chunk.token_start : chunk.token_end] = allgather(
                        tensor[first:last],
                        mapping,
                        dim=0,
                        sizes=sizes,
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
        if self.candidate_top_k is None:
            raise RuntimeError("Configure CSA2 candidate TopK from admitted geometry first")
        selected = torch.empty(
            (count, self.candidate_top_k.top_k),
            dtype=torch.int32,
            device=scores.device,
        )
        starts = torch.zeros(count, dtype=torch.int32, device=scores.device)
        self.select_prepared_scores(
            blocks,
            selected,
            starts,
            torch.full_like(starts, blocks.shape[1]),
            candidate=True,
        )
        candidates = selected.long()[..., None] * block_size + torch.arange(
            block_size, device=scores.device
        )
        valid = (selected >= 0)[..., None]
        if blocks.shape[1]:
            valid = (
                valid
                & (blocks.gather(1, selected.long().clamp(0, blocks.shape[1] - 1)) > -torch.inf)[
                    ..., None
                ]
            )
        else:
            valid = torch.zeros_like(valid)
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
        if state.index_weights.shape != (count, self.n_heads):
            raise ValueError("CSA2 index weights must match the layer's query rows and heads")
        if state.index_q_scale is None:
            if state.index_q.shape != (count, self.n_heads, self.head_dim):
                raise ValueError("CSA2 index queries must match the layer's full-query geometry")
            packed_q = pack_rows(state.index_q, "index")
            data_width = self.head_dim // 2
            q_data = packed_q[..., :data_width].contiguous().view(torch.int8)
            q_scale = packed_q[..., data_width:].contiguous()
        else:
            if (
                self.head_dim != 128
                or state.index_q.dtype not in (torch.int8, torch.uint8)
                or state.index_q.shape != (count, self.n_heads, 64)
            ):
                raise ValueError("Packed CSA2 index queries require 128D E2M1 data")
            if (
                state.index_q_scale.dtype not in (torch.int32, torch.uint8)
                or state.index_q_scale.numel() * state.index_q_scale.element_size()
                != count * self.n_heads * 4
            ):
                raise ValueError(
                    "Packed CSA2 index queries require four UE8M0 scale bytes per head"
                )
            q_data = state.index_q.contiguous().view(torch.int8)
            q_scale = state.index_q_scale.contiguous()
        return self.sparse_attn_indexer(
            metadata,
            state.swa_kv,
            q_data,
            None,
            None,
            state.index_weights.float(),
            q_scale=q_scale,
        )

    def _paged_mqa_logits(
        self,
        metadata: CSA2TrtllmMetadata,
        q_data: torch.Tensor,
        weights: torch.Tensor,
        q_scale: torch.Tensor,
        emission_kwargs: dict[str, torch.Tensor | int] | None = None,
    ) -> torch.Tensor:
        # Existing native paged specializations support 32/64 heads. Zero
        # weighted padding extends smaller CSA2 index geometries exactly.
        use_dsl = self.options.use_cute_dsl_paged_mqa_logits and IS_CUTLASS_DSL_AVAILABLE
        padded_heads = 64 if use_dsl or self.n_heads > 32 else 32
        if padded_heads != self.n_heads:
            q_data = F.pad(q_data, (0, 0, 0, padded_heads - self.n_heads))
            weights = F.pad(weights, (0, padded_heads - self.n_heads))
            q_scale = F.pad(q_scale, (0, padded_heads - self.n_heads))
        if use_dsl:
            return torch.ops.trtllm.cute_dsl_fp4_paged_mqa_logits(
                q_data.view(torch.uint8),
                q_scale,
                metadata.csa2_indexer_k_cache,
                weights,
                metadata.csa2_indexer_context_lengths.flatten(),
                metadata.csa2_indexer_block_table,
                metadata.csa2_indexer_scheduler_metadata,
                metadata.csa2_indexer_max_seq_len,
                **(emission_kwargs or {}),
            )
        return self._call_paged_mqa_logits(
            q_data,
            metadata.csa2_indexer_k_cache,
            weights,
            metadata.csa2_indexer_context_lengths,
            metadata.csa2_indexer_block_table,
            metadata.csa2_indexer_scheduler_metadata,
            metadata.csa2_indexer_max_seq_len,
            q_scale,
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
        """Schedule CSA2 phases using inherited native kernels and shared TP partitioning.

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
        prior = None
        if self._temporal_enabled(q_fp8):
            metadata.register_indexer_reset(self.layer_idx, self._reset_emission)
            prior = metadata.prepare_indexer_prior(self.layer_idx, self.index_topk)
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
            self._configure_candidate_topk(max_positions)
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

        # Host prefix bounds are valid for eager execution only. A graph must
        # admit every future replay's visibility, so use its configured bound.
        graph_mode = metadata.is_cuda_graph or (
            q_fp8.is_cuda and torch.cuda.is_current_stream_capturing()
        )
        short_bound = (
            max_positions
            if graph_mode
            else min(
                max_positions,
                max(
                    (prefix + end - begin) // layer.compress_ratio
                    for prefix, (begin, end) in zip(
                        metadata.csa2_request_start_positions, metadata.csa2_request_query_ranges
                    )
                )
                if metadata.csa2_request_query_ranges
                else 0,
            )
        )
        if (
            self.options.skip_indexer_for_short_seqs
            and score_hook is None
            and short_bound <= self.index_topk
        ):
            positions = (
                candidates
                if candidates is not None
                else torch.arange(short_bound, device=q_fp8.device).expand(count, -1)
            )
            slots = metadata.global_slot_tile(self.layer_idx, 0, count, positions)
            positions = torch.where(slots >= 0, positions, -1)
            self._select_all_visible(
                positions,
                visible,
                torch.full((count,), positions.shape[1], dtype=torch.int32, device=q_fp8.device),
                logical,
            )
            if prior is not None:
                metadata.publish_indexer_prior(self.layer_idx, logical)
            return logical

        def candidate_tile(start: int, end: int) -> _ChunkInputs:
            positions = candidates[start:end]
            slots = metadata.global_slot_tile(self.layer_idx, start, end, positions)
            keys, scales = manager.gather_indexer_keys(layer.kv_source, slots.flatten())
            width = positions.shape[1]
            starts = torch.arange(end - start, dtype=torch.int32, device=q_fp8.device) * width
            return _ChunkInputs(
                keys,
                scales,
                starts,
                starts + width,
                torch.where(slots >= 0, positions.long(), -1),
                visible[start:end],
            )

        def shared_keys(start: int, end: int, width: int) -> _ChunkInputs:
            # A request's compressed prefix is gathered once and shared by all
            # query tiles. Cached and newly published keys have the same owner.
            positions = torch.arange(width, device=q_fp8.device)
            slots = metadata.global_slot_tile(self.layer_idx, start, start + 1, positions[None, :])[
                0
            ]
            keys, scales = manager.gather_indexer_keys(layer.kv_source, slots)
            starts = torch.zeros(end - start, dtype=torch.int32, device=q_fp8.device)
            logical_positions = torch.where(slots >= 0, positions, -1).expand(end - start, -1)
            return _ChunkInputs(
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
            and 1 <= self.n_heads <= 64
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
                # CSA2 runner then bounds the Q-by-(Q*candidates) transient.
                chunks.append(
                    _QueryChunk(
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
                    _QueryChunk(
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
        self._run_csa2_chunks(
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
            prior,
            decode_start,
        )
        if paged is not None:
            decode_count = count - decode_start
            decode_scale = (
                q_scale[decode_start:]
                .contiguous()
                .view(torch.int32)
                .reshape(decode_count, 1, self.n_heads)
            )
            emission_kwargs = {}
            if (
                self.options.use_gvr_emission
                and prior is not None
                and metadata.csa2_indexer_max_seq_len % 8 == 0
            ):
                emission_kwargs = self.top_k.prepare_gvr_emission(
                    decode_count,
                    metadata.csa2_indexer_max_seq_len,
                    torch.cuda.get_device_properties(q_fp8.device).multi_processor_count,
                    metadata.csa2_indexer_prior_capacity[self.layer_idx],
                )
            logits = self._paged_mqa_logits(
                metadata,
                q_fp8[decode_start:].unsqueeze(1),
                weights[decode_start:],
                decode_scale,
                emission_kwargs=emission_kwargs,
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
                gvr_prior_indices=prior,
            )
        if prior is not None:
            metadata.publish_indexer_prior(self.layer_idx, logical)
        return logical
