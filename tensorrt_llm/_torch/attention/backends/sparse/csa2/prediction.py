# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 cache/selection adaptation for the shared sparse prediction hook."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
)

from .metadata import CSA2TrtllmMetadata
from .params import CSA2BackendForwardArgs, CSA2Layout, CSA2Mode
from .quantization import gather_rows, pack_rows

_SWA_TILE = 128
_HEAD_DIM = 512


def create_indexer(heads: int, head_dim: int, topk: int):
    from ..dsa.indexer import Indexer
    from ..dsa.params import DSAParams

    return Indexer(
        quant_config=None,
        pos_embd_params=None,
        mla_params=None,
        skip_create_weights_in_init=True,
        sparse_params=DSAParams(
            index_n_heads=heads, index_head_dim=head_dim, index_topk=topk, indexer_k_dtype="fp4"
        ),
        dtype=torch.bfloat16,
        projection_free=True,
    )


def predict_sparse_attention(
    backend,
    q: torch.Tensor,
    k: torch.Tensor | None,
    metadata: CSA2TrtllmMetadata,
    forward_args: AttentionForwardArgs,
) -> tuple[torch.Tensor, None]:
    inputs = forward_args.sparse_backend_args
    if not isinstance(inputs, CSA2BackendForwardArgs):
        raise TypeError(
            "CSA2 requires CSA2BackendForwardArgs with packed pools and selected indices"
        )
    count = metadata.num_tokens
    if (
        q.shape != (count, backend.num_heads * _HEAD_DIM)
        or q.dtype != torch.bfloat16
        or metadata.num_query_heads != backend.num_heads
    ):
        raise ValueError("CSA2 Q geometry/dtype must match its BF16 TRTLLM metadata")
    if count > backend.sparse_params.max_query_tokens:
        raise ValueError("CSA2 query tile exceeds its configured capacity")
    if forward_args.attention_input_type != AttentionInputType.generation_only:
        raise ValueError("Prepared CSA2 query metadata requires generation_only compute")
    if inputs.state is not None:
        if backend.sparse_params.layout is None:
            raise ValueError("CSA2 module inputs require the model layout")
        if backend.indexer is None and inputs.state.index_q is not None:
            iq = inputs.state.index_q
            backend.indexer = create_indexer(
                iq.shape[1], iq.shape[2], backend.sparse_params.layout.index_topk
            )
        inputs = predict_csa2_inputs(
            backend.sparse_params.layout, backend.layer_idx, inputs, count, backend.indexer
        )
    if inputs.swa_pool is None or inputs.swa_indices is None:
        raise ValueError("CSA2 requires selected SWA pool inputs")
    if metadata.swa_pool.device != q.device or inputs.swa_pool.device != q.device:
        raise ValueError("CSA2 Q, packed pools and metadata must be on the same CUDA device")
    if (
        metadata.is_cuda_graph
        and backend.compute_backend == "trtllm"
        and metadata.workspace.numel() == 0
    ):
        raise RuntimeError("Warm up CSA2 metadata with forward before CUDA Graph capture")
    if inputs.swa_indices.shape[0] != count or inputs.swa_indices.shape[1] > _SWA_TILE:
        raise ValueError("CSA2 SWA indices must match the query count and window <=128")
    swa = gather_rows(inputs.swa_pool, inputs.swa_indices, _HEAD_DIM, "swa")
    swa_valid = inputs.swa_indices >= 0
    extra = extra_valid = None
    if inputs.topk_indices is not None:
        if inputs.main_pool is None or inputs.main_pool.device != q.device:
            raise ValueError("CSA2 selected main indices require a main pool on the query device")
        if (
            inputs.topk_indices.shape[0] != count
            or inputs.topk_indices.shape[1] > metadata.num_sparse_topk - _SWA_TILE
        ):
            raise ValueError("CSA2 selected main indices exceed metadata geometry")
        extra = gather_rows(inputs.main_pool, inputs.topk_indices, _HEAD_DIM, "main")
        extra_valid = inputs.topk_indices >= 0
    metadata.prepared_counter.zero_()
    if extra is not None:
        if extra_valid is None:
            raise ValueError("Extra KV rows require a validity mask")
        rows = torch.cat((swa, extra), dim=1)
        valid = torch.cat((swa_valid, extra_valid), dim=1)
    else:
        rows, valid = swa, swa_valid
    # TG uses a dense valid prefix, split at slot 128 between its pools.
    # Compact the selected union before staging: -1 slots inside the
    # supplied extent can contribute zero logits in BF16 generation.
    # Physical source ownership no longer matters after dequantization.
    width = rows.shape[1]
    positions = torch.arange(width, device=swa.device).expand(count, -1)
    order = torch.where(valid, positions, width).argsort(dim=1, stable=True)
    packed = rows.gather(1, order[..., None].expand(-1, -1, _HEAD_DIM))
    lengths = valid.sum(1, dtype=torch.int32)
    packed = torch.where((positions < lengths[:, None])[..., None], packed, 0)
    # Zero selected rows reduce to the sink's zero value. Give TG one
    # zero KV row so it always launches a defined (nonempty) reduction.
    lengths = lengths.clamp_min(1)
    metadata.prepared_lens[:count].copy_(lengths)
    metadata.swa_pool[:count].zero_()
    metadata.extra_pool[:count].zero_()
    swa_count = min(width, _SWA_TILE)
    metadata.swa_pool[:count, :swa_count].copy_(packed[:, :swa_count])
    if width > _SWA_TILE:
        metadata.extra_pool[:count, : width - _SWA_TILE].copy_(packed[:, _SWA_TILE:])
    indices = metadata.prepared_indices[:count]
    offsets = torch.arange(count, device=swa.device)[:, None]
    swa_positions = torch.arange(_SWA_TILE, device=swa.device)[None, :]
    indices[:, :_SWA_TILE].copy_(
        torch.where(swa_positions < lengths[:, None], offsets * _SWA_TILE + swa_positions, -1)
    )
    extra_capacity = metadata.num_sparse_topk - _SWA_TILE
    if extra_capacity:
        extra_positions = torch.arange(extra_capacity, device=swa.device)[None, :]
        indices[:, _SWA_TILE:].copy_(
            torch.where(
                extra_positions + _SWA_TILE < lengths[:, None],
                offsets * extra_capacity + extra_positions,
                -1,
            )
        )

    sparse = forward_args.sparse_runtime_params
    sparse.sparse_attn_kv_lens = metadata.prepared_lens
    sparse.aux_kv_cache_pool_ptr = (
        metadata.extra_pool.data_ptr()
        if inputs.topk_indices is not None and metadata.num_sparse_topk > _SWA_TILE
        else None
    )
    forward_args.fmha_scheduler_counter = metadata.prepared_counter
    forward_args.attention_window_size = metadata.num_sparse_topk
    # Native MLA ABI inputs; projections/RoPE/cache writes are module-owned.
    forward_args.latent_cache = metadata.swa_pool[:, 0]
    forward_args.q_pe = q.view(count, backend.num_heads, _HEAD_DIM)[..., 448:]
    return metadata.prepared_indices, None


def predict_csa2_inputs(
    layout: CSA2Layout, layer_idx: int, inputs: CSA2BackendForwardArgs, count: int, indexer=None
) -> CSA2BackendForwardArgs:
    """Update owner caches and predict logical selections for one query tile.

    This is the algorithm portion of sparse_attn_predict; it does not execute
    attention. Routing carries full-query results across serialized layers.
    """
    state = inputs.state
    if state is None:
        raise ValueError("CSA2 prediction requires projected module inputs")
    layer = layout.layer(layer_idx)
    manager, batch, routing = state.cache_manager, state.batch, state.routing
    start, end = inputs.query_start, inputs.query_start + count
    total_queries = state.swa_kv.shape[0]
    if start == 0:
        routing.enter(layer)
        manager.write_swa(layer_idx, batch.swa_write_slots, state.swa_kv)
        if layer.mode == CSA2Mode.FULL:
            if state.main_kv is None or state.index_k is None:
                raise ValueError(
                    "Full mode must supply new main and index rows, including empty rows"
                )
            manager.write_global(
                layer.kv_source, batch.main_write_slots, state.main_kv, state.index_k
            )
        elif state.main_kv is not None or state.index_k is not None:
            raise ValueError("Only Full mode may write shared main/index caches")
    if layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
        if state.index_q is None or state.index_weights is None:
            raise ValueError("Full/Reindex mode requires index queries and weights")
        iq = state.index_q[start:end]
        iq_packed = pack_rows(iq, "index")
        visible = batch.visible_lengths[start:end]
        if layer.candidate_source is not None and layer_idx != layer.candidate_source:
            if layer.candidate_source not in routing.candidates:
                raise ValueError("CSA2 candidate source did not run in this forward")
            positions = routing.candidates[layer.candidate_source][start:end]
            slots = batch.global_slot_tile(start, end, positions)
        else:
            slots = batch.global_slot_tile(start, end)
            positions = torch.arange(slots.shape[-1], device=iq.device)
        key_rows = manager.get_index_buffer(layer.kv_source)[slots.clamp_min(0).long()]
        key_rows = torch.where((slots >= 0)[..., None], key_rows, 0)
        if indexer is None:
            indexer = create_indexer(iq.shape[1], iq.shape[2], layout.index_topk)
        logical_positions = positions.long().expand(count, -1)
        logical_positions = torch.where(slots >= 0, logical_positions, -1)
        width = key_rows.shape[1]
        starts = torch.arange(count, dtype=torch.int32, device=iq.device) * width
        keys = key_rows.reshape(-1, key_rows.shape[-1])
        data_width = iq.shape[-1] // 2
        q_data = iq_packed[..., :data_width].contiguous().view(torch.int8)
        q_scale = iq_packed[..., data_width:].contiguous().view(torch.int32)
        k_data = keys[:, :data_width].contiguous().view(torch.int8)
        k_scale = keys[:, data_width:].contiguous().view(torch.int32)
        logical = torch.empty((count, layout.index_topk), dtype=torch.int32, device=iq.device)

        def publish_candidates(scores: torch.Tensor) -> None:
            block_size = layout.candidate_block_size
            padded = F.pad(scores, (0, -width % block_size), value=-torch.inf)
            blocks = padded.reshape(count, padded.shape[1] // block_size, block_size).amax(-1)
            latest = (visible - 1) // block_size
            block_ids = torch.arange(blocks.shape[1], device=iq.device)
            blocks = blocks.masked_fill(block_ids[None, :] == latest[:, None], torch.inf)
            selected = torch.empty(
                (count, min(layout.candidate_topk_blocks, blocks.shape[1])),
                dtype=torch.int32,
                device=iq.device,
            )
            indexer.select_prepared_scores(
                blocks, selected, torch.zeros_like(starts), torch.full_like(starts, blocks.shape[1])
            )
            candidates = selected.long()[..., None] * block_size + torch.arange(
                block_size, device=iq.device
            )
            valid = (selected >= 0)[..., None]
            valid = valid & (blocks.gather(1, selected.long().clamp_min(0)) > -torch.inf)[..., None]
            valid = valid & (candidates < visible[:, None, None]) & (candidates < width)
            candidates = torch.where(valid, candidates, -1).flatten(1)
            if start == 0:
                routing.candidates[layer_idx] = torch.empty(
                    (total_queries, candidates.shape[1]), dtype=candidates.dtype, device=iq.device
                )
            routing.candidates[layer_idx][start:end].copy_(candidates)

        indexer.forward_prepared(
            q_data,
            k_data,
            k_scale,
            state.index_weights[start:end].float(),
            starts,
            starts + width,
            logical,
            q_scale=q_scale,
            logical_positions=logical_positions,
            visible_lengths=visible,
            score_hook=publish_candidates if layer.candidate_source == layer_idx else None,
        )
        if start == 0:
            routing.indices[layer_idx] = torch.empty(
                (total_queries, layout.index_topk), dtype=logical.dtype, device=iq.device
            )
        routing.indices[layer_idx][start:end].copy_(logical)
    elif layer.mode == CSA2Mode.REUSE:
        if layer.index_source not in routing.indices:
            raise ValueError("CSA2 index source did not run in this forward")
        source_indices = routing.indices[layer.index_source]
        if source_indices.shape[0] != total_queries:
            raise ValueError("CSA2 Reuse rows must match their source's packed query order")
        logical = source_indices[start:end]
    else:
        logical = None
    global_slots = None
    if logical is not None:
        global_slots = batch.global_slot_tile(start, end, logical)
        global_slots = torch.where(
            logical < batch.visible_lengths[start:end, None], global_slots, -1
        )
    return CSA2BackendForwardArgs(
        swa_pool=manager.get_swa_buffer(layer_idx),
        swa_indices=batch.swa_indices[start:end],
        main_pool=None if logical is None else manager.get_main_buffer(layer.kv_source),
        topk_indices=global_slots,
    )
