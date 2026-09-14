# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared CSA2 rows through DSV4's TRTLLM dynamic sparse MLA path."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention, TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._utils import is_sm_100f

from ..deepseek_v4.backend import DeepseekV4TrtllmAttention
from .params import CSA2Params

_SWA_TILE = 128
_HEAD_DIM = 512


class CSA2TrtllmMetadata(TrtllmAttentionMetadata):
    """Bounded prepared query view of module-managed sparse caches.

    Every selected query is an internal generation request with Q length one.
    Causality, request isolation and window selection are already represented
    by its indices, so the same DSV4 generation kernel serves all model phases.
    These are compute staging pools, not persistent request-owned KV caches.
    """

    @property
    def tokens_per_block(self) -> int:
        return _SWA_TILE

    @property
    def host_kv_cache_pool_pointers(self) -> torch.Tensor:
        return self.pool_pointers

    @property
    def host_kv_cache_pool_mapping(self) -> torch.Tensor:
        return self.pool_mapping

    def allocate_prepared(
        self, capacity: int, heads: int, extra_capacity: int, device: torch.device
    ) -> None:
        self.swa_pool = torch.empty(
            capacity, _SWA_TILE, _HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        self.extra_pool = torch.empty(
            capacity, max(extra_capacity, 1), _HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        self.pool_pointers = torch.tensor(
            [[self.swa_pool.data_ptr(), 0]], dtype=torch.int64, device="cpu"
        )
        self.pool_mapping = torch.zeros((1, 2), dtype=torch.int32, device="cpu")
        self.num_sparse_topk = _SWA_TILE + extra_capacity
        self.max_seq_len = self.num_sparse_topk
        self.kv_cache_params = KVCacheParams(use_cache=True)
        self.kv_cache_block_offsets = torch.zeros(
            (1, capacity, 2, (self.num_sparse_topk + _SWA_TILE - 1) // _SWA_TILE),
            dtype=torch.int32,
            device=device,
        )
        self.prepared_indices = torch.full(
            (capacity, self.num_sparse_topk), -1, dtype=torch.int32, device=device
        )
        self.prepared_lens = torch.empty(capacity, dtype=torch.int32, device=device)
        self.prepared_counter = torch.zeros(1, dtype=torch.uint32, device=device)
        self.prepared_cu_q = torch.arange(capacity + 1, dtype=torch.int32, device=device) * heads
        self.prepared_cu_kv = (
            torch.arange(capacity + 1, dtype=torch.int32, device=device) * self.num_sparse_topk
        )
        self.query_lens_host = torch.ones(capacity, dtype=torch.int32, device="cpu")
        self.query_lens_device = torch.ones(capacity, dtype=torch.int32, device=device)
        self.kv_lens.fill_(self.num_sparse_topk)
        self.kv_lens_cuda.fill_(self.num_sparse_topk)
        self.prompt_lens_cpu.fill_(1)
        self.prompt_lens_cuda.fill_(1)
        self.host_request_types.fill_(1)
        self.host_total_kv_lens.zero_()
        # THOP grows this tensor during eager warmup. Both views must retain
        # that same storage before capture; never resize a new graph buffer.
        self.cuda_graph_workspace = self.workspace
        self.warmed_query_counts: set[int] = set()

    def bind_prepared(
        self,
        swa: torch.Tensor,
        extra: torch.Tensor | None,
        swa_valid: torch.Tensor,
        extra_valid: torch.Tensor | None,
    ) -> None:
        count, swa_width, _ = swa.shape
        if swa_width > _SWA_TILE:
            raise ValueError("TRTLLM dynamic sparse MLA supports an SWA region of at most 128 rows")
        self._seq_lens = self.query_lens_host[:count]
        self._seq_lens_cuda = self.query_lens_device[:count]
        self._num_contexts = self._num_ctx_tokens = 0
        self._num_generations = self._num_tokens = count
        self._bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda[:count],
            kv_lens=self.kv_lens[:count],
            prompt_lens_cuda=self.prompt_lens_cuda[:count],
            prompt_lens_cpu=self.prompt_lens_cpu[:count],
            host_request_types=self.host_request_types[:count],
        )
        self.host_total_kv_lens[0] = 0
        self.host_total_kv_lens[1] = count * self.num_sparse_topk
        self.prepared_counter.zero_()
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
        self.prepared_lens[:count].copy_(lengths)
        self.swa_pool[:count].zero_()
        self.extra_pool[:count].zero_()
        swa_count = min(width, _SWA_TILE)
        self.swa_pool[:count, :swa_count].copy_(packed[:, :swa_count])
        if width > _SWA_TILE:
            self.extra_pool[:count, : width - _SWA_TILE].copy_(packed[:, _SWA_TILE:])
        indices = self.prepared_indices[:count]
        offsets = torch.arange(count, device=swa.device)[:, None]
        swa_positions = torch.arange(_SWA_TILE, device=swa.device)[None, :]
        indices[:, :_SWA_TILE].copy_(
            torch.where(swa_positions < lengths[:, None], offsets * _SWA_TILE + swa_positions, -1)
        )
        extra_capacity = self.num_sparse_topk - _SWA_TILE
        if extra_capacity:
            extra_positions = torch.arange(extra_capacity, device=swa.device)[None, :]
            indices[:, _SWA_TILE:].copy_(
                torch.where(
                    extra_positions + _SWA_TILE < lengths[:, None],
                    offsets * extra_capacity + extra_positions,
                    -1,
                )
            )


class CSA2TrtllmAttention(DeepseekV4TrtllmAttention):
    """Reuse DSV4's forward facade and TRTLLM/AttentionOp sparse computation.

    CSA2's module owns its compressor/indexer and persistent packed caches.
    Only selected QAT values are decoded to BF16 staging for this native kernel;
    this does not claim that trtllm-gen directly reads the packed FP4 format.
    Instances own mutable scratch and, like the base backend, are not reentrant.
    """

    Metadata = CSA2TrtllmMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int = 1,
        sparse_params: CSA2Params | None = None,
        mla_params: MLAParams | None = None,
        **kwargs,
    ) -> None:
        if not is_sm_100f():
            raise ValueError("CSA2 TRTLLM sparse MLA requires an SM100-family GPU")
        if head_dim != _HEAD_DIM:
            raise ValueError("CSA2 TRTLLM sparse MLA requires head_dim=512")
        if num_kv_heads != 1:
            raise ValueError("CSA2 TRTLLM sparse MLA requires one KV head")
        if kwargs.get("kv_cache_dtype", "auto") not in ("auto", "bfloat16"):
            raise ValueError("CSA2 TRTLLM staging uses BF16 KV, without native cache quantization")
        quant_config = kwargs.get("quant_config")
        if quant_config is not None and quant_config.kv_cache_quant_algo is not None:
            raise ValueError("CSA2 TRTLLM staging uses BF16 KV, without native cache quantization")
        sparse_params = sparse_params or CSA2Params()
        if mla_params is None:
            mla_params = MLAParams(
                q_lora_rank=1280,
                kv_lora_rank=448,
                qk_nope_head_dim=448,
                qk_rope_head_dim=64,
                v_head_dim=512,
                rope_append=False,
            )
        if (
            mla_params.kv_lora_rank,
            mla_params.qk_nope_head_dim,
            mla_params.qk_rope_head_dim,
            mla_params.v_head_dim,
            mla_params.rope_append,
        ) != (448, 448, 64, 512, False):
            raise ValueError(
                "CSA2 TRTLLM MLA requires rank/nope 448, RoPE 64, V 512 and rope_append=False"
            )
        # V4's constructor would create a second compressor/indexer and uses
        # different ratio semantics. Reuse its forward, not those owned modules.
        TrtllmAttention.__init__(
            self,
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            sparse_params=sparse_params,
            mla_params=mla_params,
            **kwargs,
        )
        self.local_layer_idx = 0  # one physical compute-pool slot in this view
        self._prepared: dict[int, CSA2TrtllmMetadata] = {}

    def _prepare_sparse_forward_args(
        self, metadata: CSA2TrtllmMetadata, forward_args: AttentionForwardArgs
    ) -> None:
        sparse = forward_args.sparse_runtime_params
        sparse.sparse_attn_kv_lens = metadata.prepared_lens[: metadata.num_tokens]
        sparse.aux_kv_cache_pool_ptr = (
            metadata.extra_pool.data_ptr() if metadata.num_sparse_topk > _SWA_TILE else None
        )

    def sparse_attn_predict(
        self, q, k, metadata: CSA2TrtllmMetadata, forward_args: AttentionForwardArgs
    ):
        return metadata.prepared_indices[: metadata.num_tokens], None

    def forward_selected(
        self,
        q: torch.Tensor,
        swa: torch.Tensor,
        extra: torch.Tensor | None,
        swa_valid: torch.Tensor,
        extra_valid: torch.Tensor | None,
        sink: torch.Tensor,
    ) -> torch.Tensor:
        with torch.cuda.device(q.device):
            return self._forward_selected(q, swa, extra, swa_valid, extra_valid, sink)

    def _forward_selected(self, q, swa, extra, swa_valid, extra_valid, sink):
        if q.ndim != 3 or q.shape[1:] != (self.num_heads, _HEAD_DIM):
            raise ValueError("CSA2 Q geometry must match its TRTLLM backend")
        count = q.shape[0]
        if count == 0:
            return torch.empty_like(q)
        if (
            q.dtype != torch.bfloat16
            or swa.dtype != torch.bfloat16
            or (extra is not None and extra.dtype != torch.bfloat16)
        ):
            raise ValueError("Prepared CSA2 Q and KV rows must be BF16")
        if count > self.sparse_params.max_query_tokens:
            raise ValueError("CSA2 query tile exceeds its configured workspace capacity")
        extra_width = 0 if extra is None else extra.shape[1]
        extra_capacity = (extra_width + _SWA_TILE - 1) // _SWA_TILE * _SWA_TILE
        capturing = torch.cuda.is_current_stream_capturing()
        metadata = self._prepared.get(extra_capacity)
        if metadata is None:
            if capturing:
                raise RuntimeError("Warm up the CSA2 TRTLLM query shape before CUDA Graph capture")
            capacity = self.sparse_params.max_query_tokens
            metadata = CSA2TrtllmMetadata(max_num_requests=capacity, max_num_tokens=capacity)
            metadata.allocate_prepared(capacity, self.num_heads, extra_capacity, q.device)
            self._prepared[extra_capacity] = metadata
            if count < capacity:
                # Size the native workspace for the largest admitted query
                # tile before any smaller shape can capture its pointer.
                self._forward_selected(
                    torch.zeros(
                        (capacity, self.num_heads, _HEAD_DIM), dtype=q.dtype, device=q.device
                    ),
                    torch.zeros(
                        (capacity, swa.shape[1], _HEAD_DIM), dtype=q.dtype, device=q.device
                    ),
                    None
                    if extra is None
                    else torch.zeros(
                        (capacity, extra_width, _HEAD_DIM), dtype=q.dtype, device=q.device
                    ),
                    torch.zeros((capacity, swa.shape[1]), dtype=torch.bool, device=q.device),
                    None
                    if extra is None
                    else torch.zeros((capacity, extra_width), dtype=torch.bool, device=q.device),
                    sink,
                )
        if metadata.swa_pool.device != q.device:
            raise ValueError("A CSA2 TRTLLM backend instance is bound to its first CUDA device")
        if capturing and count not in metadata.warmed_query_counts:
            raise RuntimeError("Warm up the CSA2 TRTLLM query count before CUDA Graph capture")
        metadata.is_cuda_graph = capturing
        metadata.bind_prepared(swa, extra, swa_valid, extra_valid)
        output = torch.empty_like(q)
        args = AttentionForwardArgs(
            attention_input_type=AttentionInputType.generation_only,
            output=output.view(count, -1),
            attention_sinks=sink.float(),
            attention_window_size=metadata.num_sparse_topk,
            cu_q_seqlens=metadata.prepared_cu_q[: count + 1],
            cu_kv_seqlens=metadata.prepared_cu_kv[: count + 1],
            fmha_scheduler_counter=metadata.prepared_counter,
            # Required by the native MLA ABI, but generation consumes the
            # already prepared Q/pools. Do not request another RoPE/append.
            latent_cache=metadata.swa_pool[:count, 0],
            q_pe=q[..., 448:],
        )
        self.forward(q.reshape(count, -1), None, None, metadata, forward_args=args)
        if not capturing:
            metadata.warmed_query_counts.add(count)
        return output
