# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 attention backends using the standard TRTLLM sparse contract."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._utils import get_sm_version, is_sm_100f

from .indexer import CSA2Indexer
from .metadata import CSA2TrtllmMetadata
from .params import CSA2BackendForwardArgs, CSA2Mode, CSA2Params, select_csa2_backend

_HEAD_DIM = 512


class CSA2TrtllmAttention(TrtllmAttention):
    """CSA2 sparse prediction with the inherited native trtllm-gen forward."""

    Metadata = CSA2TrtllmMetadata
    compute_backend = "trtllm"

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
        if sparse_params.compute_backend not in ("auto", self.compute_backend):
            raise ValueError("CSA2 backend class and requested implementation disagree")
        if self.compute_backend == "trtllm" and not is_sm_100f():
            raise ValueError("CSA2 trtllm-gen requires an SM100-family GPU")
        self.indexer = None
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
        super().__init__(
            layer_idx,
            num_heads,
            head_dim,
            num_kv_heads=num_kv_heads,
            sparse_params=sparse_params,
            mla_params=mla_params,
            **kwargs,
        )
        self.local_layer_idx = 0  # one physical compute-pool slot in this view

    def sparse_attn_predict(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        metadata: CSA2TrtllmMetadata,
        forward_args: AttentionForwardArgs,
    ) -> tuple[torch.Tensor, None]:
        inputs = forward_args.sparse_backend_args
        if not isinstance(inputs, CSA2BackendForwardArgs):
            raise TypeError("CSA2 requires CSA2BackendForwardArgs")
        count = metadata.num_tokens
        if (
            q.shape != (count, self.num_heads * _HEAD_DIM)
            or q.dtype != torch.bfloat16
            or metadata.num_query_heads != self.num_heads
        ):
            raise ValueError("CSA2 Q geometry/dtype must match its BF16 metadata")
        if count > self.sparse_params.max_query_tokens:
            raise ValueError("CSA2 query tile exceeds its configured capacity")
        if forward_args.attention_input_type != AttentionInputType.generation_only:
            raise ValueError("Prepared CSA2 query metadata requires generation_only compute")
        if (
            metadata.is_cuda_graph
            and self.compute_backend == "trtllm"
            and metadata.workspace.numel() == 0
        ):
            raise RuntimeError("Warm up CSA2 metadata with forward before CUDA Graph capture")
        if inputs.state is not None:
            state = inputs.state
            source = state.metadata
            manager = source.kv_cache_manager
            layout = self.sparse_params.layout
            if layout is None:
                raise ValueError("CSA2 module inputs require the model layout")
            layer = layout.layer(self.layer_idx)
            start, end = inputs.query_start, inputs.query_start + count
            if start == 0:
                source.enter_layer(layer)
                manager.write_swa(
                    self.layer_idx, source.csa2_swa_write_slots[self.layer_idx], state.swa_kv
                )
                if layer.mode == CSA2Mode.FULL:
                    if state.main_kv is None or state.index_k is None:
                        raise ValueError(
                            "Full mode requires main and index rows, including empty rows"
                        )
                    manager.write_global(
                        layer.kv_source,
                        source.csa2_main_write_slots[layer.kv_source],
                        state.main_kv,
                        state.index_k,
                    )
                elif state.main_kv is not None or state.index_k is not None:
                    raise ValueError("Only Full mode may write shared main/index caches")
            logical = None
            if layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
                if state.index_q is None or state.index_weights is None:
                    raise ValueError("Full/Reindex mode requires index queries and weights")
                if self.indexer is None:
                    self.indexer = CSA2Indexer(
                        layout, self.layer_idx, state.index_q.shape[1], state.index_q.shape[-1]
                    )
                # Indexer phases and TP query splitting use the complete model
                # batch. FMHA query tiles only consume the published selections.
                if start == 0:
                    self.indexer(state, 0, state.swa_kv.shape[0])
                logical = source.csa2_indices[self.layer_idx][start:end]
            elif layer.mode == CSA2Mode.REUSE:
                if layer.index_source not in source.csa2_indices:
                    raise ValueError("CSA2 index source did not run in this forward")
                indices = source.csa2_indices[layer.index_source]
                if indices.shape[0] != state.swa_kv.shape[0]:
                    raise ValueError("CSA2 Reuse rows must match their source's packed query order")
                logical = indices[start:end]
            slots = None
            if logical is not None:
                slots = source.global_slot_tile(self.layer_idx, start, end, logical)
                slots = torch.where(
                    logical < source.csa2_visible_lengths[self.layer_idx][start:end, None],
                    slots,
                    -1,
                )
            inputs = CSA2BackendForwardArgs(
                swa_pool=manager.get_swa_buffer(self.layer_idx),
                swa_indices=source.csa2_swa_indices[self.layer_idx][start:end],
                main_pool=None if logical is None else manager.get_main_buffer(layer.kv_source),
                topk_indices=slots,
            )
        if metadata.swa_pool.device != q.device:
            raise ValueError("CSA2 Q and metadata must be on the same CUDA device")
        metadata.stage_selected(inputs)
        sparse = forward_args.sparse_runtime_params
        sparse.sparse_attn_kv_lens = metadata.prepared_lens
        sparse.aux_kv_cache_pool_ptr = (
            metadata.extra_pool.data_ptr()
            if inputs.topk_indices is not None and metadata.num_sparse_topk > 128
            else None
        )
        forward_args.fmha_scheduler_counter = metadata.prepared_counter
        forward_args.attention_window_size = metadata.num_sparse_topk
        forward_args.latent_cache = metadata.swa_pool[:, 0]
        forward_args.q_pe = q.view(count, self.num_heads, _HEAD_DIM)[..., 448:]
        return metadata.prepared_indices, None


class CSA2FlashInferAttention(CSA2TrtllmAttention):
    """CSA2 sparse contract with FlashInfer BF16 FA2 computation."""

    compute_backend = "flashinfer"


class CSA2FlashMLAAttention(CSA2TrtllmAttention):
    """CSA2 sparse contract with FlashMLA BF16 sparse computation."""

    compute_backend = "flash_mla"


def get_csa2_backend(params: CSA2Params) -> type[CSA2TrtllmAttention]:
    """Select a backend class without changing the TRTLLM factory contract."""
    choice = params.compute_backend
    if choice == "auto":
        choice = select_csa2_backend(get_sm_version())
    backends = {
        "trtllm": CSA2TrtllmAttention,
        "flashinfer": CSA2FlashInferAttention,
        "flash_mla": CSA2FlashMLAAttention,
    }
    if choice not in backends:
        raise ValueError(f"Unknown CSA2 attention backend: {choice}")
    return backends[choice]
