# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 attention backends using the standard TRTLLM sparse contract."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs, MLAParams
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
from tensorrt_llm._utils import get_sm_version, is_sm_100f

from .metadata import CSA2TrtllmMetadata
from .params import CSA2Params, select_csa2_backend
from .prediction import predict_sparse_attention

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
        return predict_sparse_attention(self, q, k, metadata, forward_args)


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
