# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 attention module for integration into the causal encoder/decoder."""

from __future__ import annotations

import torch
from torch import nn
from transformers import PretrainedConfig

from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.mapping import Mapping

from ..deepseek_v4.module import project_sparse_attn_output
from .backend import CSA2Batch, CSA2Cache, CSA2Routing, DeepseekV41SparseAttention
from .compressor import CSA2CompressionBatch, CSA2Compressor
from .params import CSA2Layout, CSA2Mode


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    values = x.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight.float()).to(
        x.dtype
    )


class DeepseekV41Attention(nn.Module):
    """Attention component with explicit CED and paged-cache inputs.

    The model supplies packed query positions, cache mappings, and RoPE
    parameters (plain SWA versus compressed YaRN). Decoder global projection
    can consume encoder states independently of the decoder's query/SWA rows.
    FP8 checkpoint weights are dequantized to BF16 here; FP8 GEMM execution
    is separate from the cache quantization contract.
    """

    def __init__(
        self,
        layout: CSA2Layout,
        layer_idx: int,
        pos_embd_params: PositionalEmbeddingParams,
        hidden_size: int = 5120,
        num_heads: int = 64,
        head_dim: int = 512,
        rope_head_dim: int = 64,
        q_lora_rank: int = 1280,
        o_lora_rank: int = 1024,
        num_groups: int = 8,
        index_heads: int = 32,
        index_head_dim: int = 128,
        eps: float = 1e-20,
        mapping: Mapping | None = None,
    ) -> None:
        super().__init__()
        if num_heads % num_groups or head_dim <= rope_head_dim:
            raise ValueError("Invalid CSA2 head/group geometry")
        mapping = mapping or Mapping()
        if mapping.pp_size != 1 or mapping.cp_size != 1:
            raise ValueError(
                "CSA2 source state requires colocated layers; PP/CP transfer is not implemented"
            )
        if num_heads % mapping.tp_size or num_groups % mapping.tp_size:
            raise ValueError("CSA2 heads and output groups must be divisible by TP size")
        self.mapping = mapping
        self.num_groups = num_groups
        self.layer = layout.layer(layer_idx)
        self.backend = DeepseekV41SparseAttention(layout, layer_idx, use_flash_mla=head_dim == 512)
        self.num_heads_tp = num_heads // mapping.tp_size
        self.qk_head_dim = self.v_head_dim = head_dim
        self.qk_rope_head_dim = rope_head_dim
        self.qk_nope_head_dim = head_dim - rope_head_dim
        self.n_local_groups = num_groups // mapping.tp_size
        self.o_lora_rank = o_lora_rank
        self.eps = eps
        self.index_heads = index_heads
        self.index_head_dim = index_head_dim
        self.wq_a = nn.Linear(hidden_size, q_lora_rank, bias=False, dtype=torch.bfloat16)
        self.wq_b = Linear(
            q_lora_rank,
            num_heads * head_dim,
            bias=False,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
        )
        self.wkv = nn.Linear(hidden_size, head_dim, bias=False, dtype=torch.bfloat16)
        self.q_norm_weight = nn.Parameter(torch.ones(q_lora_rank, dtype=torch.bfloat16))
        self.kv_norm_weight = nn.Parameter(torch.ones(head_dim, dtype=torch.bfloat16))
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads_tp, dtype=torch.float32))
        # Use V4's grouped O-LoRA projection and inverse-RoPE implementation.
        self.o_a_proj = nn.Parameter(
            torch.empty(
                self.n_local_groups,
                o_lora_rank,
                num_heads * head_dim // num_groups,
                dtype=torch.bfloat16,
            )
        )
        self.o_b_proj = Linear(
            num_groups * o_lora_rank,
            hidden_size,
            bias=False,
            dtype=torch.bfloat16,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
        )
        self.rotary_emb = RotaryEmbedding(
            pos_embd_params.rope, head_dim=rope_head_dim, is_neox=pos_embd_params.is_neox
        )
        self.inverse_rotary_emb = RotaryEmbedding(
            pos_embd_params.rope,
            head_dim=rope_head_dim,
            is_neox=pos_embd_params.is_neox,
            inverse=True,
        )
        self.compressor = None
        if self.layer.mode == CSA2Mode.FULL:
            self.compressor = CSA2Compressor(hidden_size, head_dim, self.layer.compress_ratio, eps)
            self.index_wk = nn.Linear(head_dim, index_head_dim, bias=False, dtype=torch.bfloat16)
            self.index_k_norm_weight = nn.Parameter(
                torch.ones(index_head_dim, dtype=torch.bfloat16)
            )
        if self.layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            self.index_wq_b = nn.Linear(
                q_lora_rank, index_heads * index_head_dim, bias=False, dtype=torch.bfloat16
            )
            self.index_weights_proj = nn.Linear(
                hidden_size, index_heads, bias=False, dtype=torch.bfloat16
            )

    @classmethod
    def from_hf_config(
        cls, config: PretrainedConfig, layer_idx: int, mapping: Mapping | None = None
    ) -> DeepseekV41Attention:
        """Construct an attention component from the published text configuration.

        The CED model owns layer execution and passes its text_config here;
        this does not register the incomplete multimodal model with AutoModel.
        """
        layout = CSA2Layout(
            tuple(config.compress_ratios),
            tuple(config.kv_source_layer_ids),
            tuple(config.index_source_layer_ids),
            config.candidate_source_layer_id,
            config.candidate_topk_blocks,
            config.candidate_block_size,
            config.index_topk,
            config.sliding_window,
        )
        rope = RopeParams.from_config(config)
        if layout.layer(layer_idx).compress_ratio:
            rope.theta = config.compress_rope_theta
            rope.scale_type = RotaryScalingType.yarn
            rope.mscale = 0.0
            rope.mscale_all_dim = 0.0
        else:
            rope.theta = config.rope_theta
            rope.scale_type = RotaryScalingType.none
            rope.scale = 1.0
        positional = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gptj,
            rope=rope,
            is_neox=False,
        )
        return cls(
            layout,
            layer_idx,
            positional,
            config.hidden_size,
            config.num_attention_heads,
            config.head_dim,
            config.qk_rope_head_dim,
            config.q_lora_rank,
            config.o_lora_rank,
            config.o_groups,
            config.index_n_heads,
            config.index_head_dim,
            config.rms_norm_eps,
            mapping=mapping,
        )

    def load_hf_weights(self, weights: dict[str, torch.Tensor], prefix: str = "") -> None:
        """Load one HF attention with validated FP8 scales and global TP shapes."""
        from .weights import load_attention_weights

        load_attention_weights(
            self,
            weights,
            prefix,
            self.num_groups,
            self.o_lora_rank,
            self.mapping.tp_size,
            self.mapping.tp_rank,
        )

    def _rope(self, x: torch.Tensor, positions: torch.Tensor, heads: int) -> torch.Tensor:
        torch.ops.trtllm.mla_rope_inplace(
            x,
            positions.reshape(-1),
            self.rotary_emb.rotary_cos_sin,
            heads,
            x.shape[-1] - self.qk_rope_head_dim,
            self.qk_rope_head_dim,
            False,
            self.rotary_emb.is_neox,
        )
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        cache: CSA2Cache,
        batch: CSA2Batch,
        routing: CSA2Routing,
        *,
        compression: CSA2CompressionBatch | None = None,
        compressed_positions: torch.Tensor | None = None,
        global_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [tokens, hidden_size]; all packed rows must retain source order.

        ``compressed_positions`` contains the first token position of each
        completed group. ``global_hidden_states`` is the encoder output when
        decoder global KV is prepared from more rows than its SWA replay.
        """
        qr = _rms_norm(self.wq_a(hidden_states), self.q_norm_weight, self.eps)
        # Unlike V4, CSA2 does not normalize individual projected Q heads.
        q = self._rope(
            self.wq_b(qr).reshape(-1, self.num_heads_tp, self.qk_head_dim),
            positions,
            self.num_heads_tp,
        )
        swa = _rms_norm(self.wkv(hidden_states), self.kv_norm_weight, self.eps)
        swa = self._rope(swa.unsqueeze(1), positions, 1).squeeze(1)
        index_q = index_weights = main_kv = index_k = None
        if self.compressor is not None:
            if compressed_positions is None:
                raise ValueError("Full CSA2 mode requires compressed positions")
            global_input = hidden_states if global_hidden_states is None else global_hidden_states
            latent = self.compressor(global_input, compression)
            # Derive index K before mutating the main latent with RoPE.
            index_k = _rms_norm(self.index_wk(latent), self.index_k_norm_weight, self.eps)
            index_k = self._rope(index_k.unsqueeze(1), compressed_positions, 1).squeeze(1)
            main_kv = self._rope(latent.unsqueeze(1), compressed_positions, 1).squeeze(1)
        if self.layer.mode in (CSA2Mode.FULL, CSA2Mode.REINDEX):
            index_q = self._rope(
                self.index_wq_b(qr).reshape(-1, self.index_heads, self.index_head_dim),
                positions,
                self.index_heads,
            )
            index_weights = self.index_weights_proj(hidden_states) * (
                self.index_head_dim**-0.5 * self.index_heads**-0.5
            )
        output = self.backend.forward(
            q,
            swa,
            self.attn_sink,
            cache,
            batch,
            routing,
            index_q=index_q,
            index_weights=index_weights,
            main_kv=main_kv,
            index_k=index_k,
        )
        return project_sparse_attn_output(self, [output.flatten(1)], positions)
