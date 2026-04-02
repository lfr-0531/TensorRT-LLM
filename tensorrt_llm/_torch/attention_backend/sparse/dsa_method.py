# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DSA (Dynamic Sparse Attention) method for MLA.

Implements the SparseAttentionMethod protocol with DSA-specific logic:
two-phase forward (graph-capturable projections + non-capturable attention),
sparse FlashMLA kernels (SM90), and absorption-path integration (SM100+).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from tensorrt_llm._utils import get_sm_version, nvtx_range, nvtx_range_debug
from tensorrt_llm.logger import logger

from ..interface import AttentionMetadata
from .dsa import DSAtrtllmAttentionMetadata, transform_local_topk_and_prepare_pool_view

# Import FlashMLA sparse attention kernel
try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None

if TYPE_CHECKING:
    from ...modules.attention import MLA


def _should_use_short_mha(
    mla: MLA, attn_metadata: AttentionMetadata, position_ids: Optional[torch.Tensor]
) -> bool:
    """Check if the short-seq MHA optimization should be used for context.

    Uses max_ctx_kv_len (max total KV length per context sequence,
    including cached tokens) when available, to correctly account for
    chunked context where the full attention span exceeds the threshold
    even if the new token count is small.  Falls back to num_ctx_tokens
    (total new context tokens) when max_ctx_kv_len is not set.

    Disabled under torch compile so that the split DSA custom ops
    (mla_dsa_proj / mla_dsa_attn_inplace) have unconditionally
    straight-line control flow for CUDA graph capture.
    """
    from ...utils import is_torch_compiling

    if is_torch_compiling():
        return False
    if not (
        mla.short_seq_mha_threshold > 0
        and not mla.apply_rotary_emb
        and mla.mapping.cp_size == 1
        and position_ids is not None
    ):
        return False
    effective_len = getattr(attn_metadata, "max_ctx_kv_len", attn_metadata.num_ctx_tokens)
    return effective_len <= mla.short_seq_mha_threshold


def _forward_context_dsa(
    mla: MLA,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run context-phase attention for DSA models.

    Dispatches to the short-seq MHA path (forward_context) when the max
    per-sequence KV length (including cached tokens) is within the
    threshold, or falls through to the absorption/sparse MLA path
    otherwise.
    """
    if _should_use_short_mha(mla, attn_metadata, position_ids):
        return mla.forward_context(
            q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
        )

    if get_sm_version() >= 100:
        return mla.forward_absorption_context(
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            latent_cache=latent_cache,
            topk_indices=topk_indices,
        )
    else:
        return _forward_sparse_mla_kvcache_bf16(
            mla, q, latent_cache, attn_metadata, output, topk_indices, is_generation=False
        )


def _forward_generation_dsa(
    mla: MLA,
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    attn_metadata: AttentionMetadata,
    output: torch.Tensor,
    latent_cache: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run generation-phase attention for DSA models."""
    if get_sm_version() >= 100:
        return mla.forward_absorption_generation(
            q,
            compressed_kv,
            k_pe,
            attn_metadata,
            output,
            latent_cache=latent_cache,
            topk_indices=topk_indices,
        )
    else:
        return _forward_sparse_mla_kvcache_bf16(
            mla, q, latent_cache, attn_metadata, output, topk_indices, is_generation=True
        )


@nvtx_range("forward_sparse_mla_kvcache_bf16")
def _forward_sparse_mla_kvcache_bf16(
    mla: MLA,
    q: torch.Tensor,
    latent_cache: torch.Tensor,
    attn_metadata: DSAtrtllmAttentionMetadata,
    output: torch.Tensor,
    topk_indices: torch.Tensor,
    is_generation: bool = False,
) -> torch.Tensor:
    """Forward sparse MLA (DSA) for BF16 KV cache using FlashMLA kernels.

    To form the input for FlashMLA kernel and adapt our KV cache manager:
    1. Append current tokens to paged cache and apply rope to q/k via
       mla_rope_append_paged_kv_assign_q
    2. Load full kv cache from paged memory (with k rope applied)
    3. Call FlashMLA sparse attention kernel for sparse prefill/decode
    """
    from ...modules.attention import fp8_block_scaling_bmm_out

    assert isinstance(attn_metadata, DSAtrtllmAttentionMetadata), (
        "DSA requires DSAtrtllmAttentionMetadata"
    )
    trtllm_attention = mla.mqa
    with nvtx_range_debug(f"mla_rope_append_paged_kv_assign_q_is_generation={is_generation}"):
        trtllm_attention.mla_rope_append_paged_kv_assign_q(
            q, latent_cache, attn_metadata, is_generation=is_generation
        )

    num_tokens = q.shape[0]
    q_nope, q_rope = q.view(-1, mla.num_heads_tp, mla.qk_head_dim).split(
        [mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1
    )
    q_nope_out = torch.empty(
        [num_tokens, mla.num_heads_tp, mla.kv_lora_rank],
        dtype=q.dtype,
        device=q.device,
    )

    if mla.k_b_proj_trans.dtype == torch.bfloat16:
        q_nope_t = q_nope.transpose(0, 1)
        q_nope_out = q_nope_out.transpose(0, 1)
        torch.ops.trtllm.bmm_out(q_nope_t, mla.k_b_proj_trans.transpose(1, 2), q_nope_out)
    elif mla.k_b_proj_trans.dtype == torch.float8_e4m3fn:
        q_nope_out = q_nope_out.transpose(0, 1)
        fp8_block_scaling_bmm_out(
            q_nope,
            mla.k_b_proj_trans,
            mla.k_b_proj_trans_scale,
            q_nope_out,
            mla.k_b_proj_trans_dequant,
            mla.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {mla.k_b_proj_trans.dtype}.")

    q_nope_out = q_nope_out.transpose(0, 1)
    q_concat = torch.cat([q_nope_out, q_rope], dim=-1)

    sm_version = get_sm_version()
    if sm_version >= 100:
        padding = 128
        assert mla.num_heads_tp <= padding, (
            f"SM100 FlashMLA sparse kernel requires exactly {padding} heads, "
            f"got {mla.num_heads_tp}. Padding from values > {padding} is not supported."
        )
    else:  # SM90
        padding = ((mla.num_heads_tp + 63) // 64) * 64

    if mla.num_heads_tp != padding:
        logger.warning_once(
            f"Padding num_heads from {mla.num_heads_tp} to {padding} "
            f"due to FlashMLA sparse attention kernel requirement",
            key="sparse_mla_padding_warning",
        )
        q_padded = q_concat.new_empty((num_tokens, padding, q_concat.shape[2]))
        q_padded[:, : mla.num_heads_tp, :] = q_concat
        q_concat = q_padded

    topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
        topk_indices,
        attn_metadata,
        layer_idx=mla.layer_idx,
        is_generation=is_generation,
    )
    topk_indices_pool = topk_indices_pool.view(num_tokens, 1, -1)
    if flash_mla_sparse_fwd is not None:
        attn_out_latent = flash_mla_sparse_fwd(
            q_concat, kv_cache_pool, topk_indices_pool, mla.softmax_scale
        )[0]
    else:
        raise RuntimeError(
            "flash_mla_sparse_fwd not available. Please ensure FlashMLA module is built."
        )

    attn_out_latent = attn_out_latent[:, : mla.num_heads_tp, :]
    attn_out_latent = attn_out_latent.view([-1, mla.num_heads_tp, mla.kv_lora_rank])
    if mla.num_heads_tp != padding:
        attn_out_latent = attn_out_latent.contiguous()

    assert attn_out_latent.shape[0] == q.shape[0] and attn_out_latent.shape[1] == mla.num_heads_tp

    attn_output = output.view([num_tokens, mla.num_heads_tp, mla.v_head_dim])

    if mla.v_b_proj.dtype == torch.bfloat16:
        torch.ops.trtllm.bmm_out(
            attn_out_latent.transpose(0, 1),
            mla.v_b_proj.transpose(1, 2),
            attn_output.transpose(0, 1),
        )
    elif mla.v_b_proj.dtype == torch.float8_e4m3fn:
        fp8_block_scaling_bmm_out(
            attn_out_latent,
            mla.v_b_proj,
            mla.v_b_proj_scale,
            attn_output.transpose(0, 1),
            mla.v_b_proj_dequant,
            mla.use_cute_dsl_blockscaling_bmm,
        )
    else:
        raise NotImplementedError(f"Missing bmm impl for dtype: {mla.v_b_proj.dtype}.")
    return output


class DSASparseMethod:
    """DSA (Dynamic Sparse Attention) implementation of SparseAttentionMethod.

    Splits the forward pass into two phases for CUDA graph capture:
    - Op 1 (graph-capturable): token-wise projections + indexer pre-projection
    - Op 2 (non-capturable): batch-dependent slicing, sparse routing,
      attention dispatch
    """

    def __init__(self, short_seq_mha_threshold: int = 0):
        self.short_seq_mha_threshold = short_seq_mha_threshold

    def forward(
        self,
        mla: MLA,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Forward pass for DSA MLA (always in MQA mode).

        Delegates to forward_graph_capturable (token-wise projections)
        followed by forward_non_capturable (batch-dependent attention
        dispatch).
        """
        proj_outputs = self.forward_graph_capturable(
            mla, position_ids, hidden_states, attn_metadata
        )
        q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
        indexer_intermediates = proj_outputs[4:]
        self.forward_non_capturable(
            mla,
            indexer_intermediates,
            position_ids,
            attn_metadata,
            output,
            q=q,
            compressed_kv=compressed_kv,
            k_pe=k_pe,
            latent_cache=latent_cache,
        )

    def forward_graph_capturable(
        self,
        mla: MLA,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> List[torch.Tensor]:
        """Token-wise projections for DSA MLA (CUDA-graph-capturable Op 1).

        Runs kv_a_proj, layernorms, q_b_proj, and conditionally
        indexer.pre_indexer_proj().

        IMPORTANT: This method must NOT slice tensors by num_tokens or
        access batch-specific metadata, so that all operations are
        unconditionally straight-line for CUDA graph capture.  Slicing
        to num_tokens happens in forward_non_capturable (Op 2).

        Returns [q, compressed_kv, k_pe, latent_cache] when short-MHA
        handles all tokens (eager only), or
        [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale,
        weights] when the indexer runs.  Under torch compile
        _should_use_short_mha returns False so it is always length 8.
        """
        from ...modules.multi_stream_utils import maybe_execute_in_parallel

        assert mla.mqa is not None, "DSA is only supported in MQA mode"

        q, compressed_kv, k_pe = mla.kv_a_proj_with_mqa(hidden_states).split(
            [mla.q_lora_rank, mla.kv_lora_rank, mla.qk_rope_head_dim], -1
        )

        q, compressed_kv = maybe_execute_in_parallel(
            lambda: mla.q_a_layernorm(q),
            lambda: mla.kv_a_layernorm(compressed_kv),
            mla.ln_events[0],
            mla.ln_events[1],
            mla.aux_stream,
        )
        qr = q
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

        q = mla.q_b_proj(q)

        use_short_mha_for_ctx = _should_use_short_mha(mla, attn_metadata, position_ids)

        # Skip the indexer when the short MHA path handles all context
        # tokens and there are no generation tokens.
        if use_short_mha_for_ctx and attn_metadata.num_generations == 0:
            return [q, compressed_kv, k_pe, latent_cache]

        # pre_indexer_proj is the CUDA-graph-safe portion: pure token-wise
        # compute (cublas_mm, rope, FP8 quantize, weight scaling) with no
        # access to batch-specific metadata or the k cache.
        q_fp8, k_fp8, k_scale, weights = mla.mqa.indexer.pre_indexer_proj(
            qr, hidden_states, position_ids
        )

        return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights]

    def forward_non_capturable(
        self,
        mla: MLA,
        proj_outputs: List[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        *,
        q: Optional[torch.Tensor] = None,
        compressed_kv: Optional[torch.Tensor] = None,
        k_pe: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> None:
        """Batch-structure-dependent attention for DSA MLA (Op 2).

        proj_outputs is [q_fp8, k_fp8, k_scale, weights] when the
        indexer ran in Op 1, or [] when short-MHA handled all tokens.

        All num_tokens slicing happens here (not in Op 1) because
        num_tokens comes from batch-specific metadata and must not be
        baked into CUDA graph capture.
        """
        assert (
            q is not None
            and compressed_kv is not None
            and k_pe is not None
            and latent_cache is not None
        )

        indexer_intermediates = proj_outputs

        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        # Slice Op 1 outputs to actual num_tokens (Op 1 operates on the
        # full padded tensor for CUDA graph compatibility).
        q = q[:num_tokens, ...]
        compressed_kv = compressed_kv[:num_tokens, ...]
        k_pe = k_pe[:num_tokens, ...]
        latent_cache = latent_cache[:num_tokens, ...]
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        use_short_mha_for_ctx = num_contexts > 0 and _should_use_short_mha(
            mla, attn_metadata, position_ids
        )

        if use_short_mha_for_ctx and num_generations == 0:
            topk_indices = None
        else:
            q_fp8, k_fp8, k_scale, weights = indexer_intermediates
            # Slice indexer intermediates to actual num_tokens (they were
            # computed on the full padded tensor in Op 1).
            q_fp8 = q_fp8[:num_tokens, ...]
            k_fp8 = k_fp8[:num_tokens, ...]
            k_scale = k_scale[:num_tokens, ...]
            weights = weights[:num_tokens, ...]
            topk_indices = mla.mqa.indexer.sparse_attn_indexer(
                attn_metadata,
                q,  # only used for shape/device in buffer allocation
                q_fp8,
                k_fp8,
                k_scale,
                weights,
            )

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if mla.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = mla.apply_rope(q_ctx, k_pe_ctx, position_ids)

            _forward_context_dsa(
                mla,
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache_ctx,
                topk_indices=topk_indices[:num_ctx_tokens, :] if topk_indices is not None else None,
                position_ids=position_ids,
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            if mla.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = mla.apply_rope(q_gen, k_pe_gen, position_ids)

            _forward_generation_dsa(
                mla,
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                latent_cache_gen,
                topk_indices=topk_indices[num_ctx_tokens:num_tokens, :],
            )
