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
"""DSA custom ops for piecewise CUDA graph capture.

These ops split the DSA MLA forward into two phases:
- Op 1 (mla_dsa_proj): token-wise projections, CUDA-graph-capturable
- Op 2 (mla_dsa_attn_inplace): batch-dependent attention, NOT captured

They are registered as torch.library custom ops and delegate to the
DSASparseMethod via mla_layer.sparse_method.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from ...modules.attention import extract_extra_attrs


@torch.library.custom_op("trtllm::mla_dsa_proj", mutates_args=())
def mla_dsa_proj(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> List[torch.Tensor]:
    """Token-wise projections for DSA MLA (CUDA-graph-capturable).

    Runs kv_a_proj, layernorms, q_b_proj, and conditionally
    indexer.pre_indexer_proj (FP8 quantize, weight scaling).  Does NOT
    update the indexer k cache — that happens in Op 2 (mla_dsa_attn_inplace)
    because the scatter kernel accesses batch-specific metadata.

    Returns [q, compressed_kv, k_pe, latent_cache] when the short-MHA path
    handles all tokens, or [q, compressed_kv, k_pe, latent_cache, q_fp8,
    k_fp8, k_scale, weights] when the indexer runs.  Under torch compile,
    _should_use_short_mha returns False so the result is always length 8,
    keeping control flow straight-line for CUDA graph capture.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    return mla_layer.sparse_method.forward_graph_capturable(
        mla_layer, position_ids, hidden_states, metadata
    )


@mla_dsa_proj.register_fake
def _mla_dsa_proj_fake(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> List[torch.Tensor]:
    # Under torch compile _should_use_short_mha is False, so always 8 tensors.
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    num_tokens = hidden_states.shape[0]
    indexer = mla_layer.mqa.indexer
    q = hidden_states.new_empty([num_tokens, mla_layer.num_heads_tp * mla_layer.qk_head_dim])
    compressed_kv = hidden_states.new_empty([num_tokens, mla_layer.kv_lora_rank])
    k_pe = hidden_states.new_empty([num_tokens, mla_layer.qk_rope_head_dim])
    latent_cache = hidden_states.new_empty(
        [num_tokens, mla_layer.kv_lora_rank + mla_layer.qk_rope_head_dim]
    )
    # Indexer intermediates: q_fp8, k_fp8, k_scale, weights
    q_fp8 = hidden_states.new_empty(
        [num_tokens, indexer.n_heads, indexer.head_dim], dtype=torch.float8_e4m3fn
    )
    k_fp8 = hidden_states.new_empty([num_tokens, indexer.head_dim], dtype=torch.float8_e4m3fn)
    k_scale = hidden_states.new_empty([num_tokens, 1], dtype=torch.float32)
    weights = hidden_states.new_empty([num_tokens, indexer.n_heads], dtype=torch.float32)
    return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights]


@torch.library.custom_op("trtllm::mla_dsa_attn_inplace", mutates_args=("output",))
def mla_dsa_attn_inplace(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention dispatch for DSA MLA.

    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights] when the
    indexer ran in Op 1, or [] when short-MHA handled all tokens.
    Runs sparse_attn_indexer then dispatches context/generation attention.
    This op is excluded from CUDA graph capture.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    mla_layer.sparse_method.forward_non_capturable(
        mla_layer,
        indexer_intermediates,
        position_ids,
        metadata,
        output,
        q=q,
        compressed_kv=compressed_kv,
        k_pe=k_pe,
        latent_cache=latent_cache,
    )
