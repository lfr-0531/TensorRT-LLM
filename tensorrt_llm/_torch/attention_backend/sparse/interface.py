from abc import abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor

from tensorrt_llm.models.modeling_utils import QuantConfig

from ..interface import AttentionMask
from ..vanilla import VanillaAttention, VanillaAttentionMetadata
from .kernel import triton_index_gather


class SparseAttentionMetadata(VanillaAttentionMetadata):
    """Base metadata class for sparse attention algorithms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class VanillaSparseAttention(VanillaAttention):
    """
    Abstract base class for sparse attention algorithms.

    This class provides a framework for implementing different sparse attention
    algorithms by requiring subclasses to implement specific prediction and
    calculation methods.
    """

    Metadata = SparseAttentionMetadata

    _access_type = {
        1: torch.int8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64
    }

    def __init__(
            self,
            layer_idx: int,
            num_heads: int,
            head_dim: int,
            num_kv_heads: Optional[int] = None,
            quant_config: Optional[QuantConfig] = None,
            q_scaling: Optional[float] = None,
            sparse_attention_config: Optional["SparseAttentionConfig"] = None,
            **kwargs):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config, q_scaling, **kwargs)
        self.sparse_attention_config = sparse_attention_config
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

    @abstractmethod
    def single_request_sparse_attn_predict(self, q: Tensor, k: Optional[Tensor],
                                           v: Optional[Tensor],
                                           kv_cache_tensor: Tensor,
                                           metadata: SparseAttentionMetadata,
                                           past_seen_token: int, cache_idx: int,
                                           **kwargs) -> Optional[Tensor]:
        pass

    @abstractmethod
    def single_request_sparse_kv_predict(self, q: Optional[Tensor],
                                         k: Optional[Tensor],
                                         v: Optional[Tensor],
                                         metadata: SparseAttentionMetadata,
                                         past_seen_token: int, cache_idx: int,
                                         **kwargs) -> Optional[Tensor]:
        pass

    @torch.compile(dynamic=True)
    def single_request_sparse_attn_forward(self, q: Tensor, key_states: Tensor,
                                           value_states: Tensor,
                                           calc_indices: Optional[Tensor],
                                           cache_idx: int, target_seq_len: int,
                                           kv_cache_tensor: Tensor,
                                           attention_mask: AttentionMask,
                                           past_seen_token: int,
                                           metadata: SparseAttentionMetadata,
                                           **kwargs) -> Tensor:

        if calc_indices is not None:
            # Gather the selected indices
            key_states = triton_index_gather(key_states, calc_indices)
            value_states = triton_index_gather(value_states, calc_indices)

        bsz, num_heads, q_len, head_dim = q.shape

        key_states = key_states.transpose(1, 2).to(q.dtype)
        value_states = value_states.transpose(1, 2).to(q.dtype)

        is_causal, attn_mask = self._create_attention_mask(
            attention_mask, past_seen_token, target_seq_len, q.device, q_len)

        attn_output = self._compute_attention(q, key_states, value_states,
                                              is_causal, attn_mask)

        return attn_output

    def update_sparse_kv_cache(
        self,
        k: Optional[Tensor],
        v: Optional[Tensor],
        kv_cache_tensor: Tensor,
        past_seen_token: int,
        kv_indices: Tensor,
        cache_idx: int,
        metadata: Optional[SparseAttentionMetadata] = None
    ) -> Tuple[Tensor, Tensor]:
        # Select tokens from input using returned indices
        k_selected = triton_index_gather(
            k, kv_indices) if len(kv_indices) > 0 else k
        v_selected = triton_index_gather(
            v, kv_indices) if len(kv_indices) > 0 else v

        # Calculate cache write positions
        cache_write_indices = torch.arange(past_seen_token,
                                           past_seen_token + k_selected.size(1),
                                           device=k.device)

        k_out = kv_cache_tensor[cache_idx, 0, :, :, :].unsqueeze(0)
        v_out = kv_cache_tensor[cache_idx, 1, :, :, :].unsqueeze(0)

        if k is not None and v is not None:
            access_type = self._access_type[k.dtype.itemsize]
            k_out.view(dtype=access_type).index_copy_(
                1, cache_write_indices, k_selected.view(dtype=access_type))
            v_out.view(dtype=access_type).index_copy_(
                1, cache_write_indices, v_selected.view(dtype=access_type))

        # concat the new key and value states with the kv cache for gather later
        return torch.cat([k_out[:, :past_seen_token, :, :], k],
                         dim=1), torch.cat(
                             [v_out[:, :past_seen_token, :, :], v], dim=1)

    def _single_request_forward(self,
                                q,
                                k,
                                v,
                                attention_mask: AttentionMask,
                                kv_cache_tensor,
                                past_seen_token,
                                cache_idx,
                                metadata: SparseAttentionMetadata,
                                attention_window_size: Optional[int] = None,
                                **kwargs):
        q, k, v, _ = self._preprocess_inputs(q, k, v, kv_cache_tensor,
                                             past_seen_token)

        # Predict indices for writing to KV cache
        kv_indices = self.single_request_sparse_kv_predict(
            q, k, v, metadata, past_seen_token, cache_idx, **kwargs)

        target_seq_len = past_seen_token
        if kv_indices is not None:
            target_seq_len += kv_indices.size(1)

        if k is not None and v is not None:
            key_states, value_states = self.update_sparse_kv_cache(
                k, v, kv_cache_tensor, past_seen_token, kv_indices, cache_idx,
                metadata)

        # Predict indices for attention calculation (for kv cache)
        attn_indices = self.single_request_sparse_attn_predict(
            q, k, v, kv_cache_tensor, metadata, past_seen_token, cache_idx,
            **kwargs)

        # Apply sparse attention calculation
        attn_output = self.single_request_sparse_attn_forward(
            q, key_states, value_states, attn_indices, cache_idx,
            target_seq_len, kv_cache_tensor, attention_mask, past_seen_token,
            metadata, **kwargs)

        return attn_output.squeeze(0)
