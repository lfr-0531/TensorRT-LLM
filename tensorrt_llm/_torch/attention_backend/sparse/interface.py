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
    def single_request_sparse_attn_forward(
            self, q: Tensor, key_states: Tensor, value_states: Tensor,
            is_causal: bool, attn_mask: Tensor,
            sparse_indices: Optional[Tensor]) -> Tensor:
        # Select the key and value states using the sparse indices
        if sparse_indices is not None:
            key_states = triton_index_gather(key_states, sparse_indices)
            value_states = triton_index_gather(value_states, sparse_indices)

        # Attention forward
        attn_output = self._single_request_attn_forward(q, key_states,
                                                        value_states, is_causal,
                                                        attn_mask)
        return attn_output

    def single_request_update_sparse_kv_cache(
        self,
        k: Optional[Tensor],
        v: Optional[Tensor],
        kv_cache_tensor: Tensor,
        seq_len: int,
        cache_idx: int,
        cache_position: Tensor,
        sparse_kv_indices: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        # Select tokens using the sparse kv indices
        if sparse_kv_indices is not None:
            k_selected = triton_index_gather(k, sparse_kv_indices)
            v_selected = triton_index_gather(v, sparse_kv_indices)
        else:
            k_selected, v_selected = k, v

        # Get kv cache tensor
        k_out = kv_cache_tensor[cache_idx, 0, :, :, :].unsqueeze(0)
        v_out = kv_cache_tensor[cache_idx, 1, :, :, :].unsqueeze(0)

        # Update kv cache
        if k is not None and v is not None:
            access_type = self._access_type[k.dtype.itemsize]
            k_out.view(dtype=access_type).index_copy_(
                1, cache_position, k_selected.view(dtype=access_type))
            v_out.view(dtype=access_type).index_copy_(
                1, cache_position, v_selected.view(dtype=access_type))

        return k_out[:, :seq_len, :, :], v_out[:, :seq_len, :, :]

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
        # Preprocess inputs
        q, k, v, kv_len = self._single_request_preprocess_inputs(
            q, k, v, kv_cache_tensor.dtype)

        # Predict sparse kv indices
        sparse_kv_indices = self.single_request_sparse_kv_predict(
            q, k, v, metadata, past_seen_token, cache_idx)

        # Get target seq len
        target_seq_len = past_seen_token
        if sparse_kv_indices is not None:
            target_seq_len += sparse_kv_indices.size(1)
        else:
            target_seq_len += kv_len
        cache_position = torch.arange(past_seen_token,
                                      target_seq_len,
                                      device=q.device)

        # Update sparse kv cache
        key_states, value_states = self.single_request_update_sparse_kv_cache(
            k, v, kv_cache_tensor, target_seq_len, cache_idx, cache_position,
            sparse_kv_indices)

        # Predict sparse attn indices
        sparse_indices = self.single_request_sparse_attn_predict(
            q, k, v, kv_cache_tensor, metadata, past_seen_token, cache_idx)

        # Create attention mask
        attn_mask, is_causal = self._single_request_create_attention_mask(
            attention_mask, past_seen_token, target_seq_len, cache_position,
            q.device, q.size(2), attention_window_size)

        # Apply sparse attention calculation
        attn_output = self.single_request_sparse_attn_forward(
            q, key_states, value_states, is_causal, attn_mask, sparse_indices)

        return attn_output.squeeze(0)
