from abc import abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor

from tensorrt_llm.models.modeling_utils import QuantConfig

from ..interface import AttentionMask, PredefinedAttentionMask
from ..vanilla import (VanillaAttention, VanillaAttentionMetadata,
                       generate_causal_mask, repeat_kv)


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

    def __init__(self,
                 layer_idx: int,
                 num_heads: int,
                 head_dim: int,
                 num_kv_heads: Optional[int] = None,
                 quant_config: Optional[QuantConfig] = None,
                 q_scaling: Optional[float] = None,
                 **kwargs):
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config, q_scaling, **kwargs)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

    @abstractmethod
    def single_request_sparse_attn_predict(self, q: Tensor, k: Optional[Tensor],
                                           v: Optional[Tensor],
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
            key_states = key_states.gather(1, calc_indices)
            value_states = value_states.gather(1, calc_indices)

        import math

        bsz, num_heads, q_len, head_dim = q.shape

        key_states = key_states.transpose(1, 2).to(q.dtype)
        value_states = value_states.transpose(1, 2).to(q.dtype)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Handle attention mask for sparse attention
        is_causal = False
        attn_mask = None

        if attention_mask == PredefinedAttentionMask.CAUSAL:
            if past_seen_token == 0:
                is_causal = True
            elif q_len != 1:
                cache_position = torch.arange(past_seen_token,
                                              target_seq_len,
                                              device=q.device)
                # attn_mask: 4-D tensor (batch_size, 1, query_seq_len, seq_len)
                attn_mask = generate_causal_mask(bsz, target_seq_len,
                                                 cache_position, q.device)
        elif attention_mask == PredefinedAttentionMask.FULL:
            pass
        else:
            raise ValueError("Unexpected attention mask type")

        # Apply scaling
        qk_scale = None
        if self.q_scaling is not None:
            qk_scale = 1 / (math.sqrt(self.head_dim) * self.q_scaling)

        # Standard scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            key_states,
            value_states,
            is_causal=is_causal,
            attn_mask=attn_mask,
            scale=qk_scale,
        )

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
        k_selected = k.gather(1, kv_indices) if len(kv_indices) > 0 else k
        v_selected = v.gather(1, kv_indices) if len(kv_indices) > 0 else v

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

    def _single_request_forward(self, q: Tensor, k: Optional[Tensor],
                                v: Optional[Tensor],
                                attention_mask: AttentionMask,
                                kv_cache_tensor: Tensor,
                                metadata: SparseAttentionMetadata,
                                past_seen_token: int, cache_idx: int,
                                **kwargs) -> Tensor:
        bsz = 1
        q_len = q.size(0)

        # Query
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Key and Value
        target_seq_len = past_seen_token
        if k is not None and v is not None:
            kv_len = k.size(0)
            k = k.view(bsz, kv_len, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, kv_len, self.num_kv_heads, self.head_dim)

            # Handle quantization
            if self.quant_config and self.quant_config.layer_quant_mode.has_any_quant(
            ):
                qc = self.quant_config
                if qc.layer_quant_mode.has_fp8_kv_cache():
                    assert kv_cache_tensor.dtype == torch.float8_e4m3fn, f"KV cache should have fp8 dtype, but get {kv_cache_tensor.dtype}"
                    k = k.to(torch.float8_e4m3fn)
                    v = v.to(torch.float8_e4m3fn)
            assert k.dtype == v.dtype == kv_cache_tensor.dtype, f"KV cache dtype {kv_cache_tensor.dtype} does not match k/v dtype {k.dtype}/{v.dtype}"

        # Predict indices for writing to KV cache
        kv_indices = self.single_request_sparse_kv_predict(
            q, k, v, metadata, past_seen_token, cache_idx, **kwargs)

        target_seq_len += kv_indices.size(1)

        if k is not None and v is not None:
            key_states, value_states = self.update_sparse_kv_cache(
                k, v, kv_cache_tensor, past_seen_token, kv_indices, cache_idx,
                metadata)

        # Predict indices for attention calculation (for kv cache and new kv)
        attn_indices = self.single_request_sparse_attn_predict(
            q, k, v, metadata, past_seen_token, cache_idx, **kwargs)

        # Apply sparse attention calculation
        attn_output = self.single_request_sparse_attn_forward(
            q, key_states, value_states, attn_indices, cache_idx,
            target_seq_len, kv_cache_tensor, attention_mask, past_seen_token,
            metadata, **kwargs)

        return attn_output.squeeze(0)

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: SparseAttentionMetadata,
                *,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
                **kwargs) -> torch.Tensor:
        if metadata.kv_cache_manager is None:
            # Handle no KV cache case
            num_heads = self.num_heads
            num_kv_heads = self.num_kv_heads
            return self.no_kv_cache_forward(q=q,
                                            k=k,
                                            v=v,
                                            num_heads=num_heads,
                                            num_kv_heads=num_kv_heads,
                                            metadata=metadata,
                                            attention_mask=attention_mask)

        # Get cache information
        past_seen_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq
        cache_indices = [
            block_ids[0] for block_ids in metadata.block_ids_per_seq
        ]
        kv_cache_tensor = metadata.kv_cache_manager.get_buffers(self.layer_idx)

        q_len = q.size(0)
        assert len(cache_indices) == len(past_seen_tokens)
        assert len(cache_indices) == metadata.seq_lens.nelement()

        # Process each sequence in the batch
        offset = 0
        offset_kv = 0
        attn_outputs = []

        for i, (seq_len, seq_len_kv) in enumerate(
                zip(metadata.seq_lens, metadata.seq_lens_kv)):
            single_q = q[offset:offset + seq_len]
            single_k = k[
                offset_kv:offset_kv +
                seq_len_kv] if k is not None and seq_len_kv != 0 else None
            single_v = v[
                offset_kv:offset_kv +
                seq_len_kv] if v is not None and seq_len_kv != 0 else None

            past_seen_token = past_seen_tokens[i]
            cache_idx = cache_indices[i]

            attn_output = self._single_request_forward(
                single_q, single_k, single_v, attention_mask, kv_cache_tensor,
                metadata, past_seen_token, cache_idx, **kwargs)

            attn_outputs.append(attn_output)

            offset += seq_len
            offset_kv += seq_len_kv

        # Concatenate outputs
        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(q_len, -1)

        return attn_output

    @staticmethod
    def no_kv_cache_forward(
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        num_heads: int,
        num_kv_heads: int,
        metadata: SparseAttentionMetadata,
        *,
        attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL
    ) -> torch.Tensor:
        """
        Forward pass without KV cache (e.g., for BERT).
        This can be overridden by subclasses if needed.
        """
        # Default implementation falls back to vanilla attention
        from ..vanilla import VanillaAttention
        return VanillaAttention.no_kv_cache_forward(
            q=q,
            k=k,
            v=v,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            metadata=metadata,
            attention_mask=attention_mask)
