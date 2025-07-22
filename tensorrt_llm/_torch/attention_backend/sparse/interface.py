from typing import Optional, Tuple, Dict, Any, List
import torch
from torch import Tensor
from abc import abstractmethod

from ..vanilla import VanillaAttention, VanillaAttentionMetadata, repeat_kv
from ..interface import AttentionMask, PredefinedAttentionMask
from tensorrt_llm.models.modeling_utils import QuantConfig


class SparseAttentionMetadata(VanillaAttentionMetadata):
    """Base metadata class for sparse attention algorithms."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse_params: Dict[str, Any] = {}
    
    def update_sparse_params(self, **kwargs):
        self.sparse_params.update(kwargs)
    
    def get_sparse_param(self, key: str, default=None):
        return self.sparse_params.get(key, default)


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
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads, quant_config, q_scaling, **kwargs)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

    @abstractmethod
    def predict_kv_indices_for_calc(self, 
                                   q: Tensor,
                                   k: Optional[Tensor],
                                   v: Optional[Tensor],
                                   metadata: SparseAttentionMetadata,
                                   past_seen_token: int,
                                   cache_idx: int,
                                   **kwargs) -> Optional[Tensor]:
        pass

    @abstractmethod
    def predict_kv_indices_for_write(self,
                                    q: Optional[Tensor],
                                    k: Optional[Tensor],
                                    v: Optional[Tensor],
                                    metadata: SparseAttentionMetadata,
                                    past_seen_token: int,
                                    cache_idx: int,
                                    **kwargs) -> Optional[Tensor]:
        pass

    def attention_forward(self,
                                q: Tensor,
                                key_states: Tensor,
                                value_states: Tensor,
                                attention_mask: AttentionMask,
                                metadata: SparseAttentionMetadata,
                                **kwargs) -> Tensor:
        import math
        
        bsz, num_heads, q_len, head_dim = q.shape
        selected_len = key_states.shape[2]
        
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Handle attention mask for sparse attention
        is_causal = False
        attn_mask = None
        
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            # For sparse attention, we need to create a mask based on the selected indices
            # This is more complex as we need to map the sparse indices to a dense mask
            # For now, we'll use a simple approach - assume the selected tokens are in order
            if selected_len == q_len:  # Full attention case
                is_causal = True
            else:
                # Create a mask that allows attention to all selected positions
                # This is a simplified approach - in practice, you might need more sophisticated masking
                attn_mask = torch.ones(bsz, 1, q_len, selected_len, device=q.device, dtype=q.dtype)
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

    def _update_kv_cache(self,
                        k: Optional[Tensor],
                        v: Optional[Tensor],
                        kv_cache_tensor: Tensor,
                        write_indices: Tensor,
                        cache_idx: int,
                        metadata: Optional[SparseAttentionMetadata] = None) -> Tuple[Tensor, Tensor]:
        k_out = kv_cache_tensor[cache_idx, 0, :, :, :].unsqueeze(0)
        v_out = kv_cache_tensor[cache_idx, 1, :, :, :].unsqueeze(0)

        if k is not None and v is not None:
            access_type = self._access_type[k.dtype.itemsize]
            k_out.view(dtype=access_type).index_copy_(2, write_indices,
                                                      k.view(dtype=access_type))
            v_out.view(dtype=access_type).index_copy_(2, write_indices,
                                                      v.view(dtype=access_type))

        return k_out, v_out

    def _gather_kv_states(self,
                         kv_cache_tensor: Tensor,
                         calc_indices: Tensor,
                         cache_idx: int,
                         seq_len: int) -> Tuple[Tensor, Tensor]:
        k_out = kv_cache_tensor[cache_idx, 0, :, :, :].unsqueeze(0)
        v_out = kv_cache_tensor[cache_idx, 1, :, :, :].unsqueeze(0)
        
        if calc_indices is not None:
            # Gather the selected indices
            k_out = k_out.gather(2, calc_indices.unsqueeze(-1).expand(-1, -1, -1, k_out.size(-1)))
            v_out = v_out.gather(2, calc_indices.unsqueeze(-1).expand(-1, -1, -1, v_out.size(-1)))
        
        return k_out[:, :, :seq_len, :], v_out[:, :, :seq_len, :]

    def _single_request_forward(self,
                               q: Tensor,
                               k: Optional[Tensor],
                               v: Optional[Tensor],
                               attention_mask: AttentionMask,
                               kv_cache_tensor: Tensor,
                               metadata: SparseAttentionMetadata,
                               past_seen_token: int,
                               cache_idx: int,
                               **kwargs) -> Tensor:
        bsz = 1
        q_len = q.size(0)

        # Query
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Key and Value
        target_seq_len = past_seen_token
        if k is not None and v is not None:
            kv_len = k.size(0)
            k = k.view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            target_seq_len += kv_len

            # Handle quantization
            if self.quant_config and self.quant_config.layer_quant_mode.has_any_quant(
            ):
                qc = self.quant_config
                if qc.layer_quant_mode.has_fp8_kv_cache():
                    assert kv_cache_tensor.dtype == torch.float8_e4m3fn, f"KV cache should have fp8 dtype, but get {kv_cache_tensor.dtype}"
                    k = k.to(torch.float8_e4m3fn)
                    v = v.to(torch.float8_e4m3fn)
            assert k.dtype == v.dtype == kv_cache_tensor.dtype, f"KV cache dtype {kv_cache_tensor.dtype} does not match k/v dtype {k.dtype}/{v.dtype}"

        # Predict indices for attention calculation
        calc_indices = self.predict_kv_indices_for_calc(
            q, k, v, metadata, past_seen_token, cache_idx, **kwargs)
        
        # Gather key and value states for attention
        key_states, value_states = self._gather_kv_states(
            kv_cache_tensor, calc_indices, cache_idx, target_seq_len)

        # Apply sparse attention calculation
        attn_output = self.attention_forward(
            q, key_states, value_states, attention_mask, metadata, **kwargs)

        # Predict indices for writing to KV cache
        input_kv_indices = self.predict_kv_indices_for_write(
            q, k, v, metadata, past_seen_token, cache_idx, **kwargs)

        # Update KV cache
        if k is not None and v is not None:
            # Select tokens from input using returned indices
            k_selected = k.gather(2, input_kv_indices) if len(input_kv_indices) > 0 else k
            v_selected = v.gather(2, input_kv_indices) if len(input_kv_indices) > 0 else v
            
            # Calculate cache write positions
            cache_write_indices = torch.arange(past_seen_token, past_seen_token + k_selected.size(2), device=k.device)
            
            self._update_kv_cache(k_selected, v_selected, kv_cache_tensor, cache_write_indices, cache_idx, metadata)

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
            return self.no_kv_cache_forward(
                q=q, k=k, v=v, num_heads=num_heads, num_kv_heads=num_kv_heads,
                metadata=metadata, attention_mask=attention_mask)

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
        
        for i, (seq_len, seq_len_kv) in enumerate(zip(metadata.seq_lens, metadata.seq_lens_kv)):
            single_q = q[offset:offset + seq_len]
            single_k = k[offset_kv:offset_kv + seq_len_kv] if k is not None and seq_len_kv != 0 else None
            single_v = v[offset_kv:offset_kv + seq_len_kv] if v is not None and seq_len_kv != 0 else None
            
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
    def no_kv_cache_forward(q: torch.Tensor,
                           k: Optional[torch.Tensor],
                           v: Optional[torch.Tensor],
                           num_heads: int,
                           num_kv_heads: int,
                           metadata: SparseAttentionMetadata,
                           *,
                           attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL) -> torch.Tensor:
        """
        Forward pass without KV cache (e.g., for BERT).
        This can be overridden by subclasses if needed.
        """
        # Default implementation falls back to vanilla attention
        from ..vanilla import VanillaAttention
        return VanillaAttention.no_kv_cache_forward(
            q=q, k=k, v=v, num_heads=num_heads, num_kv_heads=num_kv_heads,
            metadata=metadata, attention_mask=attention_mask)