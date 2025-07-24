import math
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import KvCacheConfig as KvCacheConfigCpp
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from .interface import SparseAttentionMetadata, VanillaSparseAttention

ModelConfig = tensorrt_llm.bindings.ModelConfig


class RocketVanillaAttentionMetadata(SparseAttentionMetadata):

    sparse_attn_config: Optional["RocketSparseAttentionConfig"] = None

    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attn_config is None:
            raise ValueError("Sparse attention config is not set")
        self.real_prompt_budget = self.sparse_attn_config.prompt_budget


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch, num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


class RocketVanillaAttention(VanillaSparseAttention):
    """
    RocketKV sparse attention implementation.

    This implementation focuses on KV cache index management:
    - Context phase: Only write indices (no calc indices needed)
    - Generation phase: Both calc and write indices with RocketKV selection
    """

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
                 sparse_attention_config: "SparseAttentionConfig",
                 num_kv_heads: Optional[int] = None,
                 quant_config: Optional[QuantConfig] = None,
                 q_scaling: Optional[float] = None,
                 **kwargs):
        super().__init__(layer_idx,
                         num_heads,
                         head_dim,
                         num_kv_heads=num_kv_heads,
                         quant_config=quant_config,
                         q_scaling=q_scaling,
                         sparse_attention_config=sparse_attention_config,
                         **kwargs)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.topr = sparse_attention_config.topr
        self.topk = sparse_attention_config.topk
        self.prompt_budget = sparse_attention_config.prompt_budget
        self.window_size = sparse_attention_config.window_size
        self.kernel_size = sparse_attention_config.kernel_size
        self.page_size = sparse_attention_config.page_size

    def _single_request_update_kt_cache(self, k, kt_cache_tensor, seq_len,
                                        cache_idx, cache_position):
        """Update KT cache for RocketKV algorithm."""
        k_out = kt_cache_tensor[cache_idx, :, :, :].unsqueeze(
            0)  # (1, num_kv_heads, 2*head_dim, num_pages_per_block)

        # k: (1, seq_len, num_kv_heads, head_dim)
        if k is not None:
            padding_len = self.page_size - (
                (k.size(1) - 1) % self.page_size + 1)
            k_min = torch.cat(
                [
                    k,
                    torch.full((k.size(0), padding_len, k.size(2), k.size(3)),
                               float('inf'),
                               device=k.device,
                               dtype=k.dtype)
                ],
                dim=1)  # (1, seq_len+padding_len, num_kv_heads, head_dim)
            k_min = k_min.reshape(
                k_min.size(0),
                k_min.size(1) // self.page_size, self.page_size, k.size(2),
                k_min.size(3)
            ).amin(dim=2).permute(
                0, 2, 3, 1
            )  # (1, num_pages, num_kv_heads, head_dim)->(1, num_kv_heads, head_dim, num_pages)
            k_max = torch.cat([
                k,
                torch.full((k.size(0), padding_len, k.size(2), k.size(3)),
                           float('-inf'),
                           device=k.device,
                           dtype=k.dtype)
            ],
                              dim=1)
            k_max = k_max.reshape(
                k_max.size(0),
                k_max.size(1) // self.page_size, self.page_size, k.size(2),
                k_max.size(3)).amax(dim=2).permute(0, 2, 3, 1)
            k_value = torch.cat([
                torch.min(k_min, k_out[:, :, :k_min.size(-2), cache_position]),
                torch.max(k_max, k_out[:, :,
                                       k_max.size(-2):, cache_position])
            ],
                                dim=-2)
            access_type = self._access_type[k_value.dtype.itemsize]
            k_out.view(dtype=access_type).index_copy_(
                -1, cache_position, k_value.view(dtype=access_type))

        return k_out[:, :, :, :math.ceil(seq_len / self.page_size)]

    def single_request_sparse_kv_predict(self, q: Optional[Tensor],
                                         k: Optional[Tensor],
                                         v: Optional[Tensor],
                                         metadata: "VanillaAttentionMetadata",
                                         past_seen_token: int, cache_idx: int,
                                         **kwargs) -> Optional[Tensor]:
        """
        Predict KV indices for writing new key/value pairs.

        Returns the actual indices to use from the input k,v tensors.
        For RocketKV:
        - Context phase: Returns SnapKV selected indices from input sequence
        - Generation phase: Returns all indices (sequential)
        """
        if k is None or v is None:
            return None

        # Generation phase: Use all tokens
        if k.size(1) == 1:
            shape = (k.size(0), 1, k.size(2), k.size(-1))
            return torch.zeros(shape, device=k.device, dtype=torch.int64)

        # Context phase: Use SnapKV selection
        selected_indices = self._get_snapkv_indices(q, k, metadata)

        k_snap = k.gather(1, selected_indices)
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)
        target_seq_len = past_seen_token + k_snap.size(1)
        kt_cache_position = torch.arange(
            math.ceil(past_seen_token / self.page_size),
            math.ceil(target_seq_len / self.page_size),
            device=q.device)
        self._single_request_update_kt_cache(k_snap, kt_cache_tensor,
                                             target_seq_len, cache_idx,
                                             kt_cache_position)

        return selected_indices

    def _get_snapkv_indices(
        self,
        q: Tensor,
        k: Tensor,
        metadata: "VanillaAttentionMetadata",
    ) -> Tensor:
        """Get SnapKV selected indices from the input sequence for context phase."""
        bsz = 1
        seq_len = k.size(1)

        if seq_len <= self.prompt_budget:
            metadata.sparse_metadata.real_prompt_budget = seq_len
            return torch.arange(seq_len, device=k.device).unsqueeze(
                0).unsqueeze(-1).unsqueeze(-1).expand(bsz, -1,
                                                      self.num_kv_heads,
                                                      self.head_dim)

        metadata.sparse_metadata.real_prompt_budget = metadata.sparse_metadata.sparse_attn_config.prompt_budget
        # Use last window_size tokens as observation
        q_obs = q[:, :, -self.
                  window_size:]  # (1, num_kv_heads, window_size, head_dim)
        k_pre = repeat_kv(
            k.transpose(1,
                        2)[:, :, :-self.window_size], self.num_key_value_groups
        )  # (1, num_heads, seq_len-window_size, head_dim)

        # Compute attention scores
        score = torch.matmul(q_obs, k_pre.transpose(-1, -2)) / math.sqrt(
            self.head_dim)
        score = torch.nn.functional.softmax(score, dim=-1).sum(dim=-2)
        score = score.view(bsz, self.num_kv_heads, self.num_key_value_groups,
                           -1).sum(dim=2)
        score = torch.nn.functional.max_pool1d(score,
                                               kernel_size=self.kernel_size,
                                               padding=self.kernel_size // 2,
                                               stride=1)

        # Select top important tokens from prefix
        prefix_len = seq_len - self.window_size
        selected_prefix_indices = score.topk(self.prompt_budget -
                                             self.window_size,
                                             dim=-1).indices.sort().values

        # Combine selected prefix indices with window indices
        window_indices = torch.arange(
            prefix_len, seq_len,
            device=k.device).unsqueeze(0).unsqueeze(0).expand(
                bsz, self.num_kv_heads, -1)
        selected_indices = torch.cat([selected_prefix_indices, window_indices],
                                     dim=-1)

        return selected_indices.unsqueeze(-1).expand(-1, -1, -1,
                                                     self.head_dim).transpose(
                                                         1, 2)

    def single_request_sparse_attn_predict(self, q: Tensor, k: Optional[Tensor],
                                           v: Optional[Tensor],
                                           metadata: "VanillaAttentionMetadata",
                                           past_seen_token: int, cache_idx: int,
                                           **kwargs) -> Optional[Tensor]:
        """
        Predict KV cache indices for attention calculation.

        For RocketKV:
        - Context phase: Returns None (no calc indices needed, use full attention)
        - Generation phase: Returns RocketKV selected indices for sparse attention
        """
        if k is None or v is None:
            return None

        # all new kv indices needed for attention calculation
        kv_indices = torch.arange(
            past_seen_token, past_seen_token + k.size(1),
            device=q.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(
                1, -1, self.num_kv_heads, self.head_dim)

        # Context phase: No calc indices needed (use full attention)
        if q.size(2) > 1:
            return kv_indices

        # Get RocketKV selected indices
        calc_indices = self._rocketkv_selection(q, k, metadata, past_seen_token,
                                                cache_idx)

        # decode phase: concat the new kv indices with the kv cache calc indices
        return torch.cat([kv_indices, calc_indices], dim=1)

    def _rocketkv_selection(self, q: Tensor, k: Tensor,
                            metadata: "VanillaAttentionMetadata",
                            past_seen_token: int, cache_idx: int) -> Tensor:
        """Implement RocketKV's two-stage selection process for generation phase."""
        bsz = 1
        q_len = q.size(2)

        # Helper functions
        def _gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
            dim += (dim < 0) * t.ndim
            return t.gather(
                dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1:]))

        @torch.compile(disable=not torch.cuda.is_available())
        def _scaled_softmax(x: Tensor, divscale: Tensor | float,
                            dim: int) -> Tensor:
            return torch.softmax(x / divscale, dim=dim)

        # Get KT cache for key-token matching
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)
        target_seq_len = past_seen_token + 1  # +1 for current token

        # Update KT cache
        kt_cache_position = torch.arange(
            math.ceil(past_seen_token / self.page_size),
            math.ceil(target_seq_len / self.page_size),
            device=q.device)
        kt_states = self._single_request_update_kt_cache(
            k, kt_cache_tensor, target_seq_len, cache_idx, kt_cache_position)

        # Reshape query for multi-head processing
        qi = q.view(bsz, self.num_kv_heads, self.num_heads // self.num_kv_heads,
                    q_len, self.head_dim)
        qi_abs = torch.abs(qi)

        # Top-r selection on query features
        i1 = torch.topk(qi_abs.sum(dim=2, keepdim=True), self.topr,
                        dim=-1).indices
        qi_hat = _gather(qi, -1, i1)

        # print(f'qi_hat.shape: {qi_hat.shape}')

        # Generate signed indices for key-token matching
        i1_sign = torch.where(
            qi_hat.sum(dim=2, keepdim=True) > 0, i1 + self.head_dim,
            i1).transpose(-1, -2)

        # Gather key tokens and compute attention scores
        kt_hat = _gather(kt_states.unsqueeze(2), -2, i1_sign)
        # print(f'kt_hat.shape: {kt_hat.shape}')
        qk_hat = qi_hat @ kt_hat
        # print(f'qk_hat.shape: {qk_hat.shape}')
        qk_hat = qk_hat.repeat_interleave(self.page_size,
                                          dim=-1)[:, :, :, :, :target_seq_len]
        # print(f'target_seq_len={target_seq_len}, qk_hat.shape: {qk_hat.shape}')
        # Compute scaling factor for attention scores
        scale = torch.sqrt(self.head_dim *
                           torch.abs(qi_hat).sum(dim=-1, keepdim=True) /
                           qi_abs.sum(dim=-1, keepdim=True))

        # Apply scaled softmax
        s_hat = _scaled_softmax(
            qk_hat, scale,
            dim=-1)  # (1, num_kv_heads, num_heads, target_seq_len)

        # Top-k selection on attention scores
        topk = min(self.topk, target_seq_len)
        i2 = torch.topk(s_hat.sum(dim=2), topk, dim=-1).indices
        iKV = i2[..., 0, :, None].transpose(1, 2).expand(
            -1, -1, -1, self.head_dim)  # (1, topk, num_kv_heads, head_dim)

        return iKV


class RocketKVCacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfigCpp,
        kv_cache_type: CacheTypeCpp,
        *,
        num_layers: int,
        num_kv_heads: Union[int, List[Optional[int]]],
        head_dim: int,
        tokens_per_block: int,
        # Note that max_seq_len is not necessarily equal to kv_cache_config.num_tokens.
        # It's derived from the model's BuildConfig for consistency with the C++ backend.
        max_seq_len: int,
        max_batch_size: int,
        mapping: Mapping,
        dtype: DataType = DataType.HALF,
        spec_config: Optional["DecodingBaseConfig"] = None,
        layer_mask: Optional[List[bool]] = None,
        max_num_tokens: int = 8192,
        model_config: Optional[ModelConfig] = None,
        max_beam_width: int = 1,
        sparse_attn_config: Optional["SparseAttentionConfig"] = None,
    ) -> None:
        super().__init__(kv_cache_config=kv_cache_config,
                         kv_cache_type=kv_cache_type,
                         num_layers=num_layers,
                         num_kv_heads=num_kv_heads,
                         head_dim=head_dim,
                         tokens_per_block=tokens_per_block,
                         max_seq_len=max_seq_len,
                         max_batch_size=max_batch_size,
                         mapping=mapping,
                         dtype=dtype,
                         spec_config=spec_config,
                         layer_mask=layer_mask,
                         max_num_tokens=max_num_tokens,
                         model_config=model_config,
                         max_beam_width=max_beam_width)

        self.page_size = sparse_attn_config.page_size

        # initialize kt cache
        num_blocks = self.impl.max_num_blocks

        self.kt_cache = {}
        for layer_idx in range(self.num_local_layers):
            local_layer_idx = layer_idx
            num_kv_heads = self.num_kv_heads_per_layer[local_layer_idx]

            kt_cache_shape = (num_blocks, num_kv_heads, head_dim,
                              math.ceil(max_seq_len / self.page_size))

            self.kt_cache[layer_idx] = torch.cat([
                torch.full(kt_cache_shape,
                           float('inf'),
                           device="cuda",
                           dtype=torch.bfloat16),
                torch.full(kt_cache_shape,
                           float('-inf'),
                           device="cuda",
                           dtype=torch.bfloat16)
            ],
                                                 dim=-2).contiguous()

        self.active_cache_blocks: Dict[int, List[int]] = {
        }  # request_id -> list of cache_indices

    def get_kt_buffers(self, layer_idx: int):
        return self.kt_cache[layer_idx]

    def prepare_resources(self, scheduled_batch):
        super().prepare_resources(scheduled_batch)

        for request in scheduled_batch.all_requests():
            request_id = request.py_request_id
            if request_id not in self.active_cache_blocks:
                cache_indices = self.get_cache_indices(request)
                self.active_cache_blocks[request_id] = cache_indices

                self._clear_kt_cache_blocks(cache_indices)

    def free_resources(self, request):
        request_id = request.py_request_id

        if request_id in self.active_cache_blocks:
            cache_indices = self.active_cache_blocks[request_id]
            self._clear_kt_cache_blocks(cache_indices)
            del self.active_cache_blocks[request_id]

        super().free_resources(request)

    def _clear_kt_cache_blocks(self, cache_indices: List[int]):
        for layer_idx in range(self.num_local_layers):
            kt_cache_tensor = self.kt_cache[layer_idx]
            for cache_idx in cache_indices:
                if cache_idx < kt_cache_tensor.shape[0]:
                    head_dim = kt_cache_tensor.shape[2] // 2

                    kt_cache_tensor[cache_idx, :, :head_dim, :] = float('inf')
                    kt_cache_tensor[cache_idx, :, head_dim:, :] = float('-inf')
