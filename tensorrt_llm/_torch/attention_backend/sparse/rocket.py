import math
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import Tensor

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm._torch.attention_backend.trtllm import (
    TrtllmAttention, TrtllmAttentionMetadata)
from tensorrt_llm._torch.attention_backend.vanilla import (
    VanillaAttention, VanillaAttentionMetadata, repeat_kv)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.executor import ExecutorConfig, KvCacheConfig
from tensorrt_llm.bindings.internal.batch_manager import \
    CacheType as CacheTypeCpp
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from .kernel import (batched_update_kt_cache, flatten_sparse_indices,
                     fused_qk_split, reshape_flatten_to_batched,
                     separated_bmm_softmax, triton_index_gather)

ModelConfig = tensorrt_llm.bindings.ModelConfig


class RocketTrtllmAttentionMetadata(TrtllmAttentionMetadata):

    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget
        self.window_size = self.sparse_attention_config.window_size

        self.context_lens_cuda = torch.zeros(
            self.max_num_sequences,
            device='cuda',
            dtype=torch.int32,
        )

        self.q_cu_seqlens = torch.zeros(
            self.max_num_sequences + 1,
            device='cpu',
            dtype=torch.int32,
        )
        self.q_cu_seqlens_cuda = torch.zeros_like(self.q_cu_seqlens,
                                                  device='cuda',
                                                  dtype=torch.int32)

        self.k_cu_seqlens = torch.zeros(
            self.max_num_sequences + 1,
            device='cpu',
            dtype=torch.int32,
        )
        self.k_cu_seqlens_cuda = torch.zeros_like(self.k_cu_seqlens,
                                                  device='cuda',
                                                  dtype=torch.int32)

        self.k_context_lens = torch.zeros(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )
        self.k_context_lens_cuda = torch.zeros_like(self.k_context_lens,
                                                    device='cuda',
                                                    dtype=torch.int32)

        self.context_cumsum = torch.zeros(
            self.max_num_sequences + 1,
            device='cpu',
            dtype=torch.int32,
        )
        self.context_cumsum_cuda = torch.zeros_like(self.context_cumsum,
                                                    device='cuda',
                                                    dtype=torch.int32)

        self.q_extract_start_offsets = torch.zeros(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )
        self.q_extract_start_offsets_cuda = torch.zeros_like(
            self.q_extract_start_offsets, device='cuda', dtype=torch.int32)

        self.q_extract_lengths = torch.zeros(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )
        self.q_extract_lengths_cuda = torch.zeros_like(self.q_extract_lengths,
                                                       device='cuda',
                                                       dtype=torch.int32)

        self.q_output_offsets = torch.zeros(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )
        self.q_output_offsets_cuda = torch.zeros_like(self.q_output_offsets,
                                                      device='cuda',
                                                      dtype=torch.int32)

        self.k_extract_start_offsets = torch.zeros(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )
        self.k_extract_start_offsets_cuda = torch.zeros_like(
            self.k_extract_start_offsets, device='cuda', dtype=torch.int32)

        self.k_extract_lengths = torch.zeros(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )
        self.k_extract_lengths_cuda = torch.zeros_like(self.k_extract_lengths,
                                                       device='cuda',
                                                       dtype=torch.int32)

        self.k_output_offsets = torch.zeros(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )
        self.k_output_offsets_cuda = torch.zeros_like(self.k_output_offsets,
                                                      device='cuda',
                                                      dtype=torch.int32)

        self.sparse_offsets = torch.zeros(
            self.max_num_sequences + 1,
            device='cpu',
            dtype=torch.int32,
        )
        self.sparse_offsets_cuda = torch.zeros_like(self.sparse_offsets,
                                                    device='cuda',
                                                    dtype=torch.int32)

        self.valid_seq_indices = torch.zeros(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )
        self.valid_seq_indices_cuda = torch.zeros_like(self.valid_seq_indices,
                                                       device='cuda',
                                                       dtype=torch.int32)

        self.kt_cache_slots = torch.empty(
            self.max_num_sequences,
            device='cpu',
            dtype=torch.int32,
        )
        self.kt_cache_slots_cuda = torch.empty_like(self.kt_cache_slots,
                                                    device='cuda',
                                                    dtype=torch.int32)

    def prepare(self):
        if self.kv_cache_manager is not None:
            num_contexts = self.num_contexts
            num_generations = self.num_generations
            num_requests = num_contexts + num_generations

            for i in range(num_requests):
                if i < num_contexts:
                    self.kv_cache_params.num_cached_tokens_per_seq[i] = 0
                else:
                    if self.prompt_lens[i] > self.prompt_budget:
                        self.kv_cache_params.num_cached_tokens_per_seq[
                            i] += self.prompt_budget - self.prompt_lens[i]

        super().prepare()

        # Update prompt lens for sparse attention
        if self.kv_cache_manager is not None:
            _prompt_lens = self.prompt_lens.copy()
            for i in range(num_requests):
                if i >= num_contexts:
                    _prompt_lens[i] = min(_prompt_lens[i], self.prompt_budget)
            _prompt_lens = torch.tensor(_prompt_lens,
                                        dtype=torch.int,
                                        device='cpu')
            self.prompt_lens_cpu[:self.num_seqs].copy_(_prompt_lens)
            self.prompt_lens_cuda[:self.num_seqs].copy_(
                self.prompt_lens_cpu[:self.num_seqs], non_blocking=True)
            self.prompt_lens_cuda_runtime = self.prompt_lens_cuda[:self.
                                                                  num_seqs]
            self.prompt_lens_cpu_runtime = self.prompt_lens_cpu[:self.num_seqs]

            for i in range(num_requests):
                self.kt_cache_slots[
                    i] = self.kv_cache_manager.get_kt_cache_slot(
                        self.request_ids[i])
            self.kt_cache_slots_cuda[:num_requests].copy_(
                self.kt_cache_slots[:num_requests], non_blocking=True)

        self.context_lens_cuda[:self.num_contexts].copy_(
            self.context_lens[:self.num_contexts], non_blocking=True)

        self.context_cumsum[1:self.num_contexts + 1] = torch.cumsum(
            self.context_lens, dim=0)
        self.context_cumsum_cuda[:self.num_contexts + 1].copy_(
            self.context_cumsum[:self.num_contexts + 1], non_blocking=True)

        valid_mask = self.context_lens >= self.prompt_budget
        valid_seq_indices = torch.where(valid_mask)[0]
        invalid_seq_indices = torch.where(~valid_mask)[0]
        valid_batch_size = len(valid_seq_indices)
        self.valid_seq_indices_cuda[:valid_batch_size].copy_(valid_seq_indices,
                                                             non_blocking=True)

        self.k_context_lens[:valid_batch_size] = self.context_lens[
            valid_seq_indices] - self.window_size
        self.k_context_lens_cuda[:valid_batch_size].copy_(
            self.k_context_lens[:valid_batch_size], non_blocking=True)
        self.max_k_context_tokens = self.k_context_lens.max().item()

        sparse_counts = torch.zeros(self.num_contexts,
                                    dtype=torch.int32,
                                    device='cpu')
        sparse_counts[valid_seq_indices] = self.prompt_budget
        sparse_counts[invalid_seq_indices] = self.context_lens[
            invalid_seq_indices]

        self.sparse_offsets[1:self.num_contexts + 1] = torch.cumsum(
            sparse_counts, dim=0)
        self.sparse_offsets_cuda[:self.num_contexts + 1].copy_(
            self.sparse_offsets[:self.num_contexts + 1], non_blocking=True)

        self.q_cu_seqlens[:valid_batch_size + 1] = torch.arange(
            valid_batch_size + 1, device='cpu',
            dtype=torch.int32) * self.window_size
        self.q_extract_lengths[:valid_batch_size] = self.window_size
        self.q_extract_start_offsets[:valid_batch_size] = self.context_lens[
            valid_seq_indices] - self.window_size
        self.q_output_offsets[:
                              valid_batch_size] = self.q_cu_seqlens[:
                                                                    valid_batch_size]

        self.q_cu_seqlens_cuda[:valid_batch_size + 1].copy_(
            self.q_cu_seqlens[:valid_batch_size + 1], non_blocking=True)
        self.q_extract_lengths_cuda[:valid_batch_size].copy_(
            self.q_extract_lengths[:valid_batch_size], non_blocking=True)
        self.q_extract_start_offsets_cuda[:valid_batch_size].copy_(
            self.q_extract_start_offsets[:valid_batch_size], non_blocking=True)
        self.q_output_offsets_cuda[:valid_batch_size].copy_(
            self.q_output_offsets[:valid_batch_size], non_blocking=True)

        self.k_cu_seqlens[1:valid_batch_size + 1] = torch.cumsum(
            self.k_context_lens[:valid_batch_size], dim=0)
        self.k_extract_lengths[:
                               valid_batch_size] = self.k_context_lens[:
                                                                       valid_batch_size]
        self.k_output_offsets[:
                              valid_batch_size] = self.k_cu_seqlens[:
                                                                    valid_batch_size]

        self.k_cu_seqlens_cuda[:valid_batch_size + 1].copy_(
            self.k_cu_seqlens[:valid_batch_size + 1], non_blocking=True)
        self.k_extract_lengths_cuda[:valid_batch_size].copy_(
            self.k_extract_lengths[:valid_batch_size], non_blocking=True)
        self.k_output_offsets_cuda[:valid_batch_size].copy_(
            self.k_output_offsets[:valid_batch_size], non_blocking=True)

        self.valid_batch_size = valid_batch_size
        self.total_sparse_tokens = self.sparse_offsets[self.num_contexts].item()


class RocketTrtllmAttention(TrtllmAttention):
    Metadata = RocketTrtllmAttentionMetadata

    # Access type for different dtype sizes
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
        super().__init__(layer_idx,
                         num_heads,
                         head_dim,
                         sparse_attention_config=sparse_attention_config,
                         num_kv_heads=num_kv_heads,
                         quant_config=quant_config,
                         q_scaling=q_scaling,
                         **kwargs)
        self.topr = sparse_attention_config.topr
        self.topk = sparse_attention_config.topk
        self.prompt_budget = sparse_attention_config.prompt_budget
        self.window_size = sparse_attention_config.window_size
        self.kernel_size = sparse_attention_config.kernel_size
        self.page_size = sparse_attention_config.page_size

    def preprocess_for_context(
            self, q: torch.Tensor, k: torch.Tensor,
            context_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        num_heads, total_tokens, head_dim = q.shape
        k.shape[0]
        batch_size = len(context_lens)

        # Filter valid sequences (those with context_len >= prompt_budget)
        valid_mask = context_lens >= self.prompt_budget
        valid_seq_indices = torch.where(valid_mask)[0]
        invalid_seq_indices = torch.where(~valid_mask)[0]
        valid_batch_size = len(valid_seq_indices)

        if valid_batch_size == 0:
            return None

        valid_context_lens = context_lens[valid_seq_indices]
        k_context_lens = valid_context_lens - self.window_size
        k_context_lens.sum()
        max_k_context_tokens = k_context_lens.max()

        # Create cumulative sum for sequence offsets (original sequences)
        context_cumsum = torch.zeros(batch_size + 1,
                                     device=context_lens.device,
                                     dtype=context_lens.dtype)
        context_cumsum[1:] = torch.cumsum(context_lens, dim=0)

        # Create cumulative sum for context output offsets
        q_cu_seqlens = torch.arange(
            valid_batch_size + 1,
            device=valid_context_lens.device,
            dtype=valid_context_lens.dtype) * self.window_size
        k_cu_seqlens = torch.zeros(valid_batch_size + 1,
                                   device=k_context_lens.device,
                                   dtype=k_context_lens.dtype)
        k_cu_seqlens[1:] = torch.cumsum(k_context_lens, dim=0)

        # Prepare parameters for Q extraction (last window_size tokens)
        q_extract_start_offsets = valid_context_lens - self.window_size  # [valid_batch_size]
        q_extract_lengths = torch.full((valid_batch_size, ),
                                       self.window_size,
                                       device=q.device,
                                       dtype=torch.int32)
        q_output_offsets = q_cu_seqlens[:
                                        -1]  # Remove last element to get [valid_batch_size]

        # Prepare parameters for K extraction (first context_len - window_size tokens)
        k_extract_start_offsets = torch.zeros(valid_batch_size,
                                              device=k.device,
                                              dtype=torch.int32)
        k_extract_lengths = k_context_lens  # [valid_batch_size]
        k_output_offsets = k_cu_seqlens[:
                                        -1]  # Remove last element to get [valid_batch_size]

        # Calculate number of sparse tokens for each batch
        sparse_counts = torch.zeros(batch_size,
                                    dtype=torch.int32,
                                    device=context_lens.device)
        sparse_counts[valid_seq_indices] = self.prompt_budget
        sparse_counts[invalid_seq_indices] = context_lens[invalid_seq_indices]
        sparse_offsets = torch.zeros(batch_size + 1,
                                     dtype=torch.int32,
                                     device=context_lens.device)
        sparse_offsets[1:] = torch.cumsum(sparse_counts, dim=0)

        preprocess_inputs = {
            "context_cumsum": context_cumsum,
            "valid_seq_indices": valid_seq_indices,
            "q_extract_start_offsets": q_extract_start_offsets,
            "q_extract_lengths": q_extract_lengths,
            "q_output_offsets": q_output_offsets,
            "k_extract_start_offsets": k_extract_start_offsets,
            "k_extract_lengths": k_extract_lengths,
            "k_output_offsets": k_output_offsets,
            "q_cu_seqlens": q_cu_seqlens,
            "k_cu_seqlens": k_cu_seqlens,
            "max_k_context_tokens": max_k_context_tokens,
            "k_context_lens": k_context_lens,
            "sparse_offsets": sparse_offsets,
        }
        return preprocess_inputs

    def batched_ctx_sparse_predict(
        self, qkv: torch.Tensor, metadata: RocketTrtllmAttentionMetadata
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict sparse KV indices using optimized SnapKV algorithm.

        Uses a single Triton kernel to compute attention scores between observation window
        and prefix tokens, then selects the most important prefix tokens directly.
        """

        num_ctx_tokens = metadata.num_ctx_tokens
        if num_ctx_tokens == 0:
            return None, None

        qkv_input = qkv[:num_ctx_tokens]
        q_window, k_context = fused_qk_split(qkv_input, self.num_heads,
                                             self.num_kv_heads, self.head_dim,
                                             self.window_size,
                                             self.prompt_budget, metadata)

        scores = separated_bmm_softmax(q_window,
                                       k_context,
                                       metadata.q_cu_seqlens_cuda,
                                       metadata.k_cu_seqlens_cuda,
                                       metadata.valid_batch_size,
                                       causal=False)

        scores = scores.view(self.num_kv_heads,
                             self.num_heads // self.num_kv_heads,
                             self.window_size, -1).sum(dim=-2).sum(dim=1)

        padding_size = metadata.max_k_context_tokens

        # Reshape scores to handle variable length sequences with padding using Triton
        # scores: [num_kv_heads, total_k_tokens] -> [valid_batch_size, num_kv_heads, padding_size]
        scores = reshape_flatten_to_batched(scores,
                                            metadata.k_context_lens_cuda,
                                            metadata.k_cu_seqlens_cuda,
                                            metadata.valid_batch_size,
                                            padding_size)

        scores = torch.nn.functional.max_pool1d(scores,
                                                kernel_size=self.kernel_size,
                                                padding=self.kernel_size // 2,
                                                stride=1)

        selected_prefix_indices = scores.topk(self.prompt_budget -
                                              self.window_size,
                                              dim=-1).indices.sort().values

        # Flatten sparse indices
        selected_prefix_indices = selected_prefix_indices.to(torch.int32)
        sparse_kv_indices = flatten_sparse_indices(
            selected_prefix_indices, metadata.context_lens_cuda,
            metadata.valid_seq_indices_cuda, metadata.k_context_lens_cuda,
            metadata.sparse_offsets_cuda, metadata.num_contexts,
            metadata.total_sparse_tokens, self.window_size, self.prompt_budget)

        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)
        max_num_pages = (metadata.max_k_context_tokens + self.page_size -
                         1) // self.page_size
        batched_update_kt_cache(qkv_input, kt_cache_tensor,
                                metadata.kt_cache_slots_cuda, sparse_kv_indices,
                                metadata.sparse_offsets_cuda,
                                metadata.num_contexts, self.num_heads,
                                self.num_kv_heads, self.head_dim, max_num_pages,
                                self.page_size)

        return sparse_kv_indices, metadata.sparse_offsets_cuda[:metadata.
                                                               num_contexts + 1]

    def batched_gen_sparse_predict(
        self, q: torch.Tensor, k: torch.Tensor,
        metadata: RocketTrtllmAttentionMetadata
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass

    def batched_sparse_attention_predict(
        self, q: torch.Tensor, k: torch.Tensor,
        metadata: RocketTrtllmAttentionMetadata
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict sparse KV indices and sparse attention indices for the input sequence.
        """
        sparse_kv_indices, sparse_kv_offsets = self.batched_ctx_sparse_predict(
            q, metadata)

        return sparse_kv_indices, sparse_kv_offsets

    def sparse_attention_predict(
        self, q: torch.Tensor, k: torch.Tensor,
        metadata: RocketTrtllmAttentionMetadata
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict sparse KV indices and sparse attention indices for the input sequence.

        For RocketKV:
        - Context phase: predict SnapKV sparse kv indices
        - Generation phase: predict RocketKV sparse attention indices

        Returns:
            Tuple of (flattened_indices, batch_offsets)
            - flattened_indices: [total_selected_indices, num_kv_heads]
            - batch_offsets: [batch_size + 1] with cumulative indices count
        """
        q, k, _ = q.split([
            self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim,
            self.num_kv_heads * self.head_dim
        ],
                          dim=-1)

        if k is None or metadata is None:
            return None, None

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        seq_lens = metadata.seq_lens
        seq_lens_kv = metadata.seq_lens_kv if metadata.seq_lens_kv is not None else seq_lens
        past_seen_tokens = metadata.kv_cache_params.num_cached_tokens_per_seq

        sparse_kv_indices = []
        sparse_attn_indices = []
        sparse_kv_offsets = [0]
        sparse_attn_offsets = [0]

        q_offset = 0
        k_offset = 0

        for i in range(num_contexts + num_generations):
            seq_len = seq_lens[i].item()
            seq_len_kv = seq_lens_kv[i].item()

            if seq_len <= 0 or seq_len_kv <= 0:
                assert False, "Invalid sequence length"

            single_q = q[q_offset:q_offset + seq_len]
            single_k = k[k_offset:k_offset + seq_len_kv]

            single_q = single_q.view(1, seq_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
            single_k = single_k.view(1, seq_len_kv, self.num_kv_heads,
                                     self.head_dim)

            past_seen_token = past_seen_tokens[i]
            kt_cache_slot = metadata.kv_cache_manager.get_kt_cache_slot(
                metadata.request_ids[i])

            if i < num_contexts:
                pass
                # # Context phase: SnapKV sparse kv indices
                # _sparse_kv_indices = self._get_snapkv_indices(
                #     single_q, single_k, past_seen_token, kt_cache_slot,
                #     metadata)
                # if _sparse_kv_indices is not None:
                #     sparse_kv_indices.append(
                #         _sparse_kv_indices.squeeze(0))  # [budget, num_kv_heads]
                #     sparse_kv_offsets.append(sparse_kv_offsets[-1] +
                #                              _sparse_kv_indices.size(1))
                # else:
                #     sparse_kv_offsets.append(sparse_kv_offsets[-1])
            else:
                # Generation phase: RocketKV sparse attention indices
                _sparse_attn_indices = self._rocketkv_selection(
                    single_q, single_k, past_seen_token, kt_cache_slot,
                    metadata)
                if _sparse_attn_indices is not None:
                    sparse_attn_indices.append(
                        _sparse_attn_indices.squeeze(0))  # [topk, num_kv_heads]
                    sparse_attn_offsets.append(sparse_attn_offsets[-1] +
                                               _sparse_attn_indices.size(1))
                else:
                    sparse_attn_offsets.append(sparse_attn_offsets[-1])

            q_offset += seq_len
            k_offset += seq_len_kv

        if len(sparse_kv_indices) == 0:
            sparse_kv_indices, sparse_kv_offsets = None, None
        else:
            sparse_kv_indices = torch.cat(sparse_kv_indices,
                                          dim=0).to(torch.int32)
            sparse_kv_offsets = torch.tensor(sparse_kv_offsets,
                                             dtype=torch.int32).to(q.device)
        if len(sparse_attn_indices) == 0:
            sparse_attn_indices, sparse_attn_offsets = None, None
        else:
            sparse_attn_indices = torch.cat(sparse_attn_indices,
                                            dim=0).to(torch.int32)
            sparse_attn_offsets = torch.tensor(sparse_attn_offsets,
                                               dtype=torch.int32).to(q.device)

        return sparse_kv_indices, sparse_kv_offsets, sparse_attn_indices, sparse_attn_offsets

    def _get_snapkv_indices(
            self, q: Tensor, k: Tensor, past_seen_token: int,
            kt_cache_slot: int,
            metadata: RocketTrtllmAttentionMetadata) -> Optional[Tensor]:
        """
        Get SnapKV sparse kv indices from the input sequence for context phase.
        The shape of output is (1, prompt_budget, num_kv_heads)
        """
        bsz = 1
        seq_len = k.size(1)  # k shape: (1, seq_len, num_kv_heads, head_dim)

        # If the sequence length is less than the prompt budget, do not enable sparse kv cache
        if seq_len <= self.prompt_budget:
            return None

        # Use last window_size tokens as observation
        # (1, num_heads, window_size, head_dim)
        q_obs = q[:, :, -self.window_size:]
        # (1, num_kv_heads, seq_len, head_dim)
        k_pre = repeat_kv(k.transpose(1, 2), self.num_heads //
                          self.num_kv_heads)[:, :, :-self.window_size]

        score = torch.matmul(q_obs, k_pre.transpose(-1, -2)) / math.sqrt(
            self.head_dim)

        score = torch.nn.functional.softmax(score, dim=-1)

        score = score.sum(dim=-2)

        score = score.view(bsz, self.num_kv_heads,
                           self.num_heads // self.num_kv_heads, -1).sum(dim=2)
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
                                     dim=-1).transpose(1, 2)

        k = k.reshape(1, -1, self.num_kv_heads, self.head_dim)
        k_snap = triton_index_gather(k, selected_indices)
        # Update KT cache
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)
        target_seq_len = past_seen_token + k_snap.size(1)
        kt_cache_position = torch.arange(past_seen_token // self.page_size,
                                         math.ceil(target_seq_len /
                                                   self.page_size),
                                         device=q.device)
        self._single_request_update_kt_cache(k_snap, kt_cache_tensor,
                                             target_seq_len, kt_cache_slot,
                                             kt_cache_position)

        return selected_indices

    def _rocketkv_selection(self, q: Tensor, k: Tensor, past_seen_token: int,
                            kt_cache_slot: int,
                            metadata: RocketTrtllmAttentionMetadata) -> Tensor:
        """
        Implement RocketKV's two-stage selection process for generation phase.
        The shape of output is (1, topk, num_kv_heads)
        """
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
        kt_cache_position = torch.arange(past_seen_token // self.page_size,
                                         math.ceil(target_seq_len /
                                                   self.page_size),
                                         device=q.device)
        kt_states = self._single_request_update_kt_cache(
            k, kt_cache_tensor, target_seq_len, kt_cache_slot,
            kt_cache_position)

        # Reshape query for multi-head processing
        qi = q.view(bsz, self.num_kv_heads, self.num_heads // self.num_kv_heads,
                    q_len, self.head_dim)
        qi_abs = torch.abs(qi)

        # Top-r selection on query features
        i1 = torch.topk(qi_abs.mean(dim=2, keepdim=True), self.topr,
                        dim=-1).indices
        qi_hat = _gather(qi, -1, i1)

        # Generate signed indices for key-token matching
        i1_sign = torch.where(
            qi_hat.sum(dim=2, keepdim=True) > 0, i1 + self.head_dim,
            i1).transpose(-1, -2)

        # Gather key tokens and compute attention scores
        kt_hat = _gather(kt_states.unsqueeze(2), -2, i1_sign)
        qk_hat = qi_hat @ kt_hat
        qk_hat = qk_hat.repeat_interleave(self.page_size,
                                          dim=-1)[:, :, :, :, :target_seq_len]
        scale = torch.sqrt(self.head_dim *
                           torch.abs(qi_hat).sum(dim=-1, keepdim=True) /
                           qi_abs.sum(dim=-1, keepdim=True))

        # (1, num_kv_heads, num_heads, target_seq_len)
        s_hat = _scaled_softmax(qk_hat, scale, dim=-1)

        topk = min(self.topk, target_seq_len)
        i2 = torch.topk(s_hat.mean(dim=2, keepdim=True), topk, dim=-1).indices

        iKV = i2[:, :, 0, 0, :].transpose(1, 2)  # (1, topk, num_kv_heads)

        return iKV

    def _single_request_update_kt_cache(self, k, kt_cache_tensor, seq_len,
                                        kt_cache_slot, cache_position):
        """Update KT cache for RocketKV algorithm."""
        # (1, num_kv_heads, 2*head_dim, num_pages_per_block)
        k_out = kt_cache_tensor[kt_cache_slot, :, :, :].unsqueeze(0)

        # k: (1, seq_len, num_kv_heads, head_dim)
        if k is not None:
            padding_len = self.page_size - (
                (k.size(1) - 1) % self.page_size + 1)
            padding_tensor = torch.full(
                (k.size(0), padding_len, k.size(2), k.size(3)),
                float('inf'),
                device=k.device,
                dtype=k.dtype)
            # (1, seq_len+padding_len, num_kv_heads, head_dim)
            k_min = torch.cat([k, padding_tensor], dim=1)
            # (1, num_pages, num_kv_heads, head_dim)->(1, num_kv_heads, head_dim, num_pages)
            k_min = k_min.reshape(
                k_min.size(0),
                k_min.size(1) // self.page_size, self.page_size, k.size(2),
                k_min.size(3)).amin(dim=2).permute(0, 2, 3, 1)
            k_max = torch.cat([k, -padding_tensor], dim=1)
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


class RocketVanillaAttentionMetadata(VanillaAttentionMetadata):

    def __post_init__(self):
        super().__post_init__()
        if self.sparse_attention_config is None:
            raise ValueError("Sparse attention config is not set")
        self.prompt_budget = self.sparse_attention_config.prompt_budget

    def prepare(self) -> None:
        super().prepare()
        num_contexts = self.num_contexts
        num_generations = self.num_generations
        num_requests = num_contexts + num_generations

        for i in range(num_requests):
            if i < num_contexts:
                self.kv_cache_params.num_cached_tokens_per_seq[i] = 0
            else:
                if self.prompt_lens[i] > self.prompt_budget:
                    self.kv_cache_params.num_cached_tokens_per_seq[
                        i] += self.prompt_budget - self.prompt_lens[i]


class RocketVanillaAttention(VanillaAttention):
    """
    RocketKV sparse attention implementation.
        - Context phase: only support sparse kv cache
        - Generation phase: only support sparse attention computation
    """

    Metadata = RocketVanillaAttentionMetadata

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
        super().__init__(layer_idx,
                         num_heads,
                         head_dim,
                         sparse_attention_config=sparse_attention_config,
                         num_kv_heads=num_kv_heads,
                         quant_config=quant_config,
                         q_scaling=q_scaling,
                         **kwargs)
        self.topr = sparse_attention_config.topr
        self.topk = sparse_attention_config.topk
        self.prompt_budget = sparse_attention_config.prompt_budget
        self.window_size = sparse_attention_config.window_size
        self.kernel_size = sparse_attention_config.kernel_size
        self.page_size = sparse_attention_config.page_size

    def _single_request_sparse_kv_predict(
            self, q: Optional[Tensor], k: Optional[Tensor], v: Optional[Tensor],
            metadata: RocketVanillaAttentionMetadata, past_seen_token: int,
            cache_idx: int, **kwargs) -> Optional[Tensor]:
        """
        Predict KV indices for writing new key/value pairs.
        For RocketKV:
            - Context phase: return SnapKV sparse kv indices
            - Generation phase: return None
        """
        if k is None or v is None:
            return None, 0

        # Generation phase: do not support sparse kv cache
        if past_seen_token > 0:
            return None, k.size(1)

        # Context phase: predict SnapKV sparse kv indices
        sparse_kv_indices = self._get_snapkv_indices(q, k)

        # Gather the key states using the sparse kv indices
        if sparse_kv_indices is not None:
            k_snap = triton_index_gather(k, sparse_kv_indices)
        else:
            k_snap = k

        # Update KT cache
        kt_cache_tensor = metadata.kv_cache_manager.get_kt_buffers(
            self.layer_idx)
        target_seq_len = past_seen_token + k_snap.size(1)
        kt_cache_position = torch.arange(past_seen_token // self.page_size,
                                         math.ceil(target_seq_len /
                                                   self.page_size),
                                         device=q.device)
        kt_cache_slot = metadata.kv_cache_manager.get_kt_cache_slot_from_cache_id(
            cache_idx)
        self._single_request_update_kt_cache(k_snap, kt_cache_tensor,
                                             target_seq_len, kt_cache_slot,
                                             kt_cache_position)
        return sparse_kv_indices, k_snap.size(1)

    def _single_request_sparse_attn_predict(
            self, q: Tensor, k: Optional[Tensor], v: Optional[Tensor],
            kv_cache_tensor: Tensor, metadata: RocketVanillaAttentionMetadata,
            past_seen_token: int, cache_idx: int, **kwargs) -> Optional[Tensor]:
        """
        Predict KV cache indices for sparse attention computation.
        For RocketKV:
            - Context phase: returns None (use full attention)
            - Generation phase: return RocketKV sparse indices for sparse attention computation
        """
        if k is None or v is None:
            return None, 0

        # Context phase: use full attention
        if past_seen_token == 0:
            return None, k.size(1)

        # Get RocketKV sparse indices
        kt_cache_slot = metadata.kv_cache_manager.get_kt_cache_slot_from_cache_id(
            cache_idx)
        sparse_indices = self._rocketkv_selection(q, k, metadata,
                                                  past_seen_token,
                                                  kt_cache_slot)
        return sparse_indices, sparse_indices.size(1)

    def _get_snapkv_indices(self, q: Tensor, k: Tensor) -> Tensor:
        """Get SnapKV sparse kv indices from the input sequence for context phase."""
        bsz = 1
        seq_len = k.size(1)

        # If the sequence length is less than the prompt budget, do not enable sparse kv cache
        if seq_len <= self.prompt_budget:
            return None

        # Use last window_size tokens as observation
        # (1, num_kv_heads, window_size, head_dim)
        q_obs = q[:, :, -self.window_size:]
        # (1, num_kv_heads, seq_len, head_dim)
        k_pre = repeat_kv(k.transpose(1, 2), self.num_key_value_groups)

        dist = (torch.arange(0, self.window_size, device=q.device)[:, None] -
                torch.arange(0, seq_len, device=q.device)[None, :] + seq_len -
                self.window_size)
        attention_mask = (dist >= 0)

        score = torch.matmul(q_obs, k_pre.transpose(-1, -2)) / math.sqrt(
            self.head_dim)

        score = torch.masked_fill(
            score,
            attention_mask.view(1, 1, self.window_size, seq_len) == False,
            torch.scalar_tensor(float("-inf"),
                                device=score.device,
                                dtype=score.dtype))

        score = torch.nn.functional.softmax(score, dim=-1)

        score = torch.masked_fill(
            score,
            attention_mask.view(1, 1, self.window_size, seq_len) == False,
            torch.scalar_tensor(0, device=score.device, dtype=score.dtype))

        score = score[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)

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

        return selected_indices.transpose(1, 2)

    def _single_request_update_kt_cache(self, k, kt_cache_tensor, seq_len,
                                        kt_cache_slot, cache_position):
        """Update KT cache for RocketKV algorithm."""
        # (1, num_kv_heads, 2*head_dim, num_pages_per_block)
        k_out = kt_cache_tensor[kt_cache_slot, :, :, :].unsqueeze(0)

        # k: (1, seq_len, num_kv_heads, head_dim)
        if k is not None:
            padding_len = self.page_size - (
                (k.size(1) - 1) % self.page_size + 1)
            padding_tensor = torch.full(
                (k.size(0), padding_len, k.size(2), k.size(3)),
                float('inf'),
                device=k.device,
                dtype=k.dtype)
            # (1, seq_len+padding_len, num_kv_heads, head_dim)
            k_min = torch.cat([k, padding_tensor], dim=1)
            # (1, num_pages, num_kv_heads, head_dim)->(1, num_kv_heads, head_dim, num_pages)
            k_min = k_min.reshape(
                k_min.size(0),
                k_min.size(1) // self.page_size, self.page_size, k.size(2),
                k_min.size(3)).amin(dim=2).permute(0, 2, 3, 1)
            k_max = torch.cat([k, -padding_tensor], dim=1)
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

    def _rocketkv_selection(self, q: Tensor, k: Tensor,
                            metadata: RocketVanillaAttentionMetadata,
                            past_seen_token: int, kt_cache_slot: int) -> Tensor:
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
        kt_cache_position = torch.arange(past_seen_token // self.page_size,
                                         math.ceil(target_seq_len /
                                                   self.page_size),
                                         device=q.device)
        kt_states = self._single_request_update_kt_cache(
            k, kt_cache_tensor, target_seq_len, kt_cache_slot,
            kt_cache_position)

        # Reshape query for multi-head processing
        qi = q.view(bsz, self.num_kv_heads, self.num_heads // self.num_kv_heads,
                    q_len, self.head_dim)
        qi_abs = torch.abs(qi)

        # Top-r selection on query features
        i1 = torch.topk(qi_abs.mean(dim=2, keepdim=True), self.topr,
                        dim=-1).indices
        qi_hat = _gather(qi, -1, i1)

        # Generate signed indices for key-token matching
        i1_sign = torch.where(
            qi_hat.sum(dim=2, keepdim=True) > 0, i1 + self.head_dim,
            i1).transpose(-1, -2)

        # Gather key tokens and compute attention scores
        kt_hat = _gather(kt_states.unsqueeze(2), -2, i1_sign)
        qk_hat = qi_hat @ kt_hat
        qk_hat = qk_hat.repeat_interleave(self.page_size,
                                          dim=-1)[:, :, :, :, :target_seq_len]
        scale = torch.sqrt(self.head_dim *
                           torch.abs(qi_hat).sum(dim=-1, keepdim=True) /
                           qi_abs.sum(dim=-1, keepdim=True))

        # (1, num_kv_heads, num_heads, target_seq_len)
        s_hat = _scaled_softmax(qk_hat, scale, dim=-1)

        topk = min(self.topk, target_seq_len)
        i2 = torch.topk(s_hat.mean(dim=2, keepdim=True), topk, dim=-1).indices

        iKV = i2[:, :, 0, 0, :].transpose(1, 2)  # (1, topk, num_kv_heads)

        return iKV


class RocketKVCacheManager(KVCacheManager):

    def __init__(
        self,
        kv_cache_config: KvCacheConfig,
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
        is_draft: bool = False,
        kv_connector_manager: Optional["KvCacheConnectorManager"] = None,
        sparse_attn_config: Optional["SparseAttentionConfig"] = None,
        **kwargs,
    ) -> None:

        assert not kv_cache_config.enable_block_reuse, "RocketKV cache requires block reuse to be disabled in KV cache config"

        super().__init__(
            kv_cache_config,
            kv_cache_type,
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
            max_beam_width=max_beam_width,
            is_draft=is_draft,
            **kwargs,
        )
        self.page_size = sparse_attn_config.page_size
        self.prompt_budget = sparse_attn_config.prompt_budget
        self.max_batch_size = max_batch_size

        self.kt_cache = {}
        for layer_idx in range(self.num_local_layers):
            local_layer_idx = layer_idx
            num_kv_heads = self.num_kv_heads_per_layer[local_layer_idx]

            # Allocate kt_cache based on max_batch_size instead of num_blocks
            kt_cache_shape = (self.max_batch_size, num_kv_heads, head_dim * 2,
                              math.ceil(max_seq_len / self.page_size))

            self.kt_cache[layer_idx] = torch.empty(kt_cache_shape,
                                                   device="cuda",
                                                   dtype=torch.bfloat16)
            self._clear_kt_cache_slot(list(range(self.max_batch_size)),
                                      layer_idx)

        self.request_to_slot: Dict[int, int] = {}
        self.free_slots: List[int] = list(range(self.max_batch_size))
        self.block_id_to_slot: Dict[int, int] = {}

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
        max_beam_width: int = 1,
    ):
        requests = super().add_dummy_requests(
            request_ids=request_ids,
            token_nums=token_nums,
            is_gen=is_gen,
            prepare_resource=prepare_resource,
            max_num_draft_tokens=max_num_draft_tokens,
            use_mrope=use_mrope,
            max_beam_width=max_beam_width,
        )
        if prepare_resource:
            for req in requests:
                request_id = req.py_request_id
                if request_id not in self.request_to_slot:
                    if not self.free_slots:
                        raise RuntimeError(
                            "No free slots in KT cache, please increase max_batch_size"
                        )
                    slot = self.free_slots.pop(0)
                    self.request_to_slot[request_id] = slot
                    self._clear_kt_cache_slot([slot])
                    cache_idx = self.get_cache_indices(req)[0]
                    self.block_id_to_slot[cache_idx] = slot

        return requests

    def shutdown(self):
        self.impl.release_pools()

    def get_kt_buffers(self, layer_idx: int):
        return self.kt_cache[layer_idx]

    def get_kt_cache_slot(self, request_id: int) -> int:
        return self.request_to_slot[request_id]

    def get_kt_cache_slot_from_cache_id(self, cache_idx: int) -> int:
        return self.block_id_to_slot[cache_idx]

    def prepare_resources(self, scheduled_batch):
        super().prepare_resources(scheduled_batch)

        for request in scheduled_batch.all_requests():
            request_id = request.py_request_id
            if request_id not in self.request_to_slot:
                if not self.free_slots:
                    raise RuntimeError(
                        "No free slots in KT cache, please increase max_batch_size"
                    )
                slot = self.free_slots.pop(0)
                self.request_to_slot[request_id] = slot
                self._clear_kt_cache_slot([slot])
                cache_idx = self.get_cache_indices(request)[0]
                self.block_id_to_slot[cache_idx] = slot

    def update_resources(self, scheduled_batch):
        for request in scheduled_batch.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                seq_len = request.get_num_tokens(0)
                rewind_len = max(seq_len - 1 - self.prompt_budget, 0)
                self.rewind_kv_cache(request, rewind_len)

    def free_resources(self, request):
        request_id = request.py_request_id

        if request_id in self.request_to_slot:
            slot = self.request_to_slot.pop(request_id)
            cache_idx = self.get_cache_indices(request)[0]
            if cache_idx in self.block_id_to_slot:
                self.block_id_to_slot.pop(cache_idx)
            self.free_slots.append(slot)
            self.free_slots.sort()

        super().free_resources(request)

    def _clear_kt_cache_slot(self,
                             slot_indices: List[int],
                             layer_idx: Optional[int] = None):
        layers_to_clear = [layer_idx] if layer_idx is not None else range(
            self.num_local_layers)
        for layer in layers_to_clear:
            kt_cache_tensor = self.kt_cache[layer]
            for slot_idx in slot_indices:
                if slot_idx < kt_cache_tensor.shape[0]:
                    head_dim = kt_cache_tensor.shape[2] // 2
                    kt_cache_tensor[slot_idx, :, :head_dim, :] = float('inf')
                    kt_cache_tensor[slot_idx, :, head_dim:, :] = float('-inf')

    @staticmethod
    def get_cache_size_per_token(model_config: ModelConfig,
                                 executor_config: ExecutorConfig,
                                 mapping: Mapping):
        sparse_attn_config = executor_config.sparse_attention_config
        # get kv cache dtype bytes
        mem_per_token = 2
        quant_config = model_config.quant_config
        if quant_config is not None and quant_config.quant_mode.has_fp8_kv_cache(
        ):
            mem_per_token = 1

        # get num key value heads
        config = model_config.pretrained_config
        num_key_value_heads = getattr(config, 'num_key_value_heads',
                                      config.num_attention_heads)
        if isinstance(num_key_value_heads, Iterable):
            num_key_value_heads = sum(num_key_value_heads) / len(
                num_key_value_heads)

        # get head dim
        tp_size = 1 if mapping.enable_attention_dp else mapping.tp_size
        head_dim = getattr(config, "head_dim", None)
        if not isinstance(head_dim, int):
            head_dim = config.hidden_size // config.num_attention_heads
        head_dim = head_dim * num_key_value_heads // tp_size

        # provide at least 1 layer to prevent division by zero cache size
        num_attention_layers = max(
            len(mapping.pp_layers(model_config.get_num_attention_layers())), 1)
        mem_per_token *= num_attention_layers * head_dim

        # K and V
        # 2 for K and V, 2 / page_size for KT cache
        kv_factor = 2 + 2 / sparse_attn_config.page_size
        mem_per_token *= kv_factor
        return mem_per_token
