import math
import os
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import flashinfer
import torch
from flashinfer.jit.core import check_cuda_arch
from typing_extensions import Self

from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..metadata import KVCacheParams
from ..utils import get_global_attrs, get_model_extra_attrs
from .interface import (AttentionBackend, AttentionMask, AttentionMetadata,
                        CustomAttentionMask, PredefinedAttentionMask)

try:
    check_cuda_arch()
except RuntimeError:
    # Override TORCH_CUDA_ARCH_LIST for JIT compilation of flashinfer kernels
    # since the existed TORCH_CUDA_ARCH_LIST may be too general and flashinfer requires sm75+.
    capability = torch.cuda.get_device_capability()
    arch_list = f"{capability[0]}.{capability[1]}"
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

from tensorrt_llm._utils import prefer_pinned


@dataclass(kw_only=True, frozen=True)
class PlanParams:
    """
    Parameters that affect the flashinfer execution plan
    """

    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype

    attention_mask_type: AttentionMaskType
    attention_mask_data: Optional[torch.Tensor] = None
    sm_scale: Optional[float] = None
    window_left: Optional[int] = None


@dataclass(kw_only=True)
class FlashInferWrappers:
    is_planned: bool
    decode_wrapper: Optional[
        flashinfer.BatchDecodeWithPagedKVCacheWrapper] = None
    prefill_wrapper: Optional[
        flashinfer.BatchPrefillWithPagedKVCacheWrapper] = None
    ragged_prefill_wrapper: Optional[
        flashinfer.BatchPrefillWithRaggedKVCacheWrapper] = None


@dataclass(kw_only=True)
class FlashInferAttentionMetadata(AttentionMetadata):
    workspace_buffer: Optional[torch.Tensor] = None

    # cache concat/split kernels when using PD disaggregation
    # expects KV cache in [max_num_pages, 2, num_kv_heads, page_size, head_dim] layout,
    # so set kv_layout as "HND" here
    kv_layout: Literal["NHD", "HND"] = "HND"

    # FlashInfer kernel backend: "fa2" (default), "fa3", "trtllm-gen", or "auto".
    # "trtllm-gen" uses pre-compiled FMHA kernels and supports head_dim=512.
    flashinfer_backend: str = "fa2"

    paged_kv_indptr_decode: torch.Tensor = field(init=False)
    paged_kv_indptr_prefill: torch.Tensor = field(init=False)
    _paged_kv_indices: torch.Tensor = field(init=False, repr=False)
    _paged_kv_last_page_len: torch.Tensor = field(init=False)
    _qo_indptr: torch.Tensor = field(init=False)
    _kv_indptr: torch.Tensor = field(init=False)
    _cached_token_lens: torch.Tensor = field(init=False)

    _plan_params_to_wrappers: Dict[PlanParams,
                                   FlashInferWrappers] = field(init=False)

    def needs_plan(self, plan_params: PlanParams) -> bool:
        if plan_params not in self._plan_params_to_wrappers:
            return True

        wrappers = self._plan_params_to_wrappers[plan_params]
        return not wrappers.is_planned

    def get_prefill_wrapper(
        self, plan_params: PlanParams
    ) -> flashinfer.BatchPrefillWithPagedKVCacheWrapper:
        assert plan_params in self._plan_params_to_wrappers, "Plan params not found, make sure to call plan()"
        result = self._plan_params_to_wrappers[plan_params].prefill_wrapper
        assert result is not None, "Prefill wrapper was not created in plan()"
        return result

    def get_decode_wrapper(
        self, plan_params: PlanParams
    ) -> flashinfer.BatchDecodeWithPagedKVCacheWrapper:
        assert plan_params in self._plan_params_to_wrappers, "Plan params not found, make sure to call plan()"
        result = self._plan_params_to_wrappers[plan_params].decode_wrapper
        return result

    def get_ragged_prefill_wrapper(
        self, plan_params: PlanParams
    ) -> flashinfer.BatchPrefillWithRaggedKVCacheWrapper:
        assert plan_params in self._plan_params_to_wrappers, "Plan params not found, make sure to call plan()"
        result = self._plan_params_to_wrappers[
            plan_params].ragged_prefill_wrapper
        assert result is not None, "Ragged prefill wrapper was not created in plan()"
        return result

    @property
    def paged_kv_indices(self) -> torch.Tensor:
        return self._paged_kv_indices[:self.num_generation_blocks +
                                      self.num_context_blocks]

    def get_paged_kv_indices_for_layer(self, layer_idx: int) -> torch.Tensor:
        """Return page indices for the pool that *layer_idx* belongs to.

        For non-VSWA models this returns the default shared indices.
        For VSWA models it returns the pool-specific indices.
        """
        if (self._vswa_layer_to_pool is None
                or self._vswa_pool_indices_cache is None):
            return self.paged_kv_indices
        pool_id = self._vswa_layer_to_pool.get(layer_idx)
        if pool_id is None:
            return self.paged_kv_indices
        total_blocks = self.num_generation_blocks + self.num_context_blocks
        return self._vswa_pool_indices_cache[pool_id][:total_blocks]

    def _swap_indices_for_head_dim(self, head_dim: int) -> None:
        """Swap _paged_kv_indices to the pool matching the given head_dim.

        Used by prepare() to re-plan each cached wrapper with correct indices.
        """
        head_dim_to_pool = getattr(self, '_vswa_head_dim_to_pool', None)
        if head_dim_to_pool is None or self._vswa_pool_indices_cache is None:
            return
        pool_id = head_dim_to_pool.get(head_dim)
        if pool_id is None:
            return
        rep_layer = self._vswa_pool_to_rep_layer.get(pool_id)
        if rep_layer is not None:
            self.swap_paged_kv_indices_for_layer(rep_layer)

    def swap_paged_kv_indices_for_layer(self, layer_idx: int) -> None:
        """Copy pool-specific page indices into the shared buffer.

        The FlashInfer wrappers reference ``_paged_kv_indices`` directly,
        so we overwrite its contents with the correct pool's data before
        each layer's plan/run cycle.  We track the currently active pool
        so we only copy when the pool actually changes.
        """
        if self._vswa_layer_to_pool is None:
            return
        pool_id = self._vswa_layer_to_pool.get(layer_idx)
        if pool_id is None:
            return  # Layer not in VSWA mapping
        active = getattr(self, '_vswa_active_pool_id', None)
        if pool_id == active:
            return  # Buffer already has the right data
        total_blocks = self.num_generation_blocks + self.num_context_blocks
        src = self._vswa_pool_indices_cache[pool_id][:total_blocks]
        self._paged_kv_indices[:total_blocks].copy_(src, non_blocking=True)
        self._vswa_active_pool_id = pool_id

    @property
    def paged_kv_last_page_len(self) -> torch.Tensor:
        return self._paged_kv_last_page_len[:self.num_contexts +
                                            self.num_generations]

    @property
    def qo_indptr(self) -> torch.Tensor:
        return self._qo_indptr[:self.num_contexts + self.num_generations + 1]

    @property
    def kv_indptr(self) -> torch.Tensor:
        return self._kv_indptr[:self.num_contexts + self.num_generations + 1]

    @property
    def cached_token_lens(self) -> torch.Tensor:
        return self._cached_token_lens[:self.num_contexts +
                                       self.num_generations]

    @property
    def batch_indices(self) -> torch.Tensor:
        return self._batch_indices[:self.num_tokens]

    @property
    def positions(self) -> torch.Tensor:
        return self._positions[:self.num_tokens]

    def __post_init__(self) -> None:
        super().__post_init__()
        self._post_init_with_buffers(self.cuda_graph_buffers)

    def _post_init_with_buffers(self, buffers) -> None:
        capture_graph = self.is_cuda_graph

        if self.workspace_buffer is None:
            # Note: even though flashinfer only recommends 128 MB, we have to push it
            # a bit higher to cover all possible CUDA graph cases. If it's too small,
            # warmup will crash.
            self.workspace_buffer = self.get_empty(
                buffers,
                (320 * 1024 * 1024, ),
                dtype=torch.uint8,
                cache_name="workspace_buffer",
                capture_graph=capture_graph,
            )

        self.paged_kv_indptr_decode = torch.empty((self.max_num_requests + 1, ),
                                                  device='cuda',
                                                  dtype=torch.int)
        self.paged_kv_indptr_prefill = torch.empty(
            (self.max_num_requests + 1, ), device='cuda', dtype=torch.int)
        self._paged_kv_last_page_len = torch.empty((self.max_num_requests, ),
                                                   device='cuda',
                                                   dtype=torch.int)

        self._qo_indptr = torch.zeros(self.max_num_requests + 1,
                                      device='cuda',
                                      dtype=torch.int)

        self._kv_indptr = torch.zeros(
            self.max_num_requests + 1, device='cuda',
            dtype=torch.int) if self.is_cross else self._qo_indptr

        self._cached_token_lens = torch.empty((self.max_num_requests, ),
                                              dtype=torch.int,
                                              device='cuda')
        self._batch_indices = torch.empty((self.max_num_tokens, ),
                                          dtype=torch.int,
                                          device='cuda')
        self._positions = torch.empty((self.max_num_tokens, ),
                                      dtype=torch.int,
                                      device='cuda')
        self._plan_params_to_wrappers = {}

        # VSWA (Variable Sliding Window Attention): models with per-layer
        # max_attention_window create separate V2 pool groups with independent
        # page numbering.  We need per-pool paged_kv_indices so each layer can
        # use the indices that match its pool's buffer.
        self._vswa_layer_to_pool: Optional[Dict[int, int]] = None
        self._vswa_pool_indices_cache: Optional[Dict[int, torch.Tensor]] = None

        if self.kv_cache_manager is not None:
            max_num_pages = self.kv_cache_manager.blocks_in_primary_pool
            self._paged_kv_indices = self.get_empty(
                buffers,
                (max_num_pages, ),
                dtype=torch.int,
                cache_name="_paged_kv_indices",
                capture_graph=capture_graph,
            )

            # Detect VSWA: check if the manager has multiple pools.
            # Guard on layer_to_pool_mapping_dict which is V2-specific — V1
            # managers also expose is_vswa but lack the per-pool infrastructure.
            if (getattr(self.kv_cache_manager, 'is_vswa', False) and hasattr(
                    self.kv_cache_manager, 'layer_to_pool_mapping_dict')):
                mgr = self.kv_cache_manager
                self._vswa_layer_to_pool = {}
                self._vswa_pool_to_rep_layer: Dict[int, int] = {}
                for layer_idx in getattr(mgr, 'layer_offsets', {}):
                    layer_offset = mgr.layer_offsets[layer_idx]
                    pool_id = mgr.layer_to_pool_mapping_dict[layer_offset]
                    self._vswa_layer_to_pool[layer_idx] = pool_id
                    if pool_id not in self._vswa_pool_to_rep_layer:
                        self._vswa_pool_to_rep_layer[pool_id] = layer_idx
                # Build head_dim → pool_id mapping using V2 per-layer head_dim
                self._vswa_head_dim_to_pool: Dict[int, int] = {}
                if hasattr(mgr, 'head_dim_per_layer'):
                    for layer_idx, pool_id in self._vswa_layer_to_pool.items():
                        hd = mgr.head_dim_per_layer[
                            mgr.layer_offsets[layer_idx]]
                        if hd not in self._vswa_head_dim_to_pool:
                            self._vswa_head_dim_to_pool[hd] = pool_id

    def create_cuda_graph_metadata(self,
                                   max_batch_size: int,
                                   sub_cross_metadata: bool = False,
                                   max_draft_tokens: int = 0,
                                   buffers=None) -> Self:
        metadata = super().create_cuda_graph_metadata(max_batch_size,
                                                      sub_cross_metadata,
                                                      max_draft_tokens)
        metadata.max_num_requests = max_batch_size
        metadata.max_num_tokens = max_batch_size * (1 + max_draft_tokens)
        # Post init again to make sure all tensors are allocated
        metadata.__post_init__()
        return metadata

    @property
    def page_size(self) -> int:
        """
        Number of tokens per cache page
        """
        assert self.kv_cache_manager is not None, (
            "page_size is undefined without a KV cache manager; use the "
            "ragged prefill path instead.")
        return self.kv_cache_manager.tokens_per_block

    def _plan_ragged_cudnn_no_kv(
        self,
        plan_params: PlanParams,
        ragged_prefill_wrapper: Any,
    ) -> None:
        is_causal = plan_params.attention_mask_type == AttentionMaskType.causal
        if plan_params.attention_mask_data is not None:
            window_left = -1
        else:
            window_left = plan_params.window_left

        # Lengths are already on GPU via AttentionMetadata (seq_lens setter -> _seq_lens_cuda).
        assert self.seq_lens_cuda is not None
        assert self.seq_lens is not None

        # NOTE: When kv_cache_manager is None (e.g. ViT), ragged prefill runs only for the context phase.
        # Restrict seq_lens to the first num_contexts entries accordingly.
        q_seqlens = self.seq_lens[:self.num_contexts]
        kv_seqlens = q_seqlens

        max_query_tokens_per_sequence = int(
            self.seq_lens[:self.num_contexts].max().item())
        max_key_value_tokens_per_sequence = max_query_tokens_per_sequence

        # cuDNN ragged prefill uses *element* offsets in qo/kv indptr, not token indptr.
        num_context_sequences = int(q_seqlens.shape[0])
        query_output_element_indptr = torch.zeros(
            num_context_sequences + 1,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )
        key_value_element_indptr = torch.zeros(
            num_context_sequences + 1,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
        )
        if num_context_sequences > 0:
            num_query_output_heads = plan_params.num_heads
            num_key_value_heads = plan_params.num_kv_heads
            attention_head_dim = plan_params.head_dim
            query_output_element_indptr[1:].copy_(
                torch.cumsum(q_seqlens, dim=0).mul_(num_query_output_heads *
                                                    attention_head_dim))
            key_value_element_indptr[1:].copy_(
                torch.cumsum(kv_seqlens, dim=0).mul_(num_key_value_heads *
                                                     attention_head_dim))

        q_seqlens_cuda = self.seq_lens_cuda[:self.num_contexts]
        kv_seqlens_cuda = q_seqlens_cuda[:self.num_contexts]

        ragged_prefill_wrapper.plan(
            qo_indptr=query_output_element_indptr,
            kv_indptr=key_value_element_indptr,
            num_qo_heads=plan_params.num_heads,
            num_kv_heads=plan_params.num_kv_heads,
            head_dim_qk=plan_params.head_dim,
            custom_mask=plan_params.attention_mask_data,
            causal=is_causal,
            sm_scale=plan_params.sm_scale,
            window_left=window_left,
            q_data_type=plan_params.q_dtype,
            kv_data_type=plan_params.kv_dtype,
            seq_lens=kv_seqlens_cuda,
            seq_lens_q=q_seqlens_cuda,
            max_token_per_sequence=max_query_tokens_per_sequence,
            max_sequence_kv=max_key_value_tokens_per_sequence,
            v_indptr=key_value_element_indptr,
            o_indptr=query_output_element_indptr,
        )

    def prepare(self) -> None:
        super().prepare()
        extra_attrs = get_model_extra_attrs()
        if extra_attrs is None:
            get_global_attrs().attention_metadata = weakref.ref(self)
        # start and end indices of each sequence in the ragged query
        torch.cumsum(self.seq_lens_cuda,
                     dim=0,
                     dtype=torch.int32,
                     out=self._qo_indptr[1:self.seq_lens_cuda.size(0) + 1])

        if self.kv_cache_manager is None:
            assert self.request_ids is not None
            assert self.num_generations == 0, (
                "FlashInfer without a KV cache manager only supports context-only "
                "batches (num_generations == 0) in TRT-LLM.")
            if self.is_cross:
                raise NotImplementedError(
                    "FlashInfer without a KV cache manager is not tested for cross attention."
                )
            self.kv_cache_params = KVCacheParams(use_cache=False)
            n = self.num_seqs
            self._cached_token_lens[:n].zero_()
            for plan_params in list(self._plan_params_to_wrappers.keys()):
                if plan_params.attention_mask_data is None:
                    self._plan_params_to_wrappers[
                        plan_params].is_planned = False
                    self._plan_with_params(plan_params)
                else:
                    del self._plan_params_to_wrappers[plan_params]
            return

        # indices of used cache blocks for each sequence
        assert self.request_ids is not None
        block_ids_per_seq = self.kv_cache_manager.get_batch_cache_indices(
            self.request_ids)

        # number of tokens in the kv cache for each sequence in the batch
        cached_token_lens = torch.tensor(
            self.kv_cache_params.num_cached_tokens_per_seq, dtype=torch.int)
        self._cached_token_lens[:cached_token_lens.size(0)].copy_(
            cached_token_lens, non_blocking=True)

        # number of tokens needed in the kv cache for each sequence after the next pass
        kv_lens = self.cached_token_lens + self.seq_lens_kv_cuda

        # start and end indices of each sequence in the ragged key and value
        # for self attention it's the same as qo_indptr so avoid computing twice.
        if self.is_cross:
            # start and end indices of each sequence in the ragged key and value
            torch.cumsum(self.seq_lens_kv_cuda,
                         dim=0,
                         dtype=torch.int32,
                         out=self._kv_indptr[1:self.seq_lens_kv_cuda.size(0) +
                                             1])

        # number of cache blocks used by each sequence in the cache
        # NOTE: do not use len(block_ids) - that will give you a number
        # that can be too big if using chunked prefill/kv cache reuse
        # since we allocate all blocks ahead of time.
        num_blocks = ((kv_lens + self.page_size - 1) // self.page_size)
        self.num_blocks = num_blocks.tolist()
        self.num_context_blocks = sum(self.num_blocks[:self.num_contexts])
        self.num_generation_blocks = sum(self.num_blocks[self.num_contexts:])

        paged_kv_indices_list = []
        for i, block_ids in enumerate(block_ids_per_seq):
            paged_kv_indices_list.extend(block_ids[:self.num_blocks[i]])

        paged_kv_indices = torch.tensor(paged_kv_indices_list,
                                        dtype=torch.int32)

        self._paged_kv_indices[:paged_kv_indices.size(0)].copy_(
            paged_kv_indices, non_blocking=True)

        # VSWA: build per-pool page index CUDA tensors so each layer can use
        # the indices that match its own pool's buffer.  Tensors live on CUDA
        # so that forward_impl swap via copy_() is device-to-device (CUDA-graph
        # capturable).
        if self._vswa_layer_to_pool is not None:
            unique_pools = set(self._vswa_layer_to_pool.values())
            primary_pool_id = self._vswa_layer_to_pool.get(0, 0)
            # Allocate a dedicated buffer for the primary pool's indices.
            # Using _paged_kv_indices directly would be an alias: when a
            # secondary pool swap overwrites _paged_kv_indices the primary
            # pool's cached entry would also be corrupted.
            total_idx = paged_kv_indices.size(0)
            buf_key_primary = f'_vswa_pool_buf_{primary_pool_id}'
            primary_buf = getattr(self, buf_key_primary, None)
            if primary_buf is None or primary_buf.size(0) < total_idx:
                primary_buf = torch.empty(max(total_idx, 1),
                                          dtype=torch.int,
                                          device='cuda')
                setattr(self, buf_key_primary, primary_buf)
            primary_buf[:total_idx].copy_(self._paged_kv_indices[:total_idx],
                                          non_blocking=True)
            self._vswa_pool_indices_cache = {
                primary_pool_id: primary_buf,
            }
            for pool_id in unique_pools:
                if pool_id == primary_pool_id:
                    continue
                rep_layer = self._vswa_pool_to_rep_layer[pool_id]
                pool_block_ids = self.kv_cache_manager.get_batch_cache_indices(
                    self.request_ids, layer_id=rep_layer)
                pool_idx_list = []
                for i, blk_ids in enumerate(pool_block_ids):
                    pool_idx_list.extend(blk_ids[:self.num_blocks[i]])
                pool_indices = torch.tensor(pool_idx_list, dtype=torch.int32)
                # Allocate or reuse a CUDA buffer for this pool.
                buf_key = f'_vswa_pool_buf_{pool_id}'
                buf = getattr(self, buf_key, None)
                if buf is None or buf.size(0) < pool_indices.size(0):
                    buf = torch.empty(max(pool_indices.size(0), total_idx),
                                      dtype=torch.int,
                                      device='cuda')
                    setattr(self, buf_key, buf)
                buf[:pool_indices.size(0)].copy_(pool_indices,
                                                 non_blocking=True)
                self._vswa_pool_indices_cache[pool_id] = buf
            self._vswa_active_pool_id = primary_pool_id

        # number of tokens in the last cache block used by each sequence
        paged_kv_last_page_len = kv_lens - (num_blocks - 1) * self.page_size
        self._paged_kv_last_page_len[:paged_kv_last_page_len.size(0)].copy_(
            paged_kv_last_page_len, non_blocking=True)

        # Ragged page table, see https://docs.flashinfer.ai/tutorials/kv_layout.html#page-table-layout
        # For decoding, this MUST be allocated ahead of time (for CUDA graphs).
        # Prefill is prepared here as well just for the sake of consistency.
        paged_kv_indptr_decode = torch.cumsum(
            torch.Tensor([0] + self.num_blocks[self.num_contexts:]).int(),
            dtype=torch.int32,
            dim=0,
        )
        self.paged_kv_indptr_decode[:paged_kv_indptr_decode.size(0)].copy_(
            paged_kv_indptr_decode, non_blocking=True)

        paged_kv_indptr_prefill = torch.cumsum(
            torch.tensor([0] + self.num_blocks[:self.num_contexts],
                         dtype=torch.int32),
            dtype=torch.int32,
            dim=0,
        )
        self.paged_kv_indptr_prefill[:paged_kv_indptr_prefill.size(0)].copy_(
            paged_kv_indptr_prefill, non_blocking=True)

        # This paged_kv_indptr attribute has both prefill and decode information in it.
        # It's for the append_paged_kv_cache kernel.
        if self.num_contexts == 0:
            self.paged_kv_indptr = self.paged_kv_indptr_decode[:
                                                               paged_kv_indptr_decode
                                                               .size(0)]
        elif self.num_generations == 0:
            self.paged_kv_indptr = self.paged_kv_indptr_prefill[:
                                                                paged_kv_indptr_prefill
                                                                .size(0)]
        else:
            assert not self.is_cuda_graph, "Cannot mix decode/prefill with CUDA graphs"
            self.paged_kv_indptr = torch.cumsum(
                torch.tensor([0] + self.num_blocks, dtype=torch.int32),
                dtype=torch.int32,
                dim=0,
            ).cuda()

        # For cross attention, num_tokens is 0 during decode, and we don't need to update kv cache.
        if self.num_tokens > 0:
            batch_indices, positions = flashinfer.get_batch_indices_positions(
                self.kv_indptr,
                flashinfer.get_seq_lens(self.paged_kv_indptr,
                                        self.paged_kv_last_page_len,
                                        self.page_size),
                self.num_tokens,
            )
            self._batch_indices[:batch_indices.size(0)].copy_(batch_indices,
                                                              non_blocking=True)
            self._positions[:positions.size(0)].copy_(positions,
                                                      non_blocking=True)

        # Generally, plan_params with non-trivial attention_mask_data are relevant only the
        # corresponding forward pass. So, flush them out here as they won't be relevant for
        # subsequent forward calls.
        for plan_params in list(self._plan_params_to_wrappers.keys()):
            if plan_params.attention_mask_data is None:
                # Mark as not-planned so forward_impl will re-plan just before
                # use.  We intentionally do NOT re-plan here because multiple
                # wrappers (e.g., different head_dim for hybrid attention) share
                # the same workspace_buffer, and sequential re-planning would
                # overwrite the first wrapper's workspace with the second's.
                self._plan_params_to_wrappers[plan_params].is_planned = False
            else:
                del self._plan_params_to_wrappers[plan_params]

        # VSWA: restore primary pool indices as the default.
        if (self._vswa_layer_to_pool is not None
                and self._vswa_pool_indices_cache is not None):
            primary_pool_id = self._vswa_layer_to_pool.get(0, 0)
            total_blocks = self.num_generation_blocks + self.num_context_blocks
            src = self._vswa_pool_indices_cache[primary_pool_id][:total_blocks]
            self._paged_kv_indices[:total_blocks].copy_(src, non_blocking=True)
            self._vswa_active_pool_id = primary_pool_id

        if self.cross is not None and self.cross is not self:
            self.cross.prepare()

    def plan(self,
             num_heads: int,
             num_kv_heads: int,
             head_dim: int,
             q_dtype: torch.dtype,
             kv_dtype: torch.dtype,
             attention_mask_type: int,
             q_scaling: Optional[float] = None,
             attention_window_size: Optional[int] = None,
             attention_mask_data: Optional[torch.Tensor] = None) -> PlanParams:

        sm_scale = None
        if q_scaling is not None:
            sm_scale = 1 / (math.sqrt(head_dim) * q_scaling)

        plan_params = PlanParams(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
            sm_scale=sm_scale,
            window_left=attention_window_size
            if attention_window_size is not None else -1,
            attention_mask_type=AttentionMaskType(attention_mask_type),
            attention_mask_data=attention_mask_data)
        return self._plan_with_params(plan_params)

    def _use_tensor_cores(self, plan_params: PlanParams):
        return plan_params.kv_dtype in [
            torch.float8_e4m3fn, torch.float8_e5m2
        ] or (plan_params.num_heads // plan_params.num_kv_heads >= 4)

    def _plan_with_params(self, plan_params: PlanParams) -> PlanParams:
        if not self.needs_plan(plan_params):
            return plan_params

        if self.is_cuda_graph and torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "Cannot plan() for flashinfer kernels while stream is capturing. "
                "Make sure you run a few warmup runs before capturing the graph!"
            )

        if self.kv_cache_manager is None:
            if self.is_cuda_graph:
                raise NotImplementedError(
                    "FlashInfer without a KV cache manager does not support "
                    "CUDA graph capture; use the TRTLLM attention backend.")
            if plan_params in self._plan_params_to_wrappers:
                ragged_prefill_wrapper = self._plan_params_to_wrappers[
                    plan_params].ragged_prefill_wrapper
            else:
                ragged_prefill_wrapper = (
                    flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                        self.workspace_buffer,
                        self.kv_layout,
                        backend="cudnn",
                    ))
            torch.cuda.current_stream().synchronize()
            if self.num_contexts <= 0:
                raise ValueError(
                    "FlashInfer ragged prefill without KV cache requires "
                    "num_contexts >= 1.")
            self._plan_ragged_cudnn_no_kv(plan_params, ragged_prefill_wrapper)
            self._plan_params_to_wrappers[plan_params] = FlashInferWrappers(
                is_planned=True,
                ragged_prefill_wrapper=ragged_prefill_wrapper,
            )
            return plan_params

        if plan_params in self._plan_params_to_wrappers:
            prefill_wrapper = self._plan_params_to_wrappers[
                plan_params].prefill_wrapper
        else:
            prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer,
                self.kv_layout,
                backend=self.flashinfer_backend,
                qo_indptr_buf=self.qo_indptr,
                paged_kv_indptr_buf=self.paged_kv_indptr_prefill,
                paged_kv_indices_buf=self._paged_kv_indices,
                paged_kv_last_page_len_buf=self._paged_kv_last_page_len,
                use_cuda_graph=self.is_cuda_graph)

        is_causal = plan_params.attention_mask_type == AttentionMaskType.causal

        def prefill_plan():
            # Setting `window_left` to -1 for custom attention mask is important.
            # Else, FlashInfer proceeds to use SWA regardless of attention_mask_data.
            if plan_params.attention_mask_data is not None:
                window_left = -1
            else:
                window_left = plan_params.window_left
            prefill_wrapper.plan(
                self.qo_indptr[:self.num_contexts + 1],
                self.paged_kv_indptr_prefill[:self.num_contexts + 1],
                self._paged_kv_indices[:self.num_context_blocks],
                self._paged_kv_last_page_len[:self.num_contexts],
                plan_params.num_heads,
                plan_params.num_kv_heads,
                plan_params.head_dim,
                self.page_size,
                causal=is_causal,
                sm_scale=plan_params.sm_scale,
                window_left=window_left,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
                custom_mask=plan_params.attention_mask_data,
            )

        if plan_params in self._plan_params_to_wrappers:
            decode_wrapper = self._plan_params_to_wrappers[
                plan_params].decode_wrapper
        else:
            use_tensor_cores = self._use_tensor_cores(plan_params)

            decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                self.kv_layout,
                use_cuda_graph=self.is_cuda_graph,
                paged_kv_indptr_buffer=self.paged_kv_indptr_decode,
                paged_kv_indices_buffer=self._paged_kv_indices,
                paged_kv_last_page_len_buffer=self._paged_kv_last_page_len,
                use_tensor_cores=use_tensor_cores
                or self.flashinfer_backend == "trtllm-gen",
                backend=self.flashinfer_backend if self.flashinfer_backend
                != "fa2" else ("fa2" if torch.cuda.get_device_capability(0) == (
                    9, 0) else "auto"),
            )

        def decode_plan():
            paged_kv_indptr = torch.cumsum(
                torch.Tensor([0] +
                             self.num_blocks[self.num_contexts:]).int().cuda(),
                dtype=torch.int32,
                dim=0,
            )
            assert decode_wrapper is not None
            decode_wrapper.plan(
                paged_kv_indptr,
                self.paged_kv_indices[self.num_context_blocks:],
                self.paged_kv_last_page_len[self.num_contexts:],
                plan_params.num_heads,
                plan_params.num_kv_heads,
                plan_params.head_dim,
                self.page_size,
                sm_scale=plan_params.sm_scale,
                window_left=plan_params.window_left,
                q_data_type=plan_params.q_dtype,
                kv_data_type=plan_params.kv_dtype,
            )

        # Must sync after append_paged_kv_cache and before plan.
        torch.cuda.current_stream().synchronize()

        if self.num_contexts > 0:
            prefill_plan()

        if self.num_generations > 0:
            decode_plan()

        self._plan_params_to_wrappers[plan_params] = FlashInferWrappers(
            prefill_wrapper=prefill_wrapper,
            decode_wrapper=decode_wrapper,
            is_planned=True,
        )

        return plan_params


class FlashInferAttention(AttentionBackend[FlashInferAttentionMetadata]):

    Metadata = FlashInferAttentionMetadata

    def __init__(
        self,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        quant_config: Optional[QuantConfig] = None,
        q_scaling: Optional[float] = None,
        skip_create_weights_in_init: bool = False,
        **kwargs,
    ):
        self.flashinfer_backend = kwargs.pop('flashinfer_backend', None)
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads,
                         quant_config, **kwargs)
        if not skip_create_weights_in_init:
            self.update_quant_config(self.quant_config)
        self.q_scaling = q_scaling

    def update_quant_config(self, new_quant_config: Optional[QuantConfig]):
        self.quant_config = new_quant_config
        self.has_fp8_kv_cache = False
        if self.quant_config:
            self.has_fp8_kv_cache = self.quant_config.layer_quant_mode.has_fp8_kv_cache(
            )

    def forward_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        metadata: FlashInferAttentionMetadata,
        attention_mask_type: int,
        output: torch.Tensor,
        attention_mask_data: Optional[torch.Tensor] = None,
        attention_window_size: Optional[int] = None,
    ) -> None:
        # Query
        q = q.view(-1, self.num_heads, self.head_dim)

        if metadata.kv_cache_manager is None:
            assert k is not None and v is not None, (
                "FlashInfer without a KV cache manager requires key/value tensors."
            )
            if self.has_fp8_kv_cache:
                raise NotImplementedError(
                    "FP8 KV cache is not supported for FlashInfer without a "
                    "KV cache manager.")
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            plan_params = metadata.plan(
                self.num_heads,
                self.num_kv_heads,
                self.head_dim,
                q_dtype=q.dtype,
                kv_dtype=k.dtype,
                q_scaling=self.q_scaling,
                attention_window_size=attention_window_size,
                attention_mask_type=attention_mask_type,
                attention_mask_data=attention_mask_data,
            )
            wrapper = metadata.get_ragged_prefill_wrapper(plan_params)
            # cuDNN's ragged prefill kernel assumes contiguous NHD tensors.
            wrapper.run(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                out=output.view(-1, self.num_heads, self.head_dim),
            )
            return

        # Key and Value
        kv_cache = metadata.kv_cache_manager.get_buffers(
            self.layer_idx, kv_layout=metadata.kv_layout)

        # VSWA: swap in the correct pool's page indices for this layer
        # before both the KV cache append and the attention plan/run.
        metadata.swap_paged_kv_indices_for_layer(self.layer_idx)

        if k is not None and v is not None:
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            if self.has_fp8_kv_cache:
                assert kv_cache.dtype == torch.float8_e4m3fn, (
                    f"KV cache should have fp8 dtype, but get {kv_cache.dtype}")
                k = k.to(torch.float8_e4m3fn)
                v = v.to(torch.float8_e4m3fn)
            assert k.dtype == v.dtype == kv_cache.dtype, (
                f"KV cache dtype {kv_cache.dtype} does not match k/v dtype {k.dtype}/{v.dtype}"
            )

            flashinfer.page.append_paged_kv_cache(
                append_key=k,
                append_value=v,
                batch_indices=metadata.batch_indices,
                positions=metadata.positions,
                paged_kv_cache=kv_cache,
                kv_indices=metadata.paged_kv_indices,
                kv_indptr=metadata.paged_kv_indptr,
                kv_last_page_len=metadata.paged_kv_last_page_len,
                kv_layout=metadata.kv_layout)

        num_contexts = metadata.num_contexts
        num_generations = metadata.num_generations
        num_ctx_tokens = metadata.num_ctx_tokens

        def prefill_forward(plan_params: PlanParams, out: torch.Tensor):
            wrapper = metadata.get_prefill_wrapper(plan_params)
            wrapper.run(q[:num_ctx_tokens],
                        kv_cache,
                        out=out.view(-1, self.num_heads, self.head_dim))

        def decode_forward(plan_params: PlanParams, out: torch.Tensor):
            wrapper = metadata.get_decode_wrapper(plan_params)
            wrapper.run(q[num_ctx_tokens:],
                        kv_cache,
                        out=out.view(-1, self.num_heads, self.head_dim))

        # Override FlashInfer kernel backend if the attention layer requests it
        # (e.g., Gemma4 uses trtllm-gen for head_dim=512 support).
        if (self.flashinfer_backend is not None
                and metadata.flashinfer_backend != self.flashinfer_backend):
            metadata.flashinfer_backend = self.flashinfer_backend

        # trtllm-gen does not support non-causal (bidirectional) attention used
        # by multimodal models for image/audio tokens.  Fall back to causal
        # when trtllm-gen is active and a custom mask is provided.  This is an
        # approximation: only full attention layers (head_dim>256) are affected
        # and only during multimodal prefill.
        effective_mask_type = attention_mask_type
        effective_mask_data = attention_mask_data
        if (metadata.flashinfer_backend == "trtllm-gen"
                and attention_mask_data is not None):
            effective_mask_type = int(AttentionMaskType.causal)
            effective_mask_data = None

        # this will do nothing if the last forward pass had the same parameters
        plan_params = metadata.plan(self.num_heads,
                                    self.num_kv_heads,
                                    self.head_dim,
                                    q_dtype=q.dtype,
                                    kv_dtype=kv_cache.dtype,
                                    q_scaling=self.q_scaling,
                                    attention_window_size=attention_window_size,
                                    attention_mask_type=effective_mask_type,
                                    attention_mask_data=effective_mask_data)

        if num_contexts == 0:
            decode_forward(plan_params, output)
        elif num_generations == 0:
            prefill_forward(plan_params, output)
        else:
            prefill_forward(plan_params, output[:num_ctx_tokens, :])
            decode_forward(plan_params, output[num_ctx_tokens:, :])

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: FlashInferAttentionMetadata,
                *,
                attention_window_size: Optional[int] = None,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
                attention_mask_data: Optional[torch.Tensor] = None,
                output: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if attention_mask == CustomAttentionMask.CUSTOM:
            assert attention_mask_data is not None, "attention_mask_data is required for custom attention mask."
            attention_mask_type = int(AttentionMaskType.custom_mask)
            attention_mask_data = attention_mask_data if attention_mask_data.ndim == 1 else attention_mask_data.flatten(
            )
        elif attention_mask == PredefinedAttentionMask.CAUSAL:
            attention_mask_type = int(AttentionMaskType.causal)
            attention_mask_data = None
        elif attention_mask == PredefinedAttentionMask.FULL:
            attention_mask_type = int(AttentionMaskType.padding)
            attention_mask_data = None
        else:
            raise ValueError("Unexpected attention mask type")

        if output is None:
            output = torch.empty_like(q)

        # FlashInfer's sliding window attention is inclusive, while the attention window size defined in TRTLLM is exclusive.
        # So we need to subtract 1 from the attention window size for a consistent behavior.
        if attention_window_size is not None:
            attention_window_size = attention_window_size - 1

        self.forward_impl(q=q,
                          k=k,
                          v=v,
                          metadata=metadata,
                          attention_mask_type=attention_mask_type,
                          attention_mask_data=attention_mask_data,
                          attention_window_size=attention_window_size,
                          output=output)
        return output
