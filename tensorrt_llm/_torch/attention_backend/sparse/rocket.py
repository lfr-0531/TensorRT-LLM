import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

import tensorrt_llm
import tensorrt_llm.bindings
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.bindings.executor import KvCacheConfig as KvCacheConfigCpp
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm._torch.attention_backend.vanilla import VanillaAttention, VanillaAttentionMetadata, repeat_kv, generate_causal_mask
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.mapping import Mapping
from tensorrt_llm._torch.pyexecutor.resource_manager import BaseResourceManager
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, SamplingConfig
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._torch.attention_backend import AttentionBackend
from tensorrt_llm._torch.attention_backend.interface import AttentionMask, PredefinedAttentionMask
from tensorrt_llm.llmapi import SamplingParams

class RocketVanillaAttentionMetadata(VanillaAttentionMetadata):
    # for SnapKV
    window_size: int
    kernel_size: int

    def update_sparse_attn_param(self, window_size: int, kernel_size: int):
        self.window_size = window_size
        self.kernel_size = kernel_size
    
    def __repr__(self):
        base_repr = super().__repr__()
        sparse_params = f", window_size={self.window_size}, kernel_size={self.kernel_size}"
        if base_repr.endswith(')'):
            return base_repr[:-1] + sparse_params + ')'
        else:
            return base_repr + sparse_params

class RocketVanillaAttention(AttentionBackend[RocketVanillaAttentionMetadata]):

    Metadata = RocketVanillaAttentionMetadata

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
        super().__init__(layer_idx, num_heads, head_dim, num_kv_heads=num_kv_heads, quant_config=quant_config, q_scaling=q_scaling, **kwargs)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

    def _single_request_update_kv_cache(self, k, v, kv_cache_tensor, seq_len,
                                        cache_idx, cache_position):
        k_out = kv_cache_tensor[cache_idx, 0, :, :, :].unsqueeze(0) # (1, num_heads, seq_len, head_dim)
        v_out = kv_cache_tensor[cache_idx, 1, :, :, :].unsqueeze(0) # (1, num_heads, seq_len, head_dim)

        if k is not None and v is not None:
            access_type = self._access_type[k.dtype.itemsize]
            k_out.view(dtype=access_type).index_copy_(2, cache_position,
                                                      k.view(dtype=access_type)) # (1, num_heads, seq_len, head_dim) why it's index 2
            v_out.view(dtype=access_type).index_copy_(2, cache_position,
                                                      v.view(dtype=access_type))

        return k_out[:, :, :seq_len, :], v_out[:, :, :seq_len, :]
    
    def _single_request_update_kt_cache(self, k, kt_cache_tensor, seq_len,
                                        page_size, cache_idx, cache_position):
        k_out = kt_cache_tensor[cache_idx, :, :, :].unsqueeze(0) # (1, num_kv_heads, head_dim, num_page)

        if k is not None:
            padding_len = page_size - ((k.size(2) - 1) % page_size + 1)
            k_min = torch.cat([k, torch.full((k.size(0),k.size(1),padding_len,k.size(3)), float('inf'), device=k.device, dtype=k.dtype)], dim=2)
            k_min = k_min.reshape(k_min.size(0),k_min.size(1), k_min.size(2)//page_size, page_size, k_min.size(3)).amin(dim=-2).transpose(-1,-2)
            k_max = torch.cat([k, torch.full((k.size(0),k.size(1),padding_len,k.size(3)), float('-inf'), device=k.device, dtype=k.dtype)], dim=2)
            k_max = k_max.reshape(k_max.size(0),k_max.size(1), k_max.size(2)//page_size, page_size, k_max.size(3)).amax(dim=-2).transpose(-1,-2)
            k_value = torch.cat( 
                    [torch.min(k_min, k_out[:,:,:k_min.size(-2),cache_position]),
                     torch.max(k_max, k_out[:,:,k_max.size(-2):,cache_position])], dim=-2)
            access_type = self._access_type[k_value.dtype.itemsize]
            k_out.view(dtype=access_type).index_copy_(-1, cache_position,
                                                      k_value.view(dtype=access_type))

        return k_out[:, :, :, :seq_len//page_size]
    
    def _single_request_forward(self, q, k, v, attention_mask: AttentionMask,
                                kv_cache_tensor, kt_cache_tensor, past_seen_token, 
                                seq_len_kv, page_size, topr, topk, 
                                window_size, kernel_size, prompt_budget, cache_idx):

        bsz = 1
        q_len = q.size(0)
        # print(q.shape, k.shape, v.shape)
        is_decode = (q_len == 1)

        # Query
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Key and Value
        target_seq_len = past_seen_token
        # print(f'past_seen_token: {past_seen_token}')
        if k is not None and v is not None:
            kv_len = k.size(0)
            k = k.view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            if self.quant_config and self.quant_config.layer_quant_mode.has_any_quant(
            ):
                qc = self.quant_config
                if qc.layer_quant_mode.has_fp8_kv_cache():
                    assert kv_cache_tensor.dtype == torch.float8_e4m3fn, f"KV cache should have fp8 dtype, but get {kv_cache_tensor.dtype}"
                    k = k.to(torch.float8_e4m3fn)
                    v = v.to(torch.float8_e4m3fn)
            assert k.dtype == v.dtype == kv_cache_tensor.dtype, f"KV cache dtype {kv_cache_tensor.dtype} does not match k/v dtype {k.dtype}/{v.dtype}"

        if not is_decode:
            q_obs = q[:,:,-window_size:] # (1, num_heads, window_size, head_dim)
            k_pre = repeat_kv(k[:,:,:-window_size], self.num_key_value_groups) # (1, num_heads, seq_len-window_size, num_kv_heads, head_dim)
            # print(q.shape, k.shape, q_obs.shape, k_pre.shape)
            score = torch.matmul(q_obs, k_pre.transpose(-1,-2))/math.sqrt(self.head_dim) # (1, num_heads, window_size, seq_len-window_size)
            score = torch.nn.functional.softmax(score, dim=-1).sum(dim=-2) # (1, num_heads, window_size)
            score = score.view(bsz, self.num_kv_heads, self.num_key_value_groups, -1).sum(dim=2) # (1, num_kv_heads, window_size)
            # print(score.shape, kernel_size)
            score = torch.nn.functional.max_pool1d(score, kernel_size=kernel_size, padding=kernel_size//2, stride=1)
            indices = score.topk(prompt_budget-window_size, dim=-1).indices.sort().values
            indices = indices.unsqueeze(-1).expand(-1,-1,-1,self.head_dim)
            k_compress = k[:,:,:-window_size].gather(dim=2, index=indices)
            v_compress = k[:,:,:-window_size].gather(dim=2, index=indices)
            k_snap = torch.cat([k_compress, k[:,:,-window_size:]], dim=2)
            v_snap = torch.cat([v_compress, v[:,:,-window_size:]], dim=2)
        
            target_seq_len += k_snap.size(2)
            kt_cache_position = torch.arange(math.ceil(past_seen_token/page_size),
                                         math.ceil(target_seq_len/page_size),
                                         device=q.device)
        
            self._single_request_update_kt_cache(
                k_snap, kt_cache_tensor, target_seq_len, page_size, cache_idx, kt_cache_position)

            cache_position = torch.arange(past_seen_token,
                                      target_seq_len,
                                      device=q.device)

            self._single_request_update_kv_cache(
                k_snap, v_snap, kv_cache_tensor, target_seq_len, cache_idx, cache_position)
            key_states, value_states = k, v
        else:
            target_seq_len += k.size(2)
            # print(f'target_seq_len: {target_seq_len}')
            
            # Update cache positions
            kt_cache_position = torch.arange(math.ceil(past_seen_token/page_size),
                                         math.ceil(target_seq_len/page_size),
                                         device=q.device)
            cache_position = torch.arange(past_seen_token,
                                      target_seq_len,
                                      device=q.device)

            # Update KT cache and get KT states
            kt_states = self._single_request_update_kt_cache(
                k, kt_cache_tensor, target_seq_len, page_size, cache_idx, kt_cache_position)

            # Update KV cache and get key/value states  
            key_states, value_states = self._single_request_update_kv_cache(
                k, v, kv_cache_tensor, target_seq_len, cache_idx, cache_position)

            # Helper functions for sparse attention
            def _gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
                dim += (dim < 0) * t.ndim
                return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))

            @torch.compile(disable=not torch.cuda.is_available())
            def _scaled_softmax(x: Tensor, divscale: Tensor | float, dim: int) -> Tensor:
                return torch.softmax(x / divscale, dim=dim)

            # Reshape query for multi-head processing
            qi = q.view(bsz, self.num_kv_heads, self.num_heads//self.num_kv_heads, q_len, self.head_dim)
            qi_abs = torch.abs(qi)
            
            # print(f'qi.shape: {qi.shape}')
            # Top-r selection on query features
            i1 = torch.topk(qi_abs.sum(dim=2, keepdim=True), topr, dim=-1).indices
            # print(f'i1.shape: {i1.shape}, i1: {i1}')
            qi_hat = _gather(qi, -1, i1)
            # print(f'qi_hat.shape: {qi_hat.shape}')
            
            # Generate signed indices for key-token matching
            i1_sign = torch.where(qi_hat.sum(dim=2, keepdim=True) > 0, 
                                i1 + self.head_dim, i1).transpose(-1, -2)
            
            # Gather key tokens and compute attention scores
            # print(f'i1_sign.shape: {i1_sign.shape}')
            # print(f'kt_states.shape: {kt_states.shape}')
            kt_hat = _gather(kt_states.unsqueeze(2), -2, i1_sign)
            # print(f'after gather, kt_hat.shape: {kt_hat.shape}')
            qk_hat = qi_hat @ kt_hat
            # print(f'qk_hat.shape: {qk_hat.shape}')
            qk_hat = qk_hat.repeat_interleave(page_size, dim=-1)[:,:,:,:,:target_seq_len]
            # print(f'after repeat, qk_hat.shape: {qk_hat.shape}')
            # Compute scaling factor for attention scores
            scale = torch.sqrt(
                self.head_dim
                * torch.abs(qi_hat).sum(dim=-1, keepdim=True)
                / qi_abs.sum(dim=-1, keepdim=True)
            )
            
            # Apply scaled softmax
            s_hat = _scaled_softmax(qk_hat, scale, dim=-1)
            
            # Top-k selection on attention scores
            topk = min(topk, target_seq_len)
            # print(f's_hat.shape: {s_hat.shape}, topk: {topk}')
            i2 = torch.topk(s_hat.sum(dim=2), topk, dim=-1).indices
            iKV = i2[..., 0, :, None]
            
            # Gather final key and value states
            key_states = _gather(key_states, -2, iKV)
            value_states = _gather(value_states, -2, iKV)
         

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Attention Mask
        is_causal = False
        attn_mask = None
        if attention_mask == PredefinedAttentionMask.CAUSAL:
            if past_seen_token == 0:
                is_causal = True
            elif q_len != 1:
                # attn_mask: 4-D tensor (batch_size, 1, query_seq_len, seq_len)
                attn_mask = generate_causal_mask(bsz, target_seq_len,
                                                 cache_position, q.device)
        elif attention_mask == PredefinedAttentionMask.FULL:
            pass
        else:
            raise ValueError("Unexpected attention mask type")

 
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            key_states,
            value_states,
            is_causal=is_causal,
            attn_mask=attn_mask,
        )

        attn_output = attn_output.squeeze(0) # (1, num_heads, seq_len, head_dim)
        return attn_output

    def forward(self,
                q: torch.Tensor,
                k: Optional[torch.Tensor],
                v: Optional[torch.Tensor],
                metadata: RocketVanillaAttentionMetadata,
                *,
                attention_mask: AttentionMask = PredefinedAttentionMask.CAUSAL,
                **kwargs) -> torch.Tensor:
        if metadata.kv_cache_manager is None:
            # NOTE: WAR for no kv cache attn e.g. BERT,
            # try to separate the kv cache estimation path from no kv cache attn.
            num_heads = self.num_heads
            num_kv_heads = self.num_kv_heads
            return RocketVanillaAttention.no_kv_cache_forward(
                q=q,
                k=k,
                v=v,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                metadata=metadata,
                attention_mask=attention_mask)

        past_seen_tokens = metadata.kv_cache_manager.past_seen_tokens
        cache_indices = [
            block_ids[0] for block_ids in metadata.block_ids_per_seq
        ]
        kv_cache_tensor = metadata.kv_cache_manager.get_buffers(self.layer_idx)
        kt_cache_tensor = metadata.kv_cache_manager.kt_get_buffers(self.layer_idx)
        page_size = metadata.kv_cache_manager.get_page_size()
        topr = metadata.kv_cache_manager.get_topr()
        topk = metadata.kv_cache_manager.get_topk()
        prompt_budget = metadata.kv_cache_manager.get_prompt_budget()
        window_size = metadata.window_size
        kernel_size = metadata.kernel_size
        q_len = q.size(0)

        assert len(cache_indices) == metadata.seq_lens.nelement()

        offset = 0
        offset_kv = 0
        attn_outputs = []
        for i, (seq_len, seq_len_kv) in enumerate(
                zip(metadata.seq_lens, metadata.seq_lens_kv)):
            # print(f'seq_len: {seq_len}, seq_len_kv: {seq_len_kv}, metadata.seq_lens: {metadata.seq_lens}, metadata.seq_lens_kv: {metadata.seq_lens_kv}')
            single_q = q[offset:offset + seq_len]
            single_k = k[
                offset_kv:offset_kv +
                seq_len_kv] if k is not None and seq_len_kv != 0 else None
            single_v = v[
                offset_kv:offset_kv +
                seq_len_kv] if k is not None and seq_len_kv != 0 else None
            if i < metadata.num_contexts:
                past_seen_token = 0
                past_seen_tokens.append(past_seen_token)
            else:
                past_seen_token = past_seen_tokens[i]
            cache_idx = cache_indices[i]

            attn_output = self._single_request_forward(single_q, single_k,
                                                       single_v, attention_mask,
                                                       kv_cache_tensor,
                                                       kt_cache_tensor,
                                                       past_seen_token,
                                                       seq_len_kv,
                                                       page_size,
                                                       topr,
                                                       topk,
                                                       window_size,
                                                       kernel_size,
                                                       prompt_budget,
                                                       cache_idx)
            attn_outputs.append(attn_output)
            if i < metadata.num_contexts:
                past_seen_tokens[i] = prompt_budget
            else: 
                past_seen_tokens[i] += 1

            offset += seq_len
            offset_kv += seq_len_kv

        attn_output = torch.cat(attn_outputs, dim=1)
        attn_output = attn_output.transpose(0, 1).contiguous()
        attn_output = attn_output.view(q_len, -1)

        return attn_output

class RocketKVCacheManager(BaseResourceManager):

    def __init__(
        self,
        # kv cache parameters
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
        # Some speculative decoding methods need to use different kv lengths for the
        # draft/target layers. Add extra tokens to haddle this issue.
        num_extra_kv_tokens: int = 0,
        topr: int = 76,
        topk: int = 128,
        prompt_budget: int = 1266,
        page_size: int = 3
    ) -> None:
        self.num_layers = num_layers
        self.mapping = mapping
        self.dtype = dtype
        self.kv_cache_type = kv_cache_type

        tp_size = mapping.tp_size
        if mapping.enable_attention_dp:
            tp_size = 1

        if isinstance(num_kv_heads, int):
            self.num_kv_heads_per_layer = [
                (num_kv_heads + tp_size - 1) // tp_size
                for _ in range(num_layers)
            ]

        else:
            assert len(num_kv_heads) == self.num_layers

            self.num_kv_heads_per_layer = []
            for layer_idx, kv_head in enumerate(num_kv_heads):
                if kv_head is not None:
                    self.num_kv_heads_per_layer.append(
                        (kv_head + tp_size - 1) // tp_size)
                else:
                    self.num_kv_heads_per_layer.append(0)

        assert len(self.num_kv_heads_per_layer) > 0

        self.is_homongenous = all(val == self.num_kv_heads_per_layer[0]
                                  for val in self.num_kv_heads_per_layer[1:])

        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.tokens_per_block = tokens_per_block
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.kv_factor = 1 if kv_cache_type == CacheTypeCpp.SELFKONLY else 2
        self.num_extra_kv_tokens = num_extra_kv_tokens
        self.event_buffer_max_size = kv_cache_config.event_buffer_max_size

        # initialize kv cache manager
        cache_shape = (num_layers, max_batch_size, 2, num_kv_heads, max_seq_len, head_dim)
        self.kv_cache = torch.empty(cache_shape, device="cuda", dtype=torch.bfloat16)
        kt_cache_shape = (num_layers, max_batch_size, num_kv_heads, head_dim, math.ceil(max_seq_len/page_size))
        self.kt_cache = torch.cat(
            [torch.full(kt_cache_shape, float('inf'), device="cuda", dtype=torch.bfloat16),
             torch.full(kt_cache_shape, float('-inf'), device="cuda", dtype=torch.bfloat16)], dim=-2
        ).contiguous()
        self.page_size = page_size
        self.topr = topr
        self.topk = topk
        self.prompt_budget = prompt_budget
        self.end_pos = [0] * max_batch_size
        self.req_mapping = {}
        self.impl = None
        self.past_seen_tokens = []

    def add_sequence(self, request_id: int, prompt_len: int):
        if request_id not in self.req_mapping:
            for i,pos in enumerate(self.end_pos):
                if pos == 0:
                    self.req_mapping[request_id] = i
                    self.end_pos[i] = prompt_len
                    break

    def remove_sequence(self, request_id: int):
        if request_id in self.req_mapping:
            assert self.req_mapping[request_id] >= 0 and self.req_mapping[request_id] < len(self.end_pos)
            self.end_pos[self.req_mapping[request_id]] = 0

    def add_token(self, request_id: int):
        if request_id in self.req_mapping:
            assert self.req_mapping[request_id] >= 0 and self.req_mapping[request_id] < len(self.end_pos)
            self.end_pos[self.req_mapping[request_id]] += 1

    def add_dummy_requests(
        self,
        request_ids: List[int],
        token_nums: Optional[List[int]] = None,
        is_gen: bool = False,
        prepare_resource: bool = True,
        max_num_draft_tokens: int = 0,
        use_mrope: bool = False,
    ):
        beam_width = 1
        requests = []
        for i, req_id in enumerate(request_ids):
            sampling_params = SamplingParams()
            token_num = token_nums[
                i] if token_nums is not None else 1 + max_num_draft_tokens
            # Using 1 instead of 0 prevents NaN during warmup in e.g. Deepseek
            req = LlmRequest(request_id=req_id,
                             max_new_tokens=1,
                             input_tokens=[1] * token_num,
                             sampling_config=SamplingConfig(
                                 sampling_params._get_sampling_config()),
                             is_streaming=False,
                             encoder_input_tokens=None)
            req.paged_kv_block_ids = []
            if prepare_resource:
                self.add_sequence(req_id, token_num)
            if is_gen:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
                req.prompt_len = token_num - 1 + max_num_draft_tokens
                req.py_prompt_len = req.prompt_len
                if max_num_draft_tokens > 0:
                    req.py_draft_tokens = [0] * max_num_draft_tokens
            requests.append(req)
        return requests

    def get_batch_cache_indices(
        self,
        request_ids: List[int],
    ):
        return [[self.get_mapping_idx(request_id)] for request_id in request_ids]    

    def get_buffers(self, layer_idx: int):
        return self.kv_cache[layer_idx]
 
    def kt_get_buffers(self, layer_idx: int):
        return self.kt_cache[layer_idx]

    def get_mapping_idx(self, request_id: int):
        if request_id in self.req_mapping:
            return self.req_mapping[request_id]
        else:
            return -1

    def get_end_pos(self, mapping_idx: int):
        assert mapping_idx >= 0 and mapping_idx < len(self.end_pos)
        return self.end_pos[mapping_idx]

    def get_page_size(self):
        return self.page_size

    def get_topr(self):
        return self.topr

    def get_topk(self):
        return self.topk

    def get_prompt_budget(self):
        return self.prompt_budget

    def get_past_seen_tokens(self):
        return self.past_seen_tokens

    def get_max_resource_count(self) -> int:
        return 4096

    def get_needed_resource_to_completion(self, request: LlmRequest) -> int:
        return 0

    def get_num_free_blocks(self) -> int:
        return 4096

    def get_kv_cache_stats(self):
        return None

    def prepare_resources(self, scheduled_batch: ScheduledRequests):
        super().prepare_resources(scheduled_batch)
        context_batch = scheduled_batch.context_requests
        generation_batch = scheduled_batch.generation_requests
        # allocate KV Cache
        for req in context_batch:
            if req.is_first_context_chunk:
                self.add_sequence(req.py_request_id, req.prompt_len)
                for _ in range(self.num_extra_kv_tokens):
                    self.add_token(req.py_request_id)
                if req.py_draft_tokens is not None:
                    for _ in range(len(req.py_draft_tokens)):
                        self.add_token(req.py_request_id)

        for req in generation_batch:
            self.add_token(req.py_request_id)
            if req.py_draft_tokens is not None:
                for _ in range(len(req.py_draft_tokens)):
                    self.add_token(req.py_request_id)

    def update_resources(self, scheduled_batch: ScheduledRequests):
        pass

    def free_resources(self, request: LlmRequest):
        super().free_resources(request)
        self.remove_sequence(request.py_request_id)
    
    def get_num_available_tokens(self, max_num_draft_tokens: int = 0) -> int:        
        res = (self.get_num_free_blocks() * self.tokens_per_block -
                self.num_extra_kv_tokens - max_num_draft_tokens)
        return res