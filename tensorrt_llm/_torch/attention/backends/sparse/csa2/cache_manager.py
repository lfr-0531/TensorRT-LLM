# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 owner-aware packed caches on the shared V2 request lifecycle."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum

import torch

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    GPU_LEVEL,
    KVCacheManagerV2,
    Role,
    _estimate_full_attn_size_per_token,
    _estimate_swa_cache_size,
)
from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor, prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    LayerId,
    PageIndexMode,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._common import BAD_PAGE_INDEX

from .params import CSA2Layout
from .quantization import store_rows


class CSA2CacheRole(Enum):
    SWA = "swa"
    GLOBAL = "global"
    COMPRESSOR_KV = "compressor_kv"
    COMPRESSOR_SCORE = "compressor_score"

    @property
    def role(self) -> DataRole:
        return DataRole(f"csa2_{self.value}")

    @property
    def index_mode(self) -> PageIndexMode:
        return PageIndexMode.SHARED if self == self.GLOBAL else PageIndexMode.PER_LAYER


class CSA2CacheManager(KVCacheManagerV2):
    """Private SWA, owner-only GLOBAL records, and ratio-two compressor state.

    GLOBAL stores main and index bytes in one 356-byte record, making their
    copy-on-write, eviction and transfer atomic at the allocator page level.
    The exposed main/index views retain that record's row stride.
    """

    def __init__(
        self,
        kv_cache_config,
        kv_cache_type,
        *,
        num_layers: int,
        tokens_per_block: int,
        mapping,
        pretrained_config=None,
        sparse_attention_config=None,
        layout: CSA2Layout | None = None,
        num_kv_heads: int = 1,
        head_dim: int = 512,
        **kwargs,
    ):
        layer_mask = kwargs.get("layer_mask")
        if layer_mask is not None and not all(layer_mask):
            raise ValueError("CSA2 cache does not support disabled layers in layer_mask")
        if layout is None:
            if pretrained_config is None:
                model_config = kwargs.get("model_config")
                pretrained_config = getattr(model_config, "pretrained_config", None)
            if pretrained_config is None:
                raise ValueError("CSA2 requires pretrained_config or an explicit layout")
            layout = CSA2Layout.from_hf_config(pretrained_config)
        if len(layout.compress_ratios) < num_layers:
            raise ValueError("CSA2 layout must describe every active attention layer")
        if tokens_per_block not in (128, 256):
            raise ValueError("CSA2 cache requires tokens_per_block 128 or 256")
        if kv_cache_type != CacheType.SELFKONLY or num_kv_heads != 1 or head_dim != 512:
            raise ValueError("CSA2 cache requires SELFKONLY with one 512-dimensional KV head")
        if mapping.pp_size != 1 or mapping.cp_size != 1:
            raise ValueError("CSA2 shared cache owners currently require PP=CP=1")
        self.layout = replace(
            layout,
            compress_ratios=layout.compress_ratios[:num_layers],
            kv_source_layer_ids=tuple(i for i in layout.kv_source_layer_ids if i < num_layers),
            index_source_layer_ids=tuple(
                i for i in layout.index_source_layer_ids if i < num_layers
            ),
            candidate_source_layer_id=(
                layout.candidate_source_layer_id
                if layout.candidate_source_layer_id is not None
                and layout.candidate_source_layer_id < num_layers
                else None
            ),
        )
        self._global_buffers = {}
        super().__init__(
            kv_cache_config,
            kv_cache_type,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            mapping=mapping,
            **kwargs,
        )
        self.is_vswa = True

    def _window(self, base: int) -> int:
        return base + self.max_draft_len + self.reuse_match_backoff

    def _get_runtime_cache_size_layer_components(self):
        sizes, windows = [], []
        for layer in self.pp_layers:
            sizes.append(528)
            windows.append(self._window(self.layout.window_size))
            if layer in self.layout.kv_source_layer_ids:
                ratio = self.layout.compress_ratios[layer]
                sizes.append(356 // ratio)
                windows.append(None)
                if ratio == 2:
                    # One lifecycle group: count both equally-sized state buffers.
                    sizes.append(4096)
                    windows.append(self._window(2))
        return sizes, windows

    def _get_typical_seq_len(self, kv_cache_config):
        return kv_cache_config.avg_seq_len or self.max_seq_len

    def get_layer_bytes_per_token(self, local_layer_idx, data_role):
        # Replaced by the packed declarative layer configuration below.
        return 1

    def _build_cache_config(self, config):
        layers = []
        self._layer_roles = {}
        self._physical_roles = {}

        def add(model_layer, roles, sizes, window):
            layer_id = LayerId(len(layers))
            for role in roles:
                self._layer_roles[model_layer, role] = layer_id
                self._physical_roles[layer_id, role.role] = (model_layer, role)
            layers.append(
                AttentionLayerConfig(
                    layer_id=layer_id,
                    buffers=[
                        BufferConfig(role=role.role, size=size * self.tokens_per_block)
                        for role, size in zip(roles, sizes)
                    ],
                    sliding_window_size=window,
                    num_sink_tokens=None,
                )
            )

        for layer in self.pp_layers:
            add(layer, [CSA2CacheRole.SWA], [528], self._window(self.layout.window_size))
            if layer in self.layout.kv_source_layer_ids:
                ratio = self.layout.compress_ratios[layer]
                add(layer, [CSA2CacheRole.GLOBAL], [356 // ratio], None)
                if ratio == 2:
                    add(
                        layer,
                        [CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE],
                        [2048, 2048],
                        self._window(2),
                    )
        for layer in self.pp_layers:
            owner = self.layout.layer(layer).kv_source
            if owner is not None:
                self._layer_roles[layer, CSA2CacheRole.GLOBAL] = self._layer_roles[
                    owner, CSA2CacheRole.GLOBAL
                ]
        return replace(config, layers=layers)

    def get_buffers(self, layer_idx: int, role: CSA2CacheRole = CSA2CacheRole.SWA):
        layer_id = self._layer_roles[layer_idx, role]
        addr = self.impl.get_mem_pool_base_address(layer_id, role.role, role.index_mode)
        upper = self.impl.get_page_index_upper_bound(layer_id, role.role)
        converter = self.impl.get_page_index_converter(layer_id, role.role)
        if role.index_mode == PageIndexMode.PER_LAYER and converter.layer_offset is not None:
            upper += converter.layer_offset * converter.expansion
        if role == CSA2CacheRole.GLOBAL:
            rows, width, dtype = (
                self.tokens_per_block // self.layout.compress_ratios[layer_idx],
                356,
                DataType.UINT8,
            )
        elif role == CSA2CacheRole.SWA:
            rows, width, dtype = self.tokens_per_block, 528, DataType.UINT8
        else:
            rows, width, dtype = self.tokens_per_block, 512, DataType.FLOAT
        return convert_to_torch_tensor(TensorWrapper(addr, dtype, (upper, rows, width)))

    def get_swa_buffer(self, layer_idx: int) -> torch.Tensor:
        return self.get_buffers(layer_idx, CSA2CacheRole.SWA).view(-1, 528)

    def _get_global_buffer(self, layer_idx: int) -> torch.Tensor:
        owner = self.layout.layer(layer_idx).kv_source
        if owner is None:
            raise ValueError("SWA-only layers have no global cache")
        if owner not in self._global_buffers:
            self._global_buffers[owner] = self.get_buffers(owner, CSA2CacheRole.GLOBAL).view(
                -1, 356
            )
        return self._global_buffers[owner]

    def get_main_buffer(self, layer_idx: int) -> torch.Tensor:
        return self._get_global_buffer(layer_idx)[:, :288]

    def get_index_buffer(self, layer_idx: int) -> torch.Tensor:
        return self._get_global_buffer(layer_idx)[:, 288:]

    def write_swa(self, layer_idx, slots, values) -> None:
        store_rows(self.get_swa_buffer(layer_idx), slots, values, "swa")

    def write_global(self, layer_idx, slots, main, index) -> None:
        if self.layout.layer(layer_idx).kv_source != layer_idx:
            raise ValueError("Only the configured KV source may publish global rows")
        store_rows(self.get_main_buffer(layer_idx), slots, main, "main")
        store_rows(self.get_index_buffer(layer_idx), slots, index, "index")

    def get_cache_indices(self, request_id: int, layer_idx: int, role: CSA2CacheRole):
        layer_id = self._layer_roles[layer_idx, role]
        pool_id = self.layer_to_pool_mapping_dict[layer_id]
        cache = self.kv_cache_map[request_id]
        converter = self.impl.get_page_index_converter(layer_id, role.role)
        return converter(
            cache.get_base_page_indices(pool_id).tolist(),
            role.index_mode,
            cache.get_scratch_desc(pool_id),
        )

    def get_layer_page_index_scale(self, layer_idx: int) -> int:
        return int(
            self.impl.get_page_index_scale(
                self._layer_roles[layer_idx, CSA2CacheRole.SWA], CSA2CacheRole.SWA.role
            )
        )

    def get_batch_cache_indices(self, request_ids, layer_idx=None, num_blocks_per_seq=None):
        layer_idx = self.pp_layers[0] if layer_idx is None else layer_idx
        result = []
        for i, request_id in enumerate(request_ids):
            pages = self.get_cache_indices(request_id, layer_idx, CSA2CacheRole.SWA)
            if num_blocks_per_seq is not None:
                pages = pages[: num_blocks_per_seq[i]]
            result.append(pages)
        return result

    def _prepare_page_table_tensor(self, index_mapper_capacity):
        self.num_attention_op_pools = self.num_local_layers
        self.kv_cache_pool_pointers = torch.tensor(
            [
                [
                    self.impl.get_mem_pool_base_address(
                        self._layer_roles[layer, CSA2CacheRole.SWA],
                        CSA2CacheRole.SWA.role,
                        PageIndexMode.PER_LAYER,
                    ),
                    0,
                ]
                for layer in self.pp_layers
            ],
            dtype=torch.int64,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self.kv_cache_pool_mapping = torch.tensor(
            [[i, 0] for i in range(self.num_local_layers)],
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self.host_kv_cache_block_offsets = torch.full(
            (
                self.num_pools,
                index_mapper_capacity * self.max_beam_width,
                2,
                self.max_blocks_per_seq,
            ),
            BAD_PAGE_INDEX,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )
        self._host_swa_tables = torch.full(
            (self.num_local_layers, self.max_batch_size, self.max_blocks_per_seq),
            BAD_PAGE_INDEX,
            dtype=torch.int32,
            pin_memory=prefer_pinned(),
            device="cpu",
        )

    def copy_batch_block_offsets(
        self, dst_tensor, request_ids, beam_width, num_contexts, num_seqs, max_blocks=None
    ):
        if beam_width != 1:
            raise ValueError("CSA2 supports beam width one")
        self._host_swa_tables.fill_(BAD_PAGE_INDEX)
        width = min(dst_tensor.shape[-1], self.max_blocks_per_seq)
        for i, layer in enumerate(self.pp_layers):
            for j, request_id in enumerate(request_ids):
                pages = self.get_cache_indices(request_id, layer, CSA2CacheRole.SWA)
                count = min(width, len(pages))
                self._host_swa_tables[i, j, :count] = torch.as_tensor(
                    pages[:count], dtype=torch.int32
                )
        dst_tensor.fill_(BAD_PAGE_INDEX)
        dst_tensor[:, :num_seqs, 0, :width].copy_(
            self._host_swa_tables[:, :num_seqs, :width], non_blocking=True
        )

    @property
    def blocks_in_primary_pool(self):
        return self.impl.get_page_index_upper_bound(
            self._layer_roles[self.pp_layers[0], CSA2CacheRole.SWA], CSA2CacheRole.SWA.role
        )

    def get_num_free_blocks(self):
        assert not self.kv_cache_map, "Capacity query requires an empty cache manager"
        return max(
            self.impl.get_page_index_upper_bound(
                self._layer_roles[layer, CSA2CacheRole.SWA], CSA2CacheRole.SWA.role
            )
            for layer in self.pp_layers
        )

    def get_cache_bytes_per_token(self):
        return sum(
            356 // self.layout.compress_ratios[owner] for owner in self.layout.kv_source_layer_ids
        )

    def get_max_resource_count(self):
        return int(self.impl.get_quota(GPU_LEVEL))

    def get_needed_resource_to_completion(self, request):
        context = request.is_context_init_state
        tokens = request.prompt_len + self.num_extra_kv_tokens
        if not context:
            tokens += request.max_new_tokens
        pages = (max(0, tokens) + self.tokens_per_block - 1) // self.tokens_per_block
        sizes, windows = self._get_runtime_cache_size_layer_components()
        return sum(
            size
            * self.tokens_per_block
            * (
                pages
                if context or window is None
                else min(pages, (window + self.tokens_per_block - 1) // self.tokens_per_block + 1)
            )
            for size, window in zip(sizes, windows)
        )

    def get_disagg_role_mapper_kinds(self):
        return {
            Role.ALL: MapperKind.REPLICATED,
            **{role.role: MapperKind.REPLICATED for role in CSA2CacheRole},
        }

    def _iter_cache_buffers_for_invalid_check(self):
        for model_layer, role in self._physical_roles.values():
            yield self.get_buffers(model_layer, role)

    @classmethod
    def get_cache_size_per_token(
        cls,
        model_config,
        mapping,
        *,
        tokens_per_block,
        max_batch_size=0,
        spec_config=None,
        kv_cache_config=None,
        **kwargs,
    ):
        layout = CSA2Layout.from_hf_config(model_config.pretrained_config)
        if mapping.pp_size != 1 or mapping.cp_size != 1:
            raise ValueError("CSA2 cache sizing requires PP=CP=1")
        from tensorrt_llm._torch.speculative import draft_prompt_lookahead

        reserve = getattr(spec_config, "max_draft_len", 0)
        if kv_cache_config is not None and kv_cache_config.enable_block_reuse:
            reserve += draft_prompt_lookahead(spec_config) or 0
        sizes, windows = [], []
        for layer, ratio in enumerate(
            layout.compress_ratios[
                : kwargs.get("num_layers") or model_config.get_num_attention_layers()
            ]
        ):
            sizes.append(528)
            windows.append(layout.window_size + reserve)
            if layer in layout.kv_source_layer_ids:
                sizes.append(356 // ratio)
                windows.append(None)
                if ratio == 2:
                    sizes.append(4096)
                    windows.append(2 + reserve)
        per_token, fixed = _estimate_swa_cache_size(
            sizes, windows, tokens_per_block, context=False, scratch=False
        )
        return _estimate_full_attn_size_per_token(
            sizes, windows
        ) + per_token, fixed * max_batch_size
