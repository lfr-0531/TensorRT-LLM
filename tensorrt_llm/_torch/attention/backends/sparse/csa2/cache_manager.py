# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 owner-aware packed caches on the shared V2 request lifecycle."""

from __future__ import annotations

import weakref
from copy import copy
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
        enable_swa_bounded_replay: bool = True,
        num_kv_heads: int = 1,
        head_dim: int = 512,
        **kwargs,
    ):
        spec_config = kwargs.get("spec_config")
        if spec_config is not None:
            from tensorrt_llm._torch.speculative.utils import get_num_spec_layers

            if not spec_config.is_linear_tree or getattr(spec_config, "use_dynamic_tree", False):
                raise NotImplementedError(
                    "CSA2 supports contiguous-prefix chain verification, not token trees"
                )
            if get_num_spec_layers(spec_config):
                raise NotImplementedError(
                    "CSA2 virtual draft layers require an explicit model-layer mapping"
                )
        self._csa2_linear_speculation = spec_config is not None
        self._csa2_request_epoch_counter = 0
        self._csa2_request_epochs = {}
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
        self._reconstruction_enabled = enable_swa_bounded_replay and bool(
            self.layout.kv_source_layer_ids
        )
        self._csa2_reconstruction_plans = {}
        self._csa2_settled_replay_requests: set[int] = set()
        self._csa2_prepared_replay_steps = {}
        self._csa2_reconstruction_peers = {}
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
        self._csa2_reconstruction_members = {}
        self._csa2_reconstruction_roles = {}
        if self._reconstruction_enabled:
            for (layer, role), physical_layer in self._layer_roles.items():
                if role == CSA2CacheRole.GLOBAL:
                    continue
                group = self.layer_to_pool_mapping_dict[physical_layer]
                self._csa2_reconstruction_members.setdefault(group, set()).add(layer)
                self._csa2_reconstruction_roles.setdefault(group, set()).add(role)

    def _create_kv_cache(self, request_id, lora_task_id, input_tokens, **kwargs):
        cache = super()._create_kv_cache(request_id, lora_task_id, input_tokens, **kwargs)
        if cache is not None:
            self._csa2_request_epoch_counter += 1
            self._csa2_request_epochs[request_id] = self._csa2_request_epoch_counter
        return cache

    def request_epoch(self, request_id: int) -> int:
        """Distinguish a recycled request ID from its previous cache lifetime."""
        return self._csa2_request_epochs[request_id]

    def free_resources(self, request, pin_on_release=False):
        super().free_resources(request, pin_on_release=pin_on_release)
        self._csa2_request_epochs.pop(request.py_request_id, None)
        self._csa2_reconstruction_plans.pop(request.py_request_id, None)
        self._csa2_settled_replay_requests.discard(request.py_request_id)
        self._csa2_prepared_replay_steps.pop(request.py_request_id, None)
        self._csa2_reconstruction_peers.pop(request.py_request_id, None)

    def validate_verification(self, lengths: list[int], num_contexts: int, enabled: bool) -> None:
        if not enabled:
            return
        if not self._csa2_linear_speculation:
            raise ValueError("CSA2 verification requires configured chain rewind capacity")
        if any(length > self.max_draft_len + 1 for length in lengths[num_contexts:]):
            raise ValueError("CSA2 verification exceeds its reserved rewind window")

    def update_resources(self, scheduled_batch, attn_metadata=None, kv_cache_dtype_byte_size=None):
        # The generic relocation kernel assumes a uniform uncompressed KV pool.
        # Linear acceptance counts need only V2 resize/history updates, whereas
        # explicit token relocation would also require recomputing compressed pairs.
        for request in scheduled_batch.generation_requests:
            if request.py_num_accepted_draft_tokens_indices:
                raise NotImplementedError(
                    "CSA2 accepted-prefix updates do not support token relocation indices"
                )
        return super().update_resources(scheduled_batch, attn_metadata, kv_cache_dtype_byte_size)

    def prepare_context_cache(self, request, reuse_limit=None):
        reused = super().prepare_context_cache(request, reuse_limit)
        if reused is not None:
            cache = self.kv_cache_map[request.py_request_id]
            if (
                cache.requires_reconstruction
                and request.py_request_id not in self._csa2_reconstruction_plans
            ):
                prefix = cache.num_committed_tokens
                start = max(0, prefix - self.layout.window_size)
                groups = cache.get_reconstruction_ranges()
                members = set().union(
                    *(self._csa2_reconstruction_members[group] for group in groups)
                )
                self._csa2_reconstruction_plans[request.py_request_id] = {
                    "prefix": prefix,
                    "start": start,
                    "groups": set(groups),
                    "progress": {layer: start for layer in members},
                }
        return reused

    def apply_reconstruction_cursor(self, request, peer_manager=None) -> None:
        cache = self.kv_cache_map[request.py_request_id]
        if peer_manager is not None:
            peer = peer_manager.kv_cache_map[request.py_request_id]
            if cache.requires_reconstruction != getattr(peer, "requires_reconstruction", False):
                raise ValueError(
                    "Joint CSA2 reuse requires matching target/draft reconstruction policy"
                )
            if cache.requires_reconstruction and (
                getattr(peer_manager, "layout", None) != self.layout
                or peer.num_committed_tokens != cache.num_committed_tokens
            ):
                raise ValueError(
                    "Joint CSA2 replay requires identical layouts and a common GLOBAL prefix"
                )
        if not cache.requires_reconstruction:
            return
        plan = self._csa2_reconstruction_plans[request.py_request_id]
        if plan["prefix"] != cache.num_committed_tokens:
            raise ValueError("CSA2 reconstruction plan does not match the settled GLOBAL prefix")
        expected = min(plan["progress"].values(), default=plan["prefix"])
        if peer_manager is not None:
            peer_plan = peer_manager.automatic_replay_plan(request.py_request_id)
            expected = min(
                expected, min(peer_plan["progress"].values(), default=peer_plan["prefix"])
            )
            self._csa2_reconstruction_peers[request.py_request_id] = (
                weakref.ref(peer_manager),
                peer_manager.request_epoch(request.py_request_id),
            )
        request.py_csa2_global_reused_tokens = plan["prefix"]
        request.py_csa2_replay_tokens = plan["prefix"] - plan["start"]
        if request.context_current_position > expected and expected < plan["prefix"]:
            request.context_current_position = expected
            request.context_chunk_size = request.context_remaining_length
        self._csa2_settled_replay_requests.add(request.py_request_id)

    def has_settled_replay_prefix(self, request_id: int) -> bool:
        """Whether this cache lifetime already settled its GLOBAL reuse prefix."""
        return request_id in self._csa2_settled_replay_requests

    def prepare_context(self, request) -> bool:
        # Reaching P again after replay makes the native first-chunk predicate
        # true again. Prefix settlement must stay complete even after replay ACK.
        if self.has_settled_replay_prefix(request.py_request_id):
            if self.prepare_context_cache(request) is None:
                return False
        elif not super().prepare_context(request):
            return False
        self.apply_reconstruction_cursor(request)
        return True

    def automatic_replay_plan(self, request_id: int):
        cache = self.kv_cache_map[request_id]
        if not cache.requires_reconstruction:
            return None
        plan = self._csa2_reconstruction_plans.get(request_id)
        if plan is None:
            raise ValueError("CSA2 GLOBAL reuse requires prepare_context before attention")
        return plan

    def register_replay_step(self, metadata) -> None:
        for row, request_id in enumerate(metadata.request_ids):
            plan = self.automatic_replay_plan(request_id)
            if plan is None:
                continue
            start = metadata.csa2_request_start_positions[row]
            end = start + metadata.csa2_request_lengths[row]
            progress = min(plan["progress"].values(), default=plan["prefix"])
            if row >= metadata.num_contexts or start < plan["start"] or start > progress:
                raise ValueError("CSA2 request skipped its required SWA reconstruction interval")
            self._csa2_prepared_replay_steps[request_id] = (
                weakref.ref(metadata),
                metadata._csa2_forward_serial,
                start,
                end,
            )

    def validate_ready_query(self, request_id: int, start: int, replay: bool) -> None:
        cache = self.kv_cache_map[request_id]
        if cache.requires_reconstruction and not replay:
            raise ValueError("CSA2 cannot run ordinary attention before SWA reconstruction")
        if replay:
            return
        for group, (begin, _) in cache.get_reconstructed_ranges().items():
            roles = self._csa2_reconstruction_roles.get(group, set())
            if CSA2CacheRole.SWA in roles and max(0, start - self.layout.window_size + 1) < begin:
                raise ValueError("CSA2 query precedes the reconstructed SWA horizon")
            if (
                roles.intersection((CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE))
                and start % 2
                and start - 1 < begin
            ):
                raise ValueError("CSA2 query precedes the reconstructed compressor-state horizon")

    def try_allocate_generation(self, request) -> bool:
        cache = self.kv_cache_map.get(request.py_request_id)
        if cache is not None and cache.requires_reconstruction:
            raise ValueError("CSA2 generation requires completed SWA reconstruction")
        return super().try_allocate_generation(request)

    def _completed_replay_step(self, request_id: int) -> tuple[int, int, torch.cuda.Event]:
        plan = self._csa2_reconstruction_plans[request_id]
        step = self._csa2_prepared_replay_steps.get(request_id)
        if step is None:
            raise ValueError(
                "CSA2 reconstruction cannot be acknowledged without a prepared forward"
            )
        reference, serial, start, end = step
        metadata = reference()
        receipt = None if metadata is None else metadata.reconstruction_step_receipt(self)
        if receipt is None or receipt[0] != serial:
            raise ValueError("CSA2 reconstruction forward receipt is stale")
        _, layers, event = receipt
        if event is None or not set(plan["progress"]).issubset(layers):
            raise ValueError("CSA2 reconstruction requires every configured lifecycle member layer")
        return start, end, event

    def update_context_resources(self, scheduled_batch):
        pending = set()
        for request in scheduled_batch.context_requests:
            request_id = request.py_request_id
            cache = self.kv_cache_map.get(request_id)
            if cache is None or not cache.requires_reconstruction:
                continue
            # Overlap scheduling may suspend iteration N's cache before its
            # post-forward update. Its transient pages are no longer writable:
            # retain the unadvanced plan and replay that interval after resume.
            if not cache.is_active:
                pending.add(request_id)
                continue
            peer = None
            peer_state = self._csa2_reconstruction_peers.get(request_id)
            if peer_state is not None:
                reference, epoch = peer_state
                peer = reference()
                if peer is None or peer._csa2_request_epochs.get(request_id) != epoch:
                    raise ValueError("Joint CSA2 reconstruction peer cache lifetime changed")
                if not peer.kv_cache_map[request_id].is_active:
                    pending.add(request_id)
                    continue
            start, end, event = self._completed_replay_step(request_id)
            # Both engines share the retry cursor. Validate the peer before
            # publishing any progress or suffix while it still needs replay.
            if peer is not None and peer.kv_cache_map[request_id].requires_reconstruction:
                peer_start, peer_end, _ = peer._completed_replay_step(request_id)
                if (start, end) != (peer_start, peer_end):
                    raise ValueError(
                        "Joint CSA2 reconstruction requires matching forward intervals"
                    )
            plan = self._csa2_reconstruction_plans[request_id]
            self._stream.wait_event(event)
            for layer in plan["progress"]:
                plan["progress"][layer] = max(plan["progress"][layer], min(end, plan["prefix"]))
            for group in tuple(plan["groups"]):
                members = self._csa2_reconstruction_members[group]
                if all(plan["progress"][layer] >= plan["prefix"] for layer in members):
                    begin = (
                        plan["start"]
                        if CSA2CacheRole.SWA in self._csa2_reconstruction_roles[group]
                        else plan["prefix"] - plan["prefix"] % 2
                    )
                    cache.mark_reconstructed(group, begin, plan["prefix"])
                    plan["groups"].remove(group)
            if cache.requires_reconstruction:
                pending.add(request_id)
            else:
                self._csa2_reconstruction_plans.pop(request_id, None)
            # A partially rebuilt peer validates this same receipt when its
            # post-forward update follows ours. The next prepare replaces it.
            if peer is None or not cache.requires_reconstruction:
                self._csa2_prepared_replay_steps.pop(request_id, None)
        # Partial replay does not advance global history or recommit cached P.
        ready = copy(scheduled_batch)
        ready.context_requests_chunking = [
            r for r in scheduled_batch.context_requests_chunking if r.py_request_id not in pending
        ]
        ready.context_requests_last_chunk = [
            r for r in scheduled_batch.context_requests_last_chunk if r.py_request_id not in pending
        ]
        return super().update_context_resources(ready)

    def get_swa_replay_ranges(
        self,
        cached_prefix_lengths: list[int],
        suffix_lengths: list[int] | None = None,
        *,
        decoder: bool = False,
    ) -> tuple[tuple[int, int], ...]:
        """Return required writable query intervals for bounded reconstruction.

        The GLOBAL prefix must already be restored by the caller. Native V2
        matching is unchanged. Ordinary W-token next-query retention may omit
        the first replay token at C-W; allocate every row in the returned range
        explicitly rather than assuming the restored trailing window suffices.
        """
        suffix_lengths = (
            [0] * len(cached_prefix_lengths) if suffix_lengths is None else suffix_lengths
        )
        if len(suffix_lengths) != len(cached_prefix_lengths):
            raise ValueError("CSA2 replay prefixes and suffixes must have matching request counts")
        result = []
        for prefix, suffix in zip(cached_prefix_lengths, suffix_lengths):
            if prefix < 0 or suffix < 0 or prefix + suffix > self.max_seq_len:
                raise ValueError("CSA2 replay range exceeds its admitted context")
            if decoder and suffix:
                raise ValueError("Decoder SWA replay requires the complete prompt GLOBAL cache")
            result.append((max(0, prefix - self.layout.window_size), prefix + suffix))
        return tuple(result)

    def _window(self, base: int) -> int:
        return base + self.max_draft_len + self.reuse_match_backoff

    def _get_runtime_cache_size_layer_components(self):
        sizes, windows = [], []
        for layer in self.pp_layers:
            sizes.append(528)
            windows.append(
                self._window(self.layout.window_size + int(self._reconstruction_enabled))
            )
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
                    reconstructible=self._reconstruction_enabled and window is not None,
                )
            )

        for layer in self.pp_layers:
            add(
                layer,
                [CSA2CacheRole.SWA],
                [528],
                self._window(self.layout.window_size + int(self._reconstruction_enabled)),
            )
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
        scratch = config.swa_scratch_reuse
        if scratch is not None:
            # Linear verification may reject every draft token. Context
            # lookahead (num_extra_kv_tokens) can be one smaller than this.
            scratch = copy(scratch)
            scratch.max_rewind_len = max(scratch.max_rewind_len, self.max_draft_len)
        config = copy(config)
        config.layers = layers
        config.swa_scratch_reuse = scratch
        return config

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

    def gather_indexer_keys(
        self, layer_idx: int, slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather exact FP4 bytes with the shared native strided-cache kernel.

        Logical coordinates address the 68-byte index view; the gather kernel
        applies its actual 356-byte record stride. Its scale offset is per row,
        unlike DSA's native page-footer cache layout.
        """
        if slots.ndim != 1 or slots.dtype != torch.int64 or not slots.is_cuda:
            raise ValueError("CSA2 index gathering requires one-dimensional CUDA int64 slots")
        pool = self.get_index_buffer(layer_idx)
        owner = self.layout.layer(layer_idx).kv_source
        page_size = self.tokens_per_block // self.layout.compress_ratios[owner]
        cache = pool.view(-1, page_size, 1, 68)
        valid = (slots >= 0) & (slots < pool.shape[0])
        offsets = slots.clamp(0, pool.shape[0] - 1) * 68
        data, scales = torch.ops.trtllm.indexer_k_cache_gather_op(
            cache, offsets.contiguous(), (offsets + 64).contiguous(), 0, slots.numel(), 64
        )
        data = torch.where(valid[:, None], data.view(torch.int8), 0)
        scales = torch.where(valid[:, None], scales.view(torch.int32), 0)
        return data, scales

    def gather_indexer_pages(
        self, layer_idx: int, row_starts: torch.Tensor, output: torch.Tensor
    ) -> None:
        """Refresh caller-owned native FP4 footer pages without requantization.

        row_starts names the first physical source row of each 64-entry native
        page. Negative starts denote padding (including the reserved zero page).
        The scratch allocation is bounded by referenced requests, not pool size.
        """
        if output.dtype != torch.uint8 or output.shape != (row_starts.numel(), 64, 1, 68):
            raise ValueError("CSA2 native index scratch must be uint8 [pages,64,1,68]")
        if not output.is_contiguous() or output.device != row_starts.device:
            raise ValueError("CSA2 native index scratch must be contiguous on the slots device")
        rows = row_starts[:, None] + torch.arange(64, device=row_starts.device)
        rows = torch.where(row_starts[:, None] >= 0, rows, -1).flatten()
        data, scales = self.gather_indexer_keys(layer_idx, rows)
        flat = output.view(row_starts.numel(), 64 * 68)
        flat[:, : 64 * 64].copy_(data.view(torch.uint8).reshape(-1, 64 * 64))
        flat[:, 64 * 64 :].copy_(scales.view(torch.uint8).reshape(-1, 64 * 4))

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
            windows.append(layout.window_size + int(bool(layout.kv_source_layer_ids)) + reserve)
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
