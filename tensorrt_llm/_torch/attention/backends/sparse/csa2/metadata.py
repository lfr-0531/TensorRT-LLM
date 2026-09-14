# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 request metadata and bounded staging for TRTLLM sparse MLA."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams

from .params import CSA2Layer

_SWA_TILE = 128
_HEAD_DIM = 512


class CSA2TrtllmMetadata(TrtllmAttentionMetadata):
    """Manager-backed request metadata with bounded native query staging.

    Normal prepare resolves manager pages for the packed model forward.
    for_query_tile creates a separate compute-only metadata object.
    Every selected query is an internal generation request with Q length one.
    Causality, request isolation and window selection are already represented
    by its indices, so the same DSV4 generation kernel serves all model phases.
    These are compute staging pools, not persistent request-owned KV caches.
    """

    @property
    def tokens_per_block(self) -> int:
        return (
            self.kv_cache_manager.tokens_per_block
            if self.kv_cache_manager is not None
            else _SWA_TILE
        )

    @property
    def host_kv_cache_pool_pointers(self) -> torch.Tensor:
        return (
            self.pool_pointers
            if hasattr(self, "pool_pointers")
            else super().host_kv_cache_pool_pointers
        )

    @property
    def host_kv_cache_pool_mapping(self) -> torch.Tensor:
        return (
            self.pool_mapping
            if hasattr(self, "pool_mapping")
            else super().host_kv_cache_pool_mapping
        )

    def _allocate(
        self, capacity: int, heads: int, extra_capacity: int, device: torch.device
    ) -> None:
        self.swa_pool = torch.empty(
            capacity, _SWA_TILE, _HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        self.extra_pool = torch.empty(
            capacity, max(extra_capacity, 1), _HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        self.pool_pointers = torch.tensor(
            [[self.swa_pool.data_ptr(), 0]], dtype=torch.int64, device="cpu"
        )
        self.pool_mapping = torch.zeros((1, 2), dtype=torch.int32, device="cpu")
        self.num_sparse_topk = _SWA_TILE + extra_capacity
        self.max_seq_len = self.num_sparse_topk
        self.kv_cache_params = KVCacheParams(use_cache=True)
        self.kv_cache_block_offsets = torch.zeros(
            (1, capacity, 2, (self.num_sparse_topk + _SWA_TILE - 1) // _SWA_TILE),
            dtype=torch.int32,
            device=device,
        )
        self.prepared_indices = torch.full(
            (capacity, self.num_sparse_topk), -1, dtype=torch.int32, device=device
        )
        self.prepared_lens = torch.empty(capacity, dtype=torch.int32, device=device)
        self.prepared_counter = torch.zeros(1, dtype=torch.uint32, device=device)
        self.prepared_cu_q = torch.arange(capacity + 1, dtype=torch.int32, device=device) * heads
        self.prepared_cu_kv = (
            torch.arange(capacity + 1, dtype=torch.int32, device=device) * self.num_sparse_topk
        )
        self.query_lens_host = torch.ones(capacity, dtype=torch.int32, device="cpu")
        self.query_lens_device = torch.ones(capacity, dtype=torch.int32, device=device)
        self.kv_lens.fill_(self.num_sparse_topk)
        self.kv_lens_cuda.fill_(self.num_sparse_topk)
        self.prompt_lens_cpu.fill_(1)
        self.prompt_lens_cuda.fill_(1)
        self.host_request_types.fill_(1)
        self.host_total_kv_lens.zero_()
        # Each query count has independent native workspace storage.
        # Eager initialization and capture must use that same tensor.
        self.cuda_graph_workspace = self.workspace

    @classmethod
    def for_query_tile(cls, q: torch.Tensor, extra_width: int) -> CSA2TrtllmMetadata:
        """Prepare fixed geometry before TrtllmAttention.forward allocates output.

        q is BF16 [queries, heads, 512]. Extra capacity is a selected-main-row
        bound, not the persistent cache size. Callers retain one metadata object
        per geometry and warm it with the normal forward before graph capture.
        As with TrtllmAttentionMetadata, set is_cuda_graph for captured calls.
        """
        count, heads, dim = q.shape
        if count <= 0 or dim != _HEAD_DIM or extra_width < 0:
            raise ValueError(
                "CSA2 metadata requires a nonempty 512D query tile and nonnegative extra width"
            )
        with torch.cuda.device(q.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare and warm CSA2 metadata before CUDA Graph capture")
            metadata = cls(max_num_requests=count, max_num_tokens=count)
            metadata._allocate(count, heads, (extra_width + 127) // 128 * 128, q.device)
        metadata.num_query_heads = heads
        metadata._seq_lens = metadata.query_lens_host
        metadata._seq_lens_cuda = metadata.query_lens_device
        metadata._num_contexts = metadata._num_ctx_tokens = 0
        metadata._num_generations = metadata._num_tokens = count
        metadata._bind_runtime_views(
            kv_lens_cuda=metadata.kv_lens_cuda,
            kv_lens=metadata.kv_lens,
            prompt_lens_cuda=metadata.prompt_lens_cuda,
            prompt_lens_cpu=metadata.prompt_lens_cpu,
            host_request_types=metadata.host_request_types,
        )
        metadata.host_total_kv_lens[1] = count * metadata.num_sparse_topk
        metadata.cu_q_seqlens = metadata.prepared_cu_q
        metadata.cu_kv_seqlens = metadata.prepared_cu_kv
        return metadata

    def get_query_tile_metadata(self, q: torch.Tensor, extra_width: int) -> CSA2TrtllmMetadata:
        """Share bounded compute staging across serialized layer calls."""
        if not hasattr(self, "_csa2_query_tiles"):
            self._csa2_query_tiles = {}
        key = (q.device, q.shape[1], q.shape[0], (extra_width + 127) // 128 * 128)
        with torch.cuda.device(q.device):
            metadata = self._csa2_query_tiles.get(key)
            if metadata is None:
                metadata = self.for_query_tile(q, extra_width)
                self._csa2_query_tiles[key] = metadata
            metadata.is_cuda_graph = torch.cuda.is_current_stream_capturing()
        return metadata

    def set_source_batch(self, seq_lengths: list[int], start_positions: list[int]) -> None:
        """Set encoder source rows independently of decoder query rows.

        This host-side input is consumed by the next prepare call. The caller
        must provision the manager for these source lengths before preparing.
        """
        if len(seq_lengths) != len(start_positions) or any(
            x < 0 for x in seq_lengths + start_positions
        ):
            raise ValueError("CSA2 source lengths and positions must be nonnegative and paired")
        self._csa2_source_batch = (list(seq_lengths), list(start_positions))

    def prepare(self) -> None:
        with torch.cuda.device(self.kv_lens_cuda.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare CSA2 request metadata before CUDA Graph capture")
        super().prepare()
        self.prepare_csa2()

    def _copy_csa2_tensor(self, key: str, value: torch.Tensor) -> torch.Tensor:
        # Preparation runs outside capture. Geometry has its own persistent
        # buffers, so graph replay only observes refreshed device contents.
        if not hasattr(self, "_csa2_buffers"):
            self._csa2_buffers = {}
        cache_key = (key, tuple(value.shape), value.dtype)
        result = self._csa2_buffers.get(cache_key)
        if result is None:
            result = torch.empty_like(value, device=self.kv_lens_cuda.device)
            self._csa2_buffers[cache_key] = result
        result.copy_(value)
        return result

    def prepare_csa2(self) -> None:
        """Resolve scheduler request IDs through the real manager's converters.

        Query and source metadata have independent packed lengths. All physical
        tables include the manager's sliding scratch mappings; no modulo ring
        addressing is synthesized here.
        """
        from .cache_manager import CSA2CacheManager, CSA2CacheRole
        from .compressor import CSA2CompressionBatch

        manager = self.kv_cache_manager
        if not isinstance(manager, CSA2CacheManager):
            raise TypeError("CSA2 runtime metadata requires CSA2CacheManager")
        if self.beam_width != 1 or self.is_spec_decoding_enabled:
            raise NotImplementedError(
                "CSA2 compressor state rewind for beams/speculation is not implemented"
            )
        request_ids = list(self.request_ids)
        lengths = self.seq_lens.tolist()
        starts = list(self.kv_cache_params.num_cached_tokens_per_seq)
        if len(request_ids) != len(lengths) or len(starts) != len(lengths):
            raise ValueError("CSA2 requires one query length and cached length per request")
        source_lengths, source_starts = getattr(self, "_csa2_source_batch", (lengths, starts))
        if hasattr(self, "_csa2_source_batch"):
            del self._csa2_source_batch
        if len(source_lengths) != len(lengths):
            raise ValueError("CSA2 source and query batches must name the same requests")
        ends = [s + n for s, n in zip(source_starts, source_lengths)]
        if any(e > manager.max_seq_len for e in ends):
            raise ValueError("CSA2 source rows exceed the manager context capacity")
        if any(s < 0 or n < 0 or s + n > manager.max_seq_len for s, n in zip(starts, lengths)):
            raise ValueError("CSA2 query positions exceed the manager context capacity")
        positions = [s + j for s, n in zip(starts, lengths) for j in range(n)]
        token_requests = [r for r, n in enumerate(lengths) for _ in range(n)]
        copy = self._copy_csa2_tensor
        self.csa2_positions = copy(
            "positions", torch.tensor(positions, dtype=torch.int32, device="cpu")
        )
        self.csa2_routing = CSA2Routing()
        self._csa2_batches = {}
        self._csa2_compression = {}
        self._csa2_compressed_positions = {}
        block = manager.tokens_per_block
        page_count = (manager.max_seq_len + block - 1) // block

        def table(layer: int, role: CSA2CacheRole) -> torch.Tensor:
            result = torch.full((len(request_ids), page_count), -1, dtype=torch.int32, device="cpu")
            for r, request in enumerate(request_ids):
                pages = manager.get_cache_indices(request, layer, role)
                # Converter expansion may round allocation to a larger base
                # page bucket. Only admitted logical positions are addressable.
                pages = pages[:page_count]
                result[r, : len(pages)] = torch.tensor(pages, dtype=torch.int32, device="cpu")
            return result

        def slot(pages: torch.Tensor, request: int, position: int, page_size: int) -> int:
            if position < 0 or position // page_size >= pages.shape[1]:
                return -1
            page = int(pages[request, position // page_size])
            return -1 if page < 0 else page * page_size + position % page_size

        owner_pages = {}
        owner_writes = {}
        capacity = sum(source_lengths)
        for owner in manager.layout.kv_source_layer_ids:
            ratio = manager.layout.compress_ratios[owner]
            pages = table(owner, CSA2CacheRole.GLOBAL)
            owner_pages[owner] = copy(f"global_pages/{owner}", pages)
            groups = [
                [g for g in range(s // ratio, e // ratio)] for s, e in zip(source_starts, ends)
            ]
            counts = [len(g) for g in groups]
            write_slots = [
                slot(pages, r, g, block // ratio) for r, gs in enumerate(groups) for g in gs
            ]
            if any(value < 0 for value in write_slots):
                raise ValueError("CSA2 source output has no allocated GLOBAL page")
            compressed_positions = [g * ratio for gs in groups for g in gs]
            padding = capacity - len(write_slots)
            owner_writes[owner] = copy(
                f"writes/{owner}",
                torch.tensor(write_slots + [-1] * padding, dtype=torch.int64, device="cpu"),
            )
            self._csa2_compressed_positions[owner] = copy(
                f"compressed_positions/{owner}",
                torch.tensor(compressed_positions + [0] * padding, dtype=torch.int32, device="cpu"),
            )
            if ratio == 2:
                kv_pages_host = table(owner, CSA2CacheRole.COMPRESSOR_KV)
                score_pages_host = table(owner, CSA2CacheRole.COMPRESSOR_SCORE)
                for r, (start, end) in enumerate(zip(source_starts, ends)):
                    if end == start:
                        continue
                    first, last = (start - start % ratio) // block, (end - 1) // block + 1
                    if torch.any(kv_pages_host[r, first:last] < 0) or torch.any(
                        score_pages_host[r, first:last] < 0
                    ):
                        raise ValueError(
                            "CSA2 source rows have no allocated compressor state pages"
                        )
                kv_pages = copy(f"kv_state_pages/{owner}", kv_pages_host)
                score_pages = copy(f"score_state_pages/{owner}", score_pages_host)
                cu_source = (
                    torch.tensor([0] + source_lengths, dtype=torch.int32, device="cpu")
                    .cumsum(0)
                    .int()
                )
                cu_output = (
                    torch.tensor([0] + counts, dtype=torch.int32, device="cpu").cumsum(0).int()
                )
                self._csa2_compression[owner] = CSA2CompressionBatch(
                    manager.get_buffers(owner, CSA2CacheRole.COMPRESSOR_KV),
                    manager.get_buffers(owner, CSA2CacheRole.COMPRESSOR_SCORE),
                    kv_pages,
                    score_pages,
                    copy(
                        f"source_ends/{owner}", torch.tensor(ends, dtype=torch.int32, device="cpu")
                    ),
                    copy(
                        f"source_starts/{owner}",
                        torch.tensor(source_starts, dtype=torch.int32, device="cpu"),
                    ),
                    copy(f"source_cu/{owner}", cu_source),
                    copy(f"compressed_cu/{owner}", cu_output),
                    capacity,
                    block,
                    max(1, max(source_lengths, default=0)),
                )
        requests_host = torch.tensor(token_requests, dtype=torch.int64, device="cpu")
        positions_host = torch.tensor(positions, dtype=torch.int64, device="cpu")
        requests_device = copy("token_requests", requests_host)
        window = manager.layout.window_size
        logical_swa = positions_host[:, None] - window + 1 + torch.arange(window, device="cpu")
        logical_pages = logical_swa.clamp_min(0) // block
        for layer_idx in range(len(manager.layout.compress_ratios)):
            layer = manager.layout.layer(layer_idx)
            pages = table(layer_idx, CSA2CacheRole.SWA)
            physical = pages[requests_host[:, None], logical_pages].long()
            reads = torch.where(
                (logical_swa >= 0) & (physical >= 0), physical * block + logical_swa % block, -1
            )
            writes = reads[:, -1]
            if torch.any(writes < 0):
                raise ValueError("CSA2 query output has no allocated SWA page")
            swa_reads = copy(f"swa_reads/{layer_idx}", reads)
            swa_writes = copy(f"swa_writes/{layer_idx}", writes)
            if layer.kv_source is None:
                global_pages = copy(
                    f"empty_global/{layer_idx}",
                    torch.empty((len(positions), 0), dtype=torch.int64, device="cpu"),
                )
                visible = torch.zeros(len(positions), dtype=torch.int64, device="cpu")
                main_writes = copy(
                    f"empty_writes/{layer_idx}", torch.empty(0, dtype=torch.int64, device="cpu")
                )
            else:
                ratio = layer.compress_ratio
                global_pages = CSA2GlobalPages(
                    owner_pages[layer.kv_source],
                    requests_device,
                    block // ratio,
                    manager.max_seq_len // ratio,
                )
                visible = (positions_host + 1) // ratio
                main_writes = owner_writes[layer.kv_source]
            self._csa2_batches[layer_idx] = CSA2Batch(
                swa_reads,
                swa_writes,
                global_pages,
                copy(f"visible/{layer_idx}", visible),
                main_writes,
            )

    def get_layer_batch(self, layer_idx: int):
        return self._csa2_batches[layer_idx]

    def get_compression_batch(self, owner: int):
        return self._csa2_compression.get(owner)

    def get_compressed_positions(self, owner: int) -> torch.Tensor:
        return self._csa2_compressed_positions[owner]


@dataclass
class CSA2Routing:
    """Per-forward logical indices; construct afresh for each packed batch.

    CUDA Graph capture records producers and consumers on the same stream.
    Each replay recomputes the captured tensors; Python dictionaries are only
    traversed during capture. Eager forwards must not recycle this object.
    """

    indices: dict[int, torch.Tensor] = field(default_factory=dict)
    candidates: dict[int, torch.Tensor] = field(default_factory=dict)
    _last_layer: int = -1

    def enter(self, layer: CSA2Layer) -> None:
        if layer.layer_idx <= self._last_layer:
            raise ValueError("CSA2 routing cannot be reused across forwards or reordered layers")
        self._last_layer = layer.layer_idx


@dataclass(frozen=True)
class CSA2GlobalPages:
    """Source pool page table without a tokens-by-context mapping allocation.

    page_table is [requests, logical_pages], request_ids is [query_tokens].
    Page IDs are relative to this owner's pool; -1 denotes an absent page.
    ``tokens_per_page`` counts global entries, after ratio-two compression.
    """

    page_table: torch.Tensor
    request_ids: torch.Tensor
    tokens_per_page: int
    max_positions: int

    def __post_init__(self) -> None:
        if self.page_table.ndim != 2 or self.request_ids.ndim != 1:
            raise ValueError("CSA2 page table/request IDs have invalid ranks")
        if self.tokens_per_page <= 0 or self.max_positions < 0:
            raise ValueError("CSA2 page capacity must be positive and context bound nonnegative")

    def resolve(self, start: int, end: int, logical: torch.Tensor | None = None) -> torch.Tensor:
        requests = self.request_ids[start:end].long()
        if logical is None:
            logical = torch.arange(self.max_positions, device=self.page_table.device)
            logical = logical.expand(requests.shape[0], -1)
        logical = logical.long()
        if self.page_table.shape[0] == 0 or self.page_table.shape[1] == 0:
            return torch.full_like(logical, -1)
        pages = logical.clamp_min(0) // self.tokens_per_page
        valid = (logical >= 0) & (logical < self.max_positions) & (pages < self.page_table.shape[1])
        valid &= ((requests >= 0) & (requests < self.page_table.shape[0]))[:, None]
        physical = self.page_table[
            requests.clamp(0, self.page_table.shape[0] - 1)[:, None],
            pages.clamp(max=self.page_table.shape[1] - 1),
        ]
        slots = physical.long() * self.tokens_per_page + logical % self.tokens_per_page
        return torch.where(valid & (physical >= 0), slots, -1)


@dataclass(frozen=True)
class CSA2Batch:
    """Device metadata for one packed layer execution.

    swa_indices: [tokens, window] pool-relative row indices, -1 for padding.
    swa_write_slots: [tokens], distinct valid pool rows (including dummy rows).
    global_slots: [tokens, max_global_positions], maps logical global positions
        to the owner's pool rows. Missing entries are -1.
    visible_lengths: [tokens], floor((absolute_position + 1) / ratio).
    main_write_slots: [new_latents], distinct valid rows for completed groups.

    A consumer resolves its logical routing against its own view of the source
    pool. Prefix reuse and request reordering therefore cannot reuse stale
    physical indices. Main/index pools use the same physical slot numbering.
    """

    swa_indices: torch.Tensor
    swa_write_slots: torch.Tensor
    global_slots: torch.Tensor | CSA2GlobalPages
    visible_lengths: torch.Tensor
    main_write_slots: torch.Tensor

    def global_slot_tile(
        self, start: int, end: int, logical: torch.Tensor | None = None
    ) -> torch.Tensor:
        if isinstance(self.global_slots, CSA2GlobalPages):
            return self.global_slots.resolve(start, end, logical)
        slots = self.global_slots[start:end]
        return slots if logical is None else _resolve_slots(slots, logical)


def _resolve_slots(global_slots: torch.Tensor, logical: torch.Tensor) -> torch.Tensor:
    if global_slots.shape[1] == 0:
        return torch.full_like(logical, -1, dtype=torch.int64)
    valid = (logical >= 0) & (logical < global_slots.shape[1])
    slots = global_slots.gather(1, logical.long().clamp(0, global_slots.shape[1] - 1))
    return torch.where(valid, slots, -1).long()
