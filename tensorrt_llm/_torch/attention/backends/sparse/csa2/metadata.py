# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 request metadata and bounded staging for TRTLLM sparse MLA."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams

from .params import CSA2BackendForwardArgs, CSA2Layer

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

    indexer_max_chunk_size: int = 8192
    indexer_q_split_threshold: int = 8192

    # Request metadata is held directly on this object. Main/index physical
    # tables and write slots are owner-keyed; SWA and visibility are layer-keyed.
    csa2_indices: dict[int, torch.Tensor]
    csa2_candidates: dict[int, torch.Tensor]
    _csa2_last_layer: int
    csa2_swa_indices: dict[int, torch.Tensor]
    csa2_swa_write_slots: dict[int, torch.Tensor]
    csa2_visible_lengths: dict[int, torch.Tensor]
    csa2_main_write_slots: dict[int, torch.Tensor]
    csa2_global_page_tables: dict[int, torch.Tensor]
    csa2_token_requests: torch.Tensor
    csa2_global_page_sizes: dict[int, int]
    csa2_global_max_positions: dict[int, int]
    csa2_kv_sources: dict[int, int | None]

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

    def stage_selected(self, inputs: CSA2BackendForwardArgs) -> None:
        """Refresh bounded BF16 pools and indices for one selected query tile.

        This prepares compute buffers only; the backend publishes native ABI
        arguments and performs attention after staging completes.
        """
        from .quantization import gather_rows

        count = self.num_tokens
        if inputs.swa_pool is None or inputs.swa_indices is None:
            raise ValueError("CSA2 requires selected SWA pool inputs")
        if inputs.swa_pool.device != self.swa_pool.device:
            raise ValueError("CSA2 Q, packed pools and metadata must be on the same CUDA device")
        if (
            inputs.swa_indices.ndim != 2
            or inputs.swa_indices.shape[0] != count
            or inputs.swa_indices.shape[1] > _SWA_TILE
        ):
            raise ValueError("CSA2 SWA indices must match the query count and window <=128")
        swa = gather_rows(inputs.swa_pool, inputs.swa_indices, _HEAD_DIM, "swa")
        swa_valid = inputs.swa_indices >= 0
        extra = extra_valid = None
        if inputs.topk_indices is not None:
            if inputs.main_pool is None or inputs.main_pool.device != self.swa_pool.device:
                raise ValueError(
                    "CSA2 selected main indices require a main pool on the query device"
                )
            if (
                inputs.topk_indices.ndim != 2
                or inputs.topk_indices.shape[0] != count
                or inputs.topk_indices.shape[1] > self.num_sparse_topk - _SWA_TILE
            ):
                raise ValueError("CSA2 selected main indices exceed metadata geometry")
            extra = gather_rows(inputs.main_pool, inputs.topk_indices, _HEAD_DIM, "main")
            extra_valid = inputs.topk_indices >= 0
        self.prepared_counter.zero_()
        if extra is not None:
            if extra_valid is None:
                raise ValueError("Extra KV rows require a validity mask")
            rows = torch.cat((swa, extra), dim=1)
            valid = torch.cat((swa_valid, extra_valid), dim=1)
        else:
            rows, valid = swa, swa_valid
        # TG uses a dense valid prefix, split at slot 128 between its pools.
        # Compact the selected union before staging: -1 slots inside the
        # supplied extent can contribute zero logits in BF16 generation.
        # Physical source ownership no longer matters after dequantization.
        width = rows.shape[1]
        positions = torch.arange(width, device=swa.device).expand(count, -1)
        order = torch.where(valid, positions, width).argsort(dim=1, stable=True)
        packed = rows.gather(1, order[..., None].expand(-1, -1, _HEAD_DIM))
        lengths = valid.sum(1, dtype=torch.int32)
        packed = torch.where((positions < lengths[:, None])[..., None], packed, 0)
        # Zero selected rows reduce to the sink's zero value. Give TG one
        # zero KV row so it always launches a defined (nonempty) reduction.
        lengths = lengths.clamp_min(1)
        self.prepared_lens[:count].copy_(lengths)
        self.swa_pool[:count].zero_()
        self.extra_pool[:count].zero_()
        swa_count = min(width, _SWA_TILE)
        self.swa_pool[:count, :swa_count].copy_(packed[:, :swa_count])
        if width > _SWA_TILE:
            self.extra_pool[:count, : width - _SWA_TILE].copy_(packed[:, _SWA_TILE:])
        indices = self.prepared_indices[:count]
        offsets = torch.arange(count, device=swa.device)[:, None]
        swa_positions = torch.arange(_SWA_TILE, device=swa.device)[None, :]
        indices[:, :_SWA_TILE].copy_(
            torch.where(swa_positions < lengths[:, None], offsets * _SWA_TILE + swa_positions, -1)
        )
        extra_capacity = self.num_sparse_topk - _SWA_TILE
        if extra_capacity:
            extra_positions = torch.arange(extra_capacity, device=swa.device)[None, :]
            indices[:, _SWA_TILE:].copy_(
                torch.where(
                    extra_positions + _SWA_TILE < lengths[:, None],
                    offsets * extra_capacity + extra_positions,
                    -1,
                )
            )

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
        self.csa2_request_start_positions = tuple(starts)
        self.csa2_request_lengths = tuple(lengths)
        self.csa2_num_context_requests = self.num_contexts
        packed_start = 0
        query_ranges = []
        for length in lengths:
            query_ranges.append((packed_start, packed_start + length))
            packed_start += length
        self.csa2_request_query_ranges = tuple(query_ranges)
        self.csa2_request_last_query_indices = self._copy_csa2_tensor(
            "request_last_queries",
            torch.tensor([end - 1 for _, end in query_ranges], dtype=torch.int64, device="cpu"),
        )
        positions = [s + j for s, n in zip(starts, lengths) for j in range(n)]
        token_requests = [r for r, n in enumerate(lengths) for _ in range(n)]
        copy = self._copy_csa2_tensor
        self.csa2_positions = copy(
            "positions", torch.tensor(positions, dtype=torch.int32, device="cpu")
        )
        self.reset_routing()
        self.csa2_swa_indices = {}
        self.csa2_swa_write_slots = {}
        self.csa2_visible_lengths = {}
        self.csa2_main_write_slots = {}
        self.csa2_global_page_tables = {}
        self.csa2_global_page_sizes = {}
        self.csa2_global_max_positions = {}
        self.csa2_kv_sources = {}
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

        capacity = sum(source_lengths)
        for owner in manager.layout.kv_source_layer_ids:
            ratio = manager.layout.compress_ratios[owner]
            pages = table(owner, CSA2CacheRole.GLOBAL)
            self.csa2_global_page_tables[owner] = copy(f"global_pages/{owner}", pages)
            self.csa2_global_page_sizes[owner] = block // ratio
            self.csa2_global_max_positions[owner] = manager.max_seq_len // ratio
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
            self.csa2_main_write_slots[owner] = copy(
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
        self.csa2_token_requests = copy("token_requests", requests_host)
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
            self.csa2_swa_indices[layer_idx] = copy(f"swa_reads/{layer_idx}", reads)
            self.csa2_swa_write_slots[layer_idx] = copy(f"swa_writes/{layer_idx}", writes)
            self.csa2_kv_sources[layer_idx] = layer.kv_source
            visible = (
                torch.zeros(len(positions), dtype=torch.int64, device="cpu")
                if layer.kv_source is None
                else (positions_host + 1) // layer.compress_ratio
            )
            self.csa2_visible_lengths[layer_idx] = copy(f"visible/{layer_idx}", visible)

    def reset_routing(self) -> None:
        """Begin one packed forward; graph replay recomputes captured producers."""
        self.csa2_indices = {}
        self.csa2_candidates = {}
        self._csa2_last_layer = -1

    def enter_layer(self, layer: CSA2Layer) -> None:
        if layer.layer_idx <= self._csa2_last_layer:
            raise ValueError("CSA2 routing cannot be reused across forwards or reordered layers")
        self._csa2_last_layer = layer.layer_idx

    def global_slot_tile(
        self, layer_idx: int, start: int, end: int, logical: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Resolve one query tile through its owner's current physical pages.

        Page tables stay request-by-page; no persistent token-by-context mapping
        is allocated. Invalid logical entries, request rows and absent pages
        resolve to -1. A SWA-only layer has no global entries.
        """
        requests = self.csa2_token_requests[start:end].long()
        owner = self.csa2_kv_sources[layer_idx]
        if owner is None:
            return torch.empty((requests.shape[0], 0), dtype=torch.int64, device=requests.device)
        table = self.csa2_global_page_tables[owner]
        page_size = self.csa2_global_page_sizes[owner]
        max_positions = self.csa2_global_max_positions[owner]
        if logical is None:
            logical = torch.arange(max_positions, device=table.device)
            logical = logical.expand(requests.shape[0], -1)
        logical = logical.long()
        if table.shape[0] == 0 or table.shape[1] == 0:
            return torch.full_like(logical, -1)
        pages = logical.clamp_min(0) // page_size
        valid = (logical >= 0) & (logical < max_positions) & (pages < table.shape[1])
        valid &= ((requests >= 0) & (requests < table.shape[0]))[:, None]
        physical = table[
            requests.clamp(0, table.shape[0] - 1)[:, None],
            pages.clamp(max=table.shape[1] - 1),
        ]
        slots = physical.long() * page_size + logical % page_size
        return torch.where(valid & (physical >= 0), slots, -1)

    def prepare_indexer(self, layer_idx: int) -> CSA2TrtllmMetadata | None:
        """Refresh native paged FP4 inputs for the real generation queries.

        Invoke after owner cache publication. Repacking and scheduler updates
        execute on every replay. Eager scratch covers currently visible pages;
        graph scratch reserves each admitted request's maximum context so its
        pointers remain stable as request lengths and page mappings change.
        Identical geometry shares staging across serialized owners and layers.
        """
        from tensorrt_llm.deep_gemm import get_paged_mqa_logits_metadata

        manager = self.kv_cache_manager
        if manager is None:
            raise ValueError("CSA2 paged indexer requires a cache manager")
        if torch.cuda.is_current_stream_capturing() and not self.is_cuda_graph:
            raise RuntimeError("Set is_cuda_graph before warming CSA2 paged indexer metadata")
        owner = self.csa2_kv_sources[layer_idx]
        if owner is None:
            raise ValueError("SWA-only layers have no indexer cache")
        context_requests = self.csa2_num_context_requests
        generation_ranges = self.csa2_request_query_ranges[context_requests:]
        if not generation_ranges:
            return None
        if any(start == end for start, end in generation_ranges):
            raise ValueError("CSA2 paged indexer requires nonempty generation requests")
        decode_start = generation_ranges[0][0]
        decode_end = generation_ranges[-1][1]
        count = decode_end - decode_start
        request_count = len(generation_ranges)
        ratio = manager.layout.compress_ratios[owner]
        if self.is_cuda_graph:
            max_positions = max(1, self.csa2_global_max_positions[owner])
        else:
            max_positions = max(
                1,
                max(
                    (start + length) // ratio
                    for start, length in zip(
                        self.csa2_request_start_positions[context_requests:],
                        self.csa2_request_lengths[context_requests:],
                    )
                ),
            )
        native_page_size = 64
        required_pages = (max_positions + native_page_size - 1) // native_page_size
        if self.is_cuda_graph:
            # Reserve the largest owner geometry during warmup so another
            # serialized owner cannot replace a buffer captured by this graph.
            required_pages = max(
                1,
                (max(self.csa2_global_max_positions.values()) + native_page_size - 1)
                // native_page_size,
            )
        else:
            # Grow geometrically, replacing the old eager arena rather than
            # retaining one allocation for every context length seen so far.
            required_pages = 1 << (required_pages - 1).bit_length()
        source_page_size = self.csa2_global_page_sizes[owner]
        if source_page_size % native_page_size:
            raise ValueError("CSA2 native index pages require source page sizes divisible by 64")
        table = self.csa2_global_page_tables[owner][context_requests:]
        device = table.device
        key = (device, request_count, count, manager.layout.index_topk, self.is_cuda_graph)
        if not hasattr(self, "_csa2_indexer_workspaces"):
            self._csa2_indexer_workspaces = {}
        if not self.is_cuda_graph:
            # Runtime metadata survives many batch geometries. Retain just
            # the latest eager arena per device; active descriptors still own
            # their tensors, and serialized stream work preserves their use.
            # Graph clones may share this dictionary, so never evict graph keys.
            for previous_key in tuple(self._csa2_indexer_workspaces):
                if previous_key[0] == device and not previous_key[-1] and previous_key != key:
                    del self._csa2_indexer_workspaces[previous_key]
        scratch = self._csa2_indexer_workspaces.get(key)
        if scratch is None or scratch["page_capacity"] < required_pages:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up CSA2 paged indexer metadata before graph capture")
            page_capacity = required_pages
            pages = 1 + request_count * page_capacity
            scratch = {
                "page_capacity": page_capacity,
                "cache": torch.empty(
                    (pages, native_page_size, 1, 68), dtype=torch.uint8, device=device
                ),
                "row_starts": torch.empty(pages, dtype=torch.int64, device=device),
                "block_table": torch.empty(
                    (count, page_capacity), dtype=torch.int32, device=device
                ),
                "context_lengths": torch.empty((count, 1), dtype=torch.int32, device=device),
                "logical_positions": torch.empty(
                    (count, page_capacity * native_page_size), dtype=torch.int32, device=device
                ),
                "visible_lengths": torch.empty(count, dtype=torch.int32, device=device),
                "radix_indices": torch.empty(
                    (count, 10, manager.layout.index_topk), dtype=torch.int32, device=device
                ),
                "radix_logits": torch.empty(
                    (count, 10, manager.layout.index_topk), dtype=torch.float32, device=device
                ),
            }
            self._csa2_indexer_workspaces[key] = scratch
        page_capacity = scratch["page_capacity"]
        logical_page_starts = torch.arange(page_capacity, device=device) * native_page_size
        source_columns = logical_page_starts // source_page_size
        if not hasattr(self, "csa2_request_last_query_indices"):
            # Synthetic/prepared callers may supply the host ranges directly.
            # Normal prepare updates this persistent device buffer each step.
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prepare CSA2 request boundaries before graph capture")
            self.csa2_request_last_query_indices = torch.tensor(
                [end - 1 for _, end in self.csa2_request_query_ranges],
                dtype=torch.int64,
                device=device,
            )
        last_queries = self.csa2_request_last_query_indices[context_requests:]
        request_visible = self.csa2_visible_lengths[layer_idx][last_queries]
        if table.shape[1] == 0:
            source_pages = torch.full(
                (request_count, page_capacity), -1, dtype=torch.int64, device=device
            )
        else:
            source_pages = table[:, source_columns.clamp(max=table.shape[1] - 1)].long()
        valid_pages = (source_pages >= 0) & (source_columns[None, :] < table.shape[1])
        valid_pages &= logical_page_starts[None, :] < request_visible[:, None]
        physical_starts = source_pages * source_page_size + logical_page_starts % source_page_size
        scratch["row_starts"][0].fill_(-1)
        scratch["row_starts"][1:].copy_(torch.where(valid_pages, physical_starts, -1).flatten())
        manager.gather_indexer_pages(owner, scratch["row_starts"], scratch["cache"])
        # Missing native pages read the reserved zero page, then are removed
        # from logical output mapping so their zero logits cannot win Top-K.
        page_ids = torch.arange(
            1, 1 + request_count * page_capacity, dtype=torch.int32, device=device
        ).view(request_count, page_capacity)
        page_ids = torch.where(valid_pages, page_ids, 0)
        requests = self.csa2_token_requests[decode_start:decode_end] - context_requests
        valid_requests = (requests >= 0) & (requests < request_count)
        requests = requests.clamp(0, request_count - 1).long()
        scratch["block_table"].copy_(torch.where(valid_requests[:, None], page_ids[requests], 0))
        visible = self.csa2_visible_lengths[layer_idx][decode_start:decode_end].int()
        scratch["visible_lengths"].copy_(torch.where(valid_requests, visible, 0))
        scratch["context_lengths"].copy_(scratch["visible_lengths"].clamp_min(1)[:, None])
        logical = torch.arange(max_positions, dtype=torch.int32, device=device)
        valid = valid_pages[requests[:, None], (logical // native_page_size).long()]
        valid &= valid_requests[:, None] & (logical[None, :] < visible[:, None])
        logical_positions = scratch["logical_positions"][:, :max_positions]
        logical_positions.copy_(torch.where(valid, logical[None, :], -1))
        schedule = get_paged_mqa_logits_metadata(
            scratch["context_lengths"],
            64,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        if "schedule" not in scratch:
            scratch["schedule"] = torch.empty_like(schedule)
        scratch["schedule"].copy_(schedule)
        self.csa2_indexer_k_cache = scratch["cache"]
        self.csa2_indexer_block_table = scratch["block_table"]
        self.csa2_indexer_context_lengths = scratch["context_lengths"]
        self.csa2_indexer_scheduler_metadata = scratch["schedule"]
        self.csa2_indexer_max_seq_len = max_positions
        self.csa2_indexer_radix_aux_indices = scratch["radix_indices"]
        self.csa2_indexer_radix_aux_logits = scratch["radix_logits"]
        self.csa2_indexer_logical_positions = logical_positions
        self.csa2_indexer_visible_lengths = scratch["visible_lengths"]
        return self

    def get_compression_batch(self, owner: int):
        return self._csa2_compression.get(owner)

    def get_compressed_positions(self, owner: int) -> torch.Tensor:
        return self._csa2_compressed_positions[owner]
