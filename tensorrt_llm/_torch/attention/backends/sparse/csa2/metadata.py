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
    Eager context retains real query groups with virtual staged KV lengths.
    Generation and captured context use independent one-query generation rows.
    Source causality, request isolation and windows are encoded in selections;
    these bounded compute pools are separate from persistent manager caches.
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

    def get_query_tile_metadata(
        self, q: torch.Tensor, extra_width: int, query_start: int | None = None
    ) -> CSA2TrtllmMetadata:
        """Share bounded staging, retaining real request groups for context."""
        context_lengths = []
        if query_start is not None and query_start < self.num_ctx_tokens:
            end = query_start + q.shape[0]
            if end > self.num_ctx_tokens:
                raise ValueError("CSA2 query tiles must not cross the context/generation boundary")
            for begin, stop in self.csa2_request_query_ranges:
                lo, hi = max(begin, query_start), min(stop, end)
                if lo < hi:
                    context_lengths.append(hi - lo)
            if sum(context_lengths) != q.shape[0]:
                raise ValueError("CSA2 context tile must be covered by its packed request ranges")
        # Native context grouping currently has host-bound launch metadata.
        # Captured contexts retain the existing independent-query generation
        # path; choose it during graph warmup too, so its workspace is ready.
        with torch.cuda.device(q.device):
            capturing = torch.cuda.is_current_stream_capturing()
        if context_lengths and (self.is_cuda_graph or capturing):
            context_lengths = []
        if not hasattr(self, "_csa2_query_tiles"):
            self._csa2_query_tiles = {}
        key = (
            q.device,
            q.shape[1],
            q.shape[0],
            (extra_width + 127) // 128 * 128,
            bool(context_lengths),
        )
        with torch.cuda.device(q.device):
            capturing = torch.cuda.is_current_stream_capturing()
            metadata = self._csa2_query_tiles.get(key)
            if metadata is None:
                metadata = self.for_query_tile(q, extra_width)
                self._csa2_query_tiles[key] = metadata
            metadata.is_cuda_graph = capturing
            if capturing:
                self._csa2_replay_capture_signature = getattr(self, "csa2_replay_signature", None)
            if context_lengths:
                metadata._bind_context_tile(context_lengths)
        return metadata

    def _bind_context_tile(self, query_lengths: list[int]) -> None:
        # Physical source causality is already encoded in selected rows. The
        # context kernel still applies a causal upper bound in staged-column
        # coordinates: give even the first query the full sparse capacity,
        # otherwise short source prefixes would clip valid compressed keys.
        kv_lengths = [self.num_sparse_topk + length - 1 for length in query_lengths]
        requests = len(query_lengths)
        query = torch.tensor(query_lengths, dtype=torch.int32, device="cpu")
        kv = torch.tensor(kv_lengths, dtype=torch.int32, device="cpu")
        self.query_lens_host[:requests].copy_(query)
        self.query_lens_device[:requests].copy_(query)
        self._seq_lens = self.query_lens_host[:requests]
        self._seq_lens_cuda = self.query_lens_device[:requests]
        self._num_contexts = requests
        self._num_ctx_tokens = self._num_tokens = sum(query_lengths)
        self._num_generations = 0
        self.kv_lens[:requests].copy_(kv)
        self.kv_lens_cuda[:requests].copy_(kv)
        # THOP interprets context_lengths as Q lengths, independently of
        # past/current KV lengths. Keep them consistent with unfolded cuQ.
        self.prompt_lens_cpu[:requests].copy_(query)
        self.prompt_lens_cuda[:requests].copy_(query)
        self.host_request_types[:requests].zero_()
        self.host_total_kv_lens.zero_()
        self.host_total_kv_lens[0] = sum(kv_lengths)
        self.max_seq_len = max(self.num_sparse_topk, max(kv_lengths))
        self.prepared_cu_q[: requests + 1].copy_(
            torch.cat((query.new_zeros(1), query.cumsum(0).int()))
        )
        self.prepared_cu_kv[: requests + 1].copy_(torch.cat((kv.new_zeros(1), kv.cumsum(0).int())))
        self.cu_q_seqlens = self.prepared_cu_q[: requests + 1]
        self.cu_kv_seqlens = self.prepared_cu_kv[: requests + 1]
        self._bind_runtime_views(
            kv_lens_cuda=self.kv_lens_cuda[:requests],
            kv_lens=self.kv_lens[:requests],
            prompt_lens_cuda=self.prompt_lens_cuda[:requests],
            prompt_lens_cpu=self.prompt_lens_cpu[:requests],
            host_request_types=self.host_request_types[:requests],
        )

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
        swa_valid = (inputs.swa_indices >= 0) & (inputs.swa_indices < inputs.swa_pool.shape[0])
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
            extra_valid = (inputs.topk_indices >= 0) & (
                inputs.topk_indices < inputs.main_pool.shape[0]
            )
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

    def set_swa_bounded_replay(
        self, cached_prefix_lengths: list[int], *, decoder: bool = False
    ) -> None:
        """Prepare approximate reconstruction after an authoritative GLOBAL hit.

        The caller must allocate the manager's reported replay intervals and
        supply per-layer inputs for those absolute query positions. This does
        not perform prefix matching or change native V2 persistence policy.
        Encoder replay may include a new suffix; decoder replay assumes all
        prompt GLOBAL entries are ready and only reconstructs the last window.
        """
        if any(length < 0 for length in cached_prefix_lengths):
            raise ValueError("CSA2 cached GLOBAL prefix lengths must be nonnegative")
        if hasattr(self, "_csa2_source_batch"):
            raise ValueError(
                "Bounded replay source selection cannot be combined with set_source_batch"
            )
        self._csa2_pending_replay = (tuple(cached_prefix_lengths), decoder)

    def select_global_source(
        self,
        owner: int,
        hidden_states: torch.Tensor,
        global_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select a declared source base; never infer it from tensor lengths."""
        mode = getattr(self, "csa2_replay_mode", None)
        if mode is None:
            return hidden_states if global_hidden_states is None else global_hidden_states
        if global_hidden_states is not None:
            raise ValueError(
                "CSA2 bounded replay uses query inputs or already prepared GLOBAL cache, not an external source batch"
            )
        indices = self.csa2_global_source_indices[owner]
        if indices.numel() == 0:
            return hidden_states[:0]
        if hidden_states.shape[0] != self.csa2_positions.numel():
            raise ValueError("CSA2 replay source selectors address the full packed query input")
        return hidden_states.index_select(0, indices)

    def _swa_replay_geometry(self, prefixes, decoder, starts, lengths) -> tuple:
        owner_shapes = []
        for owner in self.kv_cache_manager.layout.kv_source_layer_ids:
            ratio = self.kv_cache_manager.layout.compress_ratios[owner]
            source_lengths = tuple(
                length
                if prefix is None
                else (
                    0 if decoder else max(0, start + length - max(start, prefix - prefix % ratio))
                )
                for prefix, start, length in zip(prefixes, starts, lengths)
            )
            owner_shapes.append((owner, source_lengths, sum(source_lengths)))
        return ("decoder" if decoder else "encoder", tuple(lengths), tuple(owner_shapes))

    def _configure_automatic_replay(self) -> None:
        manager = self.kv_cache_manager
        if manager is None or self.request_ids is None:
            return
        plans = [manager.automatic_replay_plan(request_id) for request_id in self.request_ids]
        if any(plan is not None for plan in plans):
            if self.is_cuda_graph:
                raise ValueError("Automatic CSA2 cache reconstruction requires eager execution")
            prefixes = tuple(None if plan is None else plan["prefix"] for plan in plans)
            pending = getattr(self, "_csa2_pending_replay", None)
            if pending is not None and pending != (prefixes, False):
                raise ValueError(
                    "CSA2 manual replay setup disagrees with the native GLOBAL prefix hit"
                )
            self._csa2_pending_replay = (prefixes, False)
            self._csa2_pending_automatic_replay = True
        else:
            self._csa2_pending_automatic_replay = False

    def prepare(self) -> None:
        self._configure_automatic_replay()
        if self.is_cuda_graph and hasattr(self, "_csa2_replay_capture_signature"):
            pending = getattr(self, "_csa2_pending_replay", None)
            signature = (
                None
                if pending is None
                else self._swa_replay_geometry(
                    pending[0],
                    pending[1],
                    self.kv_cache_params.num_cached_tokens_per_seq,
                    self.seq_lens.tolist(),
                )
            )
            if signature != self._csa2_replay_capture_signature:
                raise ValueError(
                    "CSA2 replay mode/source geometry changed; use fresh graph metadata and recapture"
                )
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
        cache_key = (key, tuple(value.shape), value.dtype, self.is_cuda_graph)
        if not self.is_cuda_graph:
            for previous in tuple(self._csa2_buffers):
                if previous[0] == key and not previous[-1] and previous != cache_key:
                    del self._csa2_buffers[previous]
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
        if self.beam_width != 1 or self.is_spec_dec_dynamic_tree:
            raise NotImplementedError(
                "CSA2 supports contiguous-prefix verification, not beams or dynamic trees"
            )
        request_ids = list(self.request_ids)
        lengths = self.seq_lens.tolist()
        manager.validate_verification(lengths, self.num_contexts, self.is_spec_decoding_enabled)
        starts = list(self.kv_cache_params.num_cached_tokens_per_seq)
        if len(request_ids) != len(lengths) or len(starts) != len(lengths):
            raise ValueError("CSA2 requires one query length and cached length per request")
        self._configure_automatic_replay()
        automatic_replay = getattr(self, "_csa2_pending_automatic_replay", False)
        self.csa2_automatic_replay = automatic_replay
        replay = getattr(self, "_csa2_pending_replay", None)
        if hasattr(self, "_csa2_pending_replay"):
            del self._csa2_pending_replay
        self.csa2_replay_mode = None
        self.csa2_replay_cached_lengths = ()
        replay_starts = [0] * len(lengths)
        if replay is not None:
            prefixes, decoder = replay
            if len(prefixes) != len(lengths) or (
                not automatic_replay and self.num_contexts != len(lengths)
            ):
                raise ValueError(
                    "CSA2 bounded replay requires one prefix length per context request"
                )
            if hasattr(self, "_csa2_source_batch"):
                raise ValueError("CSA2 replay cannot use an unrelated external source batch")
            replay_starts = [
                0 if prefix is None else max(0, prefix - manager.layout.window_size)
                for prefix in prefixes
            ]
            for row, (prefix, start, length) in enumerate(zip(prefixes, starts, lengths)):
                if prefix is None:
                    continue
                if row >= self.num_contexts:
                    raise ValueError("CSA2 cannot decode before native reconstruction completes")
                if start < replay_starts[row] or (
                    not automatic_replay and start != replay_starts[row]
                ):
                    raise ValueError(
                        "CSA2 replay queries must start at the reconstruction boundary"
                    )
                if not automatic_replay and (
                    start + length < prefix or (decoder and start + length != prefix)
                ):
                    raise ValueError("CSA2 replay query interval must cover its cached prefix tail")
            self.csa2_replay_mode = "decoder" if decoder else "encoder"
            self.csa2_replay_cached_lengths = prefixes
        self.csa2_replay_signature = (
            None
            if replay is None
            else self._swa_replay_geometry(
                replay[0],
                replay[1],
                starts,
                lengths,
            )
        )
        for row, request_id in enumerate(request_ids):
            replaying = replay is not None and self.csa2_replay_cached_lengths[row] is not None
            manager.validate_ready_query(request_id, starts[row], replaying)
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
        self.csa2_replay_start_positions = copy(
            "replay_starts", torch.tensor(replay_starts, dtype=torch.int64, device="cpu")
        )
        self.csa2_global_source_indices = {}
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

        for owner in manager.layout.kv_source_layer_ids:
            ratio = manager.layout.compress_ratios[owner]
            owner_starts, owner_lengths, owner_ends = source_starts, source_lengths, ends
            if self.csa2_replay_mode is not None:
                if self.csa2_replay_mode == "decoder":
                    owner_starts = list(self.csa2_replay_cached_lengths)
                    owner_lengths = [0] * len(lengths)
                    owner_ends = owner_starts
                    source_indices = []
                else:
                    owner_ends = [start + length for start, length in zip(starts, lengths)]
                    owner_starts = [
                        start if prefix is None else min(end, max(start, prefix - prefix % ratio))
                        for prefix, start, end in zip(
                            self.csa2_replay_cached_lengths, starts, owner_ends
                        )
                    ]
                    owner_lengths = [end - start for start, end in zip(owner_starts, owner_ends)]
                    source_indices = [
                        query_ranges[r][0] + position - starts[r]
                        for r, (start, end) in enumerate(zip(owner_starts, owner_ends))
                        for position in range(start, end)
                    ]
                self.csa2_global_source_indices[owner] = copy(
                    f"replay_source/{owner}",
                    torch.tensor(source_indices, dtype=torch.int64, device="cpu"),
                )
            capacity = sum(owner_lengths)
            pages = table(owner, CSA2CacheRole.GLOBAL)
            if self.csa2_replay_mode is not None:
                for request_row, prefix in enumerate(self.csa2_replay_cached_lengths):
                    if prefix is None:
                        continue
                    prefix_pages = (prefix // ratio + block // ratio - 1) // (block // ratio)
                    if torch.any(pages[request_row, :prefix_pages] < 0):
                        raise ValueError(
                            "CSA2 bounded replay requires the cached GLOBAL prefix pages to be ready"
                        )
            self.csa2_global_page_tables[owner] = copy(f"global_pages/{owner}", pages)
            self.csa2_global_page_sizes[owner] = block // ratio
            self.csa2_global_max_positions[owner] = manager.max_seq_len // ratio
            groups = [
                [g for g in range(s // ratio, e // ratio)] for s, e in zip(owner_starts, owner_ends)
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
                for r, (start, end) in enumerate(zip(owner_starts, owner_ends)):
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
                    torch.tensor([0] + owner_lengths, dtype=torch.int32, device="cpu")
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
                        f"source_ends/{owner}",
                        torch.tensor(owner_ends, dtype=torch.int32, device="cpu"),
                    ),
                    copy(
                        f"source_starts/{owner}",
                        torch.tensor(owner_starts, dtype=torch.int32, device="cpu"),
                    ),
                    copy(f"source_cu/{owner}", cu_source),
                    copy(f"compressed_cu/{owner}", cu_output),
                    capacity,
                    block,
                    max(1, max(owner_lengths, default=0)),
                )
        requests_host = torch.tensor(token_requests, dtype=torch.int64, device="cpu")
        positions_host = torch.tensor(positions, dtype=torch.int64, device="cpu")
        self.csa2_token_requests = copy("token_requests", requests_host)
        window = manager.layout.window_size
        logical_swa = positions_host[:, None] - window + 1 + torch.arange(window, device="cpu")
        logical_pages = logical_swa.clamp_min(0) // block
        replay_floor = torch.tensor(replay_starts, dtype=torch.int64, device="cpu")[
            requests_host, None
        ]
        for layer_idx in range(len(manager.layout.compress_ratios)):
            layer = manager.layout.layer(layer_idx)
            pages = table(layer_idx, CSA2CacheRole.SWA)
            physical = pages[requests_host[:, None], logical_pages].long()
            reads = torch.where(
                (logical_swa >= replay_floor) & (physical >= 0),
                physical * block + logical_swa % block,
                -1,
            )
            writes = reads[:, -1]
            if torch.any(writes < 0):
                raise ValueError(
                    "CSA2 query/replay interval requires writable SWA pages; "
                    "reserve the complete replay range before prepare"
                )
            self.csa2_swa_indices[layer_idx] = copy(f"swa_reads/{layer_idx}", reads)
            self.csa2_swa_write_slots[layer_idx] = copy(f"swa_writes/{layer_idx}", writes)
            self.csa2_kv_sources[layer_idx] = layer.kv_source
            visible = (
                torch.zeros(len(positions), dtype=torch.int64, device="cpu")
                if layer.kv_source is None
                else (positions_host + 1) // layer.compress_ratio
            )
            self.csa2_visible_lengths[layer_idx] = copy(f"visible/{layer_idx}", visible)
        self._csa2_prepared_manager = manager
        self._csa2_forward_serial = getattr(self, "_csa2_forward_serial", 0) + 1
        self._csa2_completed_layers = set()
        self._csa2_completion_event = None
        manager.register_replay_step(self)
        for layer_idx in getattr(self, "_csa2_priors", {}):
            self._refresh_indexer_prior(layer_idx)

    def _record_completion_event(self) -> None:
        if self._csa2_completion_event is None:
            self._csa2_completion_event = torch.cuda.Event()
        self._csa2_completion_event.record(torch.cuda.current_stream())

    def record_layer_completion(self, layer_idx: int) -> None:
        if torch.cuda.is_current_stream_capturing():
            return
        if getattr(self, "csa2_automatic_replay", False):
            self._csa2_completed_layers.add(layer_idx)
            self._record_completion_event()

    def reconstruction_step_receipt(self, manager):
        fields = (
            vars(self)
            if getattr(self, "_csa2_prepared_manager", None) is manager
            else getattr(self, "_csa2_manager_states", {}).get(id(manager))
        )
        if fields is None:
            return None
        return (
            fields.get("_csa2_forward_serial"),
            frozenset(fields.get("_csa2_completed_layers", ())),
            fields.get("_csa2_completion_event"),
        )

    def reconstruction_completed_layers(self) -> frozenset[int]:
        if self._csa2_completion_event is None:
            return frozenset()
        return frozenset(self._csa2_completed_layers)

    def _cache_dependent_fields(self) -> dict:
        shared = {"_csa2_query_tiles", "_csa2_indexer_workspaces", "_csa2_manager_states"}
        return {
            name: value
            for name, value in vars(self).items()
            if (name.startswith("csa2_") or name.startswith("_csa2_")) and name not in shared
        }

    def _restore_cache_fields(self, fields: dict) -> None:
        for name in self._cache_dependent_fields():
            delattr(self, name)
        for name, value in fields.items():
            setattr(self, name, value)

    def prepare_for_draft_forward(self) -> dict | None:
        """Rebuild draft-owned fields after the existing interface swaps managers.

        Only identical explicit layer layouts are representable here. Virtual
        model-layer mappings and CED draft scheduling belong to model integration.
        This hook runs before capture/replay, never inside a captured forward.
        """
        target = getattr(self, "_csa2_prepared_manager", None)
        draft = self.kv_cache_manager
        if target is None or target is draft:
            return None
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Prepare CSA2 draft cache metadata before graph capture")
        if getattr(draft, "layout", None) != target.layout:
            raise NotImplementedError(
                "CSA2 draft cache requires an identical explicit layer mapping"
            )
        saved = self._cache_dependent_fields()
        if not hasattr(self, "_csa2_manager_states"):
            self._csa2_manager_states = {}
        self._restore_cache_fields(self._csa2_manager_states.get(id(draft), {}))
        try:
            self.prepare_csa2()
            # Even contiguous draft requests may follow target execution in
            # the same TopK module. Manager transitions always invalidate its
            # emission state, independently of the restored request identities.
            callbacks = list(saved.get("_csa2_indexer_resets", {}).values())
            callbacks += list(getattr(self, "_csa2_indexer_resets", {}).values())
            for callback in callbacks:
                callback()
        except (ValueError, TypeError, KeyError, NotImplementedError):
            self._restore_cache_fields(saved)
            raise
        return saved

    def restore_after_draft_forward(self, saved_state: dict | None) -> None:
        if saved_state is None:
            return
        draft = self._csa2_prepared_manager
        self._csa2_manager_states[id(draft)] = self._cache_dependent_fields()
        self._restore_cache_fields(saved_state)
        # Target and draft may use the same module's emission buffers, while
        # their request histories remain independent. Reset before target reuse.
        for callback in getattr(self, "_csa2_indexer_resets", {}).values():
            callback()

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
        request_capacity = self.max_num_requests if self.is_cuda_graph else request_count
        draft_width = 1 + getattr(manager, "max_total_draft_tokens", 0)
        query_capacity = self.max_num_requests * draft_width if self.is_cuda_graph else count
        if self.is_cuda_graph and count > query_capacity:
            raise ValueError("CSA2 graph query count exceeds configured batch and draft width")
        key = (
            (device, manager.layout.index_topk, True)
            if self.is_cuda_graph
            else (device, request_count, count, manager.layout.index_topk, False)
        )
        if self.is_cuda_graph:
            resolved_cap = getattr(manager, "fp8_ctx_mla_kv_len_cap", None)
            if (
                resolved_cap is not None
                and resolved_cap < self.max_num_requests * manager.max_seq_len
            ):
                raise ValueError("CSA2 graph workspace requires the full admitted request KV bound")
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
            pages = 1 + request_capacity * page_capacity
            scratch = {
                "page_capacity": page_capacity,
                "cache": torch.empty(
                    (pages, native_page_size, 1, 68), dtype=torch.uint8, device=device
                ),
                "row_starts": torch.empty(pages, dtype=torch.int64, device=device),
                "block_table": torch.empty(
                    (query_capacity, page_capacity), dtype=torch.int32, device=device
                ),
                "context_lengths": torch.empty(
                    (query_capacity, 1), dtype=torch.int32, device=device
                ),
                "logical_positions": torch.empty(
                    (query_capacity, page_capacity * native_page_size),
                    dtype=torch.int32,
                    device=device,
                ),
                "visible_lengths": torch.empty(query_capacity, dtype=torch.int32, device=device),
                "radix_indices": torch.empty(
                    (query_capacity, 10, manager.layout.index_topk),
                    dtype=torch.int32,
                    device=device,
                ),
                "radix_logits": torch.empty(
                    (query_capacity, 10, manager.layout.index_topk),
                    dtype=torch.float32,
                    device=device,
                ),
            }
            self._csa2_indexer_workspaces[key] = scratch
        page_capacity = scratch["page_capacity"]
        active_pages = 1 + request_count * page_capacity
        active_cache = scratch["cache"][:active_pages]
        active_starts = scratch["row_starts"][:active_pages]
        active_blocks = scratch["block_table"][:count]
        active_context = scratch["context_lengths"][:count]
        active_visible = scratch["visible_lengths"][:count]
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
        active_starts[0].fill_(-1)
        active_starts[1:].copy_(torch.where(valid_pages, physical_starts, -1).flatten())
        manager.gather_indexer_pages(owner, active_starts, active_cache)
        # Missing native pages read the reserved zero page, then are removed
        # from logical output mapping so their zero logits cannot win Top-K.
        page_ids = torch.arange(
            1, 1 + request_count * page_capacity, dtype=torch.int32, device=device
        ).view(request_count, page_capacity)
        page_ids = torch.where(valid_pages, page_ids, 0)
        requests = self.csa2_token_requests[decode_start:decode_end] - context_requests
        valid_requests = (requests >= 0) & (requests < request_count)
        requests = requests.clamp(0, request_count - 1).long()
        active_blocks.copy_(torch.where(valid_requests[:, None], page_ids[requests], 0))
        visible = self.csa2_visible_lengths[layer_idx][decode_start:decode_end].int()
        active_visible.copy_(torch.where(valid_requests, visible, 0))
        active_context.copy_(active_visible.clamp_min(1)[:, None])
        logical = torch.arange(max_positions, dtype=torch.int32, device=device)
        valid = valid_pages[requests[:, None], (logical // native_page_size).long()]
        valid &= valid_requests[:, None] & (logical[None, :] < visible[:, None])
        logical_positions = scratch["logical_positions"][:count, :max_positions]
        logical_positions.copy_(torch.where(valid, logical[None, :], -1))
        schedule = get_paged_mqa_logits_metadata(
            active_context,
            64,
            torch.cuda.get_device_properties(device).multi_processor_count,
        )
        if "schedule" not in scratch:
            scratch["schedule"] = torch.empty_like(schedule)
        scratch["schedule"].copy_(schedule)
        self.csa2_indexer_k_cache = active_cache
        self.csa2_indexer_block_table = active_blocks
        self.csa2_indexer_context_lengths = active_context
        self.csa2_indexer_scheduler_metadata = scratch["schedule"]
        self.csa2_indexer_max_seq_len = max_positions
        self.csa2_indexer_radix_aux_indices = scratch["radix_indices"][:count]
        self.csa2_indexer_radix_aux_logits = scratch["radix_logits"][:count]
        self.csa2_indexer_logical_positions = logical_positions
        self.csa2_indexer_visible_lengths = active_visible
        return self

    @staticmethod
    def cache_gather_bytes_per_token(model_config) -> int:
        """Packed cache gather/repack component only, not an attention bound.

        Excludes query-dependent logits, BF16 fallback head intermediates,
        TopK/quantization temporaries, priors and FMHA/native workspaces. Those
        require actual serving query/head geometry and warmed peak profiling.
        This component rate must not be used as the generic backend workspace
        declaration or substituted for the fixed arena reservation below.
        """
        from .params import CSA2Layout

        layout = CSA2Layout.from_hf_config(model_config.pretrained_config)
        if not layout.kv_source_layer_ids:
            return 0
        ratio = min(layout.compress_ratios[owner] for owner in layout.kv_source_layer_ids)
        # Graph plus geometric eager packed staging, native gather/masked
        # outputs and row-index intermediates, per logical raw source token.
        return (512 + ratio - 1) // ratio

    @staticmethod
    def workspace_reservation_bytes(
        request_capacity: int, query_capacity: int, max_positions: int, topk: int, num_sms: int
    ) -> int:
        """Exact retained bytes for one native index arena, including schedule.

        Call with admitted generation capacities, not current token lengths.
        This intentionally excludes model projections, temporal priors, generic
        metadata, FMHA native/provider workspaces, and transient gather/logits;
        report those separately with get_workspace_bytes and warmed peak stats.
        """
        if min(request_capacity, query_capacity, max_positions, topk, num_sms) < 0:
            raise ValueError("CSA2 workspace capacities must be nonnegative")
        pages_per_request = max(1, (max_positions + 63) // 64)
        pages = 1 + request_capacity * pages_per_request
        return (
            pages * (64 * 68 + 8)
            + query_capacity * (pages_per_request * (4 + 64 * 4) + 8 + 80 * topk)
            + (num_sms + 1) * 2 * 4
        )

    def get_workspace_bytes(self) -> int:
        """Report retained GPU workspace once per storage, excluding KV pools."""
        seen = set()
        total = 0

        def visit(value):
            nonlocal total
            if isinstance(value, torch.Tensor):
                if value.device.type != "cuda":
                    return
                storage = value.untyped_storage()
                key = (value.device, storage.data_ptr())
                if key not in seen:
                    seen.add(key)
                    total += storage.nbytes()
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor) and name != "position_ids":
                visit(value)
        for name in (
            "workspace",
            "cuda_graph_workspace",
            "_csa2_buffers",
            "_csa2_indexer_workspaces",
            "_csa2_priors",
            "_csa2_manager_states",
        ):
            visit(getattr(self, name, None))
        for metadata in getattr(self, "_csa2_query_tiles", {}).values():
            for name, value in vars(metadata).items():
                if name != "kv_cache_manager":
                    visit(value)
        return total

    def register_indexer_reset(self, layer_idx: int, callback) -> None:
        """Register a host-prepare emission reset, before a captured replay."""
        if not hasattr(self, "_csa2_indexer_resets"):
            self._csa2_indexer_resets = {}
        self._csa2_indexer_resets[layer_idx] = callback

    def prepare_indexer_prior(self, layer_idx: int, topk: int) -> torch.Tensor:
        if not hasattr(self, "_csa2_priors"):
            self._csa2_priors = {}
            self.csa2_indexer_prior_capacity = {}
        record = self._csa2_priors.get(layer_idx)
        if record is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up CSA2 temporal priors before graph capture")
            capacity = self.max_num_tokens
            device = self.csa2_token_requests.device
            record = {
                "prior": torch.full((capacity, topk), -1, dtype=torch.int32, device=device),
                "published": torch.full((capacity, topk), -1, dtype=torch.int32, device=device),
                "published_valid": torch.zeros(capacity, dtype=torch.bool, device=device),
                "keys": (),
                "serial": -1,
            }
            self._csa2_priors[layer_idx] = record
            self.csa2_indexer_prior_capacity[layer_idx] = record["prior"]
        if record["prior"].shape[1] != topk:
            raise ValueError("CSA2 temporal prior width must remain fixed for a layer")
        if record["serial"] != getattr(self, "_csa2_forward_serial", 0):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Refresh CSA2 request prior identity before graph replay")
            self._refresh_indexer_prior(layer_idx)
        return record["prior"][: record["count"]]

    def _refresh_indexer_prior(self, layer_idx: int) -> None:
        record = self._csa2_priors[layer_idx]
        previous = {key: row for row, key in enumerate(record["keys"]) if key is not None}
        rows, identities = [], []
        publish_keys = [None] * sum(self.csa2_request_lengths)
        reset = self.csa2_num_context_requests > 0
        for request_row, length in enumerate(self.csa2_request_lengths):
            request_id = self.request_ids[request_row]
            start = self.csa2_request_start_positions[request_row]
            identity = (
                id(self.kv_cache_manager),
                self.kv_cache_manager.request_epoch(request_id),
                request_id,
            )
            begin, end = self.csa2_request_query_ranges[request_row]
            if request_row < self.csa2_num_context_requests:
                # Context source tokens are accepted; its last query safely
                # seeds the first decode after this prompt/chunk.
                if length:
                    publish_keys[end - 1] = (*identity, start + length)
                continue
            source = previous.get((*identity, start), -1) if length == 1 else -1
            reset |= source < 0
            rows.extend([source] if length == 1 else [-1] * length)
            identities.extend([identity] * length)
            if length == 1:
                publish_keys[begin] = (*identity, start + 1)
        reset |= tuple(identities) != record.get("decode_identities", ())
        record["decode_identities"] = tuple(identities)
        count = len(rows)
        if max(count, len(publish_keys)) > record["prior"].shape[0]:
            raise ValueError("CSA2 temporal prior query capacity exceeded")
        device = record["prior"].device
        record["prior"].fill_(-1)
        if count:
            source_rows = torch.tensor(rows, dtype=torch.int64, device=device)
            valid = (source_rows >= 0) & record["published_valid"][source_rows.clamp_min(0)]
            record["prior"][:count].copy_(
                torch.where(valid[:, None], record["published"][source_rows.clamp_min(0)], -1)
            )
        record["published_valid"].zero_()
        record["keys"] = tuple(publish_keys)
        record["count"] = count
        record["serial"] = getattr(self, "_csa2_forward_serial", 0)
        callbacks = getattr(self, "_csa2_indexer_resets", {})
        if reset and layer_idx in callbacks:
            callbacks[layer_idx]()

    def publish_indexer_prior(self, layer_idx: int, logical_fullbatch: torch.Tensor) -> None:
        record = self._csa2_priors[layer_idx]
        count = len(record["keys"])
        if logical_fullbatch.shape[0] != count:
            raise ValueError("CSA2 prior publication must contain the full packed batch")
        record["published"][:count].copy_(logical_fullbatch)
        record["published_valid"][:count].fill_(True)

    def get_compression_batch(self, owner: int):
        return self._csa2_compression.get(owner)

    def get_compressed_positions(self, owner: int) -> torch.Tensor:
        return self._csa2_compressed_positions[owner]
