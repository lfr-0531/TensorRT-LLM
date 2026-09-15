# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real V2 allocation, scratch tables and graph-stable CSA2 publication."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def manager_requests():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    manager = CSA2CacheManager(
        KvCacheConfig(enable_block_reuse=False, max_tokens=4096, enable_swa_scratch_reuse=True),
        CacheType.SELFKONLY,
        layout=CSA2Layout((0, 2, 2, 1), (1, 3), (1, 3)),
        num_layers=4,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=512,
        max_batch_size=2,
        max_input_len=512,
        max_num_tokens=1024,
        dtype=DataType.BF16,
        vocab_size=1024,
    )
    requests = []
    for request_id in (41, 97):
        request = LlmRequest(
            request_id=request_id,
            max_new_tokens=256,
            input_tokens=list(range(257)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        requests.append(request)
    manager._stream.synchronize()
    yield manager, requests
    for request in requests:
        manager.free_resources(request)
    manager.shutdown()


def _prepare(metadata, requests, starts, lengths):
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata.request_ids = [request.py_request_id for request in requests]
    metadata.seq_lens = torch.tensor(lengths, dtype=torch.int32)
    metadata.num_contexts = len(requests)
    metadata.prompt_lens = [257] * len(requests)
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=starts)
    metadata.prepare()


@torch.inference_mode()
def test_manager_metadata_scratch_and_source_capacity(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    _prepare(metadata, requests, [0, 0], [257, 257])
    assert metadata.csa2_swa_write_slots[1][:257].unique().numel() == 257
    # Every query retains its own past, including the beginning of a chunk
    # longer than the 128-token window.
    torch.testing.assert_close(
        metadata.csa2_swa_indices[1][128], metadata.csa2_swa_write_slots[1][1:129]
    )
    assert metadata.csa2_visible_lengths[0].count_nonzero() == 0
    assert metadata.get_compression_batch(3) is None
    assert manager.get_main_buffer(1).stride(0) == 356
    assert manager.get_index_buffer(1).stride(0) == 356
    metadata.set_source_batch([5, 3], [0, 1])
    _prepare(metadata, requests, [4, 3], [1, 1])
    compression = metadata.get_compression_batch(1)
    assert compression.output_rows == 8
    assert compression.cu_seq_lengths.tolist() == [0, 5, 8]
    assert compression.cu_compressed_lengths.tolist() == [0, 2, 4]
    assert metadata.get_compressed_positions(1).tolist() == [0, 2, 0, 2, 0, 0, 0, 0]
    assert metadata.csa2_main_write_slots[1][4:].tolist() == [-1] * 4
    _prepare(metadata, requests, [0, 1], [1, 1])
    assert metadata.get_compression_batch(1).cu_compressed_lengths.tolist() == [0, 0, 1]
    completed = metadata.csa2_main_write_slots[1].tolist()
    expected_page = manager.get_cache_indices(97, 1, CSA2CacheRole.GLOBAL)[0]
    assert completed == [expected_page * 64, -1]
    # Consumer and source resolve the same owner pages, without owning extra KV.
    assert manager.get_cache_indices(41, 1, CSA2CacheRole.GLOBAL) == manager.get_cache_indices(
        41, 2, CSA2CacheRole.GLOBAL
    )


@torch.inference_mode()
def test_manager_partial_compression_graph_and_strided_publication(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2Compressor
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        store_rows,
    )

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    metadata = metadata.create_cuda_graph_metadata(2)
    compressor = CSA2Compressor(32, 512, 2, 1e-6).cuda()
    torch.manual_seed(472)
    inputs = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
    _prepare(metadata, requests, [0, 0], [1, 1])
    manager.get_main_buffer(1).fill_(73)
    manager.get_index_buffer(1).fill_(59)
    compression = metadata.get_compression_batch(1)
    write_slots = metadata.csa2_main_write_slots[1]
    global_page_table = metadata.csa2_global_page_tables[1]
    write_slot_mapping = metadata.csa2_main_write_slots

    def forward():
        output = compressor(inputs, compression)
        store_rows(manager.get_main_buffer(1), write_slots, output, "main")
        return output

    for _ in range(3):
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    ptrs = (
        write_slots.data_ptr(),
        compression.start_positions.data_ptr(),
        global_page_table.data_ptr(),
    )
    previous = {}
    for step in range(4):
        inputs.copy_(torch.randn_like(inputs))
        ordered = requests if step < 2 else requests[::-1]
        _prepare(metadata, ordered, [step, step], [1, 1])
        refreshed = metadata.get_compression_batch(1)
        assert metadata.csa2_main_write_slots is not write_slot_mapping
        write_slot_mapping = metadata.csa2_main_write_slots
        assert ptrs == (
            metadata.csa2_main_write_slots[1].data_ptr(),
            refreshed.start_positions.data_ptr(),
            metadata.csa2_global_page_tables[1].data_ptr(),
        )
        manager.get_main_buffer(1).fill_(73)
        values = compressor.wkv(inputs.float())
        gates = compressor.wgate(inputs.float())
        expected_rows = []
        for row, request in enumerate(ordered):
            request_id = request.py_request_id
            if step % 2:
                old_values, old_gates = previous[request_id]
                weights = torch.stack((old_gates, gates[row])).softmax(0)
                expected_rows.append((torch.stack((old_values, values[row])) * weights).sum(0))
            previous[request_id] = (values[row].clone(), gates[row].clone())
        graph.replay()
        torch.cuda.synchronize()
        if step % 2 == 0:
            assert output.count_nonzero() == 0
            assert torch.all(manager.get_main_buffer(1) == 73)
        else:
            # Native pooling rounds the reduced latent before model RMSNorm.
            expected = torch.stack(expected_rows).to(torch.bfloat16).float()
            expected *= torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
            expected = (expected * compressor.norm_weight.float()).to(torch.bfloat16)
            torch.testing.assert_close(output, expected, atol=0.02, rtol=0.02)
            slots = metadata.csa2_main_write_slots[1].long()
            torch.testing.assert_close(manager.get_main_buffer(1)[slots], pack_rows(output, "main"))
        # Main stores must never overwrite the adjacent index bytes.
        assert torch.all(manager.get_index_buffer(1) == 59)


@torch.inference_mode()
def test_odd_prefix_reuse_compression_matches_fresh():
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
        CSA2CacheManager,
        CSA2CacheRole,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2Compressor
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    manager = CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=128 << 20,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            enable_swa_scratch_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
        ),
        CacheType.SELFKONLY,
        # This regression intentionally exercises the retained exact-cache
        # compatibility mode, including persisted FP32 odd-tail state.
        enable_swa_bounded_replay=False,
        layout=CSA2Layout((2,), (0,), (0,)),
        num_layers=1,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=512,
        max_batch_size=3,
        max_num_tokens=1024,
        dtype=DataType.BF16,
        vocab_size=8192,
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=3, max_num_tokens=1024, kv_cache_manager=manager)
    compressor = CSA2Compressor(32, 512, 2, 1e-6).cuda()
    torch.manual_seed(671)
    hidden = torch.randn(130, 32, dtype=torch.bfloat16, device="cuda")
    tokens = list(range(129))

    def allocate(request_id, prefix, capacity):
        cache = manager._create_kv_cache(request_id, None, prefix)
        assert cache is not None
        assert manager._resume_and_restore(request_id, cache)
        assert cache.resize(capacity)
        manager._stream.synchronize()
        return cache

    def compress(request_id, start, values):
        _prepare(metadata, [SimpleNamespace(py_request_id=request_id)], [start], [len(values)])
        output = compressor(values, metadata.get_compression_batch(0))
        manager.write_swa(
            0, metadata.csa2_swa_write_slots[0], compressor.wkv(values.float()).bfloat16()
        )
        manager.write_global(0, metadata.csa2_main_write_slots[0], output, output[:, :128])
        return output

    try:
        first = allocate(501, [], 129)
        compress(501, 0, hidden[:129])
        first.commit(tokens)
        torch.cuda.synchronize()
        manager.free_resources(SimpleNamespace(py_request_id=501))
        left = allocate(502, tokens + [7000], 130)
        right = allocate(503, tokens + [7001], 130)
        reused = left.num_committed_tokens
        assert reused == right.num_committed_tokens
        assert reused in (128, 129)
        print(
            f"CSA2 odd prefix: committed=129, actually_reused={reused}, recomputed={130 - reused}"
        )
        left_pages = manager.get_cache_indices(502, 0, CSA2CacheRole.GLOBAL)
        right_pages = manager.get_cache_indices(503, 0, CSA2CacheRole.GLOBAL)
        assert left_pages[1] != right_pages[1]
        pool = manager.get_buffers(0, CSA2CacheRole.GLOBAL)
        shared_before = pool[left_pages[0]].clone()
        right_before = pool[right_pages[1]].clone()
        continuation = compress(502, reused, hidden[reused:]).clone()
        torch.cuda.synchronize()
        # Publishing the completed odd-boundary pair must not modify either
        # the shared prefix page or another request's private writable suffix.
        torch.testing.assert_close(pool[left_pages[0]], shared_before, atol=0, rtol=0)
        torch.testing.assert_close(pool[right_pages[0]], shared_before, atol=0, rtol=0)
        torch.testing.assert_close(pool[right_pages[1]], right_before, atol=0, rtol=0)
        allocate(504, [], 130)
        fresh = compress(504, 0, hidden)
        torch.testing.assert_close(continuation[0], fresh[64], atol=0.02, rtol=0.02)
        values = compressor.wkv(hidden[128:].float())
        gates = compressor.wgate(hidden[128:].float())
        expected = (values * gates.softmax(0)).sum(0).bfloat16().float()
        expected *= torch.rsqrt(expected.square().mean() + 1e-6)
        expected = (expected * compressor.norm_weight.float()).bfloat16()
        torch.testing.assert_close(continuation[0], expected, atol=0.02, rtol=0.02)
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@pytest.mark.parametrize("start", [127, 128, 129])
@pytest.mark.parametrize("accepted_drafts", [0, 1, 4])
@torch.inference_mode()
def test_chain_rewind_preserves_compressor_and_swa(start, accepted_drafts):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2Compressor
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        gather_rows,
        pack_rows,
        unpack_rows,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DraftTargetDecodingConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    spec = DraftTargetDecodingConfig(max_draft_len=4, speculative_model="attention-test-draft")
    manager = CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=128 << 20, enable_swa_scratch_reuse=True),
        CacheType.SELFKONLY,
        layout=CSA2Layout((2,), (0,), (0,)),
        num_layers=1,
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=512,
        max_batch_size=1,
        max_num_tokens=512,
        dtype=DataType.BF16,
        vocab_size=8192,
        spec_config=spec,
    )
    request = LlmRequest(
        request_id=71,
        max_new_tokens=32,
        input_tokens=list(range(start)),
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=512, kv_cache_manager=manager)
    metadata.is_spec_decoding_enabled = True
    compressor = CSA2Compressor(32, 512, 2, 1e-6).cuda()
    torch.manual_seed(719 + start + accepted_drafts)
    hidden = torch.randn(start + 5, 32, device="cuda", dtype=torch.bfloat16)
    projected_swa = []

    def compress(position, values, context=False):
        metadata.request_ids = [71]
        metadata.seq_lens = torch.tensor([len(values)], dtype=torch.int32)
        metadata.num_contexts = int(context)
        metadata.prompt_lens = [start]
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[position]
        )
        metadata.prepare()
        output = compressor(values, metadata.get_compression_batch(0))
        swa = compressor.wkv(values.float()).bfloat16()
        projected_swa.append(swa.clone())
        manager.write_swa(0, metadata.csa2_swa_write_slots[0], swa)
        manager.write_global(0, metadata.csa2_main_write_slots[0], output, output[:, :128])
        return output

    try:
        assert manager.prepare_context(request)
        assert manager.resize_context(request, start)
        manager._stream.synchronize()
        compress(0, hidden[:start], context=True)
        cache = manager.kv_cache_map[71]
        assert cache.resize(start, start)
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        request.add_new_token(1000, 0)
        request.py_draft_tokens = [1001, 1002, 1003, 1004]
        assert manager.try_allocate_generation(request)
        manager._stream.synchronize()
        compress(start, hidden[start:])
        # The golden token is always retained; accepted_drafts counts only
        # subsequent draft tokens. The final sampled token is not cached yet.
        for offset in range(accepted_drafts + 1):
            request.add_new_token(1100 + offset, 0)
        request.py_num_accepted_draft_tokens = accepted_drafts
        request.py_rewind_len = 4 - accepted_drafts
        scheduled = ScheduledRequests()
        scheduled.generation_requests = [request]
        manager.update_resources(scheduled, metadata, 2)
        accepted_end = start + 1 + accepted_drafts
        assert cache.capacity == accepted_end
        continuation = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16)
        assert cache.resize(accepted_end + 2)
        manager._stream.synchronize()
        actual = compress(accepted_end, continuation)
        pair = (
            torch.cat((hidden[accepted_end - 1 : accepted_end], continuation[:1]))
            if accepted_end % 2
            else continuation
        )
        values, gates = compressor.wkv(pair.float()), compressor.wgate(pair.float())
        expected = (values * gates.softmax(0)).sum(0).bfloat16().float()
        expected *= torch.rsqrt(expected.square().mean() + 1e-6)
        expected = (expected * compressor.norm_weight.float()).bfloat16()
        torch.testing.assert_close(actual[0], expected, atol=0.02, rtol=0.02)
        assert torch.count_nonzero(actual[1]) == 0
        # Compare cache preservation against the exact projection inputs at
        # each publication boundary. Re-running GEMM at a different row count
        # can choose a different reduction schedule near BF16 rounding ties.
        retained = torch.cat(
            (projected_swa[0], projected_swa[1][: 1 + accepted_drafts], projected_swa[2][:1])
        )[-128:]
        swa_reference = unpack_rows(pack_rows(retained, "swa"), 512, "swa")
        slots = metadata.csa2_swa_indices[0][0]
        gathered = gather_rows(manager.get_swa_buffer(0), slots, 512, "swa")
        torch.testing.assert_close(gathered[-len(retained) :], swa_reference, atol=0, rtol=0)
    finally:
        manager.free_resources(request)
        manager.shutdown()


@torch.inference_mode()
def test_temporal_prior_identity_rewind_and_replay(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=8, kv_cache_manager=manager)
    metadata.is_cuda_graph = True
    resets = []
    metadata.register_indexer_reset(1, lambda: resets.append(True))

    def prepare(ordered, positions):
        metadata.request_ids = [request.py_request_id for request in ordered]
        metadata.seq_lens = torch.ones(2, dtype=torch.int32)
        metadata.num_contexts = 0
        metadata.prompt_lens = [257, 257]
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=positions
        )
        metadata.prepare()
        return metadata.prepare_indexer_prior(1, 4)

    prior = prepare(requests, [0, 0])
    assert torch.all(prior == -1)
    assert metadata.csa2_indexer_prior_capacity[1].shape == (8, 4)
    selected = torch.arange(8, device="cuda", dtype=torch.int32).view(2, 4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata.publish_indexer_prior(1, selected)
    graph.replay()
    prior = prepare(requests, [1, 1])
    torch.testing.assert_close(prior, selected)
    reset_count = len(resets)
    graph.replay()
    prior = prepare(requests[::-1], [2, 2])
    torch.testing.assert_close(prior, selected.flip(0))
    assert len(resets) > reset_count
    graph.replay()
    # A rewind must not carry a rejected/future query's temporal hint.
    assert torch.all(prepare(requests[::-1], [1, 1]) == -1)
    graph.replay()
    old_epoch = manager.request_epoch(requests[0].py_request_id)
    manager.free_resources(requests[0])
    assert manager.prepare_context(requests[0])
    assert manager.resize_context(requests[0], requests[0].context_chunk_size)
    assert manager.request_epoch(requests[0].py_request_id) != old_epoch
    assert torch.all(prepare(requests, [2, 2])[0] == -1)


@torch.inference_mode()
def test_retained_metadata_buffers_and_workspace_accounting(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=1024, kv_cache_manager=manager)
    for length in (1, 2, 3, 4):
        _prepare(metadata, requests, [0, 0], [length, length])
    eager_keys = [key for key in metadata._csa2_buffers if not key[-1]]
    assert len({key[0] for key in eager_keys}) == len(eager_keys)
    frame = metadata.get_query_tile_metadata(
        torch.zeros(2, 8, 512, device="cuda", dtype=torch.bfloat16), 17
    )
    before = metadata.get_workspace_bytes()
    metadata._csa2_buffers["alias", (), torch.bfloat16, False] = frame.swa_pool.view(-1)
    assert metadata.get_workspace_bytes() == before
    extra = torch.empty(129, device="cuda", dtype=torch.uint8)
    metadata._csa2_buffers["extra", (129,), torch.uint8, False] = extra
    assert metadata.get_workspace_bytes() == before + extra.untyped_storage().nbytes()
    # Exact sum of the graph arena allocations: packed pages, row starts,
    # block table, contexts, logical map, visible lengths, radix and schedule.
    expected = 3 * (64 * 68 + 8) + 4 * (260 + 8 + 80 * 32) + (148 + 1) * 8
    assert metadata.workspace_reservation_bytes(2, 4, 64, 32, 148) == expected


@torch.inference_mode()
def test_representable_draft_switch_restores_target_fields(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.speculative.interface import (
        prepare_attn_metadata_for_draft_replay,
        restore_attn_metadata_after_draft_replay,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    target, requests = manager_requests
    draft = CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=128 << 20),
        CacheType.SELFKONLY,
        layout=target.layout,
        num_layers=len(target.layout.compress_ratios),
        tokens_per_block=128,
        mapping=Mapping(),
        max_seq_len=512,
        max_batch_size=2,
        max_num_tokens=1024,
        dtype=DataType.BF16,
        vocab_size=1024,
        is_draft=True,
    )
    try:
        for request in requests:
            assert draft.prepare_context(request)
            assert draft.resize_context(request, request.context_chunk_size)
        draft._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=2,
            max_num_tokens=1024,
            kv_cache_manager=target,
            draft_kv_cache_manager=draft,
        )
        _prepare(metadata, requests, [0, 0], [1, 1])
        target_positions = metadata.csa2_positions
        target_slots = metadata.csa2_swa_write_slots[1]
        target_state = metadata.get_compression_batch(1).kv_state
        resets = []
        metadata.register_indexer_reset(1, lambda: resets.append("target"))
        saved = prepare_attn_metadata_for_draft_replay(metadata, draft)
        assert resets == ["target"]
        metadata.register_indexer_reset(1, lambda: resets.append("draft"))
        try:
            assert metadata.kv_cache_manager is draft
            assert metadata.csa2_positions.data_ptr() != target_positions.data_ptr()
            assert metadata.get_compression_batch(1).kv_state.data_ptr() != target_state.data_ptr()
        finally:
            restore_attn_metadata_after_draft_replay(metadata, saved)
        assert metadata.kv_cache_manager is target
        assert metadata.csa2_positions is target_positions
        assert metadata.csa2_swa_write_slots[1] is target_slots
        assert metadata.get_compression_batch(1).kv_state is target_state
        assert resets == ["target", "target"]
        # A later draft switch restores its own cached callback/prior state;
        # contiguous identity must not suppress reset of shared emission state.
        saved = prepare_attn_metadata_for_draft_replay(metadata, draft)
        assert resets[-2:] == ["target", "draft"]
        restore_attn_metadata_after_draft_replay(metadata, saved)
        assert resets[-1] == "target"
    finally:
        for request in requests:
            draft.free_resources(request)
        draft.shutdown()


def test_speculative_contract_rejects_unrepresentable_paths():
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager

    def construct(spec):
        return CSA2CacheManager(
            None, None, num_layers=1, tokens_per_block=128, mapping=None, spec_config=spec
        )

    with pytest.raises(NotImplementedError, match="token trees"):
        construct(SimpleNamespace(is_linear_tree=False))
    with pytest.raises(NotImplementedError, match="virtual draft layers"):
        construct(
            SimpleNamespace(
                is_linear_tree=True,
                spec_dec_mode=SimpleNamespace(is_mtp_eagle_one_model=lambda: True),
            )
        )
    manager = object.__new__(CSA2CacheManager)
    with pytest.raises(NotImplementedError, match="relocation indices"):
        manager.update_resources(
            SimpleNamespace(
                generation_requests=[SimpleNamespace(py_num_accepted_draft_tokens_indices=[1, 3])]
            )
        )


@torch.inference_mode()
def test_temporal_prefill_seeds_first_decode(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=8, kv_cache_manager=manager)
    _prepare(metadata, requests, [0, 0], [2, 2])
    assert metadata.prepare_indexer_prior(1, 4).shape == (0, 4)
    selected = torch.arange(16, dtype=torch.int32, device="cuda").view(4, 4)
    metadata.publish_indexer_prior(1, selected)
    metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    metadata.num_contexts = 0
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[2, 2])
    metadata.prepare()
    torch.testing.assert_close(metadata.prepare_indexer_prior(1, 4), selected[[1, 3]])


@torch.inference_mode()
def test_graph_workspace_rejects_underfunded_resolved_cap(manager_requests):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    manager, requests = manager_requests
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=8, kv_cache_manager=manager)
    metadata.is_cuda_graph = True
    metadata.request_ids = [request.py_request_id for request in requests]
    metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    metadata.num_contexts = 0
    metadata.prompt_lens = [257, 257]
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0, 0])
    metadata.prepare()
    manager.fp8_ctx_mla_kv_len_cap = 1
    with pytest.raises(ValueError, match="full admitted request KV bound"):
        metadata.prepare_indexer(1)
    assert not hasattr(metadata, "_csa2_indexer_workspaces")
