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
    batch = metadata.get_layer_batch(1)
    assert batch.swa_write_slots[:257].unique().numel() == 257
    # Every query retains its own past, including the beginning of a chunk
    # longer than the 128-token window.
    torch.testing.assert_close(batch.swa_indices[128], batch.swa_write_slots[1:129])
    assert metadata.get_layer_batch(0).visible_lengths.count_nonzero() == 0
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
    assert metadata.get_layer_batch(1).main_write_slots[4:].tolist() == [-1] * 4
    _prepare(metadata, requests, [0, 1], [1, 1])
    assert metadata.get_compression_batch(1).cu_compressed_lengths.tolist() == [0, 0, 1]
    completed = metadata.get_layer_batch(1).main_write_slots.tolist()
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
    batch = metadata.get_layer_batch(1)
    compression = metadata.get_compression_batch(1)

    def forward():
        output = compressor(inputs, compression)
        store_rows(manager.get_main_buffer(1), batch.main_write_slots, output, "main")
        return output

    for _ in range(3):
        forward()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = forward()
    ptrs = (
        batch.main_write_slots.data_ptr(),
        compression.start_positions.data_ptr(),
        batch.global_slots.page_table.data_ptr(),
    )
    previous = {}
    for step in range(4):
        inputs.copy_(torch.randn_like(inputs))
        ordered = requests if step < 2 else requests[::-1]
        _prepare(metadata, ordered, [step, step], [1, 1])
        refreshed = metadata.get_compression_batch(1)
        fresh_batch = metadata.get_layer_batch(1)
        assert ptrs == (
            fresh_batch.main_write_slots.data_ptr(),
            refreshed.start_positions.data_ptr(),
            fresh_batch.global_slots.page_table.data_ptr(),
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
            slots = fresh_batch.main_write_slots.long()
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
        batch = metadata.get_layer_batch(0)
        manager.write_swa(0, batch.swa_write_slots, compressor.wkv(values.float()).bfloat16())
        manager.write_global(0, batch.main_write_slots, output, output[:, :128])
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
