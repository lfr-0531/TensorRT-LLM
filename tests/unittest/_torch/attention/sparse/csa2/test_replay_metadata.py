# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded replay contracts over real allocated GLOBAL and volatile cache pools."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def replay_metadata():
    manager = _manager(4)
    for request_id in (71, 97):
        cache = manager._create_kv_cache(request_id, None, [])
        assert manager._resume_and_restore(request_id, cache)
        assert cache.resize(32)
    manager._stream.synchronize()
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=64, kv_cache_manager=manager)
    yield manager, metadata
    for request_id in list(manager.kv_cache_map):
        manager.free_resources(SimpleNamespace(py_request_id=request_id))
    manager.shutdown()


def _manager(window, *, enable_swa_bounded_replay=True):
    return CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=64 << 20, enable_swa_scratch_reuse=True),
        CacheType.SELFKONLY,
        num_layers=5,
        enable_swa_bounded_replay=enable_swa_bounded_replay,
        tokens_per_block=128,
        layout=CSA2Layout((0, 2, 2, 2, 1), (1, 4), (1, 2, 4), window_size=window, index_topk=4),
        max_seq_len=512,
        max_batch_size=2,
        max_num_tokens=512,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=8192,
    )


def _prepare(metadata, prefixes, suffixes=None, *, decoder=False, ids=None):
    suffixes = [0] * len(prefixes) if suffixes is None else suffixes
    ranges = metadata.kv_cache_manager.get_swa_replay_ranges(prefixes, suffixes, decoder=decoder)
    metadata.request_ids = ids or [71, 97][: len(prefixes)]
    metadata.num_contexts = len(prefixes)
    metadata.seq_lens = torch.tensor([end - start for start, end in ranges], dtype=torch.int32)
    metadata.prompt_lens = [end for _, end in ranges]
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[start for start, _ in ranges]
    )
    metadata.set_swa_bounded_replay(prefixes, decoder=decoder)
    metadata.prepare()
    return ranges


@torch.inference_mode()
def test_owner_source_selection_and_one_shot_reset(replay_metadata):
    manager, metadata = replay_metadata
    assert _prepare(metadata, [5, 6], [3, 2]) == ((1, 8), (2, 8))
    # Ratio2 replays one raw odd tail plus suffix; ratio1 only projects suffix.
    assert metadata.csa2_global_source_indices[1].tolist() == [3, 4, 5, 6, 11, 12]
    assert metadata.csa2_global_source_indices[4].tolist() == [4, 5, 6, 11, 12]
    compression = metadata.get_compression_batch(1)
    assert compression.start_positions.tolist() == [4, 6]
    assert compression.cu_seq_lengths.tolist() == [0, 4, 6]
    assert compression.cu_compressed_lengths.tolist() == [0, 2, 3]
    assert metadata.csa2_swa_indices[0][0, :-1].tolist() == [-1, -1, -1]
    assert metadata.csa2_swa_indices[0][7, :-1].tolist() == [-1, -1, -1]
    assert set(metadata.csa2_global_source_indices) == {1, 4}
    assert [metadata.csa2_kv_sources[layer] for layer in (0, 1, 2, 3, 4)] == [None, 1, 1, 1, 4]
    for layer in range(5):
        assert metadata.csa2_swa_indices[layer][0, :-1].tolist() == [-1, -1, -1]
        assert metadata.csa2_swa_indices[layer][7, :-1].tolist() == [-1, -1, -1]
    hidden = torch.randn(13, 32, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        metadata.select_global_source(1, hidden), hidden[[3, 4, 5, 6, 11, 12]]
    )
    with pytest.raises(ValueError, match="external source batch"):
        metadata.select_global_source(1, hidden, hidden)
    # A normal subsequent prepare consumes no stale replay floor or selector.
    metadata.seq_lens = torch.ones(2, dtype=torch.int32)
    metadata.num_contexts = 0
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[8, 8])
    metadata.prepare()
    assert metadata.csa2_replay_mode is None
    assert metadata.csa2_global_source_indices == {}
    assert metadata.csa2_replay_start_positions.tolist() == [0, 0]
    next_hidden = hidden[:2]
    assert metadata.select_global_source(1, next_hidden) is next_hidden


@torch.inference_mode()
def test_replay_graph_updates_values_but_rejects_shape_or_mode_change(replay_metadata):
    _, metadata = replay_metadata
    metadata.is_cuda_graph = True
    _prepare(metadata, [5], [3])
    hidden = torch.randn(7, 32, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros(7, 8, 512, device="cuda", dtype=torch.bfloat16)
    metadata.get_query_tile_metadata(q, 4, query_start=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata.get_query_tile_metadata(q, 4, query_start=0)
        selected = metadata.select_global_source(1, hidden)
        positions = metadata.csa2_positions.clone()
        write_slots = metadata.csa2_main_write_slots[1].clone()
    # Same source/query shape at a different absolute hit position is valid.
    _prepare(metadata, [7], [3], ids=[97])
    hidden.neg_()
    graph.replay()
    torch.testing.assert_close(selected, hidden[3:])
    torch.testing.assert_close(positions, torch.arange(3, 10, dtype=torch.int32, device="cuda"))
    torch.testing.assert_close(write_slots, metadata.csa2_main_write_slots[1])
    with pytest.raises(ValueError, match="fresh graph metadata"):
        metadata.prepare()
    with pytest.raises(ValueError, match="fresh graph metadata"):
        _prepare(metadata, [6], [3])
    with pytest.raises(ValueError, match="fresh graph metadata"):
        _prepare(metadata, [7], decoder=True)


@torch.inference_mode()
def test_replay_requires_first_window_row_to_be_physically_writable():
    # Explicit replay must still reject undersized legacy retention; automatic
    # reconstruction provisions W+1 rows and is covered in its own tests.
    manager = _manager(128, enable_swa_bounded_replay=False)
    try:
        cache = manager._create_kv_cache(71, None, [])
        assert manager._resume_and_restore(71, cache)
        cache.enable_swa_scratch_reuse = False
        # Ordinary next-query retention at C255 keeps128..254, while replay
        # requires127..254. Page0 is genuinely absent in the real allocator.
        assert cache.resize(255, 255)
        manager._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=128, kv_cache_manager=manager
        )
        assert manager.get_swa_replay_ranges([255]) == ((127, 255),)
        with pytest.raises(ValueError, match="reserve the complete replay range"):
            _prepare(metadata, [255])
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()
