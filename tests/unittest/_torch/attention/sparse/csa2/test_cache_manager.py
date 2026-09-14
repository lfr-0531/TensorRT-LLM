# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real V2 storage and request lifecycle for CSA2 packed owner caches."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping


def test_disabled_layer_mask_rejected_before_allocation():
    with pytest.raises(ValueError, match="disabled layers in layer_mask"):
        CSA2CacheManager(
            KvCacheConfig(),
            CacheType.SELFKONLY,
            num_layers=2,
            tokens_per_block=128,
            mapping=Mapping(),
            layout=CSA2Layout((0, 0), (), ()),
            layer_mask=[True, False],
        )


@pytest.fixture
def manager():
    if not torch.cuda.is_available():
        pytest.skip("CSA2 runtime cache storage requires CUDA")
    layout = CSA2Layout((0, 2, 2, 1, 1), (1, 3), (1, 3))
    result = CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=128 << 20,
            host_cache_size=0,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            enable_swa_scratch_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
        ),
        CacheType.SELFKONLY,
        num_layers=5,
        tokens_per_block=128,
        max_seq_len=2048,
        max_batch_size=3,
        max_num_tokens=1024,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=8192,
        layout=layout,
    )
    yield result
    for request_id in list(result.kv_cache_map):
        result.free_resources(SimpleNamespace(py_request_id=request_id))
    result.shutdown()


def allocate(manager, request_id, tokens, capacity):
    cache = manager._create_kv_cache(request_id, None, tokens)
    assert cache is not None
    assert manager._resume_and_restore(request_id, cache)
    assert cache.resize(capacity)
    return cache


def test_owner_views_and_lifecycle(manager):
    cache = allocate(manager, 10, [], 513)
    assert (
        len({manager.get_cache_indices(10, layer, CSA2CacheRole.SWA)[0] for layer in range(5)}) == 5
    )
    for owner in (1, 3):
        main, index = manager.get_main_buffer(owner), manager.get_index_buffer(owner)
        assert main.stride() == (356, 1)
        assert index.stride() == (356, 1)
        assert index.data_ptr() - main.data_ptr() == 288
        assert manager.get_cache_indices(
            10, owner, CSA2CacheRole.GLOBAL
        ) == manager.get_cache_indices(10, owner + 1, CSA2CacheRole.GLOBAL)
        page = manager.get_cache_indices(10, owner, CSA2CacheRole.GLOBAL)[0]
        row = page * (128 // manager.layout.compress_ratios[owner])
        main[row].fill_(37)
        index[row].fill_(91)
        torch.testing.assert_close(main[row], torch.full_like(main[row], 37))
        torch.testing.assert_close(index[row], torch.full_like(index[row], 91))
    sizes, windows = manager._get_runtime_cache_size_layer_components()
    assert sum(size for size, window in zip(sizes, windows) if window is None) == 534
    assert sizes.count(4096) == 1
    quota = manager._get_quota_from_max_tokens(512)
    assert quota > 512 * 534
    assert manager._get_max_tokens_from_quota(quota) == pytest.approx(512)
    # Long prefill resolves scratch pages, not just the final sliding ring.
    for role in (CSA2CacheRole.SWA, CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        pages = manager.get_cache_indices(10, 1, role)
        assert len(pages) >= 5
        assert len(set(pages[:5])) == 5
    assert cache.resize(514, history_length=513)
    manager.free_resources(SimpleNamespace(py_request_id=10))
    assert 10 not in manager.kv_cache_map
    allocate(manager, 11, [], 129)
    assert manager.get_cache_indices(11, 1, CSA2CacheRole.GLOBAL)


def test_prefix_reuse_preserves_combined_records(manager):
    tokens = list(range(512))
    first = allocate(manager, 20, [], len(tokens))
    for model_layer, role in manager._physical_roles.values():
        pages = manager.get_cache_indices(20, model_layer, role)
        pool = manager.get_buffers(model_layer, role)
        for page in pages:
            if page >= 0:
                pool[page].fill_(7)
    first.commit(tokens)
    torch.cuda.synchronize()
    manager.free_resources(SimpleNamespace(py_request_id=20))
    second = allocate(manager, 21, tokens + [7000], 513)
    assert second.num_committed_tokens > 0
    page = manager.get_cache_indices(21, 1, CSA2CacheRole.GLOBAL)[0]
    packed = manager.get_buffers(1, CSA2CacheRole.GLOBAL)[page]
    torch.testing.assert_close(packed, torch.full_like(packed, 7))
    # Fork an overlapping prefix and allocate independent writable suffixes.
    third = allocate(manager, 22, tokens + [7001], 513)
    assert third.num_committed_tokens > 0
    pages2 = manager.get_cache_indices(21, 1, CSA2CacheRole.GLOBAL)
    pages3 = manager.get_cache_indices(22, 1, CSA2CacheRole.GLOBAL)
    assert pages2[4] != pages3[4]
    pool = manager.get_buffers(1, CSA2CacheRole.GLOBAL)
    pool[pages2[4]].fill_(11)
    pool[pages3[4]].fill_(19)
    torch.testing.assert_close(pool[pages2[4]], torch.full_like(pool[pages2[4]], 11))
    torch.testing.assert_close(pool[pages3[4]], torch.full_like(pool[pages3[4]], 19))


def test_partial_page_copy_on_write_and_state(manager):
    tokens = list(range(129))
    source = allocate(manager, 30, [], 129)
    for model_layer, role in manager._physical_roles.values():
        pool = manager.get_buffers(model_layer, role)
        for page in manager.get_cache_indices(30, model_layer, role):
            if page >= 0:
                pool[page].fill_(23)
    source.commit(tokens)
    torch.cuda.synchronize()
    manager.free_resources(SimpleNamespace(py_request_id=30))
    left = allocate(manager, 31, tokens + [7000], 130)
    right = allocate(manager, 32, tokens + [7001], 130)
    assert left.num_committed_tokens == right.num_committed_tokens
    assert left.num_committed_tokens >= 128
    # The writable trailing page is private even when its prefix was reused.
    for role in (CSA2CacheRole.GLOBAL, CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        pool = manager.get_buffers(1, role)
        left_pages = manager.get_cache_indices(31, 1, role)
        right_pages = manager.get_cache_indices(32, 1, role)
        assert left_pages[1] != right_pages[1]
        pool[left_pages[1]].fill_(31)
        pool[right_pages[1]].fill_(47)
        torch.testing.assert_close(pool[left_pages[1]], torch.full_like(pool[left_pages[1]], 31))
    # Suspend/resume reconnects page-index buffers before the next forward.
    left.suspend()
    assert manager._resume_and_restore(31, left)
    assert manager.get_cache_indices(31, 1, CSA2CacheRole.GLOBAL)[1] >= 0
