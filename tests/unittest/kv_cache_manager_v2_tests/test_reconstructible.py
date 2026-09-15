# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-backend acceptance of real GLOBAL-only reuse and private reconstruction."""

import ctypes
import importlib
from importlib.util import find_spec

import cuda.bindings.driver as drv
import pytest

if find_spec("kv_cache_manager_v2") is not None:
    import kv_cache_manager_v2 as kv
else:
    from tensorrt_llm.runtime import kv_cache_manager_v2 as kv

utils = importlib.import_module(kv.__name__ + "._utils")
introspection = importlib.import_module(kv.__name__ + "._introspection")
LogicError = (
    kv._cpp.LogicError
    if kv.BACKEND == "cpp"
    else importlib.import_module(kv.__name__ + "._exceptions").LogicError
)
GLOBAL = kv.DataRole("global")
TRANSIENT = kv.DataRole("transient")


def _cuda(result):
    if result[0] != drv.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA operation failed: {result[0]}")
    return result[1:]


@pytest.fixture
def runtime():
    utils.init_cuda_once()
    managers, caches = [], []
    with utils.TemporaryCudaStream([]) as holder:
        stream = holder.handle

        def create(
            *,
            scratch=False,
            required=True,
            page_bytes=256,
            quota=16 << 20,
            commit_min_snapshot=False,
            with_state=False,
        ):
            layers = []
            if required:
                layers.append(
                    kv.AttentionLayerConfig(kv.LayerId(0), [kv.BufferConfig(GLOBAL, page_bytes)])
                )
            layers.append(
                kv.AttentionLayerConfig(
                    kv.LayerId(len(layers)),
                    [kv.BufferConfig(TRANSIENT, page_bytes)],
                    sliding_window_size=9,
                    reconstructible=True,
                )
            )
            if with_state:
                layers.append(
                    kv.AttentionLayerConfig(
                        kv.LayerId(len(layers)),
                        [kv.BufferConfig(kv.DataRole("state"), page_bytes)],
                        sliding_window_size=2,
                        reconstructible=True,
                    )
                )
            config = kv.KVCacheManagerConfig(
                tokens_per_block=4,
                layers=layers,
                cache_tiers=[kv.GpuCacheTierConfig(quota=quota)],
                enable_partial_reuse=True,
                max_util_for_resume=1.0,
                swa_scratch_reuse=kv.SwaScratchReuseConfig() if scratch else None,
                commit_min_snapshot=commit_min_snapshot,
            )
            manager = kv.KVCacheManager(config)
            managers.append(manager)
            return manager

        def request(manager, tokens=None):
            cache = manager.create_kv_cache(kv.ReuseScope(), tokens)
            caches.append(cache)
            return cache

        yield create, request, stream
        for cache in reversed(caches):
            cache.close()
        _cuda(drv.cuStreamSynchronize(stream))
        for manager in reversed(managers):
            manager.shutdown()
    holder.take_finish_event().synchronize()


def _pages(manager, cache, layer, role):
    layer = kv.LayerId(layer)
    group = manager.get_layer_group_id(layer)
    converter = manager.get_page_index_converter(layer, role)
    return converter(
        list(cache.get_base_page_indices(group)),
        kv.PageIndexMode.PER_LAYER,
        cache.get_scratch_desc(group),
    )


def _address(manager, cache, layer, role, ordinal):
    page = _pages(manager, cache, layer, role)[ordinal]
    assert page >= 0
    return int(
        manager.get_mem_pool_base_address(kv.LayerId(layer), role, kv.PageIndexMode.PER_LAYER)
    ) + page * manager.get_page_stride(kv.LayerId(layer), role)


def _fill(manager, cache, layer, role, value, stream, size=256):
    for ordinal, page in enumerate(_pages(manager, cache, layer, role)):
        if page >= 0:
            _cuda(
                drv.cuMemsetD8Async(
                    _address(manager, cache, layer, role, ordinal), value, size, stream
                )
            )


def _read(address, size=64):
    result = (ctypes.c_uint8 * size)()
    _cuda(drv.cuMemcpyDtoH(ctypes.addressof(result), address, size))
    return bytes(result)


def _seed(manager, request, stream, length, *, page_bytes=256):
    cache = request(manager)
    assert cache.resume(stream)
    assert cache.resize(length)
    _fill(manager, cache, 0, GLOBAL, 37, stream, page_bytes)
    _fill(manager, cache, 1, TRANSIENT, 91, stream, page_bytes)
    cache.commit(list(range(length)))
    cache.close()
    _cuda(drv.cuStreamSynchronize(stream))


@pytest.mark.parametrize("prefix", [12, 13])
@pytest.mark.parametrize("scratch", [False, True])
@pytest.mark.parametrize("commit_min_snapshot", [False, True])
def test_global_only_reuse_private_history(runtime, prefix, scratch, commit_min_snapshot):
    create, request, stream = runtime
    manager = create(scratch=scratch, commit_min_snapshot=commit_min_snapshot)
    _seed(manager, request, stream, prefix)
    transient_group = manager.get_layer_group_id(kv.LayerId(1))
    global_group = manager.get_layer_group_id(kv.LayerId(0))
    depth, global_pages = introspection.reuse_match_pages(
        manager, kv.ReuseScope(), list(range(prefix)), global_group, True
    )
    _, transient_pages = introspection.reuse_match_pages(
        manager, kv.ReuseScope(), list(range(prefix)), transient_group, True
    )
    assert depth == prefix
    assert all(page is not None for page in global_pages)
    assert all(page is None for page in transient_pages)
    left, right = request(manager, list(range(prefix))), request(manager, list(range(prefix)))
    assert left.num_committed_tokens == right.num_committed_tokens == prefix
    for cache in (left, right):
        assert cache.requires_reconstruction
        assert cache.get_reconstruction_ranges() == {transient_group: (prefix - 8, prefix)}
        assert cache.resume(stream)
        with pytest.raises(LogicError):
            cache.commit([100])
    last = (prefix - 1) // 4
    assert _address(manager, left, 1, TRANSIENT, last) != _address(
        manager, right, 1, TRANSIENT, last
    )
    _fill(manager, left, 1, TRANSIENT, 11, stream)
    _fill(manager, right, 1, TRANSIENT, 19, stream)
    left.mark_reconstructed(transient_group, prefix - 4, prefix)
    right.mark_reconstructed(transient_group)
    assert left.get_reconstructed_ranges() == {transient_group: (prefix - 4, prefix)}
    assert not left.requires_reconstruction
    _cuda(drv.cuStreamSynchronize(stream))
    assert _read(_address(manager, left, 1, TRANSIENT, last)) == bytes([11]) * 64
    assert _read(_address(manager, right, 1, TRANSIENT, last)) == bytes([19]) * 64
    assert _read(_address(manager, left, 0, GLOBAL, 0)) == bytes([37]) * 64
    # Reused partial GLOBAL pages remain private before suffix writes.
    if prefix % 4:
        assert _address(manager, left, 0, GLOBAL, last) != _address(manager, right, 0, GLOBAL, last)
        _cuda(drv.cuMemsetD8Async(_address(manager, left, 0, GLOBAL, last), 47, 64, stream))
        _cuda(drv.cuStreamSynchronize(stream))
        assert _read(_address(manager, right, 0, GLOBAL, last)) == bytes([37]) * 64
    left.suspend()
    assert left.resume(stream)
    assert _read(_address(manager, left, 1, TRANSIENT, last)) == bytes([11]) * 64
    assert not left.requires_reconstruction
    assert left.resize(prefix + 1, history_length=prefix)
    left.commit([100])


def test_readiness_lifecycle_and_reuse(runtime):
    create, request, stream = runtime
    manager = create()
    _seed(manager, request, stream, 12)
    group = manager.get_layer_group_id(kv.LayerId(1))
    cache = request(manager, list(range(12)))
    with pytest.raises(LogicError):
        cache.mark_reconstructed(group)
    # Prefetching the persistent prefix must not materialize absent transient
    # pages or mistake a memory-residency hint for completed reconstruction.
    pending = cache.get_reconstruction_ranges()
    utilization = introspection.storage_utilization(manager, kv.GPU_LEVEL)
    assert not cache.is_active
    assert cache.prefetch(kv.CacheLevel(kv.GPU_LEVEL))
    assert cache.requires_reconstruction
    assert cache.get_reconstruction_ranges() == pending
    assert not cache.is_active
    assert introspection.storage_utilization(manager, kv.GPU_LEVEL) == utilization
    assert cache.resume(stream)
    for begin, end in ((3, 12), (5, 11), (13, 12), (None, 12)):
        with pytest.raises(ValueError):
            cache.mark_reconstructed(group, begin, end)
    assert cache.requires_reconstruction
    cache.suspend()
    assert cache.resume(stream)
    assert cache.requires_reconstruction
    cache.close()
    assert not cache.requires_reconstruction
    assert cache.get_reconstructed_ranges() == {}
    later = request(manager, list(range(12)))
    assert later.num_committed_tokens == 12 and later.requires_reconstruction
    assert later.resume(stream)
    later.mark_reconstructed(group, 12, 12)
    with pytest.raises(LogicError):
        later.mark_reconstructed(group)


def test_no_required_cache_config_rejected(runtime):
    create, _, _ = runtime
    with pytest.raises((ValueError, RuntimeError)):
        create(required=False)


def test_reconstructible_config_validation():
    for window, sinks in ((None, None), (0, None), (9, 1)):
        with pytest.raises((ValueError, RuntimeError)):
            kv.AttentionLayerConfig(
                kv.LayerId(0),
                [kv.BufferConfig(TRANSIENT, 256)],
                sliding_window_size=window,
                num_sink_tokens=sinks,
                reconstructible=True,
            )
    assert not kv.AttentionLayerConfig(
        kv.LayerId(0), [kv.BufferConfig(GLOBAL, 256)]
    ).reconstructible


def test_resume_oom_rolls_back(runtime):
    create, request, stream = runtime
    manager = create(page_bytes=1 << 20)
    _seed(manager, request, stream, 12, page_bytes=1 << 20)
    target = request(manager, list(range(12)))
    blockers = []
    limit = manager.get_page_index_upper_bound(kv.LayerId(1), TRANSIENT) + 2
    for _ in range(limit):
        blocker = request(manager, list(range(12)))
        if not blocker.resume(stream):
            blocker.close()
            break
        blockers.append(blocker)
    else:
        pytest.fail("Real transient allocations did not exhaust their finite pool")
    assert blockers
    pending = target.get_reconstruction_ranges()
    utilization = introspection.storage_utilization(manager, kv.GPU_LEVEL)
    assert not target.resume(stream)
    assert target.get_reconstruction_ranges() == pending
    assert not target.is_active
    assert introspection.storage_utilization(manager, kv.GPU_LEVEL) == utilization
    blockers.pop().close()
    _cuda(drv.cuStreamSynchronize(stream))
    assert target.resume(stream)
    assert target.requires_reconstruction


def test_multiple_transient_lifecycles_require_independent_ack(runtime):
    create, request, stream = runtime
    manager = create(with_state=True)
    _seed(manager, request, stream, 12)
    cache = request(manager, list(range(12)))
    swa = manager.get_layer_group_id(kv.LayerId(1))
    state = manager.get_layer_group_id(kv.LayerId(2))
    assert cache.get_reconstruction_ranges() == {swa: (4, 12), state: (11, 12)}
    assert cache.resume(stream)
    cache.mark_reconstructed(swa, 8, 12)
    assert cache.requires_reconstruction
    with pytest.raises(LogicError):
        cache.commit([13])
    cache.mark_reconstructed(state, 12, 12)
    assert not cache.requires_reconstruction
    assert cache.get_reconstructed_ranges() == {swa: (8, 12), state: (12, 12)}
