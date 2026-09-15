# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real GLOBAL-only prefix hits through request preparation and attention replay."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import PositionalEmbeddingParams, RopeParams
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import BlockReuseConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _fixture(window):
    layout = CSA2Layout((0, 2, 2, 2), (1,), (1, 2), window_size=window, index_topk=4)
    manager = CSA2CacheManager(
        KvCacheConfig(
            max_gpu_total_bytes=256 << 20,
            enable_block_reuse=True,
            enable_partial_reuse=True,
            enable_swa_scratch_reuse=True,
            block_reuse_config=BlockReuseConfig(policy="all_reusable"),
        ),
        CacheType.SELFKONLY,
        num_layers=4,
        tokens_per_block=128,
        max_seq_len=512,
        max_batch_size=4,
        max_num_tokens=512,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=8192,
        layout=layout,
    )
    torch.manual_seed(919)
    with torch.device("cuda"):
        modules = [
            DeepseekV41Attention(
                layout,
                layer,
                PositionalEmbeddingParams(
                    type=PositionEmbeddingType.rope_gptj,
                    rope=RopeParams(dim=64, theta=160000, max_positions=512),
                    is_neox=False,
                ),
                hidden_size=32,
                num_heads=8,
                q_lora_rank=32,
                o_lora_rank=16,
                num_groups=2,
                index_heads=2,
            )
            for layer in range(4)
        ]
        for module in modules:
            for name, value in module.named_parameters():
                value.fill_(1) if "norm_weight" in name else value.normal_(std=0.1)
        embedding = torch.randn(8192, 32, dtype=torch.bfloat16)
    return manager, modules, embedding


def _request(request_id, tokens):
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=16,
        input_tokens=tokens,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )


def _metadata(manager, request, metadata=None):
    metadata = metadata or CSA2TrtllmMetadata(
        max_num_requests=1, max_num_tokens=512, kv_cache_manager=manager
    )
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.seq_lens = torch.tensor([request.context_chunk_size], dtype=torch.int32)
    metadata.prompt_lens = [request.context_chunk_size]
    metadata.kv_cache_params = KVCacheParams(
        use_cache=True, num_cached_tokens_per_seq=[request.context_current_position]
    )
    metadata.prepare()
    return metadata


def _input(embedding, request):
    start = request.context_current_position
    ids = request.get_tokens_range(0, start, start + request.context_chunk_size)
    return embedding[torch.tensor(ids, dtype=torch.int64, device="cuda")]


def _scheduled(request):
    batch = ScheduledRequests()
    if request.context_current_position + request.context_chunk_size == request.prompt_len:
        batch.context_requests_last_chunk = [request]
    else:
        batch.context_requests_chunking = [request]
    return batch


def _global_rows(manager, request_id, count):
    pages = manager.get_cache_indices(request_id, 1, CSA2CacheRole.GLOBAL)
    slots = torch.tensor(
        [pages[i // 64] * 64 + i % 64 for i in range(count // 2)], device="cuda", dtype=torch.int64
    )
    return manager.get_buffers(1, CSA2CacheRole.GLOBAL).view(-1, 356)[slots].clone()


def _seed(manager, modules, embedding, prefix):
    request = _request(101, list(range(prefix)))
    assert manager.prepare_context(request)
    assert manager.resize_context(request, request.context_chunk_size)
    metadata = _metadata(manager, request)
    hidden = _input(embedding, request)
    for module in modules:
        hidden = module(hidden, metadata.csa2_positions, metadata)
    original = _global_rows(manager, 101, prefix)
    batch = _scheduled(request)
    request.move_to_next_context_chunk()
    manager.update_context_resources(batch)
    assert manager.kv_cache_map[101].num_committed_tokens == prefix
    manager.free_resources(request)
    return original


@pytest.mark.parametrize("prefix,window", [(7, 4), (255, 128)])
@torch.inference_mode()
def test_global_only_prefix_hit_reconstructs_all_layers(prefix, window):
    manager, modules, embedding = _fixture(window)
    try:
        original = _seed(manager, modules, embedding, prefix)
        request = _request(102, list(range(prefix)) + [6000, 6001])
        assert manager.prepare_context(request)
        cache = manager.kv_cache_map[102]
        assert cache.num_committed_tokens == prefix
        assert cache.requires_reconstruction
        assert request.prepopulated_prompt_len == prefix
        assert request.context_current_position == max(0, prefix - window)
        assert request.py_csa2_global_reused_tokens == prefix
        assert request.py_csa2_replay_tokens == min(prefix, window)
        torch.testing.assert_close(_global_rows(manager, 102, prefix), original, atol=0, rtol=0)
        assert manager.resize_context(request, request.context_chunk_size)
        metadata = _metadata(manager, request)
        assert metadata.csa2_automatic_replay
        assert metadata.csa2_replay_cached_lengths == (prefix,)
        with pytest.raises(ValueError, match="completed SWA reconstruction"):
            manager.try_allocate_generation(request)
        batch = _scheduled(request)
        with pytest.raises(ValueError, match="every configured"):
            manager.update_context_resources(batch)
        hidden = _input(embedding, request)
        hidden = modules[0](hidden, metadata.csa2_positions, metadata)
        assert cache.requires_reconstruction
        with pytest.raises(ValueError, match="every configured"):
            manager.update_context_resources(batch)
        for module in modules[1:]:
            hidden = module(hidden, metadata.csa2_positions, metadata)
        assert cache.requires_reconstruction  # Completion is a post-forward lifecycle step.
        request.move_to_next_context_chunk()
        manager.update_context_resources(batch)
        assert not cache.requires_reconstruction
        assert torch.isfinite(hidden).all()
        torch.testing.assert_close(_global_rows(manager, 102, prefix), original, atol=0, rtol=0)
        assert (max(0, prefix - window), prefix) in cache.get_reconstructed_ranges().values()
        assert (prefix - prefix % 2, prefix) in cache.get_reconstructed_ranges().values()
        # Ordinary next-token decoding consumes the rebuilt state automatically.
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        request.add_new_token(7000, 0)
        assert manager.try_allocate_generation(request)
        metadata.request_ids = [102]
        metadata.num_contexts = 0
        metadata.seq_lens = torch.ones(1, dtype=torch.int32)
        metadata.prompt_lens = [request.prompt_len]
        metadata.kv_cache_params = KVCacheParams(
            use_cache=True, num_cached_tokens_per_seq=[request.max_beam_num_tokens - 1]
        )
        metadata.prepare()
        assert metadata.csa2_replay_mode is None
        hidden = embedding[7000:7001]
        for module in modules:
            hidden = module(hidden, metadata.csa2_positions, metadata)
        assert torch.isfinite(hidden).all()
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@torch.inference_mode()
def test_partial_automatic_replay_does_not_recommit_prefix():
    manager, modules, embedding = _fixture(4)
    try:
        original = _seed(manager, modules, embedding, 7)
        tokens = list(range(7)) + [6000, 6001]
        request = _request(102, tokens)
        assert manager.prepare_context(request)
        cache = manager.kv_cache_map[102]
        for begin, end in ((3, 5), (5, 7), (7, 9)):
            assert manager.prepare_context(request)
            assert request.context_current_position == begin
            assert request.prepopulated_prompt_len == 7
            assert manager.has_settled_replay_prefix(102)
            request.context_chunk_size = end - begin
            assert manager.resize_context(request, request.context_chunk_size)
            assert cache.capacity >= 7
            assert cache.history_length >= 7
            metadata = _metadata(manager, request)
            hidden = _input(embedding, request)
            for module in modules:
                hidden = module(hidden, metadata.csa2_positions, metadata)
            batch = _scheduled(request)
            request.move_to_next_context_chunk()
            manager.update_context_resources(batch)
            assert cache.num_committed_tokens == max(7, end)
            assert cache.requires_reconstruction == (end < 7)
            torch.testing.assert_close(_global_rows(manager, 102, 7), original, atol=0, rtol=0)
        assert request.context_current_position == 9
        manager.free_resources(request)
        assert not manager.has_settled_replay_prefix(102)
        # A normal token-path lookup sees exactly P+suffix, never P+replay+suffix.
        probe = _request(103, tokens + [7000])
        assert manager.prepare_context(probe)
        assert manager.kv_cache_map[103].num_committed_tokens == 9
        assert probe.py_csa2_global_reused_tokens == 9
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@torch.inference_mode()
def test_suspended_replay_defers_completion_until_resumed_forward():
    manager, modules, embedding = _fixture(4)
    try:
        _seed(manager, modules, embedding, 7)
        request = _request(102, list(range(7)) + [6000, 6001])
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        metadata = _metadata(manager, request)
        hidden = _input(embedding, request)
        for module in modules:
            hidden = module(hidden, metadata.csa2_positions, metadata)
        batch = _scheduled(request)
        request.move_to_next_context_chunk()
        manager.suspend_request(request)
        cache = manager.kv_cache_map[102]
        assert not cache.is_active
        manager.update_context_resources(batch)
        assert cache.requires_reconstruction
        assert cache.num_committed_tokens == 7
        assert set(manager.automatic_replay_plan(102)["progress"].values()) == {3}
        assert not cache.get_reconstructed_ranges()
        assert manager.prepare_context(request)
        assert request.context_current_position == 3
        assert manager.resize_context(request, request.context_chunk_size)
        metadata = _metadata(manager, request, metadata)
        hidden = _input(embedding, request)
        for module in modules:
            hidden = module(hidden, metadata.csa2_positions, metadata)
        batch = _scheduled(request)
        request.move_to_next_context_chunk()
        manager.update_context_resources(batch)
        assert not cache.requires_reconstruction
        assert cache.num_committed_tokens == 9
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@torch.inference_mode()
def test_pending_reconstruction_falls_back_before_graph_lookup():
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDAGraphRunner, KeyType

    class LookupRunner(CUDAGraphRunner):
        def __del__(self):
            pass

    manager, modules, embedding = _fixture(4)
    try:
        _seed(manager, modules, embedding, 7)
        request = _request(102, list(range(7)) + [6000, 6001])
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        metadata = _metadata(manager, request)
        runner = object.__new__(LookupRunner)
        runner.enabled = True
        runner.is_encoder_decoder = False
        runner.enable_encoder_decoder_mixed_cuda_graph = False
        runner.config = SimpleNamespace(enable_attention_dp=False, use_mrope=False)
        key = KeyType(1, 0, False)
        stored = object()
        runner.graph_metadata = {key: {"attn_metadata": stored, "spec_metadata": None}}
        lookups = []

        def graph_key(*args):
            lookups.append(True)
            return key

        runner.get_graph_key = graph_key
        # ModelEngine may promote a final context into a generation execution
        # view; the live native pending marker still forces whole-batch eager.
        promoted = ScheduledRequests()
        promoted.generation_requests = [request]
        assert promoted.can_run_cuda_graph
        assert runner.maybe_get_cuda_graph(
            promoted, False, metadata, promoted_context_request_ids=frozenset({102})
        ) == (None, None, None)
        assert not lookups
        hidden = _input(embedding, request)
        for module in modules:
            hidden = module(hidden, metadata.csa2_positions, metadata)
        batch = _scheduled(request)
        request.move_to_next_context_chunk()
        manager.update_context_resources(batch)
        assert not manager.kv_cache_map[102].requires_reconstruction
        request.state = LlmRequestState.GENERATION_IN_PROGRESS
        # Eligibility returns immediately once actual eager reconstruction has
        # completed, without changing the existing ordinary graph key family.
        assert runner.maybe_get_cuda_graph(promoted, False, metadata) == (stored, None, key)
        assert len(lookups) == 1
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@torch.inference_mode()
def test_joint_context_claim_applies_one_common_replay_cursor():
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    target, modules, embedding = _fixture(4)
    draft, draft_modules, draft_embedding = _fixture(4)
    try:
        _seed(target, modules, embedding, 7)
        _seed(draft, draft_modules, draft_embedding, 7)
        scheduler = object.__new__(KVCacheV2Scheduler)
        scheduler.kv_cache_manager = target
        scheduler._joint_draft_manager = draft
        request = _request(102, list(range(7)) + [6000, 6001])
        assert scheduler._prepare_context_pair(request)
        assert request.prepopulated_prompt_len == 7
        assert request.context_current_position == 3
        for manager, peer in ((target, draft), (draft, target)):
            assert manager.has_settled_replay_prefix(102)
            assert manager.kv_cache_map[102].num_committed_tokens == 7
            assert manager.kv_cache_map[102].requires_reconstruction
            manager.apply_reconstruction_cursor(request, peer)
            assert request.context_current_position == 3
        # Re-entry uses ordinary prepare_context, preserving the same absolute
        # common cursor instead of subtracting W again for either manager.
        assert scheduler._prepare_context_pair(request)
        assert request.context_current_position == 3
        for begin, end in ((3, 5), (5, 7), (7, 9)):
            assert scheduler._prepare_context_pair(request)
            assert request.context_current_position == begin
            assert request.prepopulated_prompt_len == 7
            request.context_chunk_size = end - begin
            # Each engine retains its metadata through post-forward ACK; the
            # manager intentionally holds only a weak reference to its receipt.
            step_metadata = {}
            for manager, layers, inputs in (
                (target, modules, embedding),
                (draft, draft_modules, draft_embedding),
            ):
                assert manager.resize_context(request, request.context_chunk_size)
                metadata = _metadata(manager, request)
                step_metadata[manager] = metadata
                hidden = _input(inputs, request)
                for module in layers:
                    hidden = module(hidden, metadata.csa2_positions, metadata)
                assert torch.isfinite(hidden).all()
            batch = _scheduled(request)
            request.move_to_next_context_chunk()
            for manager in step_metadata:
                manager.update_context_resources(batch)
                assert manager.kv_cache_map[102].requires_reconstruction == (end < 7)
                assert manager.kv_cache_map[102].num_committed_tokens == max(7, end)
        assert request.context_current_position == 9
    finally:
        for manager in (target, draft):
            for request_id in list(manager.kv_cache_map):
                manager.free_resources(SimpleNamespace(py_request_id=request_id))
            manager.shutdown()


@pytest.mark.parametrize("suspended_peer", ["target", "draft"])
@torch.inference_mode()
def test_joint_replay_defers_both_updates_when_one_peer_is_suspended(suspended_peer):
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    target, modules, embedding = _fixture(4)
    draft, draft_modules, draft_embedding = _fixture(4)
    engines = ((target, modules, embedding), (draft, draft_modules, draft_embedding))
    try:
        originals = {
            manager: _seed(manager, layers, inputs, 7) for manager, layers, inputs in engines
        }
        scheduler = object.__new__(KVCacheV2Scheduler)
        scheduler.kv_cache_manager = target
        scheduler._joint_draft_manager = draft
        request = _request(102, list(range(7)) + [6000, 6001])
        for attempt in range(2):
            assert scheduler._prepare_context_pair(request)
            assert request.context_current_position == 3
            assert request.prepopulated_prompt_len == 7
            assert request.context_chunk_size == 6
            step_metadata = {}
            for manager, layers, inputs in engines:
                assert manager.resize_context(request, request.context_chunk_size)
                metadata = _metadata(manager, request)
                step_metadata[manager] = metadata
                assert metadata.csa2_automatic_replay
                hidden = _input(inputs, request)
                for module in layers:
                    hidden = module(hidden, metadata.csa2_positions, metadata)
                assert torch.isfinite(hidden).all()
            batch = _scheduled(request)
            request.move_to_next_context_chunk()
            if attempt == 0:
                (target if suspended_peer == "target" else draft).suspend_request(request)
            for manager in step_metadata:
                manager.update_context_resources(batch)
            for manager, _, _ in engines:
                cache = manager.kv_cache_map[102]
                assert cache.requires_reconstruction == (attempt == 0)
                assert cache.num_committed_tokens == (7 if attempt == 0 else 9)
                if attempt == 0:
                    assert set(manager.automatic_replay_plan(102)["progress"].values()) == {3}
                if cache.is_active:
                    torch.testing.assert_close(
                        _global_rows(manager, 102, 7), originals[manager], atol=0, rtol=0
                    )
        assert request.context_current_position == 9
    finally:
        for manager, _, _ in engines:
            for request_id in list(manager.kv_cache_map):
                manager.free_resources(SimpleNamespace(py_request_id=request_id))
            manager.shutdown()


def test_joint_context_claim_without_reconstruction_hook():
    from tensorrt_llm._torch.pyexecutor.scheduler.scheduler_v2 import KVCacheV2Scheduler

    # The optional scheduler extension must leave ordinary manager contracts
    # unchanged. No tensors or simulated attention are involved in this hook test.
    ordinary = SimpleNamespace(
        tokens_per_block=128,
        probe_context_reuse=lambda request: 7,
        prepare_context_cache=lambda request, limit: 7,
    )
    scheduler = object.__new__(KVCacheV2Scheduler)
    scheduler.kv_cache_manager = ordinary
    scheduler._joint_draft_manager = ordinary
    request = _request(102, list(range(7)) + [6000, 6001])
    assert scheduler._prepare_context_pair(request)
    assert request.prepopulated_prompt_len == 7
    assert request.context_current_position == 7
