# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical parity of native-hit replay with an explicit bounded-replay oracle."""

from types import SimpleNamespace

import pytest
import torch
from test_automatic_replay import (
    _fixture,
    _global_rows,
    _input,
    _metadata,
    _request,
    _scheduled,
    _seed,
)

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@torch.inference_mode()
def test_automatic_odd_prefix_matches_explicit_bounded_replay_and_decode():
    automatic, modules, embedding = _fixture(4)
    reference, reference_modules, reference_embedding = _fixture(4)
    try:
        # Identical modules and inputs; the reference has no prefix tree hit.
        # Its only imported history is the canonical packed GLOBAL payload,
        # exactly the input assumed by the explicit bounded-replay algorithm.
        for actual, expected in zip(modules, reference_modules):
            for actual_parameter, expected_parameter in zip(
                actual.parameters(), expected.parameters()
            ):
                torch.testing.assert_close(actual_parameter, expected_parameter, atol=0, rtol=0)
        torch.testing.assert_close(embedding, reference_embedding, atol=0, rtol=0)
        prefix = 7
        cached_global = _seed(automatic, modules, embedding, prefix)
        tokens = list(range(prefix)) + [6000, 6001]
        actual_request = _request(102, tokens)
        reference_request = _request(202, tokens)
        assert automatic.prepare_context(actual_request)
        assert automatic.kv_cache_map[102].num_committed_tokens == prefix
        assert automatic.kv_cache_map[102].requires_reconstruction
        assert actual_request.context_current_position == 3
        assert automatic.resize_context(actual_request, actual_request.context_chunk_size)
        actual_metadata = _metadata(automatic, actual_request)
        assert actual_metadata.csa2_automatic_replay

        assert reference.prepare_context(reference_request)
        assert reference.kv_cache_map[202].num_committed_tokens == 0
        assert not reference.kv_cache_map[202].requires_reconstruction
        assert reference.resize_context(reference_request, len(tokens))
        pages = reference.get_cache_indices(202, 1, CSA2CacheRole.GLOBAL)
        rows_per_page = reference.tokens_per_block // 2
        slots = torch.tensor(
            [
                pages[row // rows_per_page] * rows_per_page + row % rows_per_page
                for row in range(prefix // 2)
            ],
            device="cuda",
            dtype=torch.int64,
        )
        reference.get_buffers(1, CSA2CacheRole.GLOBAL).view(-1, 356).index_copy_(
            0, slots, cached_global
        )
        reference_request.context_current_position = 3
        reference_request.context_chunk_size = len(tokens) - 3
        reference_metadata = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=512, kv_cache_manager=reference
        )
        reference_metadata.set_swa_bounded_replay([prefix])
        _metadata(reference, reference_request, reference_metadata)
        assert not reference_metadata.csa2_automatic_replay
        assert reference_metadata.csa2_replay_mode == "encoder"

        actual_hidden = _input(embedding, actual_request)
        expected_hidden = _input(reference_embedding, reference_request)
        # Four layers cover SWA-only, Full, Reindex, and Reuse. Compare every
        # output so an early-layer mismatch cannot be hidden downstream.
        for layer, (actual_module, expected_module) in enumerate(zip(modules, reference_modules)):
            actual_hidden = actual_module(
                actual_hidden, actual_metadata.csa2_positions, actual_metadata
            )
            expected_hidden = expected_module(
                expected_hidden, reference_metadata.csa2_positions, reference_metadata
            )
            torch.testing.assert_close(
                actual_hidden,
                expected_hidden,
                atol=0,
                rtol=0,
                msg=f"Bounded replay output differs at layer {layer}",
            )
        for manager, request in ((automatic, actual_request), (reference, reference_request)):
            torch.testing.assert_close(
                _global_rows(manager, request.py_request_id, prefix), cached_global, atol=0, rtol=0
            )
            batch = _scheduled(request)
            request.move_to_next_context_chunk()
            manager.update_context_resources(batch)
        assert not automatic.kv_cache_map[102].requires_reconstruction

        # Nine prompt positions leave raw token 8 as an incomplete compressed
        # pair. Decode at 9 must consume that FP32 tail in both implementations.
        for manager, request, metadata in (
            (automatic, actual_request, actual_metadata),
            (reference, reference_request, reference_metadata),
        ):
            request.state = LlmRequestState.GENERATION_IN_PROGRESS
            request.add_new_token(7000, 0)
            assert manager.try_allocate_generation(request)
            metadata.num_contexts = 0
            metadata.seq_lens = torch.ones(1, dtype=torch.int32)
            metadata.prompt_lens = [len(tokens)]
            metadata.kv_cache_params = KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[request.max_beam_num_tokens - 1]
            )
            metadata.prepare()
            assert metadata.csa2_replay_mode is None
        actual_hidden, expected_hidden = embedding[7000:7001], reference_embedding[7000:7001]
        for layer, (actual_module, expected_module) in enumerate(zip(modules, reference_modules)):
            actual_hidden = actual_module(
                actual_hidden, actual_metadata.csa2_positions, actual_metadata
            )
            expected_hidden = expected_module(
                expected_hidden, reference_metadata.csa2_positions, reference_metadata
            )
            torch.testing.assert_close(
                actual_hidden,
                expected_hidden,
                atol=0,
                rtol=0,
                msg=f"Post-replay decode output differs at layer {layer}",
            )
    finally:
        for manager in (automatic, reference):
            for request_id in list(manager.kv_cache_map):
                manager.free_resources(SimpleNamespace(py_request_id=request_id))
            manager.shutdown()
