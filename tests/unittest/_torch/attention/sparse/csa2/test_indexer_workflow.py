# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 scheduling, strided owner pages, and real paged indexer replay."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import (
    CSA2Indexer,
    _ChunkInputs,
    _QueryChunk,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2ForwardState, CSA2Layout
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows, unpack_rows
from tensorrt_llm._torch.modules.top_k import TopK
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class _StridedOwner:
    """Exact GLOBAL record fixture; exercise the manager's actual native gather."""

    gather_indexer_keys = CSA2CacheManager.gather_indexer_keys
    gather_indexer_pages = CSA2CacheManager.gather_indexer_pages
    tokens_per_block = 128

    def __init__(self, layout, rows):
        self.layout = layout
        self.storage = torch.full((rows.shape[0], 356), 77, dtype=torch.uint8, device=rows.device)
        self.storage[:, 288:].copy_(rows)

    def get_index_buffer(self, layer):
        assert self.layout.layer(layer).kv_source == 0
        return self.storage[:, 288:]


def _indexer():
    return CSA2Indexer(CSA2Layout((1,), (0,), (0,), index_topk=32), 0, 32, 128)


def _projected(count):
    q = torch.randn(count, 32, 128, device="cuda", dtype=torch.bfloat16)
    packed = pack_rows(q, "index")
    weights = torch.rand(count, 32, device="cuda") / 32
    return q, packed, weights


def _assert_selection(actual, scores, valid):
    """Accept boundary ties, but require the correct valid set size and ranking."""
    for row in range(actual.shape[0]):
        selected = actual[row][actual[row] >= 0].long()
        count = min(actual.shape[1], int(valid[row].sum()))
        assert selected.numel() == count
        assert selected.unique().numel() == count
        assert bool(valid[row, selected].all())
        assert bool((selected[1:] >= selected[:-1]).all())
        assert bool((actual[row, count:] == -1).all())
        if count:
            cutoff = scores[row].masked_fill(~valid[row], -torch.inf).topk(count).values[-1]
            assert bool((scores[row, selected] >= cutoff - 0.005).all())


@torch.inference_mode()
def test_cached_prefix_chunks_gather_once_and_preserve_offsets(monkeypatch):
    torch.manual_seed(4301)
    indexer = _indexer()
    q, packed, weights = _projected(5)
    keys = pack_rows(torch.randn(80, 128, device="cuda", dtype=torch.bfloat16), "index")
    visible = torch.tensor([33, 34, 35, 66, 67], device="cuda", dtype=torch.int32)
    loads = []
    calls = []
    original = indexer.forward_prepared
    topk_phases = []
    original_topk = indexer.top_k.forward

    def topk(*args, **kwargs):
        topk_phases.append(kwargs["is_prefill"])
        return original_topk(*args, **kwargs)

    monkeypatch.setattr(indexer.top_k, "forward", topk)

    def forward(*args, **kwargs):
        calls.append(args[0].shape[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(indexer, "forward_prepared", forward)

    def load(begin, end, width):
        loads.append((begin, end, width))
        starts = torch.zeros(end - begin, device="cuda", dtype=torch.int32)
        return _ChunkInputs(
            keys[:width, :64].contiguous().view(torch.int8),
            keys[:width, 64:].contiguous(),
            starts,
            torch.full_like(starts, width),
            torch.arange(width, device="cuda").expand(end - begin, -1),
            visible[begin:end],
        )

    chunks = [
        _QueryChunk(0, 3, 35, load=lambda: load(0, 3, 35), max_query_tokens=1),
        _QueryChunk(3, 5, 67, load=lambda: load(3, 5, 67), max_query_tokens=1),
    ]
    out = torch.empty(5, 32, dtype=torch.int32, device="cuda")
    indexer._run_csa2_chunks(
        chunks,
        packed[..., :64].contiguous().view(torch.int8),
        weights,
        packed[..., 64:].contiguous(),
        out,
        Mapping(),
        -1,
    )
    assert loads == [(0, 3, 35), (3, 5, 67)]
    assert calls == [1] * 5
    assert topk_phases == [True] * 5
    decoded_q = unpack_rows(packed, 128, "index").float()
    decoded_k = unpack_rows(keys, 128, "index").float()
    scores = (torch.einsum("qhd,kd->qhk", decoded_q, decoded_k).relu() * weights[..., None]).sum(1)
    _assert_selection(out, scores, torch.arange(80, device="cuda")[None] < visible[:, None])


@pytest.mark.parametrize("ratio", [1, 2])
@torch.inference_mode()
def test_native_paged_owner_indexer_and_graph(monkeypatch, ratio):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native FP4 paged CSA2 integration requires SM100 family")
    torch.manual_seed(4302)
    layout = CSA2Layout(
        (ratio,),
        (0,),
        (0,),
        index_topk=32,
        candidate_source_layer_id=0 if ratio == 1 else None,
        candidate_topk_blocks=2,
    )
    packed_keys = pack_rows(torch.randn(512, 128, device="cuda", dtype=torch.bfloat16), "index")
    manager = _StridedOwner(layout, packed_keys)
    q, _, weights = _projected(3)
    metadata = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=3)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping()
    metadata.is_cuda_graph = True
    metadata.csa2_kv_sources = {0: 0}
    metadata.csa2_num_context_requests = 0
    metadata.csa2_request_query_ranges = [(0, 2), (2, 3)]
    metadata.csa2_request_start_positions = [125, 127]
    metadata.csa2_request_lengths = [2, 1]
    metadata.csa2_global_max_positions = {0: 128}
    page_size = 128 // ratio
    metadata.csa2_global_page_sizes = {0: page_size}
    table = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)
    visible = torch.tensor([63, 63, 64], device="cuda", dtype=torch.int32)
    metadata.csa2_global_page_tables = {0: table}
    metadata.csa2_visible_lengths = {0: visible}
    metadata.csa2_token_requests = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int64)
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = -1
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(3, 512, device="cuda", dtype=torch.bfloat16),
        index_q=q,
        index_weights=weights,
    )
    indexer = CSA2Indexer(layout, 0, 32, 128)
    native_calls = []
    topk_calls = []
    original_topk = TopK.forward

    def topk(selector, *args, **kwargs):
        topk_calls.append((selector.top_k, kwargs["is_prefill"]))
        return original_topk(selector, *args, **kwargs)

    monkeypatch.setattr(TopK, "forward", topk)
    original = indexer._call_paged_mqa_logits

    def paged(*args, **kwargs):
        native_calls.append(args[0].shape)
        return original(*args, **kwargs)

    monkeypatch.setattr(indexer, "_call_paged_mqa_logits", paged)
    monkeypatch.setattr(
        indexer, "_call_mqa_logits", lambda *a, **kw: pytest.fail("Dense path used")
    )
    for _ in range(3):
        output = indexer(state, 0, 3)
    assert native_calls and all(shape[:2] == (3, 1) for shape in native_calls)
    assert (32, False) in topk_calls
    if ratio == 1:
        assert (2, True) in topk_calls  # Candidate block selection also uses TopK.
    assert manager.get_index_buffer(0).stride(0) == 356
    bound = metadata.prepare_indexer(0)
    ptrs = (
        bound.csa2_indexer_k_cache.data_ptr(),
        bound.csa2_indexer_block_table.data_ptr(),
        bound.csa2_indexer_scheduler_metadata.data_ptr(),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexer(state, 0, 3)
    for lengths, pages in (
        ([0, 1, 32], [[0, 1], [2, 3]]),
        ([64, 65, 96], [[2, 3], [0, -1]]),
        ([63, 63, 64], [[1, 0], [3, 2]]),
    ):
        visible.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
        table.copy_(torch.tensor(pages, device="cuda", dtype=torch.int32))
        q.neg_()
        graph.replay()
        torch.cuda.synchronize()
        selected = output.clone()
        refreshed = metadata.prepare_indexer(0)
        assert ptrs == (
            refreshed.csa2_indexer_k_cache.data_ptr(),
            refreshed.csa2_indexer_block_table.data_ptr(),
            refreshed.csa2_indexer_scheduler_metadata.data_ptr(),
        )
        positions = torch.arange(128, device="cuda").expand(3, -1)
        req = metadata.csa2_token_requests
        physical = table[req[:, None], positions // page_size]
        valid = (physical >= 0) & (positions < visible[:, None])
        rows = manager.get_index_buffer(0)[
            (physical.clamp_min(0) * page_size + positions % page_size).long()
        ]
        decoded_k = unpack_rows(rows, 128, "index").float()
        decoded_q = unpack_rows(pack_rows(q, "index"), 128, "index").float()
        scores = (
            torch.einsum("qhd,qkd->qhk", decoded_q, decoded_k).relu() * weights[..., None]
        ).sum(1)
        _assert_selection(selected, scores, valid)
    assert bool((manager.storage[:, :288] == 77).all())
    with pytest.raises(ValueError, match="complete model query batch"):
        indexer(state, 1, 2)


@pytest.mark.parametrize("count", [5, 1])
@torch.inference_mode()
def test_two_rank_candidate_publication_and_reindex(count):
    from tensorrt_llm._utils import mpi_rank, mpi_world_size

    if mpi_world_size() != 2:
        pytest.skip("Run with exactly two MPI ranks and one CUDA device per rank")
    torch.manual_seed(4303)
    layout = CSA2Layout(
        (1, 1),
        (0,),
        (0, 1),
        0,
        candidate_topk_blocks=8,
        candidate_block_size=4,
        index_topk=32,
    )
    keys = pack_rows(torch.randn(128, 128, device="cuda", dtype=torch.bfloat16), "index")
    manager = _StridedOwner(layout, keys)
    q, _, weights = _projected(count)
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=count)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping(world_size=2, rank=mpi_rank(), tp_size=2)
    metadata.csa2_kv_sources = {0: 0, 1: 0}
    metadata.csa2_num_context_requests = 1
    metadata.csa2_request_query_ranges = [(0, count)]
    metadata.csa2_request_start_positions = [59]
    metadata.csa2_request_lengths = [count]
    metadata.csa2_global_max_positions = {0: 128}
    metadata.csa2_global_page_sizes = {0: 128}
    metadata.csa2_global_page_tables = {0: torch.zeros(1, 1, device="cuda", dtype=torch.int32)}
    visible = torch.arange(60, 60 + count, device="cuda", dtype=torch.int32)
    metadata.csa2_visible_lengths = {0: visible, 1: visible}
    metadata.csa2_token_requests = torch.zeros(count, device="cuda", dtype=torch.int64)
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = 0
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(count, 512, device="cuda", dtype=torch.bfloat16),
        index_q=q,
        index_weights=weights,
    )
    source = CSA2Indexer(layout, 0, 32, 128)
    source_result = source(state, 0, count).clone()
    gathered_candidates = metadata.csa2_candidates[0].clone()
    consumer = CSA2Indexer(layout, 1, 32, 128)
    state.index_q = -q
    consumer_result = consumer(state, 0, count).clone()
    # Execute the same request without query splitting on each rank. This is
    # independent of the MPI gather and includes the rank with zero local Q.
    metadata.mapping = Mapping()
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    state.index_q = q
    expected_source = source(state, 0, count).clone()
    expected_candidates = metadata.csa2_candidates[0].clone()
    state.index_q = -q
    expected_consumer = consumer(state, 0, count).clone()
    torch.testing.assert_close(source_result, expected_source, atol=0, rtol=0)
    torch.testing.assert_close(gathered_candidates, expected_candidates, atol=0, rtol=0)
    torch.testing.assert_close(consumer_result, expected_consumer, atol=0, rtol=0)
    assert bool((gathered_candidates >= 0).any(1).all())


def _single_request_state(heads=8, two_owners=False):
    layout = (
        CSA2Layout((2, 1), (0, 1), (0, 1), index_topk=32)
        if two_owners
        else CSA2Layout((2,), (0,), (0,), index_topk=32)
    )
    keys = torch.full((512, 128), -1.0, device="cuda", dtype=torch.bfloat16)
    keys[64:128].fill_(1)
    keys[128:].fill_(3)
    manager = _StridedOwner(layout, pack_rows(keys, "index"))
    if two_owners:
        manager.owner_buffers = {0: manager.storage, 1: manager.storage.clone()}
        manager.owner_buffers[1][:, 288:].copy_(pack_rows(-keys, "index"))
        manager.get_index_buffer = lambda layer: manager.owner_buffers[
            layout.layer(layer).kv_source
        ][:, 288:]
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=1)
    metadata.kv_cache_manager = manager
    metadata.mapping = Mapping()
    metadata.csa2_kv_sources = {i: i for i in layout.kv_source_layer_ids}
    metadata.csa2_num_context_requests = 0
    metadata.csa2_request_query_ranges = [(0, 1)]
    metadata.csa2_request_start_positions = [3]
    metadata.csa2_request_lengths = [1]
    metadata.csa2_global_max_positions = {
        i: 256 // layout.compress_ratios[i] for i in layout.kv_source_layer_ids
    }
    metadata.csa2_global_page_sizes = {
        i: 128 // layout.compress_ratios[i] for i in layout.kv_source_layer_ids
    }
    metadata.csa2_global_page_tables = {
        i: torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
        for i in layout.kv_source_layer_ids
    }
    metadata.csa2_visible_lengths = {
        i: torch.tensor([4 // layout.compress_ratios[i]], device="cuda", dtype=torch.int32)
        for i in layout.kv_source_layer_ids
    }
    metadata.csa2_token_requests = torch.zeros(1, device="cuda", dtype=torch.int64)
    metadata.csa2_indices = {}
    metadata.csa2_candidates = {}
    metadata.indexer_max_chunk_size = 16
    metadata.indexer_q_split_threshold = -1
    state = CSA2ForwardState(
        metadata=metadata,
        swa_kv=torch.zeros(1, 512, device="cuda", dtype=torch.bfloat16),
        index_q=torch.ones(1, heads, 128, device="cuda", dtype=torch.bfloat16),
        index_weights=torch.full((1, heads), 1.0 / heads, device="cuda"),
    )
    return layout, manager, metadata, state


def _assert_state_selection(output, state, owner):
    metadata = state.metadata
    manager = metadata.kv_cache_manager
    width = metadata.csa2_global_max_positions[owner]
    positions = torch.arange(width, device="cuda").unsqueeze(0)
    slots = metadata.global_slot_tile(owner, 0, 1, positions)
    valid = (slots >= 0) & (positions < metadata.csa2_visible_lengths[owner][:, None])
    rows = manager.get_index_buffer(owner)[slots.clamp_min(0).long()]
    keys = unpack_rows(rows, 128, "index").float()
    query = unpack_rows(pack_rows(state.index_q, "index"), 128, "index").float()
    scores = (
        torch.einsum("qhd,qkd->qhk", query, keys).relu() * state.index_weights[..., None]
    ).sum(1)
    _assert_selection(output, scores, valid)


@torch.inference_mode()
def test_gathered_decode_graph_grows_past_warmup_prefix(monkeypatch):
    layout, manager, metadata, state = _single_request_state()
    metadata.is_cuda_graph = True
    indexer = CSA2Indexer(layout, 0, 8, 128)
    calls = []
    original = manager.gather_indexer_keys

    def gather(owner, slots):
        calls.append(slots.numel())
        return original(owner, slots)

    monkeypatch.setattr(manager, "gather_indexer_keys", gather)
    monkeypatch.setattr(
        indexer,
        "_call_paged_mqa_logits",
        lambda *a, **kw: pytest.fail("Eight-head decode must use bounded gathered fallback"),
    )
    for _ in range(3):
        output = indexer(state, 0, 1)
    assert calls == [128] * 3
    torch.testing.assert_close(
        output[0, :2], torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = indexer(state, 0, 1)
    for length, second_page in ((96, 1), (96, 2), (0, -1), (65, 1)):
        metadata.csa2_visible_lengths[0].fill_(length)
        metadata.csa2_global_page_tables[0][0, 1] = second_page
        graph.replay()
        torch.cuda.synchronize()
        _assert_state_selection(output, state, 0)
        if length > 64:
            assert bool((output >= 64).any())
    assert bool((manager.storage[:, :288] == 77).all())


@torch.inference_mode()
def test_paged_metadata_eager_growth_reuses_bounded_arena():
    layout, manager, metadata, state = _single_request_state(32, two_owners=True)
    metadata.is_cuda_graph = False
    previous = None
    previous_capacity = 0
    for visible in (1, 2, 31, 32, 63, 64, 65, 66, 96, 127, 128):
        metadata.csa2_request_start_positions = [visible * 2 - 1]
        metadata.csa2_visible_lengths[0].fill_(visible)
        metadata.csa2_visible_lengths[1].fill_(visible * 2)
        for owner in layout.kv_source_layer_ids:
            binding = metadata.prepare_indexer(owner)
            assert len(metadata._csa2_indexer_workspaces) == 1
            # No retained arena per exact sequence length or per owner.
            capacity = binding.csa2_indexer_k_cache.numel()
            pointer = binding.csa2_indexer_k_cache.data_ptr()
            if previous is not None and capacity == previous_capacity:
                assert pointer == previous
            previous, previous_capacity = pointer, capacity
    assert previous_capacity <= (1 + 4) * 64 * 68
    # A changed packed-query geometry must replace its eager arena too.
    metadata.csa2_request_query_ranges = [(0, 2)]
    metadata.csa2_request_lengths = [2]
    metadata.csa2_request_start_positions = [0]
    metadata.csa2_token_requests = torch.zeros(2, device="cuda", dtype=torch.int64)
    metadata.csa2_request_last_query_indices.fill_(1)
    metadata.csa2_visible_lengths = {
        owner: torch.ones(2, device="cuda", dtype=torch.int32)
        for owner in layout.kv_source_layer_ids
    }
    metadata.prepare_indexer(0)
    assert len(metadata._csa2_indexer_workspaces) == 1


@torch.inference_mode()
def test_native_graph_switches_ratio_two_and_one_owner():
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Native FP4 paged CSA2 integration requires SM100 family")
    layout, manager, metadata, state = _single_request_state(32, two_owners=True)
    metadata.is_cuda_graph = True
    indexers = [CSA2Indexer(layout, owner, 32, 128) for owner in layout.kv_source_layer_ids]

    def run():
        # Each output is independently owned even though both calls reuse
        # the same native page staging and scheduler buffers.
        return tuple(indexer(state, 0, 1) for indexer in indexers)

    for _ in range(3):
        run()
    pointers = []
    for owner in layout.kv_source_layer_ids:
        metadata.prepare_indexer(owner)
        pointers.append(
            (
                metadata.csa2_indexer_k_cache.data_ptr(),
                metadata.csa2_indexer_block_table.data_ptr(),
                metadata.csa2_indexer_scheduler_metadata.data_ptr(),
            )
        )
    assert pointers[0] == pointers[1]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = run()
    for first, second in ((65, 129), (1, 2), (0, 0), (96, 191)):
        metadata.csa2_visible_lengths[0].fill_(first)
        metadata.csa2_visible_lengths[1].fill_(second)
        state.index_q.neg_()
        graph.replay()
        torch.cuda.synchronize()
        for owner, output in enumerate(outputs):
            _assert_state_selection(output, state, owner)
            binding = metadata.prepare_indexer(owner)
            assert pointers[owner] == (
                binding.csa2_indexer_k_cache.data_ptr(),
                binding.csa2_indexer_block_table.data_ptr(),
                binding.csa2_indexer_scheduler_metadata.data_ptr(),
            )
    assert len(metadata._csa2_indexer_workspaces) == 1
