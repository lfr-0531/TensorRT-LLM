# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 ownership, quantized cache and hierarchical selection contracts."""

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import (
    CSA2Batch,
    CSA2GlobalPages,
    CSA2Routing,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
    CSA2BackendForwardArgs,
    CSA2ForwardState,
    CSA2Layout,
    CSA2Mode,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.prediction import (
    create_indexer,
    predict_csa2_inputs,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
    gather_rows,
    pack_rows,
    row_bytes,
    store_rows,
    unpack_rows,
)


def _test_pools(swa, main, index):
    """Synthetic byte pools for pure prediction tests; lifecycle has real-manager tests."""

    def write_global(owner, slots, values, keys):
        store_rows(main[owner], slots, values, "main")
        store_rows(index[owner], slots, keys, "index")

    return SimpleNamespace(
        get_swa_buffer=lambda layer: swa[layer],
        get_main_buffer=lambda owner: main[owner],
        get_index_buffer=lambda owner: index[owner],
        write_swa=lambda layer, slots, values: store_rows(swa[layer], slots, values, "swa"),
        write_global=write_global,
        indexers={},
    )


def _attention_reference(q, kv, valid, sink):
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * q.shape[-1] ** -0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    scores = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, -1)), -1)
    return torch.einsum("qhk,qkd->qhd", scores.softmax(-1)[..., :-1], kv.float()).to(q.dtype)


def _prediction_reference(layout, layer, q, swa, sink, cache, batch, routing, **kwargs):
    state = CSA2ForwardState(
        cache_manager=cache, batch=batch, routing=routing, swa_kv=swa, **kwargs
    )
    if state.index_q is not None and layer not in cache.indexers:
        cache.indexers[layer] = create_indexer(
            state.index_q.shape[1], state.index_q.shape[-1], layout.index_topk
        )
    selected = predict_csa2_inputs(
        layout, layer, CSA2BackendForwardArgs(state=state), q.shape[0], cache.indexers.get(layer)
    )
    kv = gather_rows(selected.swa_pool, selected.swa_indices, q.shape[-1], "swa")
    valid = selected.swa_indices >= 0
    if selected.topk_indices is not None:
        kv = torch.cat(
            (kv, gather_rows(selected.main_pool, selected.topk_indices, q.shape[-1], "main")), dim=1
        )
        valid = torch.cat((valid, selected.topk_indices >= 0), dim=1)
    return _attention_reference(q, kv, valid, sink)


def test_release_layout():
    layout = CSA2Layout(
        (0, 0) + (2,) * 18 + (1,) * 20 + (0,) * 3,
        (2, 8, 14, 20),
        (2, 8, 14, 20, 24, 28, 32, 36),
        20,
    )
    assert Counter(layout.layer(i).mode for i in range(40)) == {
        CSA2Mode.SWA: 2,
        CSA2Mode.FULL: 4,
        CSA2Mode.REINDEX: 4,
        CSA2Mode.REUSE: 30,
    }
    assert (
        sum(
            (row_bytes(512, "main") + row_bytes(128, "index")) // layout.compress_ratios[i]
            for i in layout.kv_source_layer_ids
        )
        == 890
    )
    assert layout.layer(39).kv_source == 20
    assert layout.layer(39).index_source == 36
    assert layout.layer(40).kv_source is None


@pytest.mark.parametrize(
    "ratios,kv,index,candidate",
    [
        ((2,), (), (), None),
        ((2, 1), (0,), (0,), None),
        ((2,), (0,), (), None),
        ((0,), (0,), (0,), None),
        ((2,), (0,), (0,), 0),
        ((1, 1), (0, 1), (0, 1), 0),
    ],
)
def test_invalid_layout(ratios, kv, index, candidate):
    with pytest.raises(ValueError):
        CSA2Layout(ratios, kv, index, candidate)


@pytest.mark.parametrize(
    "cache_format,dim,expected_bytes", [("main", 512, 288), ("index", 128, 68), ("swa", 512, 528)]
)
def test_quantized_row_layout(cache_format, dim, expected_bytes):
    x = torch.zeros(2, dim, dtype=torch.bfloat16)
    x[1] = 6
    rows = pack_rows(x, cache_format)
    assert rows.shape == (2, expected_bytes)
    assert row_bytes(dim, cache_format) == expected_bytes
    torch.testing.assert_close(unpack_rows(rows, dim, cache_format), x, atol=0, rtol=0)
    assert torch.all(rows[0, dim if cache_format == "swa" else dim // 2 :] != 0)


def test_fp4_midpoint_rounding_and_rope_tail():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2)
    x[8:] = -x[8:]
    rows = pack_rows(x[None], "main")
    assert rows[0, -1].item() == 56  # E4M3 1.0
    codes = torch.stack((rows[0, :-1] & 15, rows[0, :-1] >> 4), dim=-1).flatten()
    assert codes.tolist() == [0, 2, 2, 4, 4, 6, 6, 7, 8, 10, 10, 12, 12, 14, 14, 15]
    # No special BF16 tail: every channel, including the final RoPE group,
    # follows exactly the same packed quantization contract.
    torch.testing.assert_close(
        unpack_rows(rows, 16, "main", torch.float32),
        torch.tensor([[0, 1, 1, 2, 2, 4, 4, 6, 0, -1, -1, -2, -2, -4, -4, -6.0]]),
    )


def _selection_state(queries, keys, visible, topk=3):
    width = keys.shape[0]
    layout = CSA2Layout(
        (1, 1),
        (0,),
        (0, 1),
        0,
        candidate_topk_blocks=1,
        candidate_block_size=2,
        index_topk=topk,
        window_size=1,
    )
    manager = _test_pools(
        {
            i: torch.zeros(max(queries, 1), row_bytes(128, "swa"), dtype=torch.uint8)
            for i in range(2)
        },
        {0: torch.zeros(max(width, 1), row_bytes(128, "main"), dtype=torch.uint8)},
        {0: torch.zeros(max(width, 1), row_bytes(128, "index"), dtype=torch.uint8)},
    )
    batch = CSA2Batch(
        torch.arange(queries)[:, None],
        torch.arange(queries),
        torch.arange(width).expand(queries, -1),
        visible,
        torch.arange(width),
    )
    state = CSA2ForwardState(
        cache_manager=manager,
        batch=batch,
        routing=CSA2Routing(),
        swa_kv=torch.zeros(queries, 128, dtype=torch.bfloat16),
        index_q=torch.ones(queries, 1, 128, dtype=torch.bfloat16),
        index_weights=torch.ones(queries, 1),
        main_kv=torch.zeros(width, 128, dtype=torch.bfloat16),
        index_k=keys,
    )
    return layout, state


def test_candidates_pin_latest_block_and_mask_future():
    keys = torch.tensor([6.0, 4.0, 3.0, 2.0, 0.5, 6.0], dtype=torch.bfloat16)[:, None].expand(
        -1, 128
    )
    layout, state = _selection_state(1, keys, torch.tensor([5]))
    args = CSA2BackendForwardArgs(state=state)
    predict_csa2_inputs(layout, 0, args, 1)
    assert state.routing.candidates[0].tolist() == [[4, -1]]
    state.main_kv = state.index_k = None
    predict_csa2_inputs(layout, 1, args, 1)
    assert state.routing.indices[1].tolist() == [[4, -1, -1]]


@pytest.mark.parametrize("queries,width", [(0, 0), (0, 8), (2, 0), (2, 8)])
def test_empty_visibility(queries, width):
    layout, state = _selection_state(
        queries,
        torch.zeros(width, 128, dtype=torch.bfloat16),
        torch.zeros(queries, dtype=torch.int32),
    )
    predict_csa2_inputs(layout, 0, CSA2BackendForwardArgs(state=state), queries)
    assert state.routing.indices[0].shape == (queries, 3)
    assert torch.all(state.routing.indices[0] == -1)
    assert torch.all(state.routing.candidates[0] == -1)


def _run_modes(device, graph_replay=False, paged=False):
    layout = CSA2Layout(
        (1, 1, 1),
        (0,),
        (0, 2),
        0,
        candidate_topk_blocks=2,
        candidate_block_size=1,
        index_topk=1,
        window_size=1,
    )
    dim = 128
    cache = _test_pools(
        {
            i: torch.zeros(1, row_bytes(dim, "swa"), dtype=torch.uint8, device=device)
            for i in range(3)
        },
        {0: torch.zeros(2, row_bytes(dim, "main"), dtype=torch.uint8, device=device)},
        {0: torch.zeros(2, row_bytes(dim, "index"), dtype=torch.uint8, device=device)},
    )
    batch = CSA2Batch(
        torch.tensor([[0]], device=device),
        torch.tensor([0], device=device),
        (
            CSA2GlobalPages(
                torch.tensor([[0, 1]], device=device), torch.tensor([0], device=device), 1, 2
            )
            if paged
            else torch.tensor([[0, 1]], device=device)
        ),
        torch.tensor([2], device=device),
        torch.tensor([0, 1], device=device),
    )
    q = torch.zeros(1, 1, dim, dtype=torch.bfloat16, device=device)
    swa = torch.full((1, dim), 3.0, dtype=torch.bfloat16, device=device)
    main = torch.stack(
        (torch.full((dim,), 3.0, device=device), torch.full((dim,), 6.0, device=device))
    ).bfloat16()
    key = torch.stack((torch.ones(dim, device=device), -torch.ones(dim, device=device))).bfloat16()
    iq = torch.ones(1, 1, dim, dtype=torch.bfloat16, device=device)
    weights = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
    sink = torch.zeros(1, device=device)

    def run():
        routing = CSA2Routing()
        outputs = []
        for i in range(3):
            kwargs = dict(index_q=iq if i == 0 else -iq, index_weights=weights) if i != 1 else {}
            if i == 0:
                kwargs.update(main_kv=main, index_k=key)
            outputs.append(
                _prediction_reference(
                    layout, i, q, swa * (i + 1), sink, cache, batch, routing, **kwargs
                )
            )
        return outputs, routing

    if not graph_replay:
        outputs, routing = run()
        return outputs, routing, cache, batch
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs, routing = run()
    for sign, visible, reverse in (
        (1.0, 2, False),
        (-1.0, 2, True),
        (1.0, 1, False),
    ):
        iq.fill_(sign)
        batch.visible_lengths.fill_(visible)
        page_slots = batch.global_slots.page_table if paged else batch.global_slots
        page_slots.copy_(torch.tensor([[1, 0] if reverse else [0, 1]], device=device))
        graph.replay()
        actual = [o.clone() for o in outputs]
        actual_indices = routing.indices[0].clone()
        expected, expected_routing = run()
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=0, rtol=0)
        torch.testing.assert_close(actual_indices, expected_routing.indices[0], atol=0, rtol=0)
    if graph_replay and paged:
        for request_id in (-1, 1, 0):
            batch.global_slots.request_ids.fill_(request_id)
            batch.global_slots.page_table[0, 1] = -1
            graph.replay()
            actual = [o.clone() for o in outputs]
            expected, _ = run()
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
    return outputs, routing, cache, batch


@pytest.mark.parametrize("paged", [False, True])
def test_full_reuse_reindex_private_swa_and_shared_sink(paged):
    outputs, routing, cache, batch = _run_modes("cpu", paged=paged)
    assert routing.indices[0].tolist() == [[0]]
    assert routing.indices[2].tolist() == [[1]]
    # q=0 means selected global KV, private SWA, and sink each get 1/3.
    for actual, expected in zip(outputs, [2.0, 3.0, 5.0]):
        torch.testing.assert_close(
            actual.float(), torch.full_like(actual.float(), expected), atol=0.02, rtol=0
        )
    layout = CSA2Layout((1, 1), (0,), (0,))
    with pytest.raises(ValueError, match="across forwards"):
        routing.enter(layout.layer(0))
    q = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="source did not run"):
        _prediction_reference(layout, 1, q, q[:, 0], torch.zeros(1), cache, batch, CSA2Routing())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_modes():
    outputs, routing, _, _ = _run_modes("cuda")
    torch.cuda.synchronize()
    assert routing.indices[2].tolist() == [[1]]
    torch.testing.assert_close(outputs[-1], torch.full_like(outputs[-1], 5.0), atol=0.02, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_quantization_cuda_graph_changed_values():
    x = torch.ones(4, 128, dtype=torch.bfloat16, device="cuda")
    for _ in range(3):
        pack_rows(x, "main")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed = pack_rows(x, "main")
        restored = unpack_rows(packed, 128, "main")
    for value in (0.0, 3.0, -6.0):
        x.fill_(value)
        graph.replay()
        torch.testing.assert_close(restored, x, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("paged", [False, True])
def test_all_modes_cuda_graph_changed_routing(paged):
    _run_modes("cuda", graph_replay=True, paged=paged)


def test_paged_owner_mapping_and_request_reordering():
    pages = CSA2GlobalPages(
        torch.tensor([[2, 0], [1, -1]], dtype=torch.int32), torch.tensor([1, 0]), 2, 4
    )
    assert pages.resolve(0, 2).tolist() == [[2, 3, -1, -1], [4, 5, 0, 1]]
    logical = torch.tensor([[1, 3, -1], [2, 0, 4]], dtype=torch.int32)
    assert pages.resolve(0, 2, logical).tolist() == [[3, -1, -1], [0, 4, -1]]


def test_invalid_paged_request_does_not_alias_real_request():
    pages = CSA2GlobalPages(torch.tensor([[3], [7]]), torch.tensor([-1, 2, 1]), 2, 2)
    assert pages.resolve(0, 3).tolist() == [[-1, -1], [-1, -1], [14, 15]]
