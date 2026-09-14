# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared DSA Indexer prepared-input and unchanged prefill contracts."""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows, unpack_rows
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import Indexer
from tensorrt_llm._torch.attention.backends.sparse.dsa.params import DSAParams
from tensorrt_llm._torch.modules.top_k import TopK


# Independent mathematical references live only in this test module.
def _reference_select_candidate_positions(
    scores: torch.Tensor,
    visible_lengths: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Return bounded candidate positions [queries, blocks * block_size], int64.

    Scores are float32 [queries, positions]. Unreachable positions are masked
    before block reduction. The newest visible block is always retained, as
    required by the trained hierarchical indexer. Invalid slots contain -1.
    """
    if scores.ndim != 2 or visible_lengths.shape != scores.shape[:1]:
        raise ValueError("Candidate scores and visible lengths must have matching query rows")
    if topk_blocks <= 0 or block_size <= 0:
        raise ValueError("Candidate block count and size must be positive")
    width = scores.shape[-1]
    positions = torch.arange(width, device=scores.device)
    reachable = positions[None, :] < visible_lengths[:, None]
    scores = scores.masked_fill(~reachable, -torch.inf)
    padded = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    blocks = padded.reshape(scores.shape[0], padded.shape[-1] // block_size, block_size).amax(-1)
    block_ids = torch.arange(blocks.shape[-1], device=scores.device)
    latest = (visible_lengths - 1) // block_size
    blocks = blocks.masked_fill(block_ids[None, :] == latest[:, None], torch.inf)
    values, selected = blocks.topk(min(topk_blocks, blocks.shape[-1]), dim=-1, sorted=False)
    candidates = selected[:, :, None] * block_size + torch.arange(block_size, device=scores.device)
    valid = (values[:, :, None] > -torch.inf) & (candidates < visible_lengths[:, None, None])
    valid &= candidates < width
    return torch.where(valid, candidates, -1).flatten(1)


def _reference_select_topk_positions(
    scores: torch.Tensor,
    positions: torch.Tensor,
    visible_lengths: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Select sorted logical positions; pad unreachable selections with -1.

    ``positions`` is either [positions] or [queries, candidates]. Selection
    never publishes candidate-array offsets or cache addresses to Reuse layers.
    """
    if scores.ndim != 2 or visible_lengths.shape != scores.shape[:1]:
        raise ValueError("Top-k scores and visible lengths must have matching query rows")
    if topk <= 0 or positions.shape[-1] != scores.shape[-1]:
        raise ValueError("Invalid top-k size or candidate positions")
    positions = positions.long().expand_as(scores)
    valid = (positions >= 0) & (positions < visible_lengths[:, None])
    scores = scores.masked_fill(~valid, -torch.inf)
    values, offsets = scores.topk(min(topk, scores.shape[-1]), dim=-1, sorted=False)
    selected = positions.gather(1, offsets)
    # Sort invalid entries last. In particular, an unreachable candidate with a
    # low logical index must not become a valid selection after the gather.
    sentinel = torch.iinfo(torch.int64).max
    selected = torch.where(values > -torch.inf, selected, sentinel).sort(dim=-1).values
    selected = torch.where(selected == sentinel, -1, selected).to(torch.int32)
    return F.pad(selected, (0, topk - selected.shape[-1]), value=-1)


def _reference_index_scores(
    q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Rectified indexer scores with already-scaled per-head weights.

    Q is [queries, heads, dim], K is [positions, dim] or
    [queries, candidates, dim], weights is [queries, heads]. The gathered-K
    form bounds Reindex work by candidate count, independently of context size.
    """
    if k.ndim == 2:
        dots = torch.einsum("qhd,kd->qhk", q, k)
    elif k.ndim == 3:
        dots = torch.einsum("qhd,qkd->qhk", q, k)
    else:
        raise ValueError("Indexer keys must be shared or gathered per query")
    return (dots.relu() * weights.unsqueeze(-1)).sum(1).float()


def _indexer(heads=32, topk=32):
    if not torch.cuda.is_available():
        pytest.skip("Shared prepared indexer tests require CUDA")
    return Indexer(
        None,
        None,
        None,
        False,
        DSAParams(index_n_heads=heads, index_head_dim=128, index_topk=topk, indexer_k_dtype="fp4"),
        torch.bfloat16,
        projection_free=True,
    )


def _prepared(indexer, q, keys, weights, lengths, positions, hook=None):
    count = q.shape[0]
    width = keys.shape[-2]
    starts = torch.arange(count, dtype=torch.int32, device=q.device)
    starts = starts * width if keys.ndim == 3 else torch.zeros_like(starts)
    keys = keys.reshape(-1, 68)
    out = torch.empty((count, indexer.index_topk), dtype=torch.int32, device=q.device)
    indexer.forward_prepared(
        q[..., :64].contiguous().view(torch.int8),
        keys[:, :64].contiguous().view(torch.int8),
        keys[:, 64:].contiguous().view(torch.int32).squeeze(-1),
        weights,
        starts,
        starts + width,
        out,
        q[..., 64:].contiguous().view(torch.int32).squeeze(-1),
        logical_positions=positions,
        visible_lengths=lengths,
        score_hook=hook,
    )
    return out


def test_projection_free_lifecycle():
    indexer = _indexer()
    assert isinstance(indexer.top_k, TopK)
    assert list(indexer.parameters()) == []
    assert indexer.rotary_emb is None
    assert indexer.wq_b is indexer.wk is indexer.weights_proj is indexer.k_norm is None
    indexer.cache_derived_state()
    indexer.post_load_weights()
    with pytest.raises(RuntimeError, match="forward_prepared"):
        indexer.pre_indexer_proj(None, None, None)


@pytest.mark.parametrize("per_query", [False, True])
@pytest.mark.parametrize("heads", [2, 8, 32, 64])
def test_prepared_packed_selection(per_query, heads):
    indexer = _indexer(heads)
    torch.manual_seed(1701)
    q = pack_rows(torch.randn(3, heads, 128, device="cuda"), "index")
    shape = (3, 257, 128) if per_query else (257, 128)
    packed = pack_rows(torch.randn(shape, device="cuda"), "index")
    owner = torch.zeros((*packed.shape[:-1], 356), dtype=torch.uint8, device="cuda")
    owner[..., 288:].copy_(packed)
    keys = owner[..., 288:]
    weights = torch.rand(3, heads, device="cuda")
    lengths = torch.tensor([257, 39, 0], device="cuda", dtype=torch.int32)
    positions = torch.arange(257, device="cuda").expand(3, -1).clone()
    positions[:, 5::11] = -1
    actual = _prepared(indexer, q, keys, weights, lengths, positions)
    scores = _reference_index_scores(
        unpack_rows(q, 128, "index", torch.float32),
        unpack_rows(keys, 128, "index", torch.float32),
        weights,
    )
    expected = _reference_select_topk_positions(scores, positions, lengths, 32)
    # Low head counts can tie at zero; compare selected score multisets, with
    # exact logical output for the unambiguous 32/64-head cases.
    if heads >= 32:
        torch.testing.assert_close(actual, expected)
    else:
        valid = actual >= 0
        torch.testing.assert_close(valid.sum(-1), (expected >= 0).sum(-1))
        torch.testing.assert_close(
            scores.gather(1, actual.clamp_min(0).long())
            .masked_fill(~valid, -torch.inf)
            .sort(-1)
            .values,
            scores.gather(1, expected.clamp_min(0).long())
            .masked_fill(expected < 0, -torch.inf)
            .sort(-1)
            .values,
        )


def test_prepared_hierarchy_and_graph():
    indexer = _indexer()
    torch.manual_seed(59)
    q = pack_rows(torch.randn(3, 32, 128, device="cuda"), "index")
    keys = pack_rows(torch.randn(3, 257, 128, device="cuda"), "index")
    weights = torch.rand(3, 32, device="cuda")
    lengths = torch.tensor([257, 129, 0], device="cuda", dtype=torch.int32)
    positions = torch.arange(257, device="cuda").expand(3, -1).clone()
    candidates = torch.empty((3, 96), dtype=torch.int64, device="cuda")

    def hook(scores):
        blocks = (
            torch.nn.functional.pad(scores, (0, 31), value=-torch.inf).reshape(3, 9, 32).amax(-1)
        )
        latest = (lengths - 1) // 32
        blocks = blocks.masked_fill(
            torch.arange(9, device="cuda")[None, :] == latest[:, None], torch.inf
        )
        selected = torch.empty((3, 3), dtype=torch.int32, device="cuda")
        indexer.select_prepared_scores(
            blocks, selected, torch.zeros_like(lengths), torch.full_like(lengths, 9)
        )
        values = selected.long()[:, :, None] * 32 + torch.arange(32, device="cuda")
        valid = (values < lengths[:, None, None]) & (values < 257)
        valid = valid & (blocks.gather(1, selected.long())[:, :, None] > -torch.inf)
        candidates.copy_(torch.where(valid, values, -1).flatten(1))

    def run():
        return _prepared(indexer, q, keys, weights, lengths, positions, hook)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for visible in ([257, 129, 0], [0, 17, 256], [256, 0, 3]):
        lengths.copy_(torch.tensor(visible, device="cuda", dtype=torch.int32))
        weights.mul_(0.9)
        keys[..., :64].bitwise_xor_(0x88)
        graph.replay()
        scores = _reference_index_scores(
            unpack_rows(q, 128, "index", torch.float32),
            unpack_rows(keys, 128, "index", torch.float32),
            weights,
        )
        expected_candidates = _reference_select_candidate_positions(scores, lengths, 3, 32)
        expected = _reference_select_topk_positions(scores, positions, lengths, 32)
        torch.testing.assert_close(candidates.sort(-1).values, expected_candidates.sort(-1).values)
        torch.testing.assert_close(actual, expected)


def test_prepared_bmm_fallback(monkeypatch):
    indexer = _indexer(8)
    torch.manual_seed(7)
    q = pack_rows(torch.randn(2, 8, 128, device="cuda"), "index")
    k = pack_rows(torch.randn(2, 33, 128, device="cuda"), "index")
    weights = torch.rand(2, 8, device="cuda")
    lengths = torch.tensor([33, 3], dtype=torch.int32, device="cuda")
    positions = torch.arange(33, device="cuda").expand(2, -1)
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.dsa.indexer.get_sm_version", lambda: 90
    )
    scores = []
    actual = _prepared(indexer, q, k, weights, lengths, positions, scores.append)
    dots = torch.bmm(unpack_rows(q, 128, "index"), unpack_rows(k, 128, "index").transpose(1, 2))
    expected_scores = (dots.relu() * weights.bfloat16().unsqueeze(-1)).sum(1).float()
    expected = _reference_select_topk_positions(expected_scores, positions, lengths, 32)
    torch.testing.assert_close(actual, expected)
    for _ in range(3):
        _prepared(indexer, q, k, weights, lengths, positions)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = _prepared(indexer, q, k, weights, lengths, positions)
    lengths.copy_(torch.tensor([0, 17], dtype=torch.int32, device="cuda"))
    k[..., :64].bitwise_xor_(0x88)
    graph.replay()
    dots = torch.bmm(unpack_rows(q, 128, "index"), unpack_rows(k, 128, "index").transpose(1, 2))
    expected_scores = (dots.relu() * weights.bfloat16().unsqueeze(-1)).sum(1).float()
    expected = _reference_select_topk_positions(expected_scores, positions, lengths, 32)
    torch.testing.assert_close(replayed, expected)


@pytest.mark.parametrize("use_fp4", [False, True])
def test_prepared_default_row_local(use_fp4):
    indexer = _indexer()
    indexer.use_fp4 = use_fp4
    torch.manual_seed(913)
    q = torch.randn(2, 32, 128, device="cuda")
    k = torch.randn(258, 128, device="cuda")
    weights = torch.rand(2, 32, device="cuda")
    if use_fp4:
        qr, kr = pack_rows(q, "index"), pack_rows(k, "index")
        qd, kd = (
            qr[..., :64].contiguous().view(torch.int8),
            kr[:, :64].contiguous().view(torch.int8),
        )
        qs, ks = (
            qr[..., 64:].contiguous().view(torch.int32),
            kr[:, 64:].contiguous().view(torch.int32),
        )
        q, k = (
            unpack_rows(qr, 128, "index", torch.float32),
            unpack_rows(kr, 128, "index", torch.float32),
        )
    else:
        qd, kd = q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn)
        qs, ks = None, torch.ones(258, device="cuda")
        q, k = qd.float(), kd.float()
    starts = torch.tensor([0, 129], dtype=torch.int32, device="cuda")
    ends = starts + 129
    out = torch.empty((2, 32), dtype=torch.int32, device="cuda")
    indexer.forward_prepared(qd, kd, ks, weights, starts, ends, out, qs)
    scores = _reference_index_scores(q, k, weights)
    expected = torch.stack((scores[0, :129].topk(32).indices, scores[1, 129:].topk(32).indices))
    torch.testing.assert_close(out.long().sort(-1).values, expected.sort(-1).values)


def test_prepared_cpu_reference():
    indexer = _indexer(8, 4)
    q = pack_rows(torch.randn(2, 8, 128), "index")
    k = pack_rows(torch.randn(2, 17, 128), "index")
    weights = torch.rand(2, 8)
    lengths = torch.tensor([17, 3], dtype=torch.int32)
    positions = torch.arange(17).expand(2, -1)
    actual = _prepared(indexer, q, k, weights, lengths, positions)
    scores = _reference_index_scores(
        unpack_rows(q, 128, "index", torch.float32),
        unpack_rows(k, 128, "index", torch.float32),
        weights,
    )
    expected = _reference_select_topk_positions(scores, positions, lengths, 4)
    torch.testing.assert_close(actual, expected)


def test_csa2_specializes_shared_indexer():
    layout = CSA2Layout(
        (1, 1),
        (0,),
        (0, 1),
        candidate_source_layer_id=0,
        candidate_topk_blocks=2,
        candidate_block_size=4,
        index_topk=4,
    )
    for layer in (0, 1):
        indexer = CSA2Indexer(layout, layer, 8, 128)
        assert isinstance(indexer, Indexer)
        assert CSA2Indexer.forward_prepared is Indexer.forward_prepared
        assert CSA2Indexer.select_prepared_scores is Indexer.select_prepared_scores
        assert list(indexer.parameters()) == []
        assert isinstance(indexer.top_k, TopK)
    indexer = CSA2Indexer(layout, 0, 8, 128)
    scores = torch.tensor(
        [
            [99.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1],
            [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
        ]
    )
    visible = torch.tensor([9, 0], dtype=torch.int32)
    scores[1].fill_(-torch.inf)
    results = {}
    indexer._publish_candidates(scores[:1], visible[:1], results, 0, 2)
    indexer._publish_candidates(scores[1:], visible[1:], results, 1, 2)
    expected = _reference_select_candidate_positions(scores, visible, 2, 4)
    torch.testing.assert_close(results[0].sort(-1).values, expected.sort(-1).values)
    with pytest.raises(ValueError, match="full-query shape"):
        indexer._publish_candidates(scores[1:], visible[1:], {}, 1, 2)
