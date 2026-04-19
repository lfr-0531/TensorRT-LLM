"""Tests for KV cache token estimation in KvCacheCreator._get_token_num_for_estimation.

Guards the ADP (Attention Data Parallelism) cache-block reduction: when
enable_attention_dp is True and tp_size > 1, _create_dummy_context_requests
produces tp_size duplicate requests, but the scheduler distributes them
1-per-rank.  Each rank's KV cache therefore only needs capacity for its own
share, not all copies.
"""

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_request(num_input_tokens, beam_width=1):
    """Create a mock request with the fields _get_token_num_for_estimation reads."""
    req = Mock()
    req.input_token_ids = list(range(num_input_tokens))
    req.sampling_config.beam_width = beam_width
    return req


def _make_creator(
    tokens_per_block,
    dummy_reqs,
    enable_attention_dp,
    tp_size,
    batch_size=1,
    model_max_seq_len=1,
    max_cuda_graph_batch_size=1,
    layer_types=None,
    max_attention_window=None,
):
    """Build a minimal KvCacheCreator (bypasses __init__) wired up for
    _get_token_num_for_estimation only."""
    c = object.__new__(KvCacheCreator)

    c._tokens_per_block = tokens_per_block
    c._net_max_seq_len = 2048
    c._speculative_config = None
    c._dummy_reqs = dummy_reqs

    c._mapping = Mock(enable_attention_dp=enable_attention_dp, tp_size=tp_size, cp_config={})

    c._llm_args = Mock(disable_overlap_scheduler=True)

    pretrained = Mock()
    # spec=False so attribute access doesn't accept arbitrary fields; set only
    # the ones the production path reads.
    pretrained.layer_types = layer_types

    model_config = Mock()
    model_config.pretrained_config = pretrained

    c._model_engine = Mock(
        batch_size=batch_size,
        max_seq_len=model_max_seq_len,
        _max_cuda_graph_batch_size=max_cuda_graph_batch_size,
    )
    c._model_engine.model = Mock(model_config=model_config)

    c._kv_cache_config = Mock(
        free_gpu_memory_fraction=0.9, max_attention_window=max_attention_window
    )

    return c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_gpu():
    """Stub out CUDA memory queries and per-token KV size so the test runs on
    any machine and the memory cap never constrains the result."""
    huge = 100 * (1 << 30)
    with (
        patch("torch.cuda.mem_get_info", return_value=(huge, huge)),
        patch.object(KvCacheCreator, "_get_kv_size_per_token", return_value=1),
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_adp_reduces_blocks_to_per_rank_share():
    """With ADP + tp_size duplicated requests the result must equal a single
    rank's share, not the sum across all duplicates."""
    tpb = 64
    tp = 4
    n_in = 128  # ceil((128+1)/64) = 3 blocks per request

    baseline = _make_creator(tpb, [_make_mock_request(n_in)], enable_attention_dp=False, tp_size=1)
    adp = _make_creator(
        tpb, [_make_mock_request(n_in) for _ in range(tp)], enable_attention_dp=True, tp_size=tp
    )

    assert adp._get_token_num_for_estimation() == baseline._get_token_num_for_estimation()


def test_without_adp_all_blocks_counted():
    """Without ADP every request's blocks contribute to the total."""
    tpb = 64
    n_in = 128  # 3 blocks each
    n_reqs = 4

    c = _make_creator(
        tpb, [_make_mock_request(n_in) for _ in range(n_reqs)], enable_attention_dp=False, tp_size=1
    )

    # 4 reqs * 3 blocks * 64 tokens/block = 768
    assert c._get_token_num_for_estimation() == n_reqs * 3 * tpb


@pytest.mark.parametrize("tp_size", [2, 4, 8])
def test_adp_various_tp_sizes(tp_size):
    """ADP division must hold for several representative tp_size values."""
    tpb = 64
    n_in = 128  # 3 blocks per request

    c = _make_creator(
        tpb,
        [_make_mock_request(n_in) for _ in range(tp_size)],
        enable_attention_dp=True,
        tp_size=tp_size,
    )

    total = tp_size * 3
    expected_blocks = (total + tp_size - 1) // tp_size
    assert c._get_token_num_for_estimation() == expected_blocks * tpb


def test_regression_without_fix_would_overcount():
    """If the ADP ceil-division fix were removed, the returned
    value would be tp_size times too large.  This test guards that fix."""
    tpb = 64
    tp = 4
    n_in = 128

    c = _make_creator(
        tpb, [_make_mock_request(n_in) for _ in range(tp)], enable_attention_dp=True, tp_size=tp
    )

    result = c._get_token_num_for_estimation()

    correct = 3 * tpb  # 192  (per-rank share)
    wrong = tp * 3 * tpb  # 768  (all duplicates summed)
    assert result == correct
    assert result != wrong


# ---------------------------------------------------------------------------
# VSWA hybrid attention pool-group scaling (Gemma4 hybrid MMMU Pro hang fix)
# ---------------------------------------------------------------------------
#
# KVCacheManagerV2 creates one pool group per distinct attention-window size.
# The quota derived from max_tokens is split proportionally across pool
# groups, so each pool ends up with roughly num_cache_blocks / num_pool_groups
# blocks.  A single long-context request then overflows the full-attention
# pool and the scheduler livelocks on suspend/retry.  The fix scales
# num_cache_blocks by the number of distinct attention-window sizes inferred
# either from ``layer_types`` on the pretrained config (preferred) or from
# an explicit ``max_attention_window`` list on kv_cache_config (fallback).


def test_uniform_layer_types_no_scaling():
    """All-sliding or all-full layers stay a single pool group."""
    tpb = 32
    max_seq_len = 4096
    uniform = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=["sliding_attention"] * 26,
    )
    # num_pool_groups = 1 -> behaviour unchanged from legacy ADP-only case.
    baseline = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
    )
    assert uniform._get_token_num_for_estimation() == baseline._get_token_num_for_estimation()


def test_gemma4_hybrid_scales_by_num_pool_groups():
    """Gemma4 hybrid attention (mixed sliding/full layers) must scale the
    estimated block count by the number of distinct layer types.  Otherwise
    the per-pool quota is too small to hold a single max_seq_len request,
    which is the MMMU Pro livelock reproducer."""
    tpb = 32
    max_seq_len = 12288
    layer_types = ["sliding_attention"] * 28 + ["full_attention"] * 7
    assert len(set(layer_types)) == 2

    hybrid = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1), _make_mock_request(1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=layer_types,
    )
    uniform = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1), _make_mock_request(1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=["full_attention"] * 35,
    )

    hybrid_tokens = hybrid._get_token_num_for_estimation()
    uniform_tokens = uniform._get_token_num_for_estimation()
    assert hybrid_tokens == 2 * uniform_tokens, (
        f"Expected 2x scaling for 2 pool groups, got "
        f"hybrid={hybrid_tokens}, uniform={uniform_tokens}"
    )


def test_vswa_max_attention_window_fallback_scales():
    """When layer_types is absent but kv_cache_config.max_attention_window is
    a heterogeneous list (VSWA), we still scale by the number of distinct
    windows."""
    tpb = 32
    max_seq_len = 12288
    max_attention_window = [1024] * 28 + [max_seq_len] * 7
    assert len(set(max_attention_window)) == 2

    c = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=None,
        max_attention_window=max_attention_window,
    )
    uniform = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
    )
    assert c._get_token_num_for_estimation() == 2 * uniform._get_token_num_for_estimation()


def test_pool_scaling_prevents_mmmu_pro_underestimation():
    """Regression: with max_seq_len=12288 and max_num_tokens=12288 (MMMU Pro
    config), hybrid estimation must produce enough capacity to hold one full
    max_seq_len request per pool, with max_util_for_resume headroom."""
    tpb = 32
    max_seq_len = 12288
    layer_types = ["sliding_attention"] * 28 + ["full_attention"] * 7

    c = _make_creator(
        tpb,
        [_make_mock_request(max_seq_len - 1), _make_mock_request(1)],
        enable_attention_dp=False,
        tp_size=1,
        model_max_seq_len=max_seq_len,
        max_cuda_graph_batch_size=4,
        layer_types=layer_types,
    )

    total_tokens = c._get_token_num_for_estimation()
    per_pool_tokens = total_tokens // 2  # 2 pool groups
    # Must be enough to hold a max_seq_len request per pool.
    assert per_pool_tokens >= max_seq_len
