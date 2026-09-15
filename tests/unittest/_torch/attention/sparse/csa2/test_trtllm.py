# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CSA2 hardware policy and reuse of the native DSV4 dynamic sparse path."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
    CSA2Params,
    select_csa2_backend,
)


@pytest.mark.parametrize(
    "sm,backend",
    [
        (90, "flash_mla"),
        (100, "trtllm"),
        (103, "trtllm"),
        (107, "trtllm"),
        (120, "flashinfer"),
        (121, "flashinfer"),
    ],
)
def test_hardware_policy(sm, backend):
    assert select_csa2_backend(sm) == backend


@pytest.mark.parametrize("sm", [80, 89, 110, 130])
def test_unsupported_hardware(sm):
    with pytest.raises(ValueError, match="unsupported"):
        select_csa2_backend(sm)


def _reference(q, swa, extra, swa_valid, extra_valid, sink):
    kv, valid = swa, swa_valid
    if extra is not None:
        kv = torch.cat((kv, extra), dim=1)
        valid = torch.cat((valid, extra_valid), dim=1)
    kv = torch.where(valid[..., None], kv, 0)
    scores = torch.einsum("qhd,qkd->qhk", q.float(), kv.float()) * q.shape[-1] ** -0.5
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    probs = torch.cat((scores, sink[None, :, None].expand(q.shape[0], -1, -1)), -1).softmax(-1)[
        ..., :-1
    ]
    return torch.einsum("qhk,qkd->qhd", probs, kv.float()).to(q.dtype)


def _backend(heads, layer_idx=20, layout=None, compute_backend="auto", use_packed=False):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import get_csa2_backend
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention
    from tensorrt_llm._torch.attention.backends.utils import create_attention, get_attention_backend
    from tensorrt_llm._utils import is_sm_100f

    if compute_backend in ("auto", "trtllm") and not is_sm_100f():
        pytest.skip("TRTLLM dynamic sparse MLA requires SM100-family")
    params = CSA2Params(
        max_query_tokens=16,
        layout=layout,
        compute_backend=compute_backend,
        use_packed_sparse_attention=use_packed,
    )
    assert get_attention_backend("TRTLLM", params) is get_csa2_backend(params)
    attn = create_attention(
        "TRTLLM",
        layer_idx,
        heads,
        512,
        num_kv_heads=1,
        is_mla_enable=True,
        q_lora_rank=1280,
        kv_lora_rank=448,
        qk_nope_head_dim=448,
        qk_rope_head_dim=64,
        v_head_dim=512,
        rope_append=False,
        sparse_params=params,
    )
    assert isinstance(attn, TrtllmAttention)
    assert type(attn).forward is TrtllmAttention.forward
    assert "forward_selected" not in type(attn).__dict__
    return attn


def _helper_forward(attn, q, metadata, args):
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.backend import (
        CSA2FlashInfer,
        CSA2FlashMLA,
    )

    if attn.sparse_params.use_packed_sparse_attention:
        return attn.forward_packed(q, metadata, args)
    if not hasattr(attn, "_test_flash_helper"):
        helper = CSA2FlashMLA if attn.compute_backend == "flash_mla" else CSA2FlashInfer
        attn._test_flash_helper = helper(attn)
    compute = (
        attn._test_flash_helper.forward_context
        if args.attention_input_type == AttentionInputType.context_only
        else attn._test_flash_helper.forward_generation
    )
    return compute(q, metadata, args)


def _inputs(q, swa, extra, swa_valid, extra_valid, sink):
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2BackendForwardArgs
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    swa_ids = torch.arange(swa.shape[0] * swa.shape[1], device=q.device).reshape_as(swa_valid)
    main_ids = (
        None
        if extra is None
        else torch.arange(extra.shape[0] * extra.shape[1], device=q.device).reshape_as(extra_valid)
    )
    inputs = CSA2BackendForwardArgs(
        swa_pool=pack_rows(swa.flatten(0, 1), "swa"),
        swa_indices=torch.where(swa_valid, swa_ids, -1),
        main_pool=None if extra is None else pack_rows(extra.flatten(0, 1), "main"),
        topk_indices=None if extra is None else torch.where(extra_valid, main_ids, -1),
    )
    return AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        attention_sinks=sink,
        sparse_backend_args=inputs,
    )


def _decoded_reference(q, args, swa_valid, extra_valid, sink):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    inputs = args.sparse_backend_args
    swa = unpack_rows(inputs.swa_pool, 512, "swa").reshape(q.shape[0], -1, 512)
    main = (
        None
        if inputs.main_pool is None
        else unpack_rows(inputs.main_pool, 512, "main").reshape(q.shape[0], -1, 512)
    )
    return _reference(q, swa, main, swa_valid, extra_valid, sink)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("extra_width", [0, 17, 512])
@torch.inference_mode()
def test_native_dual_pool(heads, extra_width, monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

    calls = []
    original = FallbackFmha.forward

    def record(attn, *args, **kwargs):
        calls.append(attn.attn.sparse_params.algorithm)
        return original(attn, *args, **kwargs)

    monkeypatch.setattr(FallbackFmha, "forward", record)
    torch.manual_seed(451)
    attn = _backend(heads)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(3, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(3, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 2:] = False
    swa_valid[1] = False
    extra = (
        torch.randn(3, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(3, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[0, 1::2] = False  # holes before later valid entries
        extra_valid[1] = False
    sink = torch.randn(heads, device="cuda")
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    metadata = CSA2TrtllmMetadata.for_query_tile(q, extra_width)
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    out = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    torch.cuda.synchronize()
    assert calls == ["csa2"]
    torch.testing.assert_close(
        out, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_trtllm_graph_replay_resets_sparse_state():
    attn = _backend(64)
    q = torch.randn(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    extra = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    extra_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    sink = torch.zeros(64, device="cuda")

    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import pack_rows

    frame = CSA2TrtllmMetadata.for_query_tile(q, 17)

    def run():
        frame.is_cuda_graph = torch.cuda.is_current_stream_capturing()
        args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
        return attn.forward(q.flatten(1), None, None, frame, forward_args=args).view_as(q)

    for _ in range(3):
        run()
    pointers = frame.pool_pointers.clone()
    workspace_ptr = frame.workspace.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    # Larger eager queries use independent caller-owned metadata. No recursive
    # warmup or backend-private frame allocation is needed.
    large_q = q.repeat(8, 1, 1)
    large_frame = CSA2TrtllmMetadata.for_query_tile(large_q, 17)
    large_args = _inputs(
        large_q,
        swa.repeat(8, 1, 1),
        extra.repeat(8, 1, 1),
        swa_valid.repeat(8, 1),
        extra_valid.repeat(8, 1),
        sink,
    )
    attn.forward(large_q.flatten(1), None, None, large_frame, forward_args=large_args)
    for width in (17, 0, 3, 17):
        extra_valid.zero_()
        extra_valid[:, :width] = True
        swa_valid[0, 1:] = width != 0
        q.mul_(-1)
        extra.mul_(-1)
        graph.replay()
        # Quantize/dequantize independently of the prediction hook's gathering.
        from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

        swa_ref = unpack_rows(pack_rows(swa, "swa"), 512, "swa")
        extra_ref = unpack_rows(pack_rows(extra, "main"), 512, "main")
        torch.testing.assert_close(
            output,
            _reference(q, swa_ref, extra_ref, swa_valid, extra_valid, sink),
            atol=0.03,
            rtol=0.03,
        )
        torch.testing.assert_close(frame.pool_pointers, pointers, atol=0, rtol=0)
        assert frame.workspace.data_ptr() == workspace_ptr
        assert large_frame.workspace.data_ptr() != workspace_ptr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_standard_backend_multiple_tiles_and_reuse(monkeypatch):
    from types import SimpleNamespace

    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
        CSA2CacheManager,
        CSA2CacheRole,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.indexer import CSA2Indexer
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import (
        CSA2BackendForwardArgs,
        CSA2ForwardState,
        CSA2Layout,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import gather_rows
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    predictions = []
    original_predict = CSA2Indexer.sparse_attn_indexer

    def predict(indexer, metadata, hidden_states, *args, **kwargs):
        predictions.append((indexer.layer_idx, hidden_states.shape[0]))
        return original_predict(indexer, metadata, hidden_states, *args, **kwargs)

    monkeypatch.setattr(CSA2Indexer, "sparse_attn_indexer", predict)
    torch.manual_seed(419)
    layout = CSA2Layout((1, 1, 1), (0,), (0, 2), index_topk=4, window_size=4)
    count, heads = 19, 8
    manager = CSA2CacheManager(
        KvCacheConfig(max_gpu_total_bytes=64 << 20, host_cache_size=0),
        CacheType.SELFKONLY,
        num_layers=3,
        tokens_per_block=128,
        max_seq_len=64,
        max_batch_size=1,
        max_num_tokens=count,
        mapping=Mapping(),
        vocab_size=8192,
        layout=layout,
    )
    try:
        request = manager._create_kv_cache(100, None, [])
        assert manager._resume_and_restore(100, request)
        assert request.resize(count)
        runtime = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=count, kv_cache_manager=manager
        )
        positions = torch.arange(count, device="cuda")
        q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
        swa = torch.randn(count, 512, device="cuda", dtype=torch.bfloat16)
        main = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16)
        index_k = torch.randn(6, 128, device="cuda", dtype=torch.bfloat16)
        index_q = torch.randn(count, 2, 128, device="cuda", dtype=torch.bfloat16)
        weights = torch.ones(count, 2, device="cuda", dtype=torch.bfloat16)
        sink = torch.randn(heads, device="cuda")
        runtime.reset_routing()
        global_base = manager.get_cache_indices(100, 0, CSA2CacheRole.GLOBAL)[0] * 128
        global_slots = global_base + torch.arange(6, device="cuda")
        runtime.csa2_token_requests = torch.zeros(count, dtype=torch.int64, device="cuda")
        runtime.csa2_request_query_ranges = ((0, count),)
        runtime.csa2_request_start_positions = (0,)
        runtime.csa2_num_context_requests = 1
        runtime.csa2_global_page_tables = {
            0: torch.tensor([[global_base // 128]], dtype=torch.int32, device="cuda")
        }
        runtime.csa2_global_page_sizes = {0: 128}
        runtime.csa2_global_max_positions = {0: 6}
        runtime.csa2_main_write_slots = {0: global_slots}
        runtime.csa2_kv_sources = {0: 0, 1: 0, 2: 0}
        runtime.csa2_swa_indices = {}
        runtime.csa2_swa_write_slots = {}
        runtime.csa2_visible_lengths = {}
        for layer in range(3):
            swa_base = manager.get_cache_indices(100, layer, CSA2CacheRole.SWA)[0] * 128
            local = positions[:, None] - torch.arange(4, device="cuda")[None, :]
            swa_slots = torch.where(local >= 0, local + swa_base, -1)
            runtime.csa2_swa_indices[layer] = swa_slots
            runtime.csa2_swa_write_slots[layer] = positions + swa_base
            runtime.csa2_visible_lengths[layer] = positions.remainder(6) + 1
            args_dict = {}
            if layer != 1:
                args_dict.update(index_q=index_q * (-1 if layer else 1), index_weights=weights)
            if layer == 0:
                args_dict.update(main_kv=main, index_k=index_k)
            state = CSA2ForwardState(metadata=runtime, swa_kv=swa, **args_dict)
            backend = _backend(heads, layer, layout)
            outputs = []
            for start in range(0, count, 16):
                tile = q[start : start + 16]
                frame = runtime.get_query_tile_metadata(tile, layout.index_topk)
                args = AttentionForwardArgs(
                    attention_input_type=AttentionInputType.generation_only,
                    attention_sinks=sink,
                    sparse_backend_args=CSA2BackendForwardArgs(state=state, query_start=start),
                )
                outputs.append(
                    backend.forward(tile.flatten(1), None, None, frame, forward_args=args).view_as(
                        tile
                    )
                )
            actual = torch.cat(outputs)
            logical = runtime.csa2_indices[0 if layer == 1 else layer]
            slots = runtime.global_slot_tile(layer, 0, count, logical)
            swa_values = gather_rows(manager.get_swa_buffer(layer), swa_slots, 512, "swa")
            main_values = gather_rows(manager.get_main_buffer(0), slots, 512, "main")
            expected = _reference(q, swa_values, main_values, swa_slots >= 0, slots >= 0, sink)
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
            frame = runtime.get_query_tile_metadata(q[-3:], 4)
            assert frame.host_total_kv_lens.tolist() == [0, 3 * 256]
        assert predictions == [(0, count), (2, count)]
    finally:
        for request_id in list(manager.kv_cache_map):
            manager.free_resources(SimpleNamespace(py_request_id=request_id))
        manager.shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_shared_metadata_resets_global_to_swa_inputs():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    q = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    main_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    sink = torch.zeros(8, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17)
    first, second = _backend(8, 20), _backend(8, 21)
    args = _inputs(q, swa, main, swa_valid, main_valid, sink)
    first.forward(q.flatten(1), None, None, metadata, forward_args=args)
    assert args.sparse_runtime_params.aux_kv_cache_pool_ptr is not None
    # Reuse the runtime carrier and metadata across distinct layers, but remove
    # main selection. No stale main pointer or sparse length may survive.
    args.sparse_backend_args = _inputs(q, -swa, None, swa_valid, None, sink).sparse_backend_args
    args.output = None
    output = second.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    assert args.sparse_runtime_params.aux_kv_cache_pool_ptr is None
    torch.testing.assert_close(metadata.prepared_lens, torch.full_like(metadata.prepared_lens, 4))
    torch.testing.assert_close(
        output, _decoded_reference(q, args, swa_valid, None, sink), atol=0.03, rtol=0.03
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_cold_metadata_rejects_capture():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8)
    q = torch.zeros(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(2, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(2, 1, device="cuda", dtype=torch.bool)
    sink = torch.zeros(8, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    args = _inputs(q, swa, None, valid, None, sink)
    metadata.is_cuda_graph = True
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="Warm up CSA2 metadata"):
        with torch.cuda.graph(graph):
            attn.forward(q.flatten(1), None, None, metadata, forward_args=args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation", ["flash_mla", "flashinfer"])
@pytest.mark.parametrize("context", [False, True])
@torch.inference_mode()
def test_alternative_helper_dispatch(implementation, context, monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    def unexpected_fallback(*args, **kwargs):
        raise AssertionError("Selected CSA2 implementation fell through to native fallback")

    monkeypatch.setattr(FallbackFmha, "forward", unexpected_fallback)
    attn = _backend(8, compute_backend=implementation)
    q = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(2, 4, 512, device="cuda", dtype=torch.bfloat16)
    main = torch.randn(2, 17, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(2, 4, device="cuda", dtype=torch.bool)
    main_valid = torch.ones(2, 17, device="cuda", dtype=torch.bool)
    main_valid[0, ::2] = False
    swa_valid[1] = False
    main_valid[1] = False
    sink = torch.linspace(-2, 2, 8, device="cuda")
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 17)
    args = _inputs(q, swa, main, swa_valid, main_valid, sink)
    if context:
        from tensorrt_llm._torch.attention.backends.interface import AttentionInputType

        metadata._bind_context_tile([q.shape[0]])
        args.attention_input_type = AttentionInputType.context_only
    args.output = torch.empty_like(q).flatten(1)
    actual = _helper_forward(attn, q.flatten(1), metadata, args).view_as(q)
    assert actual.data_ptr() == args.output.data_ptr()
    torch.testing.assert_close(
        actual, _decoded_reference(q, args, swa_valid, main_valid, sink), atol=0.03, rtol=0.03
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "option", ["custom_mask", "out_scale", "output_sf", "output_shape", "output_dtype"]
)
@torch.inference_mode()
def test_alternative_backend_rejects_unsupported_options(option):
    from tensorrt_llm._torch.attention.backends.interface import CustomAttentionMask
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8, compute_backend="flashinfer")
    q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    args = _inputs(q, swa, None, valid, None, torch.zeros(8, device="cuda"))
    if option == "custom_mask":
        args.attention_mask = CustomAttentionMask.CUSTOM
    elif option == "out_scale":
        args.out_scale = torch.ones(1, device="cuda")
    elif option == "output_shape":
        args.output = torch.empty(2, q.shape[1] * 512, device=q.device, dtype=q.dtype)
    elif option == "output_dtype":
        args.output = torch.empty_like(q, dtype=torch.float32).flatten(1)
    else:
        args.output = torch.empty_like(q).flatten(1)
        args.output_sf = torch.empty(1, device="cuda", dtype=torch.uint8)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    with pytest.raises(ValueError, match="(do not support.*mask/output format|output must match)"):
        _helper_forward(attn, q.flatten(1), metadata, args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_alternative_helper_revalidates_output_scale():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(8, compute_backend="flashinfer")
    q = torch.zeros(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    args = _inputs(q, swa, None, valid, None, torch.zeros(8, device="cuda"))
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    _helper_forward(attn, q.flatten(1), metadata, args)
    torch.cuda.synchronize()
    args.out_scale = torch.ones(1, device="cuda")
    with pytest.raises(ValueError, match="do not support.*mask/output format"):
        _helper_forward(attn, q.flatten(1), metadata, args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("extra_width", [0, 17, 512])
@pytest.mark.parametrize("prefix", [0, 1024])
@pytest.mark.parametrize("tile_start", [0, 2])
@torch.inference_mode()
def test_native_context_preserves_real_query_groups(extra_width, prefix, tile_start, monkeypatch):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(459)
    heads, count = 64, 5
    attn = _backend(heads)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 1::2] = False
    swa_valid[2] = False
    extra = (
        torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(count, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[1, ::3] = False
        extra_valid[2] = False
    sink = torch.randn(heads, device="cuda")
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count + tile_start)
    source._num_contexts = 2
    source._num_ctx_tokens = source._num_tokens = count + tile_start
    source.csa2_num_context_requests = 2
    source.csa2_request_query_ranges = ((0, 3 + tile_start), (3 + tile_start, count + tile_start))
    source.csa2_request_start_positions = (prefix, prefix * 2)
    source.csa2_request_lengths = (3 + tile_start, 2)
    metadata = source.get_query_tile_metadata(q, extra_width, query_start=tile_start)
    assert metadata.num_contexts == 2
    assert metadata.num_ctx_tokens == count
    assert metadata.num_generations == 0
    torch.testing.assert_close(
        metadata.cu_q_seqlens.cpu(), torch.tensor([0, 3, 5], dtype=torch.int32)
    )
    # Native context's causal mask sees a virtual compacted K domain. Every
    # physical selected row was already filtered by source logical causality.
    topk = metadata.num_sparse_topk
    torch.testing.assert_close(
        metadata.kv_lens_runtime.cpu(), torch.tensor([topk + 2, topk + 1], dtype=torch.int32)
    )
    torch.testing.assert_close(
        metadata.cu_kv_seqlens.cpu(), torch.tensor([0, topk + 2, 2 * topk + 3], dtype=torch.int32)
    )
    calls = []
    original = FallbackFmha.forward

    def record(provider, query, key, value, meta, forward_args):
        calls.append((meta.num_contexts, meta.num_ctx_tokens, forward_args.attention_input_type))
        assert forward_args.latent_cache is None
        return original(provider, query, key, value, meta, forward_args)

    monkeypatch.setattr(FallbackFmha, "forward", record)
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    args.attention_input_type = AttentionInputType.context_only
    output = attn.forward(q.flatten(1), None, None, metadata, forward_args=args).view_as(q)
    torch.cuda.synchronize()
    assert calls == [(2, count, AttentionInputType.context_only)]
    torch.testing.assert_close(
        output, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )

    monkeypatch.setattr(FallbackFmha, "forward", original)
    generation = CSA2TrtllmMetadata.for_query_tile(q, extra_width)
    generation_args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    reference_generation = attn.forward(
        q.flatten(1), None, None, generation, forward_args=generation_args
    ).view_as(q)
    torch.testing.assert_close(output, reference_generation, atol=0.03, rtol=0.03)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("extra_width", [0, 17])
@torch.inference_mode()
def test_packed_helper_skips_bf16_staging(extra_width, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")

    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(460)
    heads, count = 32, 3
    attn = _backend(heads, use_packed=True)
    assert not any(isinstance(provider, FallbackFmha) for provider in attn._fmha_manager.fmha_libs)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    swa_valid[0, 1::2] = False
    swa_valid[1] = False
    extra = (
        torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
        if extra_width
        else None
    )
    extra_valid = (
        torch.ones(count, extra_width, device="cuda", dtype=torch.bool) if extra_width else None
    )
    if extra_valid is not None:
        extra_valid[1] = False
    sink = torch.randn(heads, device="cuda")
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    if extra is not None:
        packed = args.sparse_backend_args.main_pool
        records = torch.full((packed.shape[0], 356), 73, dtype=torch.uint8, device="cuda")
        records[:, :288].copy_(packed)
        args.sparse_backend_args.main_pool = records[:, :288]
    metadata = CSA2TrtllmMetadata.for_query_tile(q, extra_width)
    monkeypatch.setattr(
        metadata, "stage_selected", lambda *a: pytest.fail("Packed FMHA staged BF16 rows")
    )
    output = _helper_forward(attn, q.flatten(1), metadata, args).view_as(q)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
    )
    if extra is not None:
        assert bool((records[:, 288:] == 73).all())
    # Reusing the selected library must revalidate unsupported output options.
    args.out_scale = torch.ones(1, device="cuda")
    with pytest.raises(
        ValueError, match="(Packed CSA2 does not support|do not support.*mask/output format)"
    ):
        _helper_forward(attn, q.flatten(1), metadata, args)

    args.out_scale = None
    args.sparse_backend_args.output_position_ids = torch.arange(
        count, device="cuda", dtype=torch.int32
    )
    with pytest.raises(
        ValueError, match="(Packed CSA2 does not support|do not support.*mask/output format)"
    ):
        _helper_forward(attn, q.flatten(1), metadata, args)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,packed", [("trtllm", False), ("flashinfer", False), ("trtllm", True)]
)
def test_csa2_quant_update_rebuilds_local_provider_policy(implementation, packed):
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

    attn = _backend(32, compute_backend=implementation, use_packed=packed)
    native = implementation == "trtllm" and not packed
    first = attn._fmha_manager
    assert bool(first.fmha_libs) == native
    assert all(type(provider) is FallbackFmha for provider in first.fmha_libs)
    attn.update_quant_config(attn.quant_config)
    assert attn._fmha_manager is not first
    assert bool(attn._fmha_manager.fmha_libs) == native
    assert all(type(provider) is FallbackFmha for provider in attn._fmha_manager.fmha_libs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_backend_rejects_disabled_fallback(monkeypatch):
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    with pytest.raises(ValueError, match="requires the FallbackFmha"):
        _backend(32, compute_backend="trtllm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_context_graph_uses_fixed_generation_frame():
    from tensorrt_llm._torch.attention.backends.interface import AttentionInputType
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(461)
    heads, count, extra_width = 64, 5, 17
    attn = _backend(heads)
    q = torch.randn(count, heads, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.randn(count, 128, 512, device="cuda", dtype=torch.bfloat16)
    extra = torch.randn(count, extra_width, 512, device="cuda", dtype=torch.bfloat16)
    swa_valid = torch.ones(count, 128, device="cuda", dtype=torch.bool)
    extra_valid = torch.ones(count, extra_width, device="cuda", dtype=torch.bool)
    sink = torch.randn(heads, device="cuda")
    source = CSA2TrtllmMetadata(max_num_requests=2, max_num_tokens=count)
    source._num_contexts = 2
    source._num_ctx_tokens = source._num_tokens = count
    source.csa2_num_context_requests = 2
    source.csa2_request_query_ranges = ((0, 3), (3, 5))
    source.csa2_request_start_positions = (1024, 2048)
    source.csa2_request_lengths = (3, 2)
    source.is_cuda_graph = True
    metadata = source.get_query_tile_metadata(q, extra_width, query_start=0)
    assert metadata.num_contexts == 0
    assert metadata.num_generations == count
    args = _inputs(q, swa, extra, swa_valid, extra_valid, sink)
    args.attention_input_type = (
        AttentionInputType.context_only
        if metadata.num_contexts
        else AttentionInputType.generation_only
    )
    for _ in range(3):
        attn.forward(q.flatten(1), None, None, metadata, forward_args=args)
    pointers = (
        metadata.swa_pool.data_ptr(),
        metadata.extra_pool.data_ptr(),
        metadata.workspace.data_ptr(),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = source.get_query_tile_metadata(q, extra_width, query_start=0)
        assert captured is metadata
        output = attn.forward(q.flatten(1), None, None, captured, forward_args=args).view_as(q)
    swa_ids = torch.arange(count * 128, device="cuda").reshape(count, 128)
    main_ids = torch.arange(count * extra_width, device="cuda").reshape(count, extra_width)
    for active in (1, 0, 17):
        q.neg_()
        swa_valid.fill_(True)
        swa_valid[0, 1::2] = False
        swa_valid[1] = False
        extra_valid.copy_(torch.arange(extra_width, device="cuda")[None, :] < active)
        extra_valid[1] = False
        args.sparse_backend_args.swa_indices.copy_(torch.where(swa_valid, swa_ids, -1))
        args.sparse_backend_args.topk_indices.copy_(torch.where(extra_valid, main_ids, -1))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output, _decoded_reference(q, args, swa_valid, extra_valid, sink), atol=0.03, rtol=0.03
        )
        assert pointers == (
            metadata.swa_pool.data_ptr(),
            metadata.extra_pool.data_ptr(),
            metadata.workspace.data_ptr(),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("implementation,packed", [("flashinfer", False), ("trtllm", True)])
@torch.inference_mode()
def test_direct_compute_rejects_native_forward_misuse(implementation, packed):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    attn = _backend(32, compute_backend=implementation, use_packed=packed)
    q = torch.zeros(1, 32, 512, device="cuda", dtype=torch.bfloat16)
    values = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
    valid = torch.ones(1, 1, device="cuda", dtype=torch.bool)
    metadata = CSA2TrtllmMetadata.for_query_tile(q, 0)
    args = _inputs(q, values, None, valid, None, torch.zeros(32, device="cuda"))
    with pytest.raises(ValueError, match="explicit module helper"):
        attn.forward(q.flatten(1), None, None, metadata, forward_args=args)
