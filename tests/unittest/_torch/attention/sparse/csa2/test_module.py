# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual CSA2 module regression across incomplete/completed ratio-two groups."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rotate(x, positions, cos_sin, inverse=False):
    values = x.float().clone()
    rope_dim = cos_sin.shape[-1] * 2
    cos = cos_sin[positions.long(), 0].unsqueeze(1)
    sin = cos_sin[positions.long(), 1].unsqueeze(1)
    if inverse:
        sin = -sin
    even = values[..., -rope_dim::2].clone()
    odd = values[..., -rope_dim + 1 :: 2].clone()
    values[..., -rope_dim::2] = even * cos - odd * sin
    values[..., -rope_dim + 1 :: 2] = odd * cos + even * sin
    return values.to(x.dtype)


def _norm(x, weight, eps):
    return F.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps).to(x.dtype)


@pytest.fixture
def module_cache():
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    manager = CSA2CacheManager(
        KvCacheConfig(
            enable_block_reuse=False, max_gpu_total_bytes=128 << 20, enable_swa_scratch_reuse=True
        ),
        CacheType.SELFKONLY,
        num_layers=1,
        tokens_per_block=128,
        max_seq_len=128,
        max_batch_size=1,
        max_input_len=4,
        max_num_tokens=8,
        mapping=Mapping(),
        dtype=DataType.BF16,
        vocab_size=128,
        layout=CSA2Layout((2,), (0,), (0,), index_topk=2, window_size=4),
    )
    request = LlmRequest(
        request_id=41,
        max_new_tokens=4,
        input_tokens=[1, 2, 3, 4],
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    assert manager.prepare_context(request)
    assert manager.resize_context(request, request.context_chunk_size)
    manager._stream.synchronize()
    yield manager, request
    manager.free_resources(request)
    manager.shutdown()


@torch.inference_mode()
def _check_ratio2_module_partial_groups(monkeypatch, module_cache, overlap, implementation):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm.functional import PositionEmbeddingType

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.module.do_multi_stream", lambda: overlap
    )
    torch.manual_seed(818)
    manager, request = module_cache
    layout = manager.layout
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    with torch.device("cuda"):
        model = DeepseekV41Attention(
            layout,
            0,
            PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=RopeParams(dim=64, theta=160000, max_positions=8),
                is_neox=False,
            ),
            hidden_size=32,
            num_heads=8,
            head_dim=512,
            q_lora_rank=32,
            o_lora_rank=16,
            num_groups=2,
            index_heads=2,
            compute_backend=implementation,
            aux_stream=torch.cuda.Stream() if overlap else None,
        )
        for name, parameter in model.named_parameters():
            if "norm_weight" in name:
                parameter.fill_(1)
            else:
                parameter.normal_(std=0.1)
        hidden = torch.randn(4, 32, dtype=torch.bfloat16)
        for pool in (
            manager.get_swa_buffer(0),
            manager.get_main_buffer(0),
            manager.get_index_buffer(0),
        ):
            pool.zero_()

    preparation_calls = []
    original_prepare = model.backend.prepare_sparse_inputs

    def record_prepare(*args, **kwargs):
        preparation_calls.append(args[0].shape[0])
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(model.backend, "prepare_sparse_inputs", record_prepare)
    if implementation != "auto":
        from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha

        def unexpected_native(*args, **kwargs):
            pytest.fail("Module Flash helper fell through to native attention")

        monkeypatch.setattr(FallbackFmha, "forward", unexpected_native)
        assert model._flash_attention is not None
    native_rope = torch.ops.trtllm.mla_rope_inplace
    rope_rows = []

    def record_rope_rows(x, *args):
        # Observe the dispatch contract without replacing any CUDA math.
        rope_rows.append(x.shape[0])
        return native_rope(x, *args)

    monkeypatch.setattr(torch.ops.trtllm, "mla_rope_inplace", record_rope_rows)
    compressor = model.compressor
    cos_sin = model.rotary_emb.rotary_cos_sin
    for token in range(4):
        rope_rows.clear()
        completed = (token + 1) // 2
        new_rows = (token + 1) % 2 == 0
        metadata.seq_lens = torch.tensor([1], dtype=torch.int32, device="cpu")
        metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[token])
        metadata.prepare()
        positions = metadata.csa2_positions
        assert metadata.get_compression_batch(0).cu_compressed_lengths.tolist() == [
            0,
            int(new_rows),
        ]
        assert metadata.get_compression_batch(0).output_rows == 1
        old_main = manager.get_main_buffer(0).clone()
        old_index = manager.get_index_buffer(0).clone()
        # Fixed-capacity runtime padding does not generate an empty latent,
        # so exercise the zero-grid guard explicitly while recording native calls.
        empty = hidden.new_empty((0, 1, 512))
        assert model._rope(empty, positions[:0], 1) is empty
        assert not rope_rows
        actual = model(hidden[token : token + 1], positions, metadata)
        torch.cuda.synchronize()
        # Numerics alone can miss an empty native dispatch if a runtime/kernel
        # handles it as a no-op or clears its launch error in a later call.
        assert all(rows > 0 for rows in rope_rows), "Empty rows reached native RoPE"
        assert torch.isfinite(actual).all()
        if not new_rows:
            torch.testing.assert_close(manager.get_main_buffer(0), old_main, atol=0, rtol=0)
            torch.testing.assert_close(manager.get_index_buffer(0), old_index, atol=0, rtol=0)

        prefix_positions = torch.arange(token + 1, device="cuda", dtype=torch.int32)
        qr = _norm(
            F.linear(hidden[token : token + 1], model.wq_a.weight), model.q_norm_weight, model.eps
        )
        q = _rotate(F.linear(qr, model.wq_b.weight).reshape(1, 8, 512), positions, cos_sin)
        swa = _norm(
            F.linear(hidden[: token + 1], model.wkv.weight), model.kv_norm_weight, model.eps
        )
        swa = _rotate(swa.unsqueeze(1), prefix_positions, cos_sin).squeeze(1)
        selected = unpack_rows(pack_rows(swa, "swa"), 512, "swa")
        if completed:
            source = hidden[: completed * 2].float()
            values = F.linear(source, compressor.wkv.weight).reshape(completed, 2, 512)
            gates = F.linear(source, compressor.wgate.weight).reshape(completed, 2, 512)
            latent = (values * gates.softmax(1)).sum(1).bfloat16()
            latent = _norm(latent, compressor.norm_weight, model.eps)
            group_positions = torch.arange(completed, device="cuda", dtype=torch.int32) * 2
            main = _rotate(latent.unsqueeze(1), group_positions, cos_sin).squeeze(1)
            main = unpack_rows(pack_rows(main, "main"), 512, "main")
            selected = torch.cat((selected, main))
        # All visible global positions fit in top-k, so this reference needs
        # no indexer implementation. Quantization is covered independently.
        scores = torch.einsum("qhd,kd->qhk", q.float(), selected.float()) * 512**-0.5
        probs = torch.cat((scores, model.attn_sink[None, :, None]), -1).softmax(-1)[..., :-1]
        output = torch.einsum("qhk,kd->qhd", probs, selected.float()).bfloat16()
        output = _rotate(output, positions, cos_sin, inverse=True).reshape(1, 2, -1)
        projected = torch.einsum("qgd,grd->qgr", output, model.o_a_proj).flatten(1)
        expected = F.linear(projected, model.o_b_proj.weight)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        assert len(preparation_calls) == token + 1


@pytest.mark.parametrize("overlap", [False, True])
def test_ratio2_module_partial_groups(monkeypatch, module_cache, overlap):
    _check_ratio2_module_partial_groups(monkeypatch, module_cache, overlap, "auto")


@pytest.mark.parametrize("implementation", ["flash_mla", "flashinfer"])
def test_module_composed_flash_helper(monkeypatch, module_cache, implementation):
    _check_ratio2_module_partial_groups(monkeypatch, module_cache, False, implementation)


def _native_model(
    *,
    grouped_block=32,
    fused=False,
    aux_stream=None,
    layout=None,
    attention_heads=8,
    projection_quantization="mxfp8",
    sparse_options=None,
    mapping=None,
    num_groups=2,
    allreduce_strategy=None,
):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout, CSA2Params
    from tensorrt_llm.functional import PositionEmbeddingType

    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("Native MXFP8 projections require SM100/SM103")
    layout = layout or CSA2Layout((1,), (0,), (0,), index_topk=32)
    options = {"fuse_index_q": fused}
    options.update(sparse_options or {})
    with torch.device("cuda"):
        model = DeepseekV41Attention(
            layout,
            0,
            PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj,
                rope=RopeParams(dim=64, theta=160000, max_positions=32),
                is_neox=False,
            ),
            hidden_size=128,
            num_heads=attention_heads,
            q_lora_rank=128,
            o_lora_rank=128,
            num_groups=num_groups,
            index_heads=32,
            mapping=mapping,
            sparse_params=CSA2Params(layout=layout, **options),
            projection_quantization=projection_quantization,
            aux_stream=aux_stream,
            **(
                {"allreduce_strategy": allreduce_strategy} if allreduce_strategy is not None else {}
            ),
        )
    aliases = {
        "q_norm_weight": "q_norm.weight",
        "kv_norm_weight": "kv_norm.weight",
        "o_a_proj": "wo_a.weight",
        "o_b_proj.weight": "wo_b.weight",
        "compressor.norm_weight": "compressor.norm.weight",
        "index_wq_b.weight": "indexer.wq_b.weight",
        "index_weights_proj.weight": "indexer.weights_proj.weight",
        "index_wk.weight": "indexer.wk.weight",
        "index_k_norm_weight": "indexer.k_norm.weight",
    }
    weights = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".weight_scale"):
            continue
        key = aliases.get(name, name)
        shape = (
            (model.num_groups * model.o_lora_rank, parameter.shape[-1])
            if name == "o_a_proj"
            else parameter.shape
        )
        if name in ("wq_b.weight", "attn_sink"):
            shape = (parameter.shape[0] * model.mapping.tp_size, *parameter.shape[1:])
        elif name == "o_b_proj.weight":
            shape = (parameter.shape[0], parameter.shape[1] * model.mapping.tp_size)
        if name in (
            "wq_a.weight",
            "wq_b.weight",
            "wkv.weight",
            "index_wq_b.weight",
            "o_b_proj.weight",
            "o_a_proj",
        ):
            weights[key] = torch.randn(*shape, device="cuda").to(torch.float8_e4m3fn)
            block = grouped_block if name == "o_a_proj" else 32
            weights[key.removesuffix("weight") + "weight_scale_inv"] = torch.full(
                ((shape[0] + block - 1) // block, (shape[1] + block - 1) // block),
                123,
                dtype=torch.uint8,
                device="cuda",
            )
        elif "norm_weight" in name:
            weights[key] = torch.ones_like(parameter)
        else:
            weights[key] = torch.randn(shape, device=parameter.device, dtype=parameter.dtype) * 0.1
    model.load_hf_weights(weights)
    return model, weights


@torch.inference_mode()
def test_native_mxfp8_projection_checkpoint_and_atomic_load(monkeypatch):
    from tensorrt_llm._torch.modules.mxfp8_utils import dequant_mxfp8_weight, quant_bf16_to_mxfp8

    torch.manual_seed(943)
    model, weights = _native_model()
    calls = []
    native = torch.ops.trtllm.mxfp8_mxfp8_gemm

    def gemm(*args, **kwargs):
        calls.append(tuple(args[2].shape))
        return native(*args, **kwargs)

    monkeypatch.setattr(torch.ops.trtllm, "mxfp8_mxfp8_gemm", gemm)
    x = torch.randn(7, 128, dtype=torch.bfloat16, device="cuda")
    actual = model.wq_a(x)
    data, scale = quant_bf16_to_mxfp8(x)
    reference_input = dequant_mxfp8_weight(data, scale)
    reference_weight = weights["wq_a.weight"].float() / 16
    expected = F.linear(reference_input, reference_weight).bfloat16()
    assert calls == [(128, 128)]
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    torch.testing.assert_close(
        model.wq_a.weight.view(torch.uint8),
        weights["wq_a.weight"].view(torch.uint8),
        atol=0,
        rtol=0,
    )
    old = {name: value.clone() for name, value in model.named_parameters()}
    invalid = dict(weights)
    invalid["indexer.weights_proj.weight"] = torch.empty(1, device="cuda")
    with pytest.raises(ValueError, match="shape mismatch"):
        model.load_hf_weights(invalid)
    for name, parameter in model.named_parameters():
        actual_bytes = (
            parameter.view(torch.uint8) if parameter.dtype == torch.float8_e4m3fn else parameter
        )
        expected_bytes = (
            old[name].view(torch.uint8) if parameter.dtype == torch.float8_e4m3fn else old[name]
        )
        torch.testing.assert_close(actual_bytes, expected_bytes, atol=0, rtol=0)


@pytest.mark.parametrize("block", [32, 128])
@torch.inference_mode()
def test_native_grouped_output_paths_and_replay(monkeypatch, block):
    torch.manual_seed(944)
    model, weights = _native_model(grouped_block=block)
    names = ("mxfp8_mxfp8_gemm", "fused_inv_rope_fp8_quant_vllm_port", "cute_dsl_fp8_bmm_blackwell")
    calls = []
    for name in names:
        original = getattr(torch.ops.trtllm, name)

        def wrapped(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch.ops.trtllm, name, wrapped)
    values = torch.randn(3, 8, 512, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([1, 5, 11], dtype=torch.int32, device="cuda")
    for _ in range(3):
        model._project_output(values.clone(), positions)
    if block == 128:
        assert "fused_inv_rope_fp8_quant_vllm_port" in calls
        assert "cute_dsl_fp8_bmm_blackwell" in calls
    else:
        assert "fused_inv_rope_fp8_quant_vllm_port" not in calls
        assert len(calls) == 9  # two grouped GEMMs plus O-B each time
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = model._project_output(values.clone(), positions)
    for sign in (1.0, -1.0):
        values.mul_(sign)
        positions.add_(1)
        graph.replay()
        eager = model._project_output(values.clone(), positions)
        torch.testing.assert_close(captured, eager, atol=0, rtol=0)
        # A coarse independent BF16 oracle catches wrong groups/RoPE; exact
        # matched-precision native replay is checked above.
        rotated = _rotate(values, positions, model.rotary_emb.rotary_cos_sin, inverse=True)
        o_a = (weights["wo_a.weight"].float() / 16).reshape(2, 128, 2048)
        latent = torch.einsum("qgd,grd->qgr", rotated.float().reshape(3, 2, 2048), o_a).flatten(1)
        reference = F.linear(latent, weights["wo_b.weight"].float() / 16)
        error = (captured.float() - reference).norm() / reference.norm().clamp_min(1e-6)
        assert error < 0.12


@torch.inference_mode()
def test_fused_native_index_q_and_graph(monkeypatch):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )

    torch.manual_seed(945)
    model, _ = _native_model(fused=True)
    qr = torch.randn(7, 128, device="cuda", dtype=torch.bfloat16)
    qr *= torch.tensor([1e-7, 1e-4, 0.1, 1.0], device="cuda").repeat_interleave(32)
    # Diagonal projection removes reduction-rounding ambiguity while unequal
    # 32-channel blocks make a wrong 128-channel activation quantum observable.
    model.index_wq_b.weight.copy_(
        torch.eye(128, device="cuda").repeat(32, 1).to(torch.float8_e4m3fn)
    )
    positions = torch.zeros(7, device="cuda", dtype=torch.int32)
    unfused = model._rope(model.index_wq_b(qr).reshape(7, 32, 128), positions, 32)
    reference = unpack_rows(pack_rows(unfused, "index"), 128, "index")
    monkeypatch.setattr(
        model.index_wq_b,
        "forward",
        lambda *args: pytest.fail("Fused index Q must not call unfused Linear.forward"),
    )
    for _ in range(3):
        data, scales = model._project_index_q(qr, positions)
    assert data.shape == (7, 32, 64) and scales.shape == (7, 32)
    rows = torch.cat(
        (data.view(torch.uint8), scales.contiguous().view(torch.uint8).reshape(7, 32, 4)), dim=-1
    )
    actual = unpack_rows(rows, 128, "index")
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    native = torch.ops.trtllm.cute_dsl_fp8_indexer_q_gemm_rope_fp4_blackwell
    arguments = (
        qr,
        model.index_wq_b.weight,
        model.index_wq_b.weight_scale,
        positions,
        model.rotary_emb.rotary_cos_sin.view(-1, 64),
        model._fp8_alpha,
    )
    default_data, default_scales = native(*arguments)
    assert "activation_quantization_block_size" not in str(native.default._schema)
    # The nonuniform input makes default V4 128-block quantization observably
    # different from the explicit CSA2 32-block contract.
    assert not torch.equal(default_data.view(torch.uint8).reshape_as(data), data.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        data, scales = model._project_index_q(qr, positions)
    qr.neg_()
    positions.add_(1)
    graph.replay()
    expected_data, expected_scales = model._project_index_q(qr, positions)
    torch.testing.assert_close(
        data.view(torch.uint8), expected_data.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)


@torch.inference_mode()
def test_native_module_auxiliary_stream_and_graph(monkeypatch, module_cache):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    torch.manual_seed(946)
    manager, request = module_cache
    stream = torch.cuda.Stream()
    model, _ = _native_model(aux_stream=stream, layout=manager.layout)
    enabled = [False]
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.module.do_multi_stream",
        lambda: enabled[0],
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    metadata.seq_lens = torch.tensor([1], dtype=torch.int32, device="cpu")
    hidden = torch.randn(1, 128, device="cuda", dtype=torch.bfloat16)
    # Persist the first half of the compressor group, then repeatedly replace
    # its second half while comparing identical cache/history state.
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    metadata.prepare()
    model(hidden, metadata.csa2_positions, metadata)
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[1])
    metadata.num_contexts = 0
    metadata.is_cuda_graph = True
    metadata.prepare()
    positions = metadata.csa2_positions
    hidden.normal_()

    def run():
        metadata.reset_routing()
        return model(hidden, positions, metadata)

    serial = run()
    enabled[0] = True
    for _ in range(3):
        parallel = run()
    torch.testing.assert_close(parallel, serial, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for sign in (-1.0, 1.0):
        hidden.mul_(sign)
        graph.replay()
        actual = captured.clone()
        enabled[0] = False
        expected = run()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        enabled[0] = True
    torch.cuda.synchronize()


@pytest.mark.parametrize("overlap", [False, True])
@torch.inference_mode()
def test_native_module_fixed_context_graph(monkeypatch, module_cache, overlap):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    torch.manual_seed(947)
    manager, request = module_cache
    model, _ = _native_model(
        aux_stream=torch.cuda.Stream() if overlap else None, layout=manager.layout
    )
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.module.do_multi_stream", lambda: overlap
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    metadata.seq_lens = torch.tensor([4], dtype=torch.int32, device="cpu")
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    metadata.prepare()
    positions = metadata.csa2_positions
    hidden = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    phases = []
    original = model.backend.forward

    def observe(*args, **kwargs):
        phases.append(args[3].num_contexts)
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backend, "forward", observe)

    def run():
        metadata.reset_routing()
        return model(hidden, positions, metadata)

    baseline = run()
    assert phases[-1] == 1
    metadata.is_cuda_graph = True
    for _ in range(3):
        warmup = run()
    assert phases[-1] == 0
    torch.testing.assert_close(warmup, baseline, atol=0.03, rtol=0.03)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for sign in (-1.0, 1.0):
        hidden.mul_(sign)
        graph.replay()
        actual = captured.clone()
        expected = run()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()


@pytest.mark.parametrize("tokens", [17, 64])
@torch.inference_mode()
def test_large_index_q_uses_unfused_native_path(monkeypatch, tokens):
    torch.manual_seed(948)
    model, _ = _native_model(fused=True)
    qr = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32) % 16
    expected = model._rope(model.index_wq_b(qr).reshape(tokens, 32, 128), positions, 32)
    monkeypatch.setattr(
        "tensorrt_llm._torch.attention.backends.sparse.csa2.kernel.csa2_indexer_q_gemm_rope_fp4",
        lambda *args: pytest.fail("Large prefill must not force many small-M fused launches"),
    )
    actual, scales = model._project_index_q(qr, positions)
    assert scales is None
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("width", [128, 512, 1280])
@torch.inference_mode()
def test_rms_norm_native_single_rounding(monkeypatch, width):
    from tensorrt_llm._torch import custom_ops
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import _rms_norm

    calls = []
    original = custom_ops.flashinfer_rmsnorm

    def native(*args):
        calls.append(tuple(args[0].shape))
        return original(*args)

    monkeypatch.setattr(custom_ops, "flashinfer_rmsnorm", native)
    x = torch.tensor([1.0, 2.0], device="cuda", dtype=torch.bfloat16).repeat(3, width // 2)
    weight = torch.linspace(0.3, 2.1, width, device="cuda").bfloat16()
    normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
    expected = (normalized * weight.float()).bfloat16()
    extra_rounding = normalized.bfloat16() * weight
    assert bool((extra_rounding != expected).any())
    actual = _rms_norm(x, weight, 1e-20)
    assert calls == [(3, width)]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.manual_seed(949)
    x.normal_()
    reference = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20) * weight.float()
    ).bfloat16()
    # Parallel FP32 reduction/rsqrt can move a value across one BF16 boundary.
    torch.testing.assert_close(_rms_norm(x, weight, 1e-20), reference, atol=0, rtol=1 / 128)
    before = len(calls)
    cpu_x, cpu_weight = x.cpu(), weight.cpu()
    cpu_values = cpu_x.float()
    cpu_expected = (
        cpu_values
        * torch.rsqrt(cpu_values.square().mean(-1, keepdim=True) + 1e-20)
        * cpu_weight.float()
    ).bfloat16()
    torch.testing.assert_close(_rms_norm(cpu_x, cpu_weight, 1e-20), cpu_expected, atol=0, rtol=0)
    assert len(calls) == before
    assert _rms_norm(x[:0], weight, 1e-20).shape == (0, width)
    assert len(calls) == before


@pytest.mark.parametrize("quantization,block", [("bf16", 32), ("mxfp8", 32), ("mxfp8", 128)])
@torch.inference_mode()
def test_packed_output_inverse_rope_module_and_graph(
    monkeypatch, module_cache, quantization, block
):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 packed attention")

    from dataclasses import replace

    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.metadata import KVCacheParams

    torch.manual_seed(950)
    manager, request = module_cache
    model, _ = _native_model(
        layout=manager.layout,
        attention_heads=16,
        projection_quantization=quantization,
        grouped_block=block,
        sparse_options={"use_packed_sparse_attention": True, "fuse_packed_output_rope": True},
    )
    metadata = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=8, kv_cache_manager=manager)
    metadata.request_ids = [request.py_request_id]
    metadata.num_contexts = 1
    metadata.prompt_lens = [4]
    metadata.seq_lens = torch.tensor([4], dtype=torch.int32, device="cpu")
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
    metadata.is_cuda_graph = True
    metadata.prepare()
    hidden = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    positions = metadata.csa2_positions
    inverse_calls = []
    native_rope = torch.ops.trtllm.mla_rope_inplace

    def observe_rope(x, *args):
        if args[5]:
            inverse_calls.append(x.shape)
        return native_rope(x, *args)

    monkeypatch.setattr(torch.ops.trtllm, "mla_rope_inplace", observe_rope)
    projected = []
    original_projection = model._project_output

    def observe_projection(*args, **kwargs):
        projected.append(kwargs.get("inverse_rope_applied", False))
        return original_projection(*args, **kwargs)

    monkeypatch.setattr(model, "_project_output", observe_projection)

    def run():
        metadata.reset_routing()
        return model(hidden, positions, metadata)

    enabled = model.backend.sparse_params
    model.backend.sparse_params = replace(enabled, fuse_packed_output_rope=False)
    baseline = run()
    assert projected[-1] is False
    model.backend.sparse_params = enabled
    inverse_calls.clear()
    for _ in range(3):
        actual = run()
    expected_fusion = not (quantization == "mxfp8" and block == 128)
    assert projected[-1] == expected_fusion
    assert not inverse_calls  # reducer rotates, or grouped128 fused quantizer owns rotation
    torch.testing.assert_close(actual, baseline, atol=0.02, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for sign in (-1.0, 1.0):
        hidden.mul_(sign)
        graph.replay()
        actual = captured.clone()
        expected = run()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.cuda.synchronize()


@pytest.mark.parametrize("quantization", ["bf16", "mxfp8"])
@torch.inference_mode()
def test_two_rank_module_projection_parity(monkeypatch, quantization):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm._utils import mpi_rank, mpi_world_size
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    if mpi_world_size() != 2:
        pytest.skip("Requires exactly two MPI ranks, each bound to its own GPU")
    torch.manual_seed(951)
    layout = CSA2Layout((1,), (0,), (0,), index_topk=32, window_size=4)
    tp_mapping = Mapping(world_size=2, rank=mpi_rank(), tp_size=2)
    model, checkpoint = _native_model(
        layout=layout,
        attention_heads=16,
        num_groups=4,
        mapping=tp_mapping,
        projection_quantization=quantization,
    )
    reference, _ = _native_model(
        layout=layout,
        attention_heads=16,
        num_groups=4,
        mapping=Mapping(),
        projection_quantization=quantization,
    )
    reference.load_hf_weights(checkpoint)
    assert model.wq_b.out_features * 2 == reference.wq_b.out_features
    assert model.o_b_proj.in_features * 2 == reference.o_b_proj.in_features
    assert model.o_b_proj.all_reduce is not None
    reductions = []
    native_reduce = model.o_b_proj.all_reduce.forward

    def observe_reduction(*args, **kwargs):
        reductions.append(args[0].shape)
        return native_reduce(*args, **kwargs)

    monkeypatch.setattr(model.o_b_proj.all_reduce, "forward", observe_reduction)
    generator = torch.Generator(device="cuda").manual_seed(952)
    hidden = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16, generator=generator)
    resources = []

    def fresh_metadata(mapping):
        manager = CSA2CacheManager(
            KvCacheConfig(
                enable_block_reuse=False,
                max_gpu_total_bytes=128 << 20,
                enable_swa_scratch_reuse=True,
            ),
            CacheType.SELFKONLY,
            num_layers=1,
            tokens_per_block=128,
            max_seq_len=128,
            max_batch_size=1,
            max_input_len=3,
            max_num_tokens=3,
            mapping=mapping,
            dtype=DataType.BF16,
            vocab_size=128,
            layout=layout,
        )
        request = LlmRequest(
            request_id=953,
            max_new_tokens=1,
            input_tokens=[1, 2, 3],
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        resources.append((manager, request))
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        manager._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=3, kv_cache_manager=manager
        )
        metadata.request_ids = [request.py_request_id]
        metadata.seq_lens = torch.tensor([3], dtype=torch.int32, device="cpu")
        metadata.num_contexts = 1
        metadata.prompt_lens = [3]
        metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0])
        metadata.prepare()
        return metadata

    try:
        tp_metadata = fresh_metadata(tp_mapping)
        reference_metadata = fresh_metadata(Mapping())
        # Both ranks execute both calls, including any shared startup/tuning
        # collectives. The reference has replicated heads and no TP reduction.
        actual = model(hidden, tp_metadata.csa2_positions, tp_metadata)
        expected = reference(hidden, reference_metadata.csa2_positions, reference_metadata)
        torch.cuda.synchronize()
        assert reductions == [torch.Size([3, 128])]
        # Each TP shard rounds its O-B accumulation to BF16 before summation;
        # the reference rounds once after a full-width accumulation.
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.04)
    finally:
        for manager, request in resources:
            manager.free_resources(request)
            manager.shutdown()


@pytest.fixture
def replay_module_factory():
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata
    from tensorrt_llm._torch.attention.backends.sparse.csa2.module import DeepseekV41Attention
    from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import DataType, SamplingConfig
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.functional import PositionEmbeddingType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    resources = []

    def create(ratio, input_length=8):
        layout = CSA2Layout((ratio,), (0,), (0,), index_topk=16, window_size=4)
        manager = CSA2CacheManager(
            KvCacheConfig(
                enable_block_reuse=False,
                max_gpu_total_bytes=128 << 20,
                enable_swa_scratch_reuse=True,
            ),
            CacheType.SELFKONLY,
            num_layers=1,
            tokens_per_block=128,
            max_seq_len=128,
            max_batch_size=1,
            max_input_len=16,
            max_num_tokens=16,
            mapping=Mapping(),
            dtype=DataType.BF16,
            vocab_size=128,
            layout=layout,
        )
        request = LlmRequest(
            request_id=954,
            max_new_tokens=8,
            input_tokens=list(range(input_length)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
        )
        resources.append((manager, request))
        assert manager.prepare_context(request)
        assert manager.resize_context(request, request.context_chunk_size)
        manager._stream.synchronize()
        metadata = CSA2TrtllmMetadata(
            max_num_requests=1, max_num_tokens=16, kv_cache_manager=manager
        )
        metadata.request_ids = [request.py_request_id]
        metadata.num_contexts = 1
        metadata.prompt_lens = [input_length]
        with torch.device("cuda"):
            model = DeepseekV41Attention(
                layout,
                0,
                PositionalEmbeddingParams(
                    type=PositionEmbeddingType.rope_gptj,
                    rope=RopeParams(dim=64, theta=160000, max_positions=128),
                    is_neox=False,
                ),
                hidden_size=32,
                num_heads=8,
                q_lora_rank=32,
                o_lora_rank=16,
                num_groups=2,
                index_heads=8,
            )
            for name, parameter in model.named_parameters():
                parameter.fill_(1) if "norm_weight" in name else parameter.normal_(std=0.1)
            if ratio == 2:
                # Equal gates isolate replay boundary/state restoration from
                # softmax rounding (nonuniform gates have separate coverage).
                model.compressor.wgate.weight.zero_()
            hidden = torch.randn(input_length + 1, 32, dtype=torch.bfloat16)
        return model, manager, metadata, hidden

    yield create
    for manager, request in resources:
        manager.free_resources(request)
        manager.shutdown()


def _prepare_replay_step(metadata, start, end, cached=None, decoder=False):
    from tensorrt_llm._torch.metadata import KVCacheParams

    metadata.seq_lens = torch.tensor([end - start], dtype=torch.int32, device="cpu")
    metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[start])
    if cached is not None:
        metadata.set_swa_bounded_replay([cached], decoder=decoder)
    metadata.prepare()


def _reference_global_rows(model, source, first_position):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )

    ratio = model.layer.compress_ratio
    projected = model.compressor.wkv(source.float() if ratio == 2 else source)
    if ratio == 2:
        complete = source.shape[0] // 2
        values = projected[: complete * 2].reshape(complete, 2, 512)
        gates = model.compressor.wgate(source.float())[: complete * 2].reshape(complete, 2, 512)
        projected = (values * gates.softmax(1)).sum(1).bfloat16()
    latent = _norm(projected, model.compressor.norm_weight, model.eps)
    positions = (
        torch.arange(latent.shape[0], device="cuda", dtype=torch.int32) * ratio + first_position
    )
    rotated = _rotate(latent.unsqueeze(1), positions, model.rotary_emb.rotary_cos_sin).squeeze(1)
    return unpack_rows(pack_rows(rotated, "main"), 512, "main")


def _reference_bounded_output(model, hidden, positions, global_rows):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import (
        pack_rows,
        unpack_rows,
    )

    qrank = _norm(F.linear(hidden, model.wq_a.weight), model.q_norm_weight, model.eps)
    query = _rotate(
        F.linear(qrank, model.wq_b.weight).reshape(-1, 8, 512),
        positions,
        model.rotary_emb.rotary_cos_sin,
    )
    swa = _norm(F.linear(hidden, model.wkv.weight), model.kv_norm_weight, model.eps)
    swa = _rotate(swa.unsqueeze(1), positions, model.rotary_emb.rotary_cos_sin).squeeze(1)
    swa = unpack_rows(pack_rows(swa, "swa"), 512, "swa")
    outputs = []
    for row, position in enumerate(positions.tolist()):
        visible = (position + 1) // model.layer.compress_ratio
        selected = torch.cat((swa[max(0, row - 3) : row + 1], global_rows[:visible]))
        scores = torch.einsum("hd,kd->hk", query[row].float(), selected.float()) * 512**-0.5
        probs = torch.cat((scores, model.attn_sink[:, None]), -1).softmax(-1)[..., :-1]
        outputs.append(torch.einsum("hk,kd->hd", probs, selected.float()).bfloat16())
    output = _rotate(torch.stack(outputs), positions, model.rotary_emb.rotary_cos_sin, inverse=True)
    latent = torch.einsum("qgd,grd->qgr", output.reshape(-1, 2, 2048), model.o_a_proj).flatten(1)
    return F.linear(latent, model.o_b_proj.weight)


@torch.inference_mode()
def test_encoder_bounded_replay_preserves_global_and_truncates_swa(
    monkeypatch, replay_module_factory
):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    torch.manual_seed(955)
    model, manager, metadata, hidden = replay_module_factory(2)
    _prepare_replay_step(metadata, 0, 5)
    model(hidden[:5], metadata.csa2_positions, metadata)
    old_slots = metadata.csa2_main_write_slots[0]
    old_slots = old_slots[old_slots >= 0].clone()
    old_main = manager.get_main_buffer(0)[old_slots].clone()
    old_index = manager.get_index_buffer(0)[old_slots].clone()
    manager.get_swa_buffer(0).fill_(255)
    for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
        manager.get_buffers(0, role).fill_(float("nan"))
    replay = hidden[1:8].clone()
    replay[:4].add_(0.125)  # model-layer replay inputs may be approximate
    seen = []
    compressor = model.compressor.forward

    def observe(x, batch=None):
        seen.append(x.clone())
        return compressor(x, batch)

    monkeypatch.setattr(model.compressor, "forward", observe)
    _prepare_replay_step(metadata, 1, 8, cached=5)
    actual = model(replay, metadata.csa2_positions, metadata)
    assert len(seen) == 1 and seen[0].shape[0] == 4
    torch.testing.assert_close(seen[0], replay[3:], atol=0, rtol=0)
    torch.testing.assert_close(manager.get_main_buffer(0)[old_slots], old_main, atol=0, rtol=0)
    torch.testing.assert_close(manager.get_index_buffer(0)[old_slots], old_index, atol=0, rtol=0)
    assert int((metadata.csa2_swa_indices[0][0] >= 0).sum()) == 1
    expected_global = torch.cat(
        (unpack_rows(old_main, 512, "main"), _reference_global_rows(model, replay[3:], 4))
    )
    expected = _reference_bounded_output(model, replay, metadata.csa2_positions, expected_global)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)


@pytest.mark.parametrize(
    "ratio,cached,decoder", [(1, 5, False), (2, 4, False), (2, 5, False), (1, 5, True)]
)
@torch.inference_mode()
def test_pure_replay_skips_global_projection_and_restores_odd_state(
    monkeypatch, replay_module_factory, ratio, cached, decoder
):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
    from tensorrt_llm._torch.attention.backends.sparse.csa2.quantization import unpack_rows

    torch.manual_seed(956)
    model, manager, metadata, hidden = replay_module_factory(ratio)
    _prepare_replay_step(metadata, 0, cached)
    model(hidden[:cached], metadata.csa2_positions, metadata)
    slots = metadata.csa2_main_write_slots[0]
    slots = slots[slots >= 0].clone()
    old_main = manager.get_main_buffer(0)[slots].clone()
    old_index = manager.get_index_buffer(0)[slots].clone()
    manager.get_swa_buffer(0).fill_(255)
    if ratio == 2:
        for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
            manager.get_buffers(0, role).fill_(float("nan"))
    start = max(cached - 4, 0)
    replay = hidden[start:cached].clone().add_(0.125)
    counts = []
    projection = model.compressor.wkv.forward

    def observe(x):
        counts.append(x.shape[0])
        return projection(x)

    monkeypatch.setattr(model.compressor.wkv, "forward", observe)
    _prepare_replay_step(metadata, start, cached, cached=cached, decoder=decoder)
    actual = model(replay, metadata.csa2_positions, metadata)
    assert counts == ([1] if ratio == 2 and cached % 2 and not decoder else [])
    torch.testing.assert_close(manager.get_main_buffer(0)[slots], old_main, atol=0, rtol=0)
    torch.testing.assert_close(manager.get_index_buffer(0)[slots], old_index, atol=0, rtol=0)
    assert bool((metadata.csa2_main_write_slots[0] < 0).all())
    expected = _reference_bounded_output(
        model, replay, metadata.csa2_positions, unpack_rows(old_main, 512, "main")
    )
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    if ratio == 2 and cached % 2:
        # Ordinary continuation must consume the reconstructed raw tail, not
        # the poisoned historical state or an overwritten cached global row.
        _prepare_replay_step(metadata, cached, cached + 1)
        model(hidden[cached : cached + 1], metadata.csa2_positions, metadata)
        new_slots = metadata.csa2_main_write_slots[0]
        new_slots = new_slots[new_slots >= 0]
        expected_row = _reference_global_rows(
            model, torch.cat((replay[-1:], hidden[cached : cached + 1])), cached - 1
        )
        actual_row = unpack_rows(manager.get_main_buffer(0)[new_slots], 512, "main")
        torch.testing.assert_close(actual_row, expected_row, atol=0.02, rtol=0.02)


@torch.inference_mode()
def test_bounded_replay_full_module_graph_changes_hit_position(replay_module_factory):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import CSA2CacheRole
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    torch.manual_seed(957)
    model, manager, seed_metadata, hidden = replay_module_factory(2, input_length=12)

    def new_metadata(graph=False):
        result = CSA2TrtllmMetadata(max_num_requests=1, max_num_tokens=16, kv_cache_manager=manager)
        result.request_ids = list(seed_metadata.request_ids)
        result.num_contexts = 1
        result.prompt_lens = [12]
        result.is_cuda_graph = graph
        return result

    eager_metadata = new_metadata()
    graph_metadata = new_metadata(graph=True)
    replay_input = torch.empty(7, 32, device="cuda", dtype=torch.bfloat16)

    def seed(cached):
        _prepare_replay_step(seed_metadata, 0, cached)
        model(hidden[:cached], seed_metadata.csa2_positions, seed_metadata)
        slots = seed_metadata.csa2_main_write_slots[0]
        return slots[slots >= 0].clone()

    def drop_local_state():
        manager.get_swa_buffer(0).fill_(255)
        for role in (CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE):
            manager.get_buffers(0, role).fill_(float("nan"))

    def run_graph_body():
        graph_metadata.reset_routing()
        return model(replay_input, graph_metadata.csa2_positions, graph_metadata)

    graph = None
    pointers = None
    for cached in (5, 7):
        start, end = cached - 4, cached + 3
        replay_input.copy_(hidden[start:end])
        replay_input[:4].add_(0.125)
        seed(cached)
        drop_local_state()
        _prepare_replay_step(eager_metadata, start, end, cached=cached)
        expected = model(replay_input, eager_metadata.csa2_positions, eager_metadata).clone()
        # Reset real GLOBAL contents to this hit position's seeded prefix,
        # then remove local state again before executing the captured path.
        slots = seed(cached)
        old_main = manager.get_main_buffer(0)[slots].clone()
        old_index = manager.get_index_buffer(0)[slots].clone()
        _prepare_replay_step(graph_metadata, start, end, cached=cached)
        current_pointers = (
            graph_metadata.csa2_positions.data_ptr(),
            graph_metadata.get_compressed_positions(0).data_ptr(),
            graph_metadata.get_compression_batch(0).start_positions.data_ptr(),
        )
        if graph is None:
            for _ in range(3):
                drop_local_state()
                run_graph_body()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = run_graph_body()
            pointers = current_pointers
        else:
            assert current_pointers == pointers
        drop_local_state()
        graph.replay()
        torch.cuda.synchronize()
        # The oracle uses the same approximate/truncated replay policy,
        # not a full-prefix reconstruction. Context/gen kernels can differ
        # by ordinary BF16 reduction rounding.
        torch.testing.assert_close(captured, expected, atol=0.03, rtol=0.03)
        torch.testing.assert_close(manager.get_main_buffer(0)[slots], old_main, atol=0, rtol=0)
        torch.testing.assert_close(manager.get_index_buffer(0)[slots], old_index, atol=0, rtol=0)


@pytest.mark.parametrize("strategy", [None, "NCCL"], ids=["default-auto", "explicit-nccl"])
def test_tp_output_projection_uses_requested_allreduce_strategy(strategy):
    from tensorrt_llm._torch.distributed import AllReduceStrategy
    from tensorrt_llm._utils import mpi_rank, mpi_world_size
    from tensorrt_llm.mapping import Mapping

    if mpi_world_size() != 2:
        pytest.skip("Requires two MPI ranks")
    torch.cuda.set_device(mpi_rank())
    requested = None if strategy is None else AllReduceStrategy.NCCL
    model, _ = _native_model(
        fused=False,
        projection_quantization="bf16",
        mapping=Mapping(world_size=2, rank=mpi_rank(), tp_size=2),
        allreduce_strategy=requested,
    )
    assert model.o_b_proj.all_reduce is not None
    assert model.o_b_proj.all_reduce.strategy == (
        AllReduceStrategy.AUTO if requested is None else requested
    )
    assert not model.o_b_proj.use_fused_gemm_allreduce
