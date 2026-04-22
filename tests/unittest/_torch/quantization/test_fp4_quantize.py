# Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for `fp4_quantize_1x32_sf_transpose`.

These tests pin the indexer's FP4 quantization to DeepGEMM's reference
implementation. Any divergence would make the C++ fp8_fp4 MQA logits
kernels reinterpret the bytes incorrectly.
"""

import pytest
import torch

from tensorrt_llm.quantization.utils.fp4_utils import fp4_quantize_1x32_sf_transpose


def _deepgemm_reference():
    """Import DeepGEMM's per_token_cast_to_fp4 from the fetched submodule."""
    import importlib.util
    import pathlib

    deepgemm_src = pathlib.Path("/home/scratch.fanrongl_gpu/GitRepo/TRT_LLM/main/DeepGEMM")
    math_path = deepgemm_src / "deep_gemm" / "utils" / "math.py"
    if not math_path.exists():
        pytest.skip("DeepGEMM reference not available at expected path")
    spec = importlib.util.spec_from_file_location("_dg_math", math_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.per_token_cast_to_fp4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device")
@pytest.mark.parametrize(
    "shape",
    [
        (1, 128),
        (16, 128),
        (7, 128),
        (4, 32, 128),
        (2, 64, 128),
    ],
)
def test_bit_exact_match_with_deepgemm(shape):
    """Output must match DeepGEMM's per_token_cast_to_fp4 byte-for-byte."""
    per_token_cast_to_fp4 = _deepgemm_reference()
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * 3.0

    packed_ours, scale_ours = fp4_quantize_1x32_sf_transpose(x)

    head_dim = shape[-1]
    x_flat = x.view(-1, head_dim)
    packed_ref, scale_ref = per_token_cast_to_fp4(
        x_flat, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    packed_ref = packed_ref.view(*shape[:-1], head_dim // 2)
    scale_ref = scale_ref.view(*shape[:-1], head_dim // 128)

    assert packed_ours.dtype == torch.int8
    assert scale_ours.dtype == torch.int32
    assert torch.equal(packed_ours, packed_ref), "FP4 packed bytes diverged"
    assert torch.equal(scale_ours, scale_ref), "UE8M0 packed scales diverged"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device")
def test_nibble_order_even_low_odd_high():
    """Even-index elements occupy the low nibble, odd-index the high nibble."""
    # Hand-crafted: first token = +1.0, second = +2.0, third = -3.0, fourth = +6.0
    # After scale = max / 6 = 1.0 (UE8M0 rounds up to 1.0 = 2^0), the codes are:
    #   +1.0 → E2M1 code 2 (0b0010)
    #   +2.0 → code 4 (0b0100)
    #   -3.0 → code 5 | sign-bit = 0b1101
    #   +6.0 → code 7 (0b0111)
    x = torch.zeros(1, 128, device="cuda", dtype=torch.bfloat16)
    x[0, 0] = 1.0
    x[0, 1] = 2.0
    x[0, 2] = -3.0
    x[0, 3] = 6.0
    packed, _scale = fp4_quantize_1x32_sf_transpose(x)
    byte0 = packed[0, 0].item() & 0xFF
    byte1 = packed[0, 1].item() & 0xFF
    # byte0 = (odd=2 << 4) | (even=1) but E2M1 codes are 0b0010 and 0b0100
    assert (byte0 & 0x0F) == 0b0010, f"low nibble of byte0 should be +1.0, got {byte0:#b}"
    assert (byte0 >> 4) == 0b0100, f"high nibble of byte0 should be +2.0, got {byte0:#b}"
    assert (byte1 & 0x0F) == 0b1101, f"low nibble of byte1 should be -3.0, got {byte1:#b}"
    assert (byte1 >> 4) == 0b0111, f"high nibble of byte1 should be +6.0, got {byte1:#b}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device")
def test_roundtrip_error_within_tolerance():
    """FP4 dequantize must reconstruct the input within ~1.0 mean abs error.

    E2M1 has only 8 levels per nibble, so rough parity is all we expect;
    this guards against coarse implementation bugs (wrong boundaries,
    dropped signs, etc.).
    """
    torch.manual_seed(0)
    x = torch.randn(16, 128, device="cuda", dtype=torch.float32) * 2.0

    packed, scale = fp4_quantize_1x32_sf_transpose(x)

    # Dequantize inline (mirrors deep_gemm.utils.math.cast_back_from_fp4
    # for gran_k=32 with packed UE8M0 scales).
    fp4_values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device="cuda",
        dtype=torch.float32,
    )
    packed_u8 = packed.view(torch.uint8)
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F
    codes = torch.empty(16, 128, device="cuda", dtype=torch.uint8)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    value_idx = (codes & 0x07).to(torch.int64)
    sign = (codes & 0x08) != 0
    values = fp4_values[value_idx]
    values = torch.where(sign & (value_idx != 0), -values, values)
    # Unpack scales (4 UE8M0 bytes per int32 → 4 scales per token).
    scale_bytes = scale.view(torch.uint8).view(16, 4).to(torch.int32)
    scale_fp32 = (scale_bytes << 23).view(torch.float32)
    scale_expanded = scale_fp32.unsqueeze(-1).expand(16, 4, 32).reshape(16, 128)
    reconstructed = values * scale_expanded

    mae = (reconstructed - x).abs().mean().item()
    assert mae < 1.0, f"FP4 roundtrip mean abs error too large: {mae}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device")
def test_output_shapes_and_dtypes():
    """2D, 3D, and 4D inputs all preserve their leading dims."""
    for shape in [(8, 128), (3, 16, 128), (2, 4, 32, 128)]:
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        packed, scale = fp4_quantize_1x32_sf_transpose(x)
        assert packed.shape == (*shape[:-1], shape[-1] // 2)
        assert scale.shape == (*shape[:-1], shape[-1] // 128)
        assert packed.dtype == torch.int8
        assert scale.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA device")
def test_cuda_graph_capture_matches_eager():
    torch.manual_seed(0)
    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16) * 2.0

    packed_eager, scale_eager = fp4_quantize_1x32_sf_transpose(x)

    for _ in range(2):
        fp4_quantize_1x32_sf_transpose(x)
    torch.cuda.synchronize()

    packed_buf = torch.zeros_like(packed_eager)
    scale_buf = torch.zeros_like(scale_eager)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        x_capture = x.clone()
        with torch.cuda.graph(graph):
            p, s = fp4_quantize_1x32_sf_transpose(x_capture)
            packed_buf.copy_(p)
            scale_buf.copy_(s)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(packed_buf, packed_eager)
    assert torch.equal(scale_buf, scale_eager)
