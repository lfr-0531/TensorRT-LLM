from enum import IntEnum
from typing import Tuple

import torch
import triton
import triton.language as tl

# The declarations must be aligned with thUtils.h
SF_DTYPE = torch.uint8
FLOAT4_E2M1X2 = torch.uint8

# For GEMM autotuning.
# Taken from https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime//modelConfig.h#L38
# TODO: move to model config, tune for blackwell hardware
FP4_BUCKETS = [64, 128, 256, 512, 1024]

# Export
float4_e2m1x2 = FLOAT4_E2M1X2
float4_sf_dtype = SF_DTYPE
fp4_buckets = FP4_BUCKETS

__all__ = [
    'float4_e2m1x2', 'float4_sf_dtype', 'pad_up', 'fp4_buckets',
    'fp4_quantize_1x32_sf_transpose'
]


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


class FP4GemmType(IntEnum):
    W4A4_NVFP4_NVFP4 = 0
    W4A8_MXFP4_MXFP8 = 1


def get_fp4_shape(input_shape, sf_vec_size, is_swizzled_layout=True):
    m = 1
    for i in range(len(input_shape) - 1):
        m *= input_shape[i]

    output_shape = [i for i in input_shape]
    output_shape[-1] //= 2

    scale_shape = pad_up(m, 128) * pad_up(
        input_shape[-1] // sf_vec_size,
        4) if is_swizzled_layout else m * (input_shape[-1] // sf_vec_size)
    return output_shape, scale_shape


def get_reorder_rows_for_gated_act_gemm_row_indices(x) -> torch.Tensor:
    """
    Reorders rows in the gemm/MOE_gemm weight matrix for min-latency
    [r0, r1, r2, r3, ..., rN/2, r(N/2+1), .. r(N-1)]
    to
    [r0, rN/2, r1, rN/2+1, ..., r(N/2-1), r(N-1)]
    """
    M = x.shape[0]
    assert M % 2 == 0, f"x.shape[0] must be even, not {M}"

    row_indices = torch.arange(M, dtype=torch.long)

    # We split into top half and bottom half, but if M is odd,
    # the bottom half is one row larger.
    top = row_indices[:(M + 1) // 2]  # round up
    bot = row_indices[(M + 1) // 2:]  # remainder

    # Create the output
    permuted_row_indices = torch.empty_like(row_indices)

    # We'll place rows of `top` and `bot` in alternation
    permuted_row_indices[0::2] = top
    permuted_row_indices[1::2] = bot

    return permuted_row_indices


def reorder_rows_for_gated_act_gemm(x):
    """
    PyTorch implementation of trt-llm gen `reorderRowsForGatedActGemm`
    """
    row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(x)

    permute = lambda x: x[row_indices]

    return permute(x)


# yapf: disable
srcToDstBlk16RowMap = [
    0,  8,
    1,  9,
    2, 10,
    3, 11,
    4, 12,
    5, 13,
    6, 14,
    7, 15
]

srcToDstBlk32RowMap = [
    0,  8, 16, 24,
    1,  9, 17, 25,
    2, 10, 18, 26,
    3, 11, 19, 27,
    4, 12, 20, 28,
    5, 13, 21, 29,
    6, 14, 22, 30,
    7, 15, 23, 31
]
# yapf: enable


def get_shuffle_block_size(epilogue_tile_m: int) -> int:
    shuffle_block_size = 16
    if epilogue_tile_m % 128 == 0:
        shuffle_block_size = 32
    return shuffle_block_size


def get_shuffle_matrix_a_row_indices(input_tensor: torch.Tensor,
                                     epilogue_tile_m: int) -> torch.Tensor:
    """
    Higher-level PyTorch approach to reorder the rows in blocks of size 16 or 32.
    - We do NOT try to handle custom e2m1 memory usage (i.e. no 'K/2' bytes).
    - Instead, we purely reorder rows in a standard PyTorch shape [M, K].
    """
    # M from the input
    M = input_tensor.shape[0]

    # Choose block size 16 or 32
    shuffle_block_size = get_shuffle_block_size(epilogue_tile_m)
    row_map = (srcToDstBlk16RowMap
               if shuffle_block_size == 16 else srcToDstBlk32RowMap)

    assert M % shuffle_block_size == 0, f"input_tensor.shape[0] must be multiples of {shuffle_block_size}"

    # row_indices[new_row] = old_row
    # so row_indices is an array of size M telling us from which old_row
    # the new_row should be taken.
    row_indices = torch.empty(M, dtype=torch.long)

    for old_row in range(M):
        block_idx = old_row // shuffle_block_size
        row_in_block = old_row % shuffle_block_size
        mapped_row_in_block = row_map[row_in_block]

        new_row = block_idx * shuffle_block_size + mapped_row_in_block

        row_indices[new_row] = old_row

    return row_indices


def shuffle_matrix_a(input_tensor: torch.Tensor,
                     epilogue_tile_m: int) -> torch.Tensor:
    """
    PyTorch equivalent of trtllm-gen `shuffleMatrixA`
    """
    row_indices = get_shuffle_matrix_a_row_indices(input_tensor,
                                                   epilogue_tile_m)

    return torch.ops.trtllm.shuffle_matrix(input_tensor,
                                           row_indices.to(input_tensor.device))


def get_shuffle_matrix_sf_a_row_indices(
        input_tensor: torch.Tensor,
        epilogue_tile_m: int,
        num_elts_per_sf: int = 16) -> torch.Tensor:

    assert input_tensor.dtype == float4_sf_dtype
    assert num_elts_per_sf == 16 or num_elts_per_sf == 32

    assert input_tensor.dim(
    ) == 2, f"input_tensor should be a 2D tensor, not {input_tensor.dim()}"

    # M, K from the input
    M, K = input_tensor.shape
    assert M % 128 == 0
    assert K % 4 == 0

    row_indices = get_shuffle_matrix_a_row_indices(input_tensor,
                                                   epilogue_tile_m)

    return row_indices


def shuffle_matrix_sf_a(
    input_tensor: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: int = 16,
):
    """
    Cuda implementation of trtllm-gen `shuffleMatrixSfA` but with a caveat.
    `shuffleMatrixSfA` expects the input to be in 128x4 layout and then
    apply the same shuffling in `shuffleMatrixA` and writes out in 128x4
    layout.
    This function expects the input to be in linear layout. It's done this
    way because the scaling factors in the NVFP4 checkpoints are quantized
    and are in linear layout.
    This function doesn't add padding.
    """

    row_indices = get_shuffle_matrix_sf_a_row_indices(input_tensor,
                                                      epilogue_tile_m)

    w_shuffled = torch.ops.trtllm.shuffle_matrix(
        input_tensor, row_indices.to(input_tensor.device))

    # 128x4
    return torch.ops.trtllm.block_scale_interleave(w_shuffled)


# ---------------------------------------------------------------------------
# FP4 E2M1 per-block-32 quantization for the DSA indexer (B200 only).
#
# DeepGEMM's fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits kernels expect
# data packed as int8 with two E2M1 nibbles per byte (even-index in the low
# nibble, odd-index in the high nibble) and per-block-32 UE8M0 scales packed
# four-per-int32. Bit-exact parity with DeepGEMM's testing.per_token_cast_to_fp4
# (gran_k=32, use_packed_ue8m0=True) is required for the kernel to interpret
# the cache correctly.


@triton.jit
def _fp4_quantize_1x32_kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        x_stride_row,
        packed_stride_row,
        scale_stride_row,
        HEAD_DIM: tl.constexpr,  # must be 128
        NUM_BLOCKS: tl.constexpr,  # HEAD_DIM // 32 = 4 at HD=128
):
    """One program instance = one token.

    Block_size=32 is hardcoded because DeepGEMM's FP4 MQA logits kernel
    asserts it. Producing bit-exact output relative to DeepGEMM's
    testing.per_token_cast_to_fp4(..., gran_k=32, use_packed_ue8m0=True)
    requires matching the exact math:

    - ceil_to_ue8m0(amax/6): keep the mantissa-OR-round trick so tied
      midpoints round up to the next power of two, not down.
    - E2M1 code: 8-entry LUT indexed by the bucketize midpoints
      {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0}, sign goes into bit 3 unless
      the magnitude index is zero.
    - Packing: even-index element in the low nibble, odd-index in the high
      nibble.
    - Scale packing: the UE8M0 float exponent byte (upper 8 bits after the
      mantissa shift) is stored little-endian across the four blocks into
      one int32.
    """
    tid = tl.program_id(0)

    # Load the whole token (HEAD_DIM elements). Use float32 so ceil_to_ue8m0
    # and bucketize operate in the same precision as the torch reference.
    offs = tl.arange(0, HEAD_DIM)
    row_ptr = x_ptr + tid * x_stride_row
    x = tl.load(row_ptr + offs).to(tl.float32)

    # Per-block-32 amax.
    x_view = tl.reshape(x, (NUM_BLOCKS, 32))
    x_amax = tl.max(tl.abs(x_view), axis=1)
    x_amax = tl.where(x_amax < 1e-4, 1e-4, x_amax)
    sf_fp32 = x_amax / 6.0

    # ceil_to_ue8m0: exp = ((bits >> 23) & 0xFF) + (mantissa != 0)
    sf_bits = sf_fp32.to(tl.int32, bitcast=True)
    exp = ((sf_bits >> 23) & 0xFF) + ((sf_bits & 0x7FFFFF) != 0).to(tl.int32)
    exp = tl.maximum(1, tl.minimum(254, exp))
    sf_ue8m0 = (exp << 23).to(tl.float32, bitcast=True)  # (NUM_BLOCKS,)

    # Broadcast scale across its 32 elements and divide.
    sf_per_elem = tl.reshape(
        sf_ue8m0[:, None] + tl.zeros((NUM_BLOCKS, 32), tl.float32),
        (HEAD_DIM, ))
    x_scaled = x / sf_per_elem

    # E2M1 quantize via bucketize on midpoints, then OR in the sign bit.
    # torch.bucketize(..., right=False) returns the smallest i with
    # boundaries[i] >= x, so we count boundaries strictly less than x to
    # produce the same index — matching DeepGEMM's reference byte-for-byte.
    ax = tl.minimum(tl.abs(x_scaled), 6.0)
    idx = ((ax > 0.25).to(tl.int32) + (ax > 0.75).to(tl.int32) +
           (ax > 1.25).to(tl.int32) + (ax > 1.75).to(tl.int32) +
           (ax > 2.5).to(tl.int32) + (ax > 3.5).to(tl.int32) +
           (ax > 5.0).to(tl.int32))
    code = idx & 0x07
    sign = ((x_scaled < 0) & (idx != 0)).to(tl.int32)
    code = code | (sign << 3)  # (HEAD_DIM,) int32 with FP4 nibbles in [0,15]

    # Pack two nibbles per byte: even → low, odd → high.
    even_offs = tl.arange(0, HEAD_DIM // 2) * 2
    odd_offs = even_offs + 1
    low = tl.load(row_ptr + even_offs,
                  mask=tl.full([HEAD_DIM // 2], False, tl.int1),
                  other=0.0)  # dummy to satisfy static pass
    # Re-gather from the already-computed code vector rather than reloading.
    code_even = tl.sum(tl.where(offs[None, :] == even_offs[:, None],
                                code[None, :].to(tl.int32), 0),
                       axis=1)
    code_odd = tl.sum(tl.where(offs[None, :] == odd_offs[:, None],
                               code[None, :].to(tl.int32), 0),
                      axis=1)
    packed = ((code_even & 0x0F) | ((code_odd & 0x0F) << 4)).to(tl.int8)
    packed_row_ptr = packed_ptr + tid * packed_stride_row
    tl.store(packed_row_ptr + tl.arange(0, HEAD_DIM // 2), packed)

    # Pack the four UE8M0 exponent bytes into one int32 (little-endian).
    # For HD=128, NUM_BLOCKS=4, so exp_lo holds 4 bytes in [0,255].
    exp_lo = exp & 0xFF  # (NUM_BLOCKS,)
    byte_shifts = tl.arange(0, NUM_BLOCKS) * 8
    packed_scale = tl.sum(exp_lo << byte_shifts, axis=0)  # int32 scalar
    # Output has trailing dim NUM_BLOCKS // 4 (= 1 at HD=128). Store one int32
    # per token.
    scale_row_ptr = scale_ptr + tid * scale_stride_row
    tl.store(scale_row_ptr + tl.arange(0, 1), packed_scale.to(tl.int32))


def fp4_quantize_1x32_sf_transpose(
        x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize `x` to FP4 E2M1 with UE8M0 scales packed four-per-int32.

    The trailing dim of `x` must be 128 — DeepGEMM's FP4 MQA logits kernel
    hard-asserts `head_dim == 128` and block_size=32, so each token carries
    exactly four UE8M0 scales that pack into one int32.

    Implementation: a Triton kernel that fuses the amax, UE8M0 ceil, FP4
    quantize, and nibble/byte packing into a single grid launch. This path
    is CUDA-graph-capturable (the torch-based reference used torch.bucketize,
    which is not). Output remains bit-identical to DeepGEMM's
    `testing.per_token_cast_to_fp4(..., gran_k=32, use_packed_ue8m0=True)`.

    Returns:
        packed: ``torch.int8`` tensor with the same leading dims as `x` and a
            trailing dim of ``x.shape[-1] // 2``. Two FP4 E2M1 codes are
            packed per byte: the even-index element in the low nibble and the
            odd-index element in the high nibble.
        scale: ``torch.int32`` tensor with the same leading dims as `x` and a
            trailing dim of ``x.shape[-1] // 128``. Each int32 carries four
            UE8M0 exponents, one per block of 32 elements.
    """
    gran_k = 32
    assert x.dim() >= 1
    head_dim = x.shape[-1]
    assert head_dim == 128, (
        "fp4_quantize_1x32_sf_transpose is specialized to head_dim=128 to "
        f"match DeepGEMM's FP4 MQA logits kernel constraint; got {head_dim}")
    num_blocks = head_dim // gran_k

    leading = x.shape[:-1]
    x_flat = x.reshape(-1, head_dim).contiguous()
    n_tokens = x_flat.shape[0]

    # Fast-path an empty call so Triton doesn't launch a zero-grid kernel
    # (which would still allocate output tensors but produce no writes —
    # harmless but confuses some graph-capture diagnostics).
    packed = torch.empty((n_tokens, head_dim // 2),
                         dtype=torch.int8,
                         device=x.device)
    scale = torch.empty((n_tokens, num_blocks // 4),
                        dtype=torch.int32,
                        device=x.device)
    if n_tokens == 0:
        return packed.view(*leading, head_dim // 2), \
            scale.view(*leading, num_blocks // 4)

    _fp4_quantize_1x32_kernel[(n_tokens, )](
        x_flat,
        packed,
        scale,
        x_flat.stride(0),
        packed.stride(0),
        scale.stride(0),
        HEAD_DIM=head_dim,
        NUM_BLOCKS=num_blocks,
    )

    packed = packed.view(*leading, head_dim // 2)
    sf_packed = scale.view(*leading, num_blocks // 4)
    return packed, sf_packed
