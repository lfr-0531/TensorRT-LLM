# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""CSA2 cache, index projection and direct packed attention kernels.

CuTe projection/attention types are constructed lazily, so importing Triton
cache publication does not import optional Cutlass bindings or initialize CUDA.
"""

import functools
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_packed_rows(
    pool,
    slots,
    rows,
    pool_stride: tl.constexpr,
    row_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    capacity: tl.constexpr,
    width: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row * slot_stride).to(tl.int64)
    if (slot >= 0) & (slot < capacity):
        offsets = tl.arange(0, block)
        values = tl.load(rows + row * row_stride + offsets, offsets < width, 0)
        tl.store(pool + slot * pool_stride + offsets, values, offsets < width)


def scatter_packed_rows(pool: torch.Tensor, slots: torch.Tensor, rows: torch.Tensor) -> None:
    """Ignore padding slots before any store, including under graph replay."""
    if rows.shape[0] == 0:
        return
    if pool.stride(1) != 1 or rows.stride(1) != 1:
        raise ValueError("CSA2 packed cache columns must be contiguous")
    _scatter_packed_rows[(rows.shape[0],)](
        pool,
        slots,
        rows,
        pool.stride(0),
        rows.stride(0),
        slots.stride(0),
        pool.shape[0],
        rows.shape[1],
        triton.next_power_of_2(rows.shape[1]),
    )


@triton.autotune(
    configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)],
    key=["head_dim", "cache_format", "num_rows"],
)
@triton.jit
def _quantize_scatter(
    pool,
    slots,
    values,
    pool_stride: tl.constexpr,
    value_stride: tl.constexpr,
    slot_stride: tl.constexpr,
    capacity: tl.constexpr,
    head_dim: tl.constexpr,
    cache_format: tl.constexpr,
    num_rows: tl.constexpr,
    group: tl.constexpr,
    groups: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(slots + row * slot_stride).to(tl.int64)
    # Rejected/padded slots are skipped before destination address calculation.
    if (slot >= 0) & (slot < capacity):
        group_ids = tl.arange(0, groups)
        offsets = group_ids[:, None] * group + tl.arange(0, group)[None, :]
        x = tl.load(values + row * value_stride + offsets, offsets < head_dim, 0).to(tl.float32)
        maximum = tl.max(tl.abs(x), axis=1)
        # Triton's max/clamp ignore NaNs by default; Torch amax propagates
        # them to every value in the quantization group.
        group_nan = tl.sum((x != x).to(tl.int32), axis=1) != 0
        if cache_format == "main":
            pre_scale = tl.div_rn(tl.maximum(maximum, 6.0 * 2.0**-9), 6.0)
            scales_fp8 = pre_scale.to(tl.float8e4nv)
            # Native E4M3 conversion matches Torch CUDA on validated SM100,
            # including saturation beyond the largest finite value.
            scale_bytes = scales_fp8.to(tl.uint8, bitcast=True)
            scales = scale_bytes.to(tl.uint8).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        else:
            if cache_format == "swa":
                divisor: tl.constexpr = 448.0
                floor: tl.constexpr = 1.0e-4
            else:
                divisor: tl.constexpr = 6.0
                floor: tl.constexpr = 6.0 * 2.0**-126
            exponent = tl.ceil(tl.log2(tl.div_rn(tl.maximum(maximum, floor), divisor)))
            scale_bytes = (exponent + 127).to(tl.uint8)
            scales = tl.exp2(exponent)
        if cache_format == "main":
            scale_bytes = tl.where(group_nan, 127, scale_bytes)
        else:
            scale_bytes = tl.where(group_nan, 0, scale_bytes)
        scaled = tl.div_rn(x, scales[:, None])
        if cache_format == "swa":
            quantized = tl.minimum(tl.maximum(scaled, -448.0), 448.0).to(tl.float8e4nv)
            payload = quantized.to(tl.uint8, bitcast=True)
            payload = tl.where(group_nan[:, None] | (scaled != scaled), 127, payload).to(tl.uint8)
            tl.store(pool + slot * pool_stride + offsets, payload, offsets < head_dim)
            data_bytes: tl.constexpr = head_dim
        else:
            magnitude = tl.abs(scaled)
            # RNE at E2M1 midpoints: ties choose the even adjacent code.
            codes = (magnitude > 0.25).to(tl.int32)
            codes += (magnitude >= 0.75).to(tl.int32)
            codes += (magnitude > 1.25).to(tl.int32)
            codes += (magnitude >= 1.75).to(tl.int32)
            codes += (magnitude > 2.5).to(tl.int32)
            codes += (magnitude >= 3.5).to(tl.int32)
            codes += (magnitude > 5.0).to(tl.int32)
            # bucketize(NaN) selects the last bin in the Torch reference.
            codes = tl.where(magnitude != magnitude, 7, codes)
            codes |= (scaled.to(tl.int32, bitcast=True) >> 28) & 8
            codes = tl.where(group_nan[:, None], 7, codes)
            pairs = tl.reshape(codes, (groups * group // 2, 2))
            low, high = tl.split(pairs)
            packed = (low | (high << 4)).to(tl.uint8)
            byte_ids = tl.arange(0, groups * group // 2)
            tl.store(pool + slot * pool_stride + byte_ids, packed, byte_ids < head_dim // 2)
            data_bytes: tl.constexpr = head_dim // 2
        tl.store(
            pool + slot * pool_stride + data_bytes + group_ids,
            scale_bytes.to(tl.uint8),
            group_ids < head_dim // group,
        )


def quantize_scatter_rows(pool, slots, values, cache_format="main"):
    """Publish BF16 rows directly; caller dispatches only on validated hardware."""
    if values.shape[0] == 0:
        return pool
    group = 16 if cache_format == "main" else 32
    _quantize_scatter[(values.shape[0],)](
        pool,
        slots,
        values,
        pool.stride(0),
        values.stride(0),
        slots.stride(0),
        pool.shape[0],
        values.shape[1],
        cache_format,
        values.shape[0],
        group,
        triton.next_power_of_2(values.shape[1] // group),
    )
    return pool


@triton.jit
def _gather_dequant_rows(
    pool,
    slots,
    output,
    pool_stride: tl.constexpr,
    slot_row_stride: tl.constexpr,
    slot_col_stride: tl.constexpr,
    slot_columns: tl.constexpr,
    capacity: tl.constexpr,
    head_dim: tl.constexpr,
    cache_format: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    slot_offset = row // slot_columns * slot_row_stride + row % slot_columns * slot_col_stride
    slot = tl.load(slots + slot_offset).to(tl.int64)
    valid = (slot >= 0) & (slot < capacity)
    safe_slot = tl.where(valid, slot, 0)
    channels = tl.arange(0, block)
    mask = valid & (channels < head_dim)
    if cache_format == "swa":
        payload = tl.load(pool + safe_slot * pool_stride + channels, mask, 0)
        values = payload.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_offset = head_dim + channels // 32
    else:
        payload = tl.load(pool + safe_slot * pool_stride + channels // 2, mask, 0).to(tl.int32)
        codes = (payload >> (channels % 2 * 4)) & 15
        magnitude = codes & 7
        # Same E2M1 nibble layout as the existing NVFP4 dequantizer, preserving -0.
        values = tl.where(
            magnitude < 4,
            magnitude.to(tl.float32) * 0.5,
            (1.0 + (magnitude % 2).to(tl.float32) * 0.5)
            * ((magnitude // 2 + 126) << 23).to(tl.float32, bitcast=True),
        )
        value_bits = values.to(tl.int32, bitcast=True) ^ ((codes & 8) << 28)
        values = value_bits.to(tl.float32, bitcast=True)
        group: tl.constexpr = 16 if cache_format == "main" else 32
        scale_offset = head_dim // 2 + channels // group
    scale_bytes = tl.load(pool + safe_slot * pool_stride + scale_offset, mask, 0)
    if cache_format == "main":
        scales = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    else:
        # Exact 2**(byte-127), including the FP32 subnormal at byte zero.
        # Unlike the E8M0 numeric cast, byte255 follows unpack_rows and means inf.
        scale_bits = tl.where(scale_bytes == 0, 0x00400000, scale_bytes.to(tl.int32) << 23)
        scales = scale_bits.to(tl.float32, bitcast=True)
    # Preserve subnormal channels; approximate exp2/multiply must not flush them.
    result = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[values, scales],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    result = tl.where(valid, result, 0.0)
    tl.store(output + row * head_dim + channels, result, channels < head_dim)


def gather_dequant_rows(pool, slots, head_dim, cache_format="main"):
    """Gather row-strided CSA2 bytes into BF16, without packed-row intermediates."""
    output = torch.empty((*slots.shape, head_dim), dtype=torch.bfloat16, device=pool.device)
    if slots.numel() == 0:
        return output
    if slots.ndim == 1:
        columns, row_stride, col_stride = slots.shape[0], 0, slots.stride(0)
    else:
        columns, row_stride, col_stride = slots.shape[1], slots.stride(0), slots.stride(1)
    _gather_dequant_rows[(slots.numel(),)](
        pool,
        slots,
        output,
        pool.stride(0),
        row_stride,
        col_stride,
        columns,
        pool.shape[0],
        head_dim,
        cache_format,
        triton.next_power_of_2(head_dim),
    )
    return output


@functools.lru_cache(maxsize=1)
def _indexer_projection_runner_type():
    """Import optional CuTe dependencies only when fused projection executes."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass._mlir.dialects import llvm

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLIndexerQBlackwellRunner
    from tensorrt_llm._torch.cute_dsl_kernels.blackwell.dense_blockscaled_gemm_persistent import (
        Sm100BlockScaledPersistentDenseGemmKernel,
        _indexer_q_pack_fp4x4,
    )
    from tensorrt_llm.quantization.utils.fp4_utils import pad_up

    class _CSA2IndexerQKernel(Sm100BlockScaledPersistentDenseGemmKernel):
        """Existing small-M MMA with CSA2 nearest-even FP4 and its scale floor."""

        @cute.jit
        def _indexer_q_transform_rows(
            self,
            sC: cute.Tensor,
            c_buffer: cutlass.Int32,
            row_start: cutlass.Int32,
            row_stride: cutlass.Constexpr,
            head_idx: cutlass.Int32,
            token_tile_idx: cutlass.Int32,
            batch_idx: cutlass.Int32,
            real_subtile_idx: cutlass.Int32,
            mPacked_nml: cute.Tensor,
            mIndexerScale_nml: cute.Tensor,
            mPositionIds: cute.Tensor,
            mCosSinCache: cute.Tensor,
        ):
            """Transform complete BF16 token rows from the shared epilogue tile."""
            lane_idx = cute.arch.lane_idx()
            for row in cutlass.range(row_start, self.epi_tile_n, row_stride, unroll_full=True):
                token_idx = (
                    token_tile_idx * self.cta_tile_shape_mnk[1]
                    + real_subtile_idx * self.epi_tile_n
                    + row
                )
                if token_idx < mPositionIds.shape[0]:
                    values = cute.make_rmem_tensor((4,), cutlass.Float32)
                    for value_idx in cutlass.range_constexpr(4):
                        feature_idx = lane_idx * 4 + value_idx
                        values[value_idx] = cutlass.Float32(sC[(feature_idx, row, c_buffer)])

                    if lane_idx >= 16:
                        position = mPositionIds[token_idx]
                        pair_base = (lane_idx * 4 - 64) // 2
                        for value_idx in cutlass.range_constexpr(0, 4, 2):
                            cosine = mCosSinCache[position, pair_base + value_idx // 2]
                            sine = mCosSinCache[position, pair_base + value_idx // 2 + 32]
                            x = values[value_idx]
                            y = values[value_idx + 1]
                            values[value_idx] = (
                                (cosine * x - sine * y).to(cutlass.BFloat16).to(cutlass.Float32)
                            )
                            values[value_idx + 1] = (
                                (cosine * y + sine * x).to(cutlass.BFloat16).to(cutlass.Float32)
                            )

                    amax = cute.arch.fmax(
                        cute.arch.fmax(values[0], -values[0]),
                        cute.arch.fmax(values[1], -values[1]),
                    )
                    amax = cute.arch.fmax(
                        amax,
                        cute.arch.fmax(
                            cute.arch.fmax(values[2], -values[2]),
                            cute.arch.fmax(values[3], -values[3]),
                        ),
                    )
                    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
                    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
                    amax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=4))
                    if amax < cutlass.Float32(6.0 * 2.0**-126):
                        amax = cutlass.Float32(6.0 * 2.0**-126)

                    scale_reg = cute.make_rmem_tensor((1,), cutlass.Float8E8M0FNU)
                    scale_reg[0] = (amax * cutlass.Float32(1.0 / 6.0)).to(cutlass.Float8E8M0FNU)
                    scale_byte = cute.recast_tensor(scale_reg, cutlass.Uint8)[0]
                    exponent = cutlass.Uint32(scale_byte)
                    inverse_bits = cutlass.Uint32(0)
                    if exponent == cutlass.Uint32(254):
                        inverse_bits = cutlass.Uint32(0x00400000)
                    else:
                        inverse_bits = (cutlass.Uint32(254) - exponent) << 23
                    inverse_scale = cutlass.Float32(
                        llvm.bitcast(cutlass.Float32.mlir_type, inverse_bits.ir_value())
                    )
                    for value_idx in cutlass.range_constexpr(4):
                        values[value_idx] = values[value_idx] * inverse_scale

                    packed = _indexer_q_pack_fp4x4(
                        values[0],
                        values[1],
                        values[2],
                        values[3],
                    )
                    mPacked_nml[head_idx * 32 + lane_idx, token_idx, batch_idx] = packed

                    exponent0 = cutlass.Uint32(cute.arch.shuffle_sync(exponent, cutlass.Int32(0)))
                    exponent1 = cutlass.Uint32(cute.arch.shuffle_sync(exponent, cutlass.Int32(8)))
                    exponent2 = cutlass.Uint32(cute.arch.shuffle_sync(exponent, cutlass.Int32(16)))
                    exponent3 = cutlass.Uint32(cute.arch.shuffle_sync(exponent, cutlass.Int32(24)))
                    if lane_idx == 0:
                        mIndexerScale_nml[
                            head_idx,
                            token_idx,
                            batch_idx,
                        ] = exponent0 | (exponent1 << 8) | (exponent2 << 16) | (exponent3 << 24)

    class _CSA2IndexerQRunner(CuteDSLIndexerQBlackwellRunner):
        """Reuse tuning/launch contracts, with local quantization and epilogue."""

        small_m_kernel_class = _CSA2IndexerQKernel
        kernel_cache = {}

        def unique_id(self):
            return ("csa2_rne_32", self.use_tvm_ffi)

        def get_valid_tactics(self, inputs, profile, **kwargs):
            m, k = inputs[0].shape
            n = inputs[1].shape[0]
            return list(self._small_m_tactics) if self._small_m_kernel_is_supported(m, n, k) else []

        def forward(self, inputs, tactic):
            x, weight, weight_scale, positions, cos_sin, alpha = inputs
            m, k = x.shape
            n = weight.shape[0]
            if not self._small_m_kernel_is_supported(m, n, k):
                raise ValueError("Fused CSA2 index Q requires 1..16 rows and N/K divisible by 128")
            if tactic == -1:
                tactic = self._small_m_tactics[0 if m <= 4 else 1]
            kind, tile, cluster, prefetch, warps = tactic
            if kind != "swap_ab":
                raise ValueError("CSA2 index Q supports only the small-M epilogue specialization")
            # Unlike V4, both activation scaling and output FP4 rounding follow the
            # 32-channel CSA2 contract. The GEMM/transform kernel remains fused.
            data, scales = torch.ops.trtllm.mxfp8_quantize(x, True)
            packed = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
            output_scales = torch.empty((m, n // 32), dtype=torch.uint8, device=x.device)
            pointers = (
                self._ptr(data, cutlass.Float8E4M3FN),
                self._ptr(weight, cutlass.Float8E4M3FN),
                self._ptr(scales, cutlass.Float8E8M0FNU),
                self._ptr(weight_scale, cutlass.Float8E8M0FNU),
                self._ptr(packed, cutlass.Uint8),
                self._ptr(output_scales, cutlass.Float8E8M0FNU),
                self._ptr(positions, cutlass.Int32, 4),
                self._ptr(cos_sin, cutlass.Float32, 32),
            )
            alpha_cute = cute.runtime.from_dlpack(alpha)
            stream = (
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
                if self.use_tvm_ffi
                else cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            )
            key = (tile, cluster, prefetch, warps, self.use_tvm_ffi)
            dynamic = (
                m,
                n,
                k,
                pad_up(m, 128) // 128,
                pad_up(n, 128) // 128,
                pad_up(k // 32, 4) // 4,
                cos_sin.shape[0],
            )
            if key not in self.kernel_cache:
                kernel = self.small_m_kernel_class(
                    32,
                    tile,
                    cluster,
                    use_prefetch=prefetch,
                    indexer_q_fusion=True,
                    indexer_transform_warps=warps,
                )
                clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
                    cluster[0] * cluster[1]
                )
                self.kernel_cache[key] = cute.compile(
                    kernel.wrapper_indexer_q_swap_ab,
                    *dynamic,
                    1,
                    *pointers,
                    alpha_cute,
                    clusters,
                    stream,
                    options="--opt-level 2 --enable-tvm-ffi"
                    if self.use_tvm_ffi
                    else "--opt-level 2",
                )
            compiled = self.kernel_cache[key]
            if self.use_tvm_ffi:
                compiled(
                    *dynamic,
                    data.data_ptr(),
                    weight.data_ptr(),
                    scales.data_ptr(),
                    weight_scale.data_ptr(),
                    packed.data_ptr(),
                    output_scales.data_ptr(),
                    positions.data_ptr(),
                    cos_sin.data_ptr(),
                    alpha,
                )
            else:
                compiled(*dynamic, *pointers, alpha_cute, stream)
            return packed.view(torch.int8), output_scales.view(torch.int32)

    return _CSA2IndexerQRunner


@torch.library.custom_op(
    "trtllm::csa2_indexer_q_gemm_rope_fp4", mutates_args=(), device_types="cuda"
)
def csa2_indexer_q_gemm_rope_fp4(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native small-M projection with exact CSA2 activation/FP4 semantics."""
    from tensorrt_llm._torch.autotuner import AutoTuner

    runner = _indexer_projection_runner_type()()
    inputs = [x, weight, weight_scale, positions, cos_sin, alpha]
    _, tactic = AutoTuner.get().choose_one(
        "trtllm::csa2_indexer_q_gemm_rope_fp4", [runner], runner.tuning_config, inputs
    )
    return runner(inputs, tactic=tactic)


@csa2_indexer_q_gemm_rope_fp4.register_fake
def _fake_indexer_projection(x, weight, weight_scale, positions, cos_sin, alpha):
    return (
        x.new_empty((x.shape[0], weight.shape[0] // 2), dtype=torch.int8),
        x.new_empty((x.shape[0], weight.shape[0] // 128), dtype=torch.int32),
    )


def supports_packed_attention(
    q,
    swa_pool,
    main_pool,
    swa_indices,
    main_indices,
    sink,
    scale,
    *,
    position_ids=None,
    rotary_cos_sin=None,
):
    if not isinstance(q, torch.Tensor) or not q.is_cuda or q.ndim != 3:
        return False
    no_main = main_pool is None and main_indices is None
    if not no_main and (main_pool is None or main_indices is None):
        return False
    pools = (swa_pool,) if no_main else (swa_pool, main_pool)
    indices = (swa_indices,) if no_main else (swa_indices, main_indices)
    tensors = (q, sink, *pools, *indices)
    if position_ids is not None or rotary_cos_sin is not None:
        if position_ids is None or rotary_cos_sin is None:
            return False
        if not (
            isinstance(position_ids, torch.Tensor)
            and isinstance(rotary_cos_sin, torch.Tensor)
            and position_ids.dtype in (torch.int32, torch.int64)
            and position_ids.shape == (q.shape[0],)
            and position_ids.is_contiguous()
            and rotary_cos_sin.dtype == torch.float32
            and rotary_cos_sin.ndim == 3
            and rotary_cos_sin.shape[0] > 0
            and rotary_cos_sin.shape[1:] == (2, 32)
            and rotary_cos_sin.is_contiguous()
        ):
            return False
        tensors += (position_ids, rotary_cos_sin)
    return (
        all(isinstance(t, torch.Tensor) and t.is_cuda and t.device == q.device for t in tensors)
        and torch.cuda.get_device_capability(q.device) == (10, 0)
        and q.dtype == torch.bfloat16
        and q.ndim == 3
        and q.shape[2] == 512
        and q.shape[1] > 0
        and q.shape[1] % 16 == 0
        and q.is_contiguous()
        and all(
            p.dtype == torch.uint8 and p.ndim == 2 and p.shape[0] > 0 and p.stride(1) == 1
            for p in pools
        )
        and swa_pool.shape[1] == 528
        and (no_main or main_pool.shape[1] == 288)
        and all(
            i.dtype in (torch.int32, torch.int64)
            and i.ndim == 2
            and i.shape[0] == q.shape[0]
            and i.is_contiguous()
            for i in indices
        )
        and sink.dtype == torch.float32
        and sink.shape == (q.shape[1],)
        and sink.is_contiguous()
        and isinstance(scale, (int, float))
    )


def packed_attention_workspace_bytes(
    queries: int, heads: int, swa_width: int, main_width: int, num_sms: int
) -> int:
    """Transient FP32 split accumulators, maxima and denominators (no KV staging)."""
    if queries == 0:
        return 0
    splits = max(
        1,
        min(
            math.ceil((swa_width + main_width) / 64), max(1, 2 * num_sms // (queries * heads // 16))
        ),
    )
    return queries * heads * splits * 514 * 4


@functools.lru_cache(maxsize=1)
def _packed_attention_kernel_type():
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass.cute.nvgpu import warp

    class PackedAttention:
        @cute.jit
        def __call__(
            self,
            q,
            swa,
            main,
            si,
            mi,
            sink,
            out,
            partial,
            positions,
            cos_sin,
            scale: cutlass.Float32,
            stream,
        ):
            mma = cute.make_tiled_mma(
                warp.MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, (16, 8, 16)),
                cute.make_layout((1, 4, 1)),
            )
            self.kernel(q, swa, main, si, mi, sink, partial, scale, mma).launch(
                grid=(q.shape[0], q.shape[1] // 16, partial.shape[2]),
                block=(128, 1, 1),
                stream=stream,
            )

            self.reduce(partial, sink, out, positions, cos_sin).launch(
                grid=(q.shape[0], q.shape[1] // 4, 1), block=(128, 1, 1), stream=stream
            )

        @cute.kernel
        def kernel(
            self, q, swa, main, si, mi, sink, partial, scale: cutlass.Float32, mma: cute.TiledMma
        ):
            tid, _, _ = cute.arch.thread_idx()
            row, head_block, split = cute.arch.block_idx()
            swa_fp8 = cute.recast_tensor(swa, cutlass.Float8E4M3FN)
            main_fp8 = cute.recast_tensor(main, cutlass.Float8E4M3FN)
            smem = utils.SmemAllocator()
            sq = smem.allocate_tensor(
                cutlass.BFloat16, cute.make_layout((16, 512), stride=(512, 1)), 16
            )
            sk = smem.allocate_tensor(
                cutlass.BFloat16, cute.make_layout((64, 512), stride=(512, 1)), 16
            )
            sp = smem.allocate_tensor(
                cutlass.BFloat16, cute.make_layout((16, 64), stride=(64, 1)), 16
            )
            ss = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((16, 64), stride=(64, 1)), 16
            )
            stats = smem.allocate_tensor(
                cutlass.Float32, cute.make_layout((16, 3), stride=(3, 1)), 16
            )
            valid = smem.allocate_tensor(cutlass.Int32, cute.make_layout(64), 16)
            for j in cutlass.range_constexpr(64):
                offset = tid + j * 128
                sq[offset // 512, offset % 512] = q[
                    row, head_block * 16 + offset // 512, offset % 512
                ]
            if tid < 16:
                stats[tid, 0] = cutlass.Float32(float("-inf"))
                stats[tid, 1] = 0.0
            thr = mma.get_slice(tid)
            acc = cute.make_rmem_tensor(thr.partition_shape_C((16, 512)), cutlass.Float32)
            acc.fill(0.0)
            co = thr.partition_C(cute.make_identity_tensor((16, 512)))
            cs = thr.partition_C(cute.make_identity_tensor((16, 64)))
            sv = cute.make_tensor(sk.iterator, cute.make_layout((512, 64), stride=(1, 512)))
            cute.arch.barrier()
            for tile in range(
                split, cute.ceil_div(si.shape[1] + mi.shape[1], 64), partial.shape[2]
            ):
                if tid < 64:
                    column = tile * 64 + tid
                    slot = cutlass.Int64(-1)
                    capacity = cutlass.Int64(0)
                    if column < si.shape[1]:
                        slot = cutlass.Int64(si[row, column])
                        capacity = cutlass.Int64(swa.shape[0])
                    elif column < si.shape[1] + mi.shape[1]:
                        slot = cutlass.Int64(mi[row, column - si.shape[1]])
                        capacity = cutlass.Int64(main.shape[0])
                    valid[tid] = cutlass.Int32((slot >= 0) & (slot < capacity))
                for j in range(256):
                    offset = tid + j * 128
                    key = offset // 512
                    d = offset % 512
                    column = tile * 64 + key
                    value = cutlass.Float32(0.0)
                    if column < si.shape[1]:
                        slot = cutlass.Int64(si[row, column])
                        if (slot >= 0) & (slot < swa.shape[0]):
                            exponent = cutlass.Int32(swa[slot, 512 + d // 32]) - 127
                            value = cutlass.Float32(swa_fp8[slot, d]) * cute.exp2(
                                cutlass.Float32(exponent)
                            )
                    elif column < si.shape[1] + mi.shape[1]:
                        slot = cutlass.Int64(mi[row, column - si.shape[1]])
                        if (slot >= 0) & (slot < main.shape[0]):
                            packed = cutlass.Int32(main[slot, d // 2])
                            code = (packed >> ((d % 2) * 4)) & 15
                            mag = code & 7
                            value = cutlass.Float32(mag) * 0.5
                            if mag >= 4:
                                value = (1.0 + cutlass.Float32(mag % 2) * 0.5) * cute.exp2(
                                    cutlass.Float32(mag // 2 - 1)
                                )
                            if (code & 8) != 0:
                                value = -value
                            value = value * cutlass.Float32(main_fp8[slot, 256 + d // 16])
                    sk[key, d] = cutlass.BFloat16(value)
                cute.arch.barrier()
                scores = cute.make_rmem_tensor(thr.partition_shape_C((16, 64)), cutlass.Float32)
                scores.fill(0.0)
                for k in range(32):
                    qa = cute.local_tile(sq, (16, 16), (0, k))
                    kb = cute.local_tile(sk, (64, 16), (0, k))
                    pa = thr.partition_A(qa)
                    pb = thr.partition_B(kb)
                    ra = thr.make_fragment_A(pa)
                    rb = thr.make_fragment_B(pb)
                    cute.autovec_copy(pa, ra)
                    cute.autovec_copy(pb, rb)
                    cute.gemm(mma, scores, ra, rb, scores)
                for j in cutlass.range_constexpr(cute.size(scores)):
                    h, k = cs[j]
                    value = cutlass.Float32(float("-inf"))
                    if valid[k] != 0:
                        value = scores[j] * scale
                    ss[h, k] = value
                cute.arch.barrier()
                # One warp per head, four heads at a time. The reducer adds
                # the sink exactly once after merging disjoint key splits.
                lane = tid % 32
                for j in cutlass.range_constexpr(4):
                    h = tid // 32 + j * 4
                    a = ss[h, lane]
                    b = ss[h, lane + 32]
                    mx = cute.arch.warp_reduction_max(cute.arch.fmax(a, b))
                    mx = cute.arch.fmax(mx, stats[h, 0])
                    old_scale = cutlass.Float32(0.0)
                    if stats[h, 1] != 0.0:
                        old_scale = cute.exp2((stats[h, 0] - mx) * 1.4426950408889634)
                    safe_mx = mx
                    if mx == cutlass.Float32(float("-inf")):
                        safe_mx = cutlass.Float32(0.0)
                    a = cute.exp2((a - safe_mx) * 1.4426950408889634)
                    b = cute.exp2((b - safe_mx) * 1.4426950408889634)
                    denom = cute.arch.warp_reduction_sum(a + b)
                    sp[h, lane] = cutlass.BFloat16(a)
                    sp[h, lane + 32] = cutlass.BFloat16(b)
                    if lane == 0:
                        stats[h, 0] = mx
                        stats[h, 1] = stats[h, 1] * old_scale + denom
                        stats[h, 2] = old_scale
                cute.arch.barrier()
                for j in cutlass.range_constexpr(cute.size(acc)):
                    h, _ = co[j]
                    acc[j] = acc[j] * stats[h, 2]
                for k in range(4):
                    pa = thr.partition_A(cute.local_tile(sp, (16, 16), (0, k)))
                    pb = thr.partition_B(cute.local_tile(sv, (512, 16), (0, k)))
                    ra = thr.make_fragment_A(pa)
                    rb = thr.make_fragment_B(pb)
                    cute.autovec_copy(pa, ra)
                    cute.autovec_copy(pb, rb)
                    cute.gemm(mma, acc, ra, rb, acc)
                cute.arch.barrier()
            for j in cutlass.range_constexpr(cute.size(acc)):
                h, d = co[j]
                partial[row, head_block * 16 + h, split, d] = acc[j]

            if tid < 16:
                partial[row, head_block * 16 + tid, split, 512] = stats[tid, 0]
                partial[row, head_block * 16 + tid, split, 513] = stats[tid, 1]

        @cute.kernel
        def reduce(self, partial, sink, out, positions, cos_sin):
            tid, _, _ = cute.arch.thread_idx()
            row, heads, _ = cute.arch.block_idx()
            head = heads * 4 + tid // 32
            lane = tid % 32
            mx = sink[head]
            for split in range(partial.shape[2]):
                mx = cute.arch.fmax(mx, partial[row, head, split, 512])
            denom = cute.exp2((sink[head] - mx) * 1.4426950408889634)
            for split in range(partial.shape[2]):
                factor = cute.exp2((partial[row, head, split, 512] - mx) * 1.4426950408889634)
                denom += factor * partial[row, head, split, 513]
            for dblock in cutlass.range_constexpr(16):
                d = lane + 32 * dblock
                value = cutlass.Float32(0.0)
                for split in range(partial.shape[2]):
                    factor = cute.exp2((partial[row, head, split, 512] - mx) * 1.4426950408889634)
                    value += factor * partial[row, head, split, d]
                rounded = cutlass.BFloat16(value / denom)
                if cutlass.const_expr(positions is not None and dblock >= 14):
                    # Preserve the BF16 attention boundary before inverse RoPE.
                    own = cutlass.Float32(rounded)
                    other = cute.arch.shuffle_sync_bfly(own, offset=1)
                    position = cutlass.Int64(positions[row])
                    rotated = cutlass.Float32(float("nan"))
                    # Invalid positions never form out-of-bounds cache reads.
                    # Runtime metadata owns position validation.
                    if (position >= 0) & (position < cos_sin.shape[0]):
                        cosine = cos_sin[position, 0, (d - 448) // 2]
                        sine = cos_sin[position, 1, (d - 448) // 2]
                        if lane % 2 != 0:
                            sine = -sine
                        rotated = own * cosine + other * sine
                    rounded = cutlass.BFloat16(rotated)
                out[row, head, d] = rounded

    return PackedAttention


_PACKED_ATTENTION_COMPILED = {}


def packed_sparse_attention(
    q,
    swa_pool,
    main_pool,
    swa_indices,
    main_indices,
    sink,
    scale,
    *,
    output=None,
    workspace=None,
    position_ids=None,
    rotary_cos_sin=None,
):
    """Run direct packed attention; invalid indices have zero probability.

    Optional CUDA positions[Q] and FP32 cosine/sine[max_position,2,32]
    fuse inverse interleaved RoPE on the final 64 channels. Positions must
    be in range; invalid positions produce NaN in the rotated tail.
    """
    if not supports_packed_attention(
        q,
        swa_pool,
        main_pool,
        swa_indices,
        main_indices,
        sink,
        scale,
        position_ids=position_ids,
        rotary_cos_sin=rotary_cos_sin,
    ):
        raise ValueError("Unsupported CSA2 packed attention geometry, dtype or device")
    if main_pool is None:
        main_pool = torch.empty((1, 288), dtype=torch.uint8, device=q.device)
        main_indices = torch.empty((q.shape[0], 0), dtype=torch.int32, device=q.device)
    if output is None:
        output = torch.empty_like(q)
    if (
        output.shape != q.shape
        or output.dtype != q.dtype
        or output.device != q.device
        or not output.is_contiguous()
    ):
        raise ValueError("CSA2 packed output must match contiguous BF16 query shape/device")
    if q.shape[0] == 0:
        return output
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    workspace_bytes = packed_attention_workspace_bytes(
        q.shape[0],
        q.shape[1],
        swa_indices.shape[1],
        main_indices.shape[1],
        torch.cuda.get_device_properties(q.device).multi_processor_count,
    )
    splits = workspace_bytes // (q.shape[0] * q.shape[1] * 514 * 4)
    shape = (q.shape[0], q.shape[1], splits, 514)
    if workspace is None:
        workspace = torch.empty(shape, dtype=torch.float32, device=q.device)
    elif (
        workspace.shape != shape
        or workspace.dtype != torch.float32
        or workspace.device != q.device
        or not workspace.is_contiguous()
    ):
        raise ValueError("CSA2 packed workspace must match FP32 split geometry/device")
    partial = workspace
    tensors = (
        q,
        swa_pool,
        main_pool,
        swa_indices,
        main_indices,
        sink,
        output,
        partial,
        position_ids,
        rotary_cos_sin,
    )
    key = (
        q.device.index,
        tuple(None if t is None else (tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors),
    )
    views = tuple(None if t is None else from_dlpack(t.detach(), assumed_align=1) for t in tensors)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    if key not in _PACKED_ATTENTION_COMPILED:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm CSA2 packed attention before CUDA Graph capture")
        _PACKED_ATTENTION_COMPILED[key] = cute.compile(
            _packed_attention_kernel_type()(), *views, cutlass.Float32(scale), stream
        )
    _PACKED_ATTENTION_COMPILED[key](*views, cutlass.Float32(scale), stream)
    return output
