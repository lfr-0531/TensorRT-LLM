<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compressed Sparse Attention 2 (CSA2)

This package contains the CSA2 attention component implementation. It is not
registered as a complete `DeepseekV41ForCausalLM` model. Executor metadata and
cache allocation integration are still required before serving the checkpoint.

## Computation and ownership

`CSA2Layout` separates SWA-only, Full, Reindex and Reuse layers. Ratio zero means
SWA-only; ratios two and one describe encoder and decoder global caches. Only
KV sources allocate main and index pools. Every attention layer has private
SWA storage. `CSA2Cache.global_bytes_per_token` accounts for the published
layout's 890 bytes per token of persistent main/index cache, excluding bounded
SWA and compression state.

`DeepseekV41Attention.from_hf_config` accepts the model's text configuration.
`load_hf_weights` validates global shapes before TP slicing, including grouped
O-LoRA, and converts checkpoint FP8 block scales to the component's BF16/FP32
weights. The indexer is replicated across TP ranks; attention heads and output
projection are sharded using TRT-LLM `Linear`. PP and CP require shared-state
transfer and are rejected.

The module omits V4's projected-Q head normalization. `CSA2Compressor` reuses the
V4 native non-overlap compressor, extended to ratio two, with FP32 projection
and persistent state. Its zero APE is a constant buffer. Ratio one has no gate.
Both paths return the normalized, unrotated main latent; index K is derived
before main-KV RoPE and quantization. The grouped output projection and inverse
RoPE reuse V4's implementation.

Main KV stores E2M1 values with E4M3 scales per 16 channels. Index Q/K use
UE8M0 scales per 32 channels. SWA uses E4M3 values with power-of-two scales per
32 channels. All formats include the RoPE channels. The packing helpers are
Torch implementations; their fusion and performance have not been evaluated.

The hierarchical indexer pins the latest visible block and uses block maxima
for candidate selection. Reindex gathers only candidate keys, then publishes
logical positions. Reuse shares these logical positions and resolves physical
slots afresh. `CSA2Routing` belongs to one packed forward; it must not survive
into the next eager forward or be shared between concurrent batches.

For 512-dimensional heads, the module's automatic compute policy follows V4:

| GPU architecture | CSA2 compute path |
| --- | --- |
| Hopper (SM90) | Repository-pinned FlashMLA sparse BF16 |
| Blackwell datacenter (SM100 family) | TRTLLM backend and native trtllm-gen dynamic sparse MLA |
| SM120/SM121 | FlashInfer BF16 ragged FA2 |

`CSA2TrtllmAttention` inherits V4's sparse forward facade and uses its existing
TRTLLM FMHA dispatch and native AttentionOp. The CSA2 controller supplies the
selected, dequantized rows through bounded BF16 staging pools. Valid SWA and
global rows are compacted together, then split at the native 128-row pool
boundary; padding is outside the actual sparse length. The packed persistent
FP4/FP8 caches retain their original CSA2 formats. This path does not imply
native trtllm-gen support for reading CSA2 FP4 bytes directly.

The SM120/SM121 implementation uses FlashInfer's 512-dimensional BF16 FA2 path.
V4's dedicated FlashInfer sparse MLA kernel requires its different FP8 footer
layout; repacking CSA2 into that layout would introduce extra quantization.
The BF16 path preserves decoded values and includes the sink using FA2's
returned softmax normalization statistics.

All paths compute one combined SWA/global attention with one sink. The explicit
`compute_backend="torch"` option is available for component/reference testing.
TRTLLM and FlashInfer own mutable staging/planning state and require serialized
calls on one stream per instance. Warm up each query shape before CUDA Graph
capture. Unsupported architectures fail explicitly.

## Caller contracts still needing executor integration

- Supply `CSA2Batch` pool-relative mappings and exact causal visible lengths.
  Source and consumer query rows must have identical packed order.
- Allocate distinct SWA write slots for a whole prefill chunk, including
  staging storage. Mapping a long chunk directly to a circular cache would
  overwrite history needed by earlier queries. Retire staging after use.
- Allocate main/index pools exactly once per KV source and bind both to the
  same physical slot mapping. Prefix reuse and transfer must preserve source
  ownership, quantization scales and partial compressor state.
- Supply completed-group positions at each group's first source token.
  CUDA Graph padding rows require valid dummy positions and private write
  slots; no uninitialized positions may reach RoPE.
- For decoder SWA replay, `global_hidden_states` can contain the encoder rows
  used to prepare global KV while query/SWA computation consumes fewer rows.
  Full bounded-replay scheduling, cache-hit reconstruction and DSpark request
  lifecycle are not implemented here.
- Supply `CSA2GlobalPages` for paged source caches. Full selection resolves
  page tables per query tile; Reindex resolves only selected candidate rows,
  avoiding a persistent tokens-by-context mapping.

## Validation boundaries

Tests in `tests/unittest/_torch/attention/sparse/csa2/` cover ownership,
quantized layouts, candidate selection, all three reuse modes, weight loading,
native ratio-2 compression, ratio-4/128 regression, hardware routing, and
FlashMLA/TRTLLM/FlashInfer numerics and graph replay. Native TRTLLM tests also
exercise multiple query tiles and Full/Reindex/Reuse layers.
A synthetic ratio-two Full-module regression covers incomplete/completed
compression groups, cache preservation and projected output against an unfused
reference. These tests do not establish released-checkpoint parity, complete
model inference, distributed execution, prefix-cache scheduling or performance
parity.

Numerical definitions follow the official
[reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)
and [quantization kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py).
