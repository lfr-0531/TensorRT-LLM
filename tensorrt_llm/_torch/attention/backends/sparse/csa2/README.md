<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compressed Sparse Attention 2 (CSA2)

This package implements CSA2 attention backends, cache management and runtime
metadata. It does not register a complete `DeepseekV41ForCausalLM` model or
implement the model's full causal encoder/decoder scheduling.

## Framework integration

`backend.py` defines three backends using the standard `TrtllmAttention`
forward and sparse prediction contract:

| Architecture | Backend | Computation |
| --- | --- | --- |
| SM100 family | `CSA2TrtllmAttention` | Native trtllm-gen dynamic sparse MLA |
| SM120/121 | `CSA2FlashInferAttention` | FlashInfer BF16 FA2 |
| SM90 | `CSA2FlashMLAAttention` | FlashMLA sparse BF16 |

The inherited forward owns output allocation, prediction-hook dispatch and
FMHA selection. `prediction.py` prepares native sparse inputs.
`attention/backends/fmha/csa2.py` supplies the FlashInfer/FlashMLA compute
libraries. There is no separate attention controller or alternate backend
forward API.

`CSA2SparseAttentionConfig(algorithm="csa2")` selects the cache manager through
the existing sparse registry. Cache geometry comes from checkpoint text
configuration rather than an independent set of LLM-argument overrides.
`ModelConfig` selects CSA2 for `deepseek_v41`/`deepseek_v41_text` configurations.

## Module and indexer

`DeepseekV41Attention` owns projections, RoPE, compression and grouped output
projection. Its forward consumes prepared `CSA2TrtllmMetadata` and directly
calls the selected backend over bounded query tiles. The module omits V4's
projected-Q head normalization. Index K is derived from the compressed main
latent before main-KV RoPE and quantization. Ratio two reuses the native V4
non-overlap compressor with FP32 state and zero APE; ratio one has no gate.

Prediction uses the existing DSA `Indexer` class in projection-free mode.
Its shared prepared-input QK-to-TopK flow is also used by DSA's existing prefill
paths. Native MQA dispatch, exact TopK and output handling belong to that class.
CSA2 adds its block-max candidate/latest-block rule, logical-position mapping
and Full/Reindex/Reuse decisions. Shared layers consume logical selections and
resolve physical pages afresh for their request view.

The selected main/SWA values are decoded into bounded BF16 compute pools.
Native trtllm-gen does not consume CSA2 FP4 bytes directly. FlashInfer FA2 also
avoids extra quantization into V4's different FP8 footer layout. FlashInfer and
FlashMLA emit BF16 output; custom masks and scaled/quantized outputs are rejected.

## Cache ownership and lifecycle

`CSA2CacheManager` specializes `KVCacheManagerV2`. It reuses the shared request
allocation, commit, prefix reuse, copy-on-write, scratch, release and tier
storage mechanisms. Only KV-source layers allocate global storage; every
attention layer owns private SWA storage. Only ratio-two KV owners allocate
FP32 KV/score compressor state.

Each global record stores 288 main bytes and 68 index bytes together. Main
uses E2M1/E4M3 scales per 16 channels; index uses E2M1/UE8M0 per 32 channels.
The main/index views have a 356-byte row stride and share physical page numbers
by construction. Copy-on-write and transfer operate on the complete record,
including both sets of scales. The published layout requires 890 global bytes
per original token, excluding bounded SWA and partial state.

SWA uses 528-byte E4M3/power-of-two-scale rows. All CSA2 formats include the
RoPE channels. Long prefill reads scratch-aware page mappings instead of
writing directly into a short circular window. Partial-state groups retain
both FP32 values and scores. No separate cache container manages allocations
or writes outside the cache manager.

## Runtime metadata

`CSA2TrtllmMetadata.prepare()` resolves scheduler request IDs, cached lengths
and V2 page converters into layer-specific `CSA2Batch` and compression inputs.
`CSA2Routing` belongs to one packed forward and cannot be recycled across eager
forwards. Source and consumer queries retain the same packed order.

Encoder source rows can be supplied independently of decoder query rows with
`set_source_batch()` before prepare. Compression output capacity follows source
rows. Incomplete groups produce zero-filled padding with position zero and
write slot -1; the packed-row store skips invalid slots and respects strided
views. Caller-provided source hidden states must match the prepared source batch.

Runtime metadata owns fixed-shape compute views shared by serialized layers.
Different query counts use independent native workspaces. Warm each view with
the standard forward before capture and set `is_cuda_graph` for captured calls.
Prepare refreshes persistent device metadata outside capture before replay.

## Scope and validation

Targeted tests exercise all CSA2 layer modes, quantized layouts, logical/paged
selection, inherited backend dispatch, shared DSA indexer computation, real V2
allocation/scratch/prefix reuse/COW, partial compressor state, changed graph
replay inputs and an actual attention module against an unfused reference.
Tests are under `tests/unittest/_torch/attention/sparse/csa2/`.

PP/CP and disabled-layer masks are rejected by the current cache integration.
Beam/speculative compressor-state rewind is rejected by runtime metadata.
Disaggregation role mappings preserve byte layout, but do not establish full
model disaggregation, CED scheduling or DSpark inference support. Whole-model
checkpoint parity and performance remain separate validation work. Before
whole-model serving, verify that profiling exercises every Full layer at the
maximum query tile and configured global width: gathered index rows and dense
logits workspace scale with those bounds.

Numerical definitions follow the official
[reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)
and [quantization kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py).
