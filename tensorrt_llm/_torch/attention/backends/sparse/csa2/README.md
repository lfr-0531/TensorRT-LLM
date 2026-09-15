<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Compressed Sparse Attention 2 (CSA2)

This package implements DeepSeek-V4.1 CSA2 attention modules, backends,
indexing, cache ownership and runtime metadata. Complete model registration
and causal encoder/decoder (CED) execution remain separate integration work.

## Backend and hardware routing

`CSA2TrtllmAttention` owns the shared cache/indexer preparation. On the default
SM100 path it inherits `TrtllmAttention.forward`, including output allocation
and native FMHA selection. The module directly invokes its composed Flash
helper on other hardware; there is no CSA2 FMHA registration or controller.

| Architecture | Module compute path | Computation |
| --- | --- | --- |
| SM100 family | `CSA2TrtllmAttention.forward` | Native trtllm-gen dynamic sparse MLA |
| SM120/121 | `CSA2FlashInfer` | FlashInfer BF16 FA2 |
| SM90 | `CSA2FlashMLA` | FlashMLA sparse BF16 |

`backend.py` contains the single sparse backend and both plain Flash helper
classes. The opt-in packed path uses the backend's explicit `forward_packed`
method. Every path invokes the same sparse preparation once per tile;
CSA2-specific GPU kernels are collected in `kernel.py`.

The default compute path decodes selected SWA/main rows into bounded BF16
staging pools. Native trtllm-gen does not consume the persistent CSA2 FP4
format directly. FlashInfer likewise avoids requantizing into V4's different
FP8 footer cache format. The library adapters return BF16; unsupported custom
masks and scaled/quantized output contracts are rejected.

Eager native context preserves real request query groups. Its virtual staged
KV coordinates prevent the native causal mask from clipping already selected
rows; actual source causality remains encoded in selection and visibility.
Generation and fixed-shape context graphs use independent-query generation
staging. A tile contains only one compute phase.

`CSA2SparseAttentionConfig(algorithm="csa2")` selects the cache manager through
the existing sparse registry. Geometry comes from the checkpoint text
configuration. `ModelConfig` recognizes `deepseek_v41`/`deepseek_v41_text` for
this attention configuration; this does not register a complete causal LM.

## Projections, compression and indexing

`DeepseekV41Attention` owns projections, normalization, RoPE, compression and
grouped output projection. It consumes prepared `CSA2TrtllmMetadata`, retains
CSA2's omission of projected-Q head normalization, and derives index K from
the compressed main latent before main-KV RoPE. Ratio-two compression reuses
the native V4 non-overlap compressor with FP32 value/gate state and zero APE.
Ratio one has an uncompressed global cache and no compression gate.

BF16 projection execution is the default. Optional
`projection_quantization="mxfp8"` uses native MXFP8 linear/grouped output
operations on SM100/SM103, with checkpoint scale validation and support for
the admitted 32/128-block weight layouts. Eligible small-query MXFP8 index-Q
projection can fuse GEMM, interleaved RoPE and CSA2 nearest-even FP4 conversion
with `fuse_index_q`; larger queries retain the unfused native projection path.
An optional auxiliary stream overlaps compression/index preparation with Q
work when the existing multi-stream policy is enabled.

`CSA2Indexer` subclasses DSA `Indexer` in projection-free mode and reuses its
MQA kernels and `TopK` module. CSA2-specific phase orchestration, candidate
loading and logical mapping remain local. Indexing runs once for the complete
layer batch before attention tiles consume the results. Full prefill gathers
cached and new keys once per request chunk; bounded query tiling limits
logits workspace. TP query splitting uses the shared partition/allgather
primitives and synchronizes candidate-source outputs as well as selections.

Unrestricted SM100 decode uses native paged FP4 MQA for supported index-head
geometry, including padding smaller head counts into native specializations.
Metadata repacks exact index bytes into native 64-entry footer pages. Missing
pages map to a zero page and are excluded from logical selection. Restricted
candidates and other hardware use bounded gathering with inherited MQA;
decode retains decode TopK semantics. Reuse layers consume logical selections
from their index source and resolve current physical pages without recomputing
indexer logits.

Exact CUDA TopK is the default; short sequences whose keys all fit can
enumerate visible positions without logits or TopK. Internal `CSA2Params` also exposes eligible CuTe DSL exact
TopK, self-sampling/temporal GVR, and CuTe paged MQA/emission options. Temporal
priors use request allocation epochs and logical positions, seed first decode
from accepted prefill, and follow request identity across reordering. Rewind
and multi-query verification invalidate unsafe hints. Emission state resets
before replay when row ownership changes and on target/draft transitions. Candidate-restricted Reindex does not consume temporal hints from
an incompatible selection domain.

## Cache lifecycle and speculation

`CSA2CacheManager` specializes `KVCacheManagerV2` and reuses allocation,
commit, prefix reuse, copy-on-write, scratch, release and tier storage. Every
attention layer has private SWA storage; only KV-source layers own global
main/index storage, and only ratio-two owners allocate FP32 compressor state.

Each global record combines 288 main bytes and 68 index bytes. Main uses
NVFP4 E2M1 values with E4M3 scales per 16 channels; index uses E2M1 values
with UE8M0 scales per 32 channels. Their
views retain a 356-byte row stride and identical physical page numbering, so
copy-on-write and transfer preserve both payloads and scales together. The
published layout uses 890 global bytes per original token, excluding SWA,
compressor state and compute workspace. SWA uses 528-byte E4M3/power-of-two
rows. All formats include their RoPE channels.
These persistent formats are part of CSA2's cache contract and do not require
a generic NVFP4 KV-cache flag. Projection precision and the selected attention
compute path do not change the persistent main-cache encoding.

Long prefill uses scratch-aware page mappings. Partial groups retain raw FP32
values and scores. CUDA BF16 cache publication on SM100 uses fused exact-byte
quantize/scatter; other paths retain the existing encoder and masked scatter.
Invalid slots, including padding slot -1, never write a cache row.
The default staged attention path also uses fused gather/dequantization on
SM100 to read selected packed rows directly into BF16, including strided
main/index views. Invalid read slots produce zero rows. The packing and
unpacking helpers in `quantization.py` remain the reference fallback for
unsupported gather layouts or hardware.

Attention-local contiguous chain verification supports accepted-prefix rewind
through V2 resource updates. SWA/state windows and scratch rewind capacity
cover the configured draft tail; continuation recomputes incomplete groups
and cannot expose rejected compressed rows. Target/draft metadata hooks
support identical explicit layer layouts and independent cache-dependent
state. Non-linear trees, beams, explicit token relocation indices and implicit
virtual draft-layer mappings remain rejected. These local mechanisms do not
establish complete model-level MTP drafting or generation.

## SWA bounded replay

After an authoritative cached GLOBAL prefix of length C, the caller can use
`get_swa_replay_ranges()` to plan writable query intervals and
`set_swa_bounded_replay()` to prepare their reconstruction. Encoder replay
starts at `max(0, C - window_size)` and may include an uncached suffix. Each
query's SWA is restricted to the replay segment. This reconstruction is
approximate across layers; its reference is truncated replay, not a complete
historical forward.

Cached main/index records remain read-only. Only uncached source rows produce
new GLOBAL entries. At an odd ratio-two boundary, the last cached raw token
is included to reconstruct partial compressor state before continuation.
Pure replay with no required source rows skips global projection. Decoder
replay reconstructs only the final window and requires prompt GLOBAL entries
to be already prepared.

Replay preparation is one-shot. The caller supplies the declared query inputs
and provisions every writable row in the returned interval; an ordinary
retained SWA window may omit its first reconstruction token. Captured replay
can refresh same-geometry positions and page mappings, while changes to replay
mode or source geometry require fresh metadata and recapture.

Automatic GLOBAL-only reuse is enabled by default when the layout has a
GLOBAL owner. Native and Python V2 prefix matching use persistent GLOBAL
coverage; private SWA and compressor-state pages are reconstructed instead of
being published into the reuse trie. SWA-only layouts retain ordinary matching.
The manager can explicitly disable this policy with
`enable_swa_bounded_replay=False`.

For a GLOBAL hit at C, request preparation preserves C as the reused-prefix
length and rewinds the compute cursor to `max(0, C - window_size)`. Metadata
prepares replay intervals from actual request IDs, including partial chunks.
Physical SWA retention includes one extra row for the first replay query.
The manager acknowledges reconstruction only after every member layer has
completed its forward and its CUDA event has completed; pending reconstruction
cannot be committed. Target and draft claims settle a common replay cursor.
Reaching C again after replay does not repeat initial prefix settlement.

Automatic reconstruction runs eagerly, including a final context token promoted
to generation. Normal CUDA Graph eligibility resumes after acknowledgement.
The explicit fixed-geometry replay interface above retains its separate graph
contract. These runtime mechanisms do not supply the complete V4.1 model's
encoder-to-decoder activation handoff.

## Metadata, graphs and workspace

`CSA2TrtllmMetadata.prepare()` resolves real request IDs, cached lengths and V2
page converters. Layer-specific SWA/visibility, owner page tables/write slots,
compression inputs, candidates and routing results live directly on metadata.
`global_slot_tile()` resolves pages without a persistent token-by-context map.

`set_source_batch()` permits encoder source lengths to differ from decoder
query lengths. Compressor output capacity follows source rows; incomplete
outputs are zero-filled with position zero and write slot -1. Supplied source
hidden states must match that prepared source batch. Full CED scheduling is
not supplied by this interface.

Set outer metadata's `is_cuda_graph` before warmup, and warm the standard
forward before capture. Host prepare refreshes persistent device metadata
before replay. Calls and replays sharing metadata must remain serialized.
Eager buffers replace obsolete geometry; native indexer staging uses one
configured maximum graph arena per device/topk across graph batch sizes and
serialized owners. Replays refresh packed pages, visibility and scheduling.

`workspace_reservation_bytes()` reports one retained native index arena from
explicit serving capacities. `get_workspace_bytes()` deduplicates retained
metadata/frame/arena/prior storage separately from manager pools. Cache-gather
component estimates exclude query-dependent logits and other temporaries;
the optional packed kernel has its own workspace-size helper. A resolved cap
that cannot cover the admitted graph arena is rejected before allocation.
These attention-local APIs do not install generic executor admission or a
complete serving-memory reserve. Model integration must account for fixed
arenas, projection/provider/native workspace and measured transient peaks at
the actual serving geometry.

## Optional packed attention and remaining scope

`use_packed_sparse_attention` enables an eligible SM100 BF16-Q kernel that
reads packed SWA/main rows directly, bypassing BF16 selected-KV staging. It
uses split-KV partial workspace. `fuse_packed_output_rope` additionally fuses
inverse interleaved RoPE into the output reduction. Both are off by default;
select them with workload-specific numerical validation and profiling.

PP/CP and disabled-layer masks are unsupported. Disaggregation role mappings
preserve cache bytes but do not establish whole-model disaggregation, CED or
DSpark execution. Whole-model checkpoint parity, model/executor integration
and whole-model performance remain outside this package's validated scope.
Component/runtime tests live in `tests/unittest/_torch/attention/sparse/csa2/`.

Numerical definitions follow the official
[reference implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/model.py)
and [quantization kernels](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/kernel.py).
