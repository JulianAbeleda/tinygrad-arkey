# Memory-Adaptive Prefill Architecture: Full Overlay vs Bounded Packed Tiles

Date: 2026-07-15
Scope: model-agnostic route selection at the device-memory boundary
Concrete evidence example: Qwen3 Q4_K_M on AMD gfx1100 with 24 GB VRAM

## Executive summary

tinygrad can use different weight representations for prefill and decode when the complete resident set fits:

- prefill expands the quantized linear weights into a model-sized FP16 overlay and uses WMMA/tensor-core GEMM;
- decode retains the packed Q4_K/Q6_K weights and uses generated single-token GEMV kernels.

The user still explicitly selects a model. The execution-route choice must not be keyed to that model's name or a
parameter-count label. It is a decision over the selected model's actual packed bytes,
overlay bytes, KV-cache bytes, peak workspace, available device memory, and target kernel capabilities. On the current
24 GB machine, Qwen3-8B and Qwen3-14B happen to exercise opposite branches and are useful evidence fixtures; they are
not the routing rule.

The intended non-fitting solution is not a smaller whole-model overlay. It is llama.cpp's bounded-lifetime architecture:
retain the complete model in packed form, quantize prompt activations to Q8_1, cooperatively stage only the current
weight and activation tiles in LDS, compute and accumulate the output tile, then reuse the same LDS allocation for the
next K epoch. The temporary decoded representation is tens of kilobytes per resident workgroup rather than tens of
gigabytes per model.

## The memory-fit decision

The route is admissible only when all simultaneously resident allocations fit the device budget:

```text
packed decode weights
+ FP16 prefill overlay
+ KV cache
+ activations, outputs, compiler/runtime scratch
<= admitted VRAM budget
```

The packed weights and FP16 overlay cannot be treated as alternatives in the current dual-path design. Packed weights
remain necessary after prefill because single-token decode is bandwidth-sensitive and uses the Q4_K/Q6_K
representation directly.

If the FP16 route is forced when this inequality does not hold, realizing the overlay eventually fails with a device
allocation/OOM. The intended behavior is to reject that route before allocation or select a memory-safe packed
prefill profile.

The safety planner should consume facts, not identities, and enumerate feasible strategies:

```text
device_budget = admissible free VRAM after safety reserve
base_residency = packed_weights + KV_cache + persistent_runtime_state
overlay_peak = base_residency + fp16_overlay + prefill_peak_workspace

if base_residency > device_budget:
    decline load, reduce context/KV representation, or use an explicit offload policy
else:
    feasible = []
    if overlay_peak <= device_budget and dense_overlay_kernel_is_supported:
        feasible += FULL_RESIDENT_OVERLAY candidates
    if bounded_packed_kernel_covers_every_required role/quant/shape:
        feasible += BOUNDED_PACKED_TILES candidates
    if correct direct-packed fallback_is_supported:
        feasible += DIRECT_PACKED_FALLBACK candidates
    if not feasible:
        fail closed; do not attempt a hidden full dequantization
    selected = machine_search(feasible, actual GPU, tensor inventory, workload objective)
```

Fit is a hard constraint, not a performance ordering. A full overlay that fits is not automatically fastest, and a
bounded packed route is not automatically preferable merely because the overlay does not fit. The planner proves
which complete routes are safe and semantically supported. Target-local machine search measures the feasible routes
and selects the best route for the requested objective.

For the current project the primary objective is steady-state end-to-end token throughput, with correctness, no OOM,
GPU health, and resource limits as hard gates. Kernel-only timing is attribution evidence, not the final selector.

Required decision inputs include:

- exact packed tensor allocation bytes, including alignment;
- exact overlay tensor bytes for the roles the dense route would cover;
- KV-cache bytes derived from batch, layers, KV heads, head dimension, context capacity, dtype, and scale metadata;
- peak prefill activations, outputs, temporary buffers, compiler/runtime reservations, and a safety margin;
- target support for the dense-overlay, bounded-packed, and fallback kernels;
- the quant format and shape of every routed tensor.

Parameter count may help estimate memory before loading metadata, but it is not an admission authority. Two models with
the same parameter count can have different quant mixes, tensor padding, context reservations, or device budgets and
therefore select different branches. Likewise, the same model can select different branches on 16 GB, 24 GB, and
48 GB devices or at different requested context capacities.

## Current implementation status and remaining debt

The user supplies the GGUF path. `from_gguf` now unconditionally scans the opened device and live VRAM itself; its
production API no longer accepts a replacement device scanner or a caller-provided baseline memory tier. Packed base
residency follows the real loader topology: one selected whole-file backing allocation rounded to the allocator
granularity reported by the device scan; per-tensor GGML payload accounting remains available for sidecars. The
production safety reserve is likewise derived from live occupied bytes and allocator granularity, rather than a fixed
percentage or absolute VRAM threshold. The planner compares:

```text
aligned selected-file backing allocation + route sidecars/overlay + KV/runtime peak + scanned dynamic reserve
```

against live free device memory. It does not branch on a filename, `8B`, `14B`, or a named VRAM class.

The production machine-search controller and CLI now expose the same boundary: model path plus workload/cache controls,
with no injectable device, VRAM facts, reserve policy, or execution seam. They construct the tinygrad execution seam
internally and perform one live hardware scan. The selected-model inventory also derives non-square attention Q/O
geometry from `head_count * head_dim` rather than assuming hidden-size-square matrices, and records a tied embedding as
an explicit fixed LM-head route when the GGUF omits `output.weight`.

The remaining debt is execution and evidence completeness, not model-size routing:

- only candidate-local physical M=512 is currently validated for the resident-overlay/direct bindings, with explicit
  logical-remainder-to-physical-M mapping;
- accelerated cached policies must still acquire a complete measured allocation-fact bridge before production binding;
- exact physical-buffer ownership and final schedule-lifetime manifests are being added; phase-level counter deltas
  remain reconciliation evidence and are not accepted as allocation-category authority;
- the bounded Q6_K direct packed fallback emits and executes, but cooperative promotion remains fail-closed;
- the full bounded Q4_K oracle-derived callback is still advancing through real gfx1100 compiler/resource blockers;
- no production-complete bounded packed-tile policy yet covers every required Q4_K/Q6_K invocation and tail.

The required refactor is one fact-derived `PrefillMemoryPlan` (name illustrative) that returns both the admitted
context/KV representation, the feasible strategy/candidate set, and the machine-selected strategy:

```text
FULL_RESIDENT_OVERLAY
BOUNDED_PACKED_TILES
DIRECT_PACKED_FALLBACK
REFUSE
```

The planner must carry its byte accounting and capability/coverage proof into runtime route selection. Profile IDs may
name reproducible measurements and candidate artifacts, but they must never determine semantic eligibility. Exact
model fixtures remain necessary for validation without becoming production branches.

The machine-search cache key must be made from canonical facts: GPU/backend/architecture and relevant resource facts,
the selected model's tensor-content/inventory identity and shapes, quant ABIs, context/prefill workload, objective,
candidate identities, and compiler/runtime revision. A model filename or `8B`/`14B` label must not participate. The
content identity distinguishes genuinely different selected models; renaming the same model does not change routing.

## Architecture A: the complete overlay fits

This is the full-overlay branch. The shipped Qwen3-8B route on the current 24 GB device is one concrete instance.

```text
                           GGUF model
                      packed Q4_K / Q6_K
                               |
             +-----------------+-----------------+
             |                                   |
      prefill, T > 1                       decode, T = 1
             |                                   |
 dequantize covered weights once          retain packed weights
             |                                   |
 resident model-sized FP16 overlay        generated Q4/Q6 GEMV
             |                                   |
 generated FP16 WMMA GEMM                 one activation vector
             |                                   |
 high reuse across prompt tokens          minimum weight bytes/token
             +-----------------+-----------------+
                               |
                            KV cache
```

Approximate fitting evidence fixture, Qwen3-8B on the current device:

```text
packed Q4_K_M model                 4.7 GB
covered FP16 prefill overlay       ~14 GB
KV cache and runtime buffers        remainder
                                  --------
total                               fits 24 GB
```

This representation split matches the two phases:

1. Prefill has hundreds of activation rows. A dense FP16 weight tile can be reused across those rows, so the one-time
   expansion cost buys high tensor-core utilization.
2. Decode has one activation row. There is little cross-token reuse, so retaining compressed weights minimizes global
   memory traffic per generated token.

For this fixture, the promoted prefill route is `prefill_wmma_lds_dbuf_generated`; it uses FP16 operands, FP32
accumulation, two LDS slots, and exact role/shape admission. The decode side uses the packed Q4_K G3 and generated
Q6_K GEMV families. A different model selects this architecture only if its own memory and capability facts pass.

## Architecture B: the complete overlay does not fit

Qwen3-14B on the current device is a concrete non-fitting evidence fixture:

```text
packed Q4_K_M model                 8.4 GB
covered/full FP16 weight overlay  ~26-28 GB
KV cache and runtime buffers        additional
                                  ---------
total                              ~35 GB or more
```

This fixture's allocation is impossible on a 24 GB device. Discarding the packed model does not solve it: the FP16 weights alone
exceed the practical device budget, and the packed representation is still required for bandwidth-efficient decode.
Reloading or rebuilding representations between prefill and decode would introduce a large phase-transition cost and
would still not make the FP16 model resident.

The current memory-safe tinygrad fallback therefore keeps only packed model weights:

```text
                         packed Q4_K / Q6_K model
                                    |
                 +------------------+------------------+
                 |                                     |
          prefill, T > 1                         decode, T = 1
                 |                                     |
       direct-packed generated GEMM             packed generated GEMV
                 |                                     |
 load/decode weights inside output work         Q4 G3 / Q6 generated
                 |                                     |
          no FP16 model overlay                  no FP16 model overlay
```

This branch is selected because of its calculated residency, not because the model is named `14B`. The same branch
must apply to any model/context/device combination whose overlay peak exceeds its admitted budget.

The direct-packed route is correct and memory-safe. Its present prefill structure does not yet provide llama MMQ's cooperative
decode/stage/reuse lifecycle: packed loads and dequantization occur inside a per-output GEMM body, so related output
work can repeat representation and load costs. It therefore avoids OOM but gives up much of the reuse that made the
full-overlay route fast.

## How llama.cpp handles the non-fitting case

The pinned llama.cpp prefill route retains packed model weights and constructs only bounded working tiles.

```text
                              packed model in VRAM
                                      |
prompt activations -> Q8_1 records    |
                 |                    |
                 +---------+----------+
                           |
               select one 128 x 128 output tile
                           |
                 loop over K in 256-wide epochs
                           |
          +----------------+----------------+
          |                                 |
 cooperative Q4_K load/decode       cooperative Q8_1 load
          |                                 |
          +--------------> LDS <------------+
                           |
                        barrier
                           |
          chained int8 WMMA groups + FP32 correction
                           |
                 accumulate output tile
                           |
             overwrite LDS for next K epoch
```

For the inspected Q4_K oracle, one workgroup's exact stage is approximately:

```text
decoded Q4_K weight tile             38,912 bytes
Q8 activation records                18,432 bytes
remaining stage/ABI state               512 bytes
                                      ------
total LDS stage                      57,856 bytes
```

The LDS allocation is on-chip workgroup-local storage. It is not allocated once for every model tensor in VRAM. GPU
hardware keeps only a bounded number of workgroups resident, and each workgroup repeatedly overwrites its stage while
walking K and later output tiles.

At pp512 with hidden size 5120, a complete Q8_1 prompt panel is only on the order of 3 MB including per-block metadata.
It is small relative to a model-sized FP16 overlay and can be reused across matrix multiplications where the runtime
contract permits it.

The key amortization is:

```text
decode one packed weight tile once
                |
reuse it across many prompt rows and output columns
                |
perform integer tensor-core products
                |
discard only when the workgroup advances to the next tile
```

## Decode is a separate packed architecture

Batch-1 generation does not use the prefill MMQ tile. llama.cpp selects MMVQ:

```text
activation vector -> Q8_1 once
                  -> one/few waves per output row
                  -> stream packed Q4_K/Q6_K weights
                  -> integer dot products and wave reduction
                  -> output vector
```

tinygrad's Q4_K G3 path is already MMVQ-like at a high level. The Q6_K path currently emits partials and then uses an
external reduction, and tinygrad consumes FP16 activations rather than a shared Q8_1 representation. Those are decode
design differences, not the cause of an overlay admission failure. In the current Qwen3-14B fixture, retained shallow
decode is approximately at llama parity; fixed-live-context measurements remain required before claiming a decode
performance deficit.

## Target tinygrad architecture for the non-fitting branch

The desired steady state keeps the packed model as the sole model-sized weight representation for both phases:

```text
                            packed Q4_K/Q6_K model
                                      |
                   +------------------+------------------+
                   |                                     |
            prefill MMQ                            decode MMVQ/GEMV
                   |                                     |
Q8 prompt + bounded cooperative LDS tiles       streamed packed weights
                   |                                     |
       integer WMMA + FP32 correction            register accumulation
                   |                                     |
          no full FP16 overlay                    no full FP16 overlay
```

For the current Qwen3-14B development fixture, the exact Q4_K/Q8_1 MMQ substrate now describes the physical record ABI, cooperative producers, one 128x128x256
epoch, the two-K16-per-K32 recurrence, and the runtime address/tail contract. It is not yet a complete emitted and
production-routed outer-K kernel. Completion requires joining those pieces into the executable full kernel, proving
correctness and target resource evidence, binding the actual admitted role inventory, and demonstrating whole-model
tok/s improvement. The resulting emitter and admission contract must remain model-agnostic.

## Architectural invariants

Any implementation of the non-fitting branch should preserve these properties:

1. No model-sized FP16 weight overlay or hidden full dequantization.
2. Packed Q4_K/Q6_K weights remain the authoritative resident representation.
3. Temporary decoded state has a bounded tile lifetime and an explicit storage budget.
4. Cooperative production, barriers, fragment ownership, and correction terms are represented by the generated
   kernel vocabulary rather than profile-specific address constants.
5. Prefill MMQ and batch-1 decode MMVQ remain separate routes with separate evidence.
6. Admission is derived from memory/capability facts; model names, parameter-count labels, and copied profile grids are
   not semantic selectors.
7. Promotion requires full-output correctness, route-census proof, final target resource evidence, GPU health, and
   end-to-end measurements from the same model and workload definition.
