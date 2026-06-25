# Low-Level Wall Exposure Audit Goal

## Goal

Build an audit layer that makes tinygrad's hidden GPU control-surface limits visible.

The purpose is not just to record that a candidate won or lost. The purpose is to prove when a candidate lost because the winning primitive required lower-level control that current tinygrad machine search, scheduling, or codegen could not express.

One-line goal:

```text
Expose whether the optimal primitive is outside the current searchable/codegen surface,
then convert that hidden primitive into either a searchable primitive, an owned escape-hatch kernel, or a documented non-actionable wall.
```

## Problem Statement

Current machine search can only tune what tinygrad exposes.

For GPU performance, some wins live below that exposed search space. If those controls are not visible to search, then search can fail even when a faster implementation is known to be possible.

The audit must distinguish these cases:

| class | meaning |
|---|---|
| Search-space miss | The right choice existed conceptually, but was not part of the current search space. |
| Codegen miss | The schedule requires instructions or lowering patterns the renderer does not emit. |
| Runtime/dataflow miss | The kernel is fast locally, but required lifecycle work is outside the measured primitive. |
| Harness miss | The benchmark measured the wrong boundary or omitted required costs. |
| True wall | The path is bounded by hardware, bandwidth, occupancy, correctness, or quality constraints. |

## Low-Level Wall Classes

| wall type | examples |
|---|---|
| Instruction wall | `v_dot2`, MFMA/WMMA variant choice, DPP, `ds_bpermute`, packed dot, fp8/fp16 conversion intrinsics. |
| Memory hierarchy wall | LDS staging, vectorized LDS/global loads, cache behavior, double buffering, operand reuse. |
| Scheduling wall | `waitcnt` placement, barrier placement, software pipelining, register lifetime, occupancy tradeoffs. |
| Dataflow wall | QK/softmax/PV fusion, split/combine economics, q8 pack lifecycle, dequant placement, KV materialization avoidance. |
| Runtime wall | graph boundaries, launch count, cache identity, persistent KV lifecycle, host/device synchronization. |

## External Grounding

The design follows the same lesson as modern GPU compiler and kernel systems: search works only when the right control surface is exposed.

| reference | lesson |
|---|---|
| DeepSeek-V3 Technical Report, Sections 3.1-3.3, https://arxiv.org/html/2412.19437v1 | DeepSeek used low-level infrastructure work, including customized PTX instructions, custom communication kernels, FP8 dataflow, and compute/communication overlap. This is the model for dropping below framework abstractions when needed. |
| Ansor, https://www.usenix.org/conference/osdi20/presentation/zheng | Autotuning depends on search-space construction. A search system cannot find programs outside the represented space. |
| TVM MetaSchedule, https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html | Practical autotuning exposes explicit schedule primitives such as tiling, vectorization, thread binding, and tensorization. |
| Triton paper, https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf | Productive GPU programming still requires explicit tile-level control so the compiler can shape dataflow and memory behavior. |
| Triton intro, https://openai.com/index/triton/ | The goal is efficient custom GPU kernels through a higher-productivity abstraction, not unconstrained pure search. |

## Repo Grounding

| repo source | role |
|---|---|
| `structure/Development/performance-primitive-research-principles.md` | Defines a performance primitive as math plus layout, memory path, decomposition, compiler lowering, scheduling, and integration boundary. |
| `docs/gpu-lifecycle-primitive-coverage-tracker-20260624.md` | Tracks primitive coverage and marks codegen/ISA control and native-codegen portability as incomplete. |
| `bench/qk-decode-eval/HARNESS_GUIDE.md` | Defines benchmark authority: local A/B is diagnostic; clean W==D is decode promotion authority. |
| `docs/decode-campaign-final-synthesis-20260623.md` | Records decode evidence that owned AMDGCN primitives such as `v_dot2`, LDS, and cross-lane behavior matter. |

## Audit Question

For each decode or prefill performance gap, answer:

```text
Was this gap caused by a bad choice inside the current tinygrad search space,
or by a missing lower-level control surface?
```

Then classify the gap as one of:

| label | meaning |
|---|---|
| `INSIDE_SEARCH_SPACE` | Current search/codegen could express the faster path; search or policy failed. |
| `MISSING_SCHEDULE_PRIMITIVE` | The schedule needs a tunable primitive that is not represented. |
| `MISSING_RENDERER_PRIMITIVE` | The renderer cannot emit the required instruction/lowering. |
| `MISSING_RUNTIME_PRIMITIVE` | The kernel exists, but route integration, cache identity, launch graph, or lifecycle control is missing. |
| `HARNESS_BOUNDARY_ERROR` | The observed win/loss was caused by measuring the wrong boundary. |
| `TRUE_HARDWARE_WALL` | The path is bounded by roofline, occupancy, memory bandwidth, quality, or correctness constraints. |
| `UNKNOWN_WALL` | Evidence is insufficient; more instrumentation is required. |

## Required Artifacts

| artifact | purpose |
|---|---|
| `low_level_wall_inventory.json` | Machine-readable list of known wall classes and affected primitives. |
| `primitive_exposure_matrix.md` | Human-readable matrix of primitive, exposed, searched, emitted, measured, and needed status. |
| `decode_low_level_wall_audit.json` | Decode-specific result across GEMV, attention, KV, small ops, runtime, and ctx slope. |
| `prefill_low_level_wall_audit.json` | Prefill-specific result across GEMM, attention, layout, long-context stability, and runtime. |
| `search_space_gap_report.md` | Explicit explanation of where machine search failed because the needed primitive was not in the search space. |
| `promotion_decision.md` | Decision for each wall: expose in codegen, keep owned kernel, defer, or close. |

## Evidence Contract

A wall claim must include at least one structural source and one behavioral source.

Structural sources:

- source/render audit
- emitted kernel source
- disassembly or ISA summary
- renderer/codegen capability check
- route/materialization inspection

Behavioral sources:

- W==D decode benchmark
- whole-prefill benchmark
- local A/B diagnostic with correctness
- profiler attribution
- role timing by context
- reproducibility band

Do not promote a primitive from structural evidence alone. Do not claim a codegen wall from a benchmark alone.

## Success Criteria

| criterion | pass condition |
|---|---|
| Search boundary is explicit | Every candidate states which primitives were actually searchable. |
| Wall is measurable | Each claimed wall has benchmark evidence plus structural evidence. |
| Wall is classified | ISA, memory hierarchy, scheduling, dataflow, runtime, harness, or true hardware wall. |
| Action is decided | Each wall maps to expose, hand-own, defer, close, or investigate. |
| No fake wins | Local A/B cannot promote without W==D or whole-prefill authority. |
| Regression guard exists | Shipped or closed walls have an artifact or check that can catch reintroduction. |

## Decision Policy

| finding | decision |
|---|---|
| Faster path is inside current search space | Fix search policy, candidate generation, or evaluator pruning. |
| Faster path needs one missing schedule primitive | Add that primitive to the searchable schedule space. |
| Faster path needs missing renderer/ISA lowering | Add renderer/codegen support or keep an owned kernel while documenting the gap. |
| Faster path only wins outside lifecycle boundary | Do not promote; either integrate lifecycle costs or close. |
| Faster path is diagnostic but non-portable | Keep as oracle/reference, not default. |
| Faster path is bounded by true hardware wall | Close the lane and record the proof. |

## Intended Outcome

After this audit exists, future decode and prefill work should not say only:

```text
candidate failed
```

It should say:

```text
candidate failed because the required primitive was not expressible by current search/codegen,
and the missing control surface is X.
```

Or:

```text
candidate failed even though the primitive was expressible, so the issue is search policy, route integration, or a true wall.
```

This makes the project honest about what is machine-searchable today and what still requires a lower-level DeepSeek-style escape hatch.
