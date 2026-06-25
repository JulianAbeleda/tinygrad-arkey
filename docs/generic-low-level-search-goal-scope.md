# Generic Low-Level Search Goal Scope

## Goal

Build toward a generic GPU search system that can discover decode and prefill performance primitives only after the relevant low-level control surfaces are explicitly exposed.

The goal is not generic search over the current tinygrad surface. The goal is generic search over a richer, declared hardware/compiler/lifecycle primitive space.

One-line goal:

```text
Make the winning GPU primitive searchable by first exposing the instruction, memory, scheduling, dataflow, and runtime controls that define it.
```

## Core Principle

Generic search works only if the optimal program is inside the represented search space.

If the search space lacks the needed primitive, the search can still be exhaustive and still fail.

Bad target:

```text
Run more generic search over the current exposed knobs.
```

Better target:

```text
Expose the missing low-level primitive classes, then run generic search over those classes with correctness and lifecycle gates.
```

## Why This Matters

The current project has two layers that should not be confused:

| layer | current role | limitation |
|---|---|---|
| Audit/evaluator | Measures candidates and decides promotion/refutation. | Does not generate lower-level programs. |
| Candidate registry | Encodes curated candidate lanes. | Not generic kernel search. |
| tinygrad codegen/BEAM | Searches/tunes some exposed compiler choices. | Cannot find primitives that UOps/renderers/scheduler cannot express. |
| Owned kernels | Prove the lower-level target exists. | Not generic or portable by default. |
| Profile system | Makes benchmark authority portable across model/device/workload profiles. | Does not itself make kernel search generic. |

The generic-search goal is to connect these layers:

```text
audit identifies missing control -> expose primitive -> generic search explores it -> evaluator promotes/refutes by profile
```

## Search-Space Completeness Requirement

A search system is only valid for a performance claim if the artifact states what was searchable.

Each search run must declare:

| field | question |
|---|---|
| `search_space_id` | Which primitive space was searched? |
| `exposed_instruction_primitives` | Which ISA/tensor/vector instructions could be selected? |
| `exposed_memory_primitives` | Could the search choose LDS/global/register staging and vector loads? |
| `exposed_scheduling_primitives` | Could it choose waitcnt/barriers/software pipeline/register lifetime knobs? |
| `exposed_dataflow_primitives` | Could it choose fusion/split/combine/online softmax/q8 lifecycle? |
| `exposed_runtime_primitives` | Could it choose graph boundaries/cache identity/persistent buffers? |
| `excluded_primitives` | What known relevant controls were not searchable? |
| `proof_of_coverage` | Why is this space sufficient for the claim being made? |

## Primitive Vocabulary To Expose

### Instruction primitives

| primitive | examples |
|---|---|
| Tensor/matrix instructions | WMMA, MFMA, tensorized dot variants. |
| Vector dot instructions | `v_dot2`, packed dot, dot4-style operations where valid. |
| Cross-lane movement | `ds_bpermute`, DPP, lane shuffle/reduction idioms. |
| Conversion intrinsics | fp8/fp16/f32 conversion, packed load/decode patterns. |
| Special math | exp/log approximations and softmax-friendly approximations where quality permits. |

### Memory hierarchy primitives

| primitive | examples |
|---|---|
| Global load form | scalar, vectorized, packed, coalesced, cached/non-cached if available. |
| LDS staging | global-to-LDS tile, LDS vector reads, bank-conflict-aware layout. |
| Register staging | direct register tile, accumulator layout, operand reuse. |
| Double buffering | ping-pong LDS/register buffers, prefetch distance. |
| Cache identity | whole-buffer read, no materializing slice/view buffers. |

### Scheduling primitives

| primitive | examples |
|---|---|
| Workgroup shape | threads/block, waves/block, rows/head/splits per block. |
| Occupancy tradeoff | VGPR pressure, LDS bytes, spills, waves per CU. |
| Wait/barrier schedule | `s_waitcnt`, `s_barrier`, dependency placement. |
| Software pipeline | prologue, steady-state K loop, epilogue. |
| Reduction schedule | lane reduction, LDS reduction, multi-stage combine. |

### Dataflow primitives

| primitive | examples |
|---|---|
| Decode attention | QK, online softmax, PV, split-KV, combine. |
| Prefill GEMM | tiled WMMA/MFMA, LDS-pipelined K loop, operand reuse. |
| Quant lifecycle | q8 pack, dequant, scale/min decode, qsum/min correction. |
| Fusion boundary | norm/RoPE/residual/activation/copy fusion where measurable. |
| Layout lifecycle | avoid format conversions that erase kernel wins. |

### Runtime primitives

| primitive | examples |
|---|---|
| Graph boundary | number of programs per token, JIT replay, launch grouping. |
| Persistent state | KV cache allocation, append/read ordering, buffer reuse. |
| Synchronization | item sync, host/device sync, profile/debug contamination. |
| Route policy | ctx thresholds, shape guards, fallback behavior. |

## Generic Search Loop

The desired loop is:

```text
1. Select benchmark profile.
2. Load candidate/search-space definition.
3. Validate that the profile shape is supported.
4. Generate candidates from exposed primitive vocabulary.
5. Run structural checks before timing.
6. Run correctness/quality gate.
7. Run local diagnostic gate.
8. Run W==D or whole-prefill authority gate.
9. Classify result: promote, opt-in, refute, defer, or expose-more-primitives.
10. Record excluded primitives so failed search does not masquerade as a true wall.
```

## Search Result Labels

A generic search result should use labels that distinguish search failure from primitive exposure failure.

| label | meaning |
|---|---|
| `SEARCH_FOUND_PROMOTABLE` | Search found a candidate that passed profile authority gates. |
| `SEARCH_FOUND_LOCAL_ONLY` | Search found a local win that failed lifecycle authority. |
| `SEARCH_EXHAUSTED_SPACE` | Search exhausted the declared space and found no winner. |
| `SEARCH_SPACE_INCOMPLETE` | Audit evidence shows the needed primitive was excluded. |
| `SEARCH_BLOCKED_BY_CODEGEN` | Search wanted a primitive the renderer/scheduler cannot emit. |
| `SEARCH_BLOCKED_BY_RUNTIME` | Candidate requires graph/cache/lifecycle control not exposed. |
| `SEARCH_BLOCKED_BY_CORRECTNESS` | Candidate violates exactness/quality tolerance. |
| `SEARCH_BLOCKED_BY_PROFILE` | Candidate is valid for another profile but not this model/device/workload. |

## Relationship To Profile System

The profile system defines the benchmark question.

The generic search system defines the candidate space.

They are orthogonal:

```text
profile = model/device/workload/comparator/thresholds
search space = primitive vocabulary and generation rules
candidate = one sampled/generated point in that space
verdict = profile-relative decision
```

This separation is required for model-agnostic behavior.

A candidate does not need to support every profile. But the search framework must be able to say:

```text
This candidate/search space is unsupported for this profile.
```

Instead of silently relying on hardcoded shape assumptions.

## Relationship To Owned Kernels

Owned kernels are not the end state of generic search, but they are useful evidence.

Use owned kernels as:

| use | valid? | reason |
|---|---|---|
| Oracle target | yes | Proves a lower-level primitive can win. |
| Structural guide | yes | Shows which instructions/memory/schedule choices matter. |
| Default route | yes, if W==D/whole-prefill gates pass and fallback is safe. |
| Proof of generic search | no | Hand-owned code is not evidence that the generic search space can express the same primitive. |

The generic search milestone is reached only when the search space can represent the owned-kernel lesson without hand-writing the whole kernel.

## Practical Milestones

### Milestone 1: Explicit search-space manifests

Add machine-readable search-space manifests.

Example:

`bench/qk-search-spaces/decode_attention_gfx1100_v1.json`

Fields:

- primitive family
- supported profiles
- exposed instruction primitives
- exposed memory primitives
- exposed scheduling primitives
- exposed dataflow primitives
- exposed runtime primitives
- excluded known primitives
- correctness gate
- lifecycle authority gate

### Milestone 2: Bind candidates to search spaces

Every candidate in `bench/qk-decode-eval/candidates.json` should state either:

```json
"search_space_id": "decode_attention_gfx1100_v1"
```

Or:

```json
"search_space_id": "manual_oracle_not_search_generated"
```

This prevents hand-owned or oracle candidates from being mistaken for generic search results.

### Milestone 3: Add primitive-exposure audit to every failed search

A failed generic search must report:

- what was searched
- what was not searched
- whether an oracle/owned kernel used excluded primitives
- whether the failure is a true wall or a search-space incompleteness

### Milestone 4: First generic low-level search lane

Choose one bounded lane where the vocabulary is small enough to be real.

Recommended first lane:

```text
decode attention split/combine policy + KV identity + route thresholds
```

Reason:

- existing W==D authority exists
- correctness gates exist
- known ctx ladder exists
- search dimensions are bounded
- this lane exercises lifecycle search without requiring full ISA generation immediately

### Milestone 5: First codegen-exposure lane

Choose one missing lower-level primitive and expose it as a searchable/codegen option.

Recommended first targets:

| target | reason |
|---|---|
| cross-lane reduction primitive | Owned GEMV/attention lessons repeatedly need lane reductions. |
| `v_dot2`/packed dot lowering | Decode attention and GEMV audits identify this as a key missing low-level control. |
| LDS/vector-load schedule primitive | Prefill and attention walls repeatedly point at memory staging. |

### Milestone 6: Compare search-generated vs owned/oracle target

A search-generated candidate should be compared against:

- current default
- owned/oracle target, if one exists
- profile thresholds
- structural evidence that the same primitive class was emitted

This is the moment where the project can claim movement from hand-owned primitive to generic searchable primitive.

## Success Criteria

| criterion | pass condition |
|---|---|
| Search space declared | Every generic search run has a search-space manifest. |
| Exclusions recorded | Known missing primitives are listed, not hidden. |
| Profile separated | Model/device/workload thresholds come from profile, not candidate code. |
| Candidate provenance clear | Candidate says whether it was generated, manual, oracle, or owned. |
| Failure classified correctly | A failed search cannot be called a true wall if the winning oracle used excluded primitives. |
| First generated primitive lands | At least one candidate generated from an exposed low-level search space passes local and lifecycle gates. |
| Owned-kernel lesson transferred | At least one previously hand-owned primitive class becomes representable by search/codegen. |

## Non-Goals

This scope does not require:

- full ISA superoptimization
- replacing all owned AMDGCN kernels
- supporting every GPU backend at once
- discovering tensor-core intrinsics from first principles
- removing profile-specific shape guards
- proving upstream-style BEAM is worse or better

The immediate goal is narrower and more useful:

```text
Make search honest about what it can and cannot express, then expand the expressible space where audits show it matters.
```

## Final Target Statement

The long-term generic-search target is:

```text
Given a declared profile and a declared low-level primitive vocabulary,
the system can generate, evaluate, prune, and remember GPU candidates,
and when it fails, it can prove whether the failure was a true performance wall
or a missing exposed primitive.
```

That is the practical version of generic GPU search for this project.
