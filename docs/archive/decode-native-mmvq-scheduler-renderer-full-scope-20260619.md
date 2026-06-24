# Decode Native MMVQ Scheduler/Renderer Full Scope

Date: 2026-06-19

## Purpose

Fully scope the only remaining dependency-free decode path after the complete tooling atlas:

```text
make tinygrad natively preserve the llama-style MMVQ lifecycle contract in-model
```

This is not a reduce/glue fusion project and not another small kernel search. The current tooling result already says:

- ATT/HCQ visibility works;
- runtime/cache identity is closed;
- reduce/glue is visible but does not clear the build gate;
- imported llama Q4 consumers are correct and graph-safe but lose in model;
- q8 fused artifact route is the only measured decode speed route, but it is external-artifact/research-only;
- the native route is project-level scheduler/renderer work.

This document scopes that native route completely.

## Authority

| Fact | Value |
|---|---:|
| tinygrad standalone MMVQ/GEMV ceiling | about `76%` HBM peak |
| tinygrad in-model weight-GEMV bucket | about `44%` HBM |
| llama standalone MMVQ/GEMV | about `57%` HBM |
| llama in-model MMVQ/GEMV | about `54%` HBM |
| weight-GEMV share of decode GPU time | about `85%` |
| target if tinygrad reaches llama retention | about `1.187x` decode |
| theoretical full standalone transfer | about `1.557x` decode |
| proven q8 artifact W==D movement | `1.05-1.06x`, dNLL `+0.002887` |
| native q8 consumer gap | tinygrad AMD DSL `166.65us` vs hipcc/LLD artifact `93.54us` |

## Core Hypothesis

The remaining decode gap is a **contract preservation** problem:

```text
activation lifecycle + low-VGPR/high-grid MMVQ consumer + scheduler/resource behavior + graph-safe integration
```

llama preserves that contract in model. tinygrad can produce strong standalone surfaces, but loses the contract when the
primitive is embedded in the decode graph.

## Non-Goals

- Do not build direct-output/reduce fusion unless a new timing-grade ledger clears the gate.
- Do not tune old env knobs (`Q4K_COOP_RT`, `Q6K_COOP_RT`, coop on/off); that surface is closed.
- Do not start another standalone MMVQ benchmark as proof of model movement.
- Do not promote packet counts as timing.
- Do not make external artifacts default.
- Do not reopen q8 producer work until the native consumer/scheduler wall moves.

## What Must Be True To Win

A dependency-free native route must satisfy all of these:

1. generate a low-VGPR/high-grid MMVQ consumer with llama/hipcc-class schedule quality;
2. preserve graph replay without per-token Python/setup overhead;
3. avoid separate lifecycle taxes that erase the kernel win;
4. pass W==D decode, not just role-local timing;
5. pass dNLL for lossy q8 paths;
6. generalize at least across the Qwen3-8B high-share roles or provide a strong reason to stay role-specific.

## Workstreams

There are three workstreams. They can overlap, but only one is allowed to change performance code at a time.

### Workstream O - Oracle and Attribution

Goal: maintain a stable target and prevent blind compiler work.

Inputs:

- q8 hipcc/LLD artifact route;
- imported llama MMVQ object route;
- llama HIP trace/kernarg capture;
- complete tooling atlas;
- native tinygrad AMD DSL/COMGR outputs;
- prefill/Tensile evidence as a cross-regime scheduler oracle.

Output:

- machine-readable contracts for target schedules and current tinygrad schedules.

### Workstream C - Compiler/Scheduler Capability

Goal: add the smallest reusable AMD backend capability that moves measured decode or a shared oracle.

Candidate capability classes:

- dependency-aware wait placement;
- instruction scheduling around dot/load/reduce;
- register/live-range control;
- launch/resource contract preservation;
- explicit schedule metadata IR;
- late AMD-specific lowering for packed quant dot bodies;
- optional software-pipelined loop capability if shared with prefill.

### Workstream M - Model Integration

Goal: ensure any local improvement survives the actual decode lifecycle.

Required gates:

- same-process interleaved role A/B;
- graph replay stability;
- W==D ctx sweep;
- dNLL where lossy;
- default unchanged.

## Phases

### NSR-0 - Freeze The Source Of Truth

Goal: prevent stale decode narratives from steering the project.

Deliverables:

- pointer to `decode-complete-tooling-result-20260619.md`;
- pointer to `decode-mmvq-large-project-scope-20260619.md`;
- pointer to `q8-ffn-amd-scheduler-codegen-project-scope-20260619.md`;
- one `bench/qk-decode-native-scheduler/authority.json`.

Gate:

- all current closed surfaces are listed:
  - runtime/cache identity;
  - env launch knobs;
  - imported Q4 route as local timing win;
  - reduce/glue fusion without new timing;
  - bounded q8 N2 patch after N1.

Kill:

- if the scope cannot identify a live surface distinct from closed surfaces, stop.

### NSR-1 - Unified Oracle Matrix

Goal: collect every relevant oracle in one table.

Rows:

| Oracle | Role | Why |
|---|---|---|
| q8 hipcc/LLD artifact | `ffn_gate/up` | only measured decode-speed route; scheduler target |
| tinygrad AMD DSL q8 | `ffn_gate/up` | native baseline that fails |
| COMGR fused-C q8 | `ffn_gate/up` | source-level compiler baseline |
| imported llama Q4 consumer | `attn_output`, `ffn_gate/up` | correct/fast standalone but loses in-model |
| native tinygrad Q4/Q6 coop | high-share roles | current default contract |
| llama HIP MMVQ | Q4/Q6 | external lifecycle target |
| Tensile/prefill if in scope | prefill GEMM | cross-regime scheduler/latency target |

Deliverable:

- `bench/qk-decode-native-scheduler/oracle_matrix.json`

Gate:

- each row has correctness status, timing status, launch geometry, resource metadata where available, and graph status.

Kill:

- if no oracle shows a native-codegen-relevant gap, do not start compiler work.

### NSR-2 - Feature Attribution Contract

Goal: name candidate backend features with movement budgets.

Candidate labels:

| Label | Evidence required |
|---|---|
| `load_shape` | isolated movement from load width/coalescing, not just static diff |
| `wait_schedule` | timing change from wait grouping/placement |
| `instruction_order` | body schedule changes with same instruction set and measurable timing |
| `register_lifetime` | VGPR/occupancy changes tied to timing |
| `resource_descriptor` | launch/resource metadata changes tied to timing |
| `reduction_shape` | reduction topology changes tied to timing |
| `activation_lifecycle` | producer/consumer fusion removes a measured lifecycle cost |
| `graph_boundary` | replay/integration change removes measured overhead |

Deliverable:

- `bench/qk-decode-native-scheduler/feature_attribution.json`

Gate:

- one label has either:
  - `>=30us` attributed movement on q8 lifecycle, or
  - `>=5%` projected W==D movement, or
  - a shared decode+prefill capability with independent oracle movement.

Kill:

- if all labels are `unknown` or below gate, do not build a feature; keep native route as roadmap.

### NSR-3 - Minimal Backend Feature Proof

Goal: implement exactly one backend capability, behind a research/env flag or isolated probe.

Allowed first features:

1. **schedule metadata IR**: attach explicit load/wait/live-range/schedule groups to UOps or AMD-lowered fragments;
2. **AMD wait/instruction scheduler**: controlled `s_waitcnt`, `s_clause`, `s_delay_alu`, and load/dot ordering;
3. **register/live-range control**: prevent schedule collapse or VGPR pressure cliffs;
4. **packed quant dot lowering**: renderer-level lowering that preserves q8/MMVQ body shape without external artifacts;
5. **graph-safe lifecycle fusion primitive**: only if timing shows lifecycle, not scheduler, is the binding tax.

Deliverable:

- one proof artifact under `bench/qk-decode-native-scheduler/feature_proof.json`

Gate:

- feature changes generated ISA in the intended way;
- correctness passes;
- local movement:
  - q8 consumer improves by `>=25us`, or
  - role-local high-share surface improves by `>=10%`, or
  - shared prefill/decode oracle clears its own gate.

Kill:

- if the feature is body-insensitive or moves `<15us`, stop that feature and return to NSR-2.

### NSR-4 - Native Consumer Rebuild

Goal: rebuild the fused q8 gate/up consumer natively using the new capability.

Why q8 first:

- it has the clearest oracle gap;
- it has a measured W==D decode route;
- it tests activation lifecycle and scheduler quality together;
- it is already default-off/research-gated.

Deliverable:

- `bench/qk-decode-native-scheduler/native_q8_consumer.json`

Gates:

| Gate | Continue | Strong pass |
|---|---:|---:|
| consumer time | `<=75us` | `<=60us` |
| lifecycle producer + gate/up | `<=129.2us` | close to artifact `115us` |
| correctness | max_abs `<=2e-3` | same |
| runtime | HCQ/no HIP runtime | same |

Kill:

- if native consumer remains `>100us`, q8 native ownership is still closed.

### NSR-5 - Producer/Lifecycle Reopen

Run only if NSR-4 passes.

Goal: remove the q8 producer lifecycle tax natively.

Required capabilities:

- per-row RMSNorm reduce;
- barrier/LDS or register broadcast;
- per-32 q8 max/quantize;
- multi-output stores;
- graph-safe side-channel lifetime;
- no separate q8 pack launch if that erases the win.

Deliverable:

- `bench/qk-decode-native-scheduler/native_q8_lifecycle.json`

Gate:

- producer + gate/up lifecycle `<=129.2us`;
- role-local `>=1.10x` vs baseline gate/up;
- W==D projected `>=3%`.

Kill:

- if producer must remain separate and lifecycle exceeds the gate, keep the artifact route as research-only.

### NSR-6 - Multi-Role Expansion

Goal: decide whether the backend capability generalizes beyond q8 gate/up.

Targets:

- Q4_K `attn_q/o`;
- Q6_K `ffn_down`;
- Q6_K `lm_head`;
- optional `attn_k/v` only if long-context share justifies it.

Deliverable:

- `bench/qk-decode-native-scheduler/role_matrix.json`

Gate:

- combined projected W==D movement `>=5%`;
- no role regresses enough to erase q8 gain;
- graph replay remains stable.

Kill:

- if the feature only helps q8 gate/up and cannot generalize, keep it scoped to the q8 research flag.

### NSR-7 - In-Model Decode Gate

Goal: final authority.

Deliverable:

- `bench/qk-decode-native-scheduler/wd_decode_result.json`

Required measurements:

- ctx `128`, `512`, `1024`, `4096`;
- same clock/DPM policy as baseline;
- W==D;
- flag off baseline and flag on candidate interleaved where possible;
- dNLL for lossy q8 paths;
- graph replay, not eager-only.

Gate:

- sustained W==D `>=3%` to keep as research flag;
- sustained W==D `>=5%` to call it a meaningful decode primitive win;
- dNLL `<=0.01`;
- default unchanged.

Kill:

- if only isolated/role-local speed moves but W==D does not, the native feature does not count as a decode win.

### NSR-8 - Maintenance and Ownership Decision

Goal: decide whether this becomes tinygrad-native capability, research flag, or roadmap only.

Outcomes:

| Outcome | Meaning |
|---|---|
| `NATIVE_RESEARCH_PASS` | native route works behind a flag, default off |
| `NATIVE_GENERAL_CAPABILITY` | feature helps decode and another AMD primitive class, worth upstream-style maintenance |
| `Q8_ONLY_RESEARCH` | feature only helps q8 gate/up; keep narrow |
| `ARTIFACT_ONLY` | native route fails; keep external q8 artifact as research |
| `ROADMAP_ONLY` | no bounded feature clears gate |

Deliverable:

- `docs/decode-native-mmvq-scheduler-renderer-result-20260619.md`

## Parallelism Plan

Can run in parallel:

- NSR-1 oracle matrix;
- NSR-2 feature attribution from existing artifacts;
- prefill/Tensile cross-oracle extraction if used only as evidence;
- q8 artifact reproducibility hardening.

Must be sequential:

- NSR-3 depends on a feature label from NSR-2;
- NSR-4 depends on NSR-3;
- NSR-5 depends on NSR-4;
- NSR-7 depends on NSR-4/5/6.

Do not run multiple performance-code experiments at once. It makes attribution ambiguous.

## Expected Potential

| Path | Expected decode movement | Confidence | Notes |
|---|---:|---|---|
| q8 artifact research flag | `1.05-1.06x` | high | already measured; external artifact |
| native q8 parity with artifact | `1.03-1.06x` | medium-low | requires consumer scheduler and producer lifecycle |
| reach llama in-model retention `44% -> 54%` | about `1.187x` | low | requires broader MMVQ contract preservation across roles |
| transfer full standalone `44% -> 76%` | about `1.557x` | very low | not earned; theoretical ceiling |
| reduce/glue only | low single digits | high | currently below build gate |

## Staffing/Risk

This is a compiler/backend project, not a model-kernel patch.

Expected effort if NSR-2 finds a bounded feature:

- feature proof: days to one week;
- native consumer rebuild: one to two weeks;
- lifecycle producer: one to two weeks;
- full role/model gate: several days.

Expected effort if NSR-2 does not find a bounded feature:

- broad AMD scheduler/renderer project: multi-week to month-scale;
- high regression surface across AMD kernels;
- requires a real test matrix beyond this decode model.

## Start Criteria

Start implementation only if one of these is true:

1. NSR-2 identifies a `>=30us` q8 or `>=5%` W==D feature;
2. the project explicitly accepts a broad AMD backend investment without feature attribution;
3. the goal is only to harden the already measured q8 artifact route, not native ownership.

Without one of those, the correct state is:

```text
decode primitive work is exhausted at bounded scope; native route is roadmap-only
```

## Completion Criteria

The scope is complete when it produces one of:

1. a native flag with W==D `>=3%` and dNLL pass;
2. a native general backend capability with decode plus cross-primitive payoff;
3. a documented kill stating that no bounded native feature clears movement gates;
4. an explicit decision to keep only the q8 artifact research flag.

