# AMD ISA Loop-Carried Regalloc Feature Scope - 2026-06-29

## Purpose

Build the missing backend capability behind the remaining native AMD ISA decode-attention speed gap:

```text
loop-carried physical accumulators
```

This is the feature required to revisit N5A/register accumulators. The native tile is already correct, route-bound,
search-owned, and measured at roughly 61-66% of owned. R0 proved the bounded local levers are exhausted or too small.
The remaining structural gap is the LDS-backed accumulator state.

## Current Ground Truth

| artifact | verdict | conclusion |
|---|---|---|
| `bench/amd-isa-backend-phase-n5/native_tile_residual/latest.json` | `AMD_ISA_PHASE_N5A_BLOCKED_REGALLOC` | direct register accumulators are blocked by regalloc semantics |
| `bench/amd-isa-backend-residual-lever-triage/latest.json` | `AMD_ISA_RESIDUAL_TRIAGE_INCONCLUSIVE_NO_LIVE_LEVER` | no bounded live lever remains; the real lever is regalloc-backed accumulators |
| `bench/amd-isa-backend-pc-source-trace/latest.json` | `AMD_ISA_PC_SOURCE_TRACE_PASS_SOURCE_ROWS_PINNED` | `lds_accum_stage` is a top structural hot row |
| `bench/amd-isa-backend-phase-n4/latest.json` | `AMD_ISA_PHASE_N4_PASS_WHOLE_STEP_ATTRIBUTION_PINNED` | native attention tile is the native-vs-owned delta |

Current N7 W==D:

| ctx | native | owned | native % |
|---:|---:|---:|---:|
| 512 | 67.09 | 103.15 | 65.0% |
| 1024 | 66.72 | 101.22 | 65.9% |
| 2048 | 64.46 | 98.92 | 65.2% |
| 4096 | 57.40 | 94.10 | 61.0% |

## Why This Is Not A Tile Peephole

The current AMD ISA backend maps `Ops.DEFINE_REG` accumulator state to LDS. See `tinygrad/renderer/isa/amd.py` around
the LDS-backed reduction accumulator comment. That path was chosen deliberately because memory can represent a
read-modify-write value across loop iterations without fighting SSA register allocation.

The desired replacement is a register-resident accumulator:

```text
init acc_reg
for token/split loop:
  read acc_reg
  compute new_acc
  write acc_reg
after loop:
  use acc_reg
```

That is stateful and multi-write. The current late regalloc is single-def SSA-oriented.

## N5A Blocker Walls

N5A identified three concrete walls:

1. **Single-def assert.** `tinygrad/codegen/late/regalloc.py` asserts each tagged def has `lr[v][0] == i`. A pinned
   accumulator written every iteration would be tagged as a def more than once and trip the invariant.
2. **No fixed-register operand reference.** The renderer has no class for "read this physical VGPR, but do not allocate
   or track it as a normal virtual def." Everything tuple-tagged enters live-range/regalloc as a virtual definition.
3. **No real VGPR-to-VGPR move / loop carry model.** `AMDOps.MOV` is currently a coalescing no-op. Writing a pinned
   accumulator from a computed VGPR needs a real `v_mov_b32 dst, src`, and regalloc must model that the pinned state is
   live through `RANGE/END`.

Therefore the next work is a regalloc feature with microgates, not another direct tile edit.

## Design Principle

Do not break the existing generic regalloc path for ordinary SSA virtual registers.

Add a small, explicit representation for loop-carried physical state, ideally backend-local first and generic only if the
abstraction proves clean. The feature should be opt-in and off by default until all gates pass.

## Phase RA0 - Design Audit

Before writing feature code, produce:

```text
bench/amd-isa-backend-regalloc-accum/design.json
bench/amd-isa-backend-regalloc-accum/design.md
```

The design must answer:

| question | required answer |
|---|---|
| representation | How is a loop-carried accumulator represented in UOps or AMDOps? |
| ownership | Is it backend-local AMD-only, or a generic regalloc concept? |
| defs/uses | Which nodes read the physical accumulator and which nodes write it? |
| live range | How does the accumulator stay live across `RANGE/END` without tripping the single-def invariant? |
| interference | How are pinned VGPRs removed from the normal VGPR pool and protected from allocation? |
| initialization | Where is the accumulator initialized? |
| final use | How is the final accumulator value consumed after the loop? |
| moves | Which real move instruction writes computed values into the pinned accumulator? |
| fallback | How does the old LDS accumulator path remain the default? |

Allowed RA0 verdicts:

```text
AMD_ISA_REGALLOC_ACCUM_RA0_PASS_DESIGN_READY
AMD_ISA_REGALLOC_ACCUM_RA0_BLOCKED_BAD_ABSTRACTION
AMD_ISA_REGALLOC_ACCUM_RA0_BLOCKED_REQUIRES_TILE_REWRITE
```

Stop if RA0 does not pass.

## Phase RA1 - Minimal Backend Primitive

Implement the smallest opt-in primitive that can run a loop-carried register accumulator microkernel.

Suggested interface:

```text
AMD_ISA_REG_ACCUM=1
```

This flag must only affect the new microgates at first. It must not change the native decode tile until RA2.

Possible implementation shape:

- reserve a high VGPR range, for example `v240..v255`, and remove it from the normal AMD ISA VGPR pool;
- add explicit AMDOps for:
  - read pinned accumulator as a source;
  - write pinned accumulator from a VGPR using real `v_mov_b32`;
  - initialize pinned accumulator;
- ensure pinned accumulator references do not enter the normal single-def live-range map as ordinary virtual defs;
- preserve correctness through scheduler and waitcnt insertion.

Do not edit `autogen/**`.

RA1 microgate:

```text
extra/amd_isa_regalloc_accum_microgate.py
```

Required microkernels:

1. Single accumulator:

```text
acc = 0
for i in range(N):
  acc = acc + x[i]
out[0] = acc
```

2. Two independent accumulators:

```text
acc0 = 0
acc1 = 1
for i in range(N):
  acc0 = acc0 + x[i]
  acc1 = acc1 * scale + y[i]
out[0] = acc0
out[1] = acc1
```

3. Nested loop accumulator:

```text
acc = 0
for outer:
  for inner:
    acc = acc + f(outer, inner)
out[0] = acc
```

RA1 artifacts:

```text
bench/amd-isa-backend-regalloc-accum/ra1_latest.json
bench/amd-isa-backend-regalloc-accum/ra1_summary.md
```

Allowed RA1 verdicts:

```text
AMD_ISA_REGALLOC_ACCUM_RA1_PASS_MICROGATES
AMD_ISA_REGALLOC_ACCUM_RA1_BLOCKED_SINGLE_DEF
AMD_ISA_REGALLOC_ACCUM_RA1_BLOCKED_FIXED_REG_REFERENCE
AMD_ISA_REGALLOC_ACCUM_RA1_BLOCKED_MOVE_OR_LOOP_CARRY
AMD_ISA_REGALLOC_ACCUM_RA1_BLOCKED_CORRECTNESS
```

Stop if RA1 does not pass.

## Phase RA2 - DEFINE_REG Accumulator Opt-In

Wire the feature to the AMD ISA `Ops.DEFINE_REG` accumulator path, still default-off.

Goal:

```text
AMD_ISA_REG_ACCUM=1
```

When safe, `DEFINE_REG` accumulator loads/stores in the native tile should use pinned register state instead of LDS.

Rules:

- Only target accumulator state, not `DEFINE_LOCAL` K/V staging.
- Do not remove the LDS fallback.
- Do not use the feature for unknown/multi-thread cases until the microgates prove the addressing/lane ownership model.
- If per-thread accumulators need one physical VGPR per lane, stop and record the design as invalid. A physical VGPR is
  already per-lane in SIMD execution; do not allocate 128 separate physical registers for 128 lanes.

RA2 gate:

```text
extra/amd_isa_regalloc_accum_define_reg_gate.py
```

Required proof:

- `lds_accum_stage` DS load/store count decreases in `extra/amd_isa_pc_source_trace.py`;
- token match holds for the native tile microgate;
- route-bound/no-fallback holds;
- existing Phase B/C/F/G/H gates still pass with `AMD_ISA_REG_ACCUM=0`.

Allowed RA2 verdicts:

```text
AMD_ISA_REGALLOC_ACCUM_RA2_PASS_DEFINE_REG_OPT_IN
AMD_ISA_REGALLOC_ACCUM_RA2_BLOCKED_TILE_CORRECTNESS
AMD_ISA_REGALLOC_ACCUM_RA2_BLOCKED_LDS_COUNT_NO_MOVEMENT
AMD_ISA_REGALLOC_ACCUM_RA2_BLOCKED_UNSAFE_THREAD_MODEL
```

Stop if RA2 does not pass.

## Phase RA3 - Native Tile W==D Measurement

Only after RA2 passes, measure W==D with the feature enabled:

```text
AMD_ISA_REG_ACCUM=1
```

Required measurements:

- ctx512 native vs owned;
- ctx4096 native vs owned;
- token match;
- deterministic repeated runs;
- route-bound/no fallback;
- PC/source trace before/after for `lds_accum_stage`;
- N4 whole-step attribution before/after for native attention tile GPU-compute.

Expected but not guaranteed outcome:

The wall-clock ceiling may be modest because N3F/N4 showed the tile is only part of the decode step. A successful feature
should primarily reduce native tile cost, not necessarily reach owned parity.

Allowed RA3 verdicts:

```text
AMD_ISA_REGALLOC_ACCUM_RA3_PASS_WD_MOVEMENT
AMD_ISA_REGALLOC_ACCUM_RA3_PASS_CORRECT_NO_MOVEMENT
AMD_ISA_REGALLOC_ACCUM_RA3_BLOCKED_TOKEN_MATCH
AMD_ISA_REGALLOC_ACCUM_RA3_BLOCKED_ROUTE_ATTRIBUTION
AMD_ISA_REGALLOC_ACCUM_RA3_BLOCKED_NONDETERMINISM
```

## Required Regression Ladder

Run at minimum:

```text
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_regalloc_accum_microgate.py
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_pc_source_trace.py
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_phase_n4_whole_step_attribution.py
```

If RA2/RA3 touches the native tile, also run the existing cheap correctness ladder used in the N phases:

```text
Inc 0/1/2/3
Phase B
Phase C
Phase F
Phase G
Phase H token route
```

Use the existing scripts if present. Do not invent pass claims from partial runs.

## Success Criteria

The full feature is successful only if:

1. RA0 design passes.
2. RA1 microgates pass.
3. RA2 reduces `lds_accum_stage` DS load/store traffic with `AMD_ISA_REG_ACCUM=1`.
4. RA3 preserves token match, route binding, no fallback, and determinism.
5. W==D is measured honestly, even if movement is small.
6. The default route with `AMD_ISA_REG_ACCUM=0` remains unchanged.

## Stop Conditions

Stop immediately and record a blocker if:

- the implementation requires weakening the single-def assert for ordinary virtual registers;
- pinned physical registers can be allocated by normal regalloc;
- accumulator state is corrupted across `RANGE/END`;
- per-thread state is modeled as one physical VGPR per workitem instead of one SIMD VGPR with per-lane values;
- correctness requires a tile algorithm rewrite instead of a regalloc feature;
- W==D cannot be measured because route attribution or token match fails.

## Claude Prompt

Use this prompt verbatim:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Read and follow:

  docs/amd-isa-loop-carried-regalloc-scope-20260629.md

Context:
The native AMD ISA decode-attention route is correct, route-bound, search-owned, and measured at roughly 61-66% of owned. R0 residual triage proved there is no bounded local lever left. The remaining structural lever is lds_accum_stage: native uses LDS-backed accumulator state, while the desired path is register-resident loop-carried accumulators.

Important blocker from N5A:
This is not a local tile peephole. N5A proved direct register accumulators are blocked by the current single-def linear-scan regalloc model:

  1. single-def assert in tinygrad/codegen/late/regalloc.py
  2. no "read fixed physical VGPR without allocating/tracking it as a normal virtual def" class
  3. no real VGPR-to-VGPR move / loop-carried pinned state model

Task:
Implement this in phases. Do not skip phases.

Phase RA0:
Write the design artifact first:

  bench/amd-isa-backend-regalloc-accum/design.json
  bench/amd-isa-backend-regalloc-accum/design.md

The design must answer representation, defs/uses, live range across RANGE/END, interference, initialization, final use, real moves, fallback, and whether this is AMD-local or generic. If the abstraction is not sound, stop with a blocker.

Phase RA1:
Only if RA0 passes, implement the smallest opt-in primitive behind:

  AMD_ISA_REG_ACCUM=1

Add:

  extra/amd_isa_regalloc_accum_microgate.py

Microgates:

  1. single loop-carried sum accumulator
  2. two independent loop-carried accumulators
  3. nested loop accumulator

Do not wire the native decode tile yet. Pass RA1 first.

Phase RA2:
Only if RA1 passes, wire the feature to AMD ISA DEFINE_REG accumulator state, still default-off. Do not apply to DEFINE_LOCAL K/V staging. Keep the old LDS path as fallback.

Add:

  extra/amd_isa_regalloc_accum_define_reg_gate.py

Required proof: lds_accum_stage DS load/store count decreases in the PC/source trace, token match holds, route-bound/no-fallback holds.

Phase RA3:
Only if RA2 passes, measure W==D with AMD_ISA_REG_ACCUM=1 at ctx512 and ctx4096. Re-run PC/source trace and N4 whole-step attribution before/after.

Constraints:

  - Do not edit autogen/**
  - Do not make native attention the shipped default
  - Do not weaken the single-def invariant for ordinary virtual registers
  - Do not let pinned physical registers enter the normal allocation pool
  - Do not model per-thread accumulator state as one physical register per workitem; a SIMD VGPR already has per-lane values
  - Stop at the first hard blocker

Final report must include:

  - RA0 design verdict
  - RA1 microgate verdicts
  - RA2 DEFINE_REG/tile correctness verdict, if reached
  - RA3 W==D ctx512/ctx4096, if reached
  - static DS load/store movement for lds_accum_stage
  - token_match / route_bound / no fallback / determinism
  - exact files changed
  - final verdict
```

