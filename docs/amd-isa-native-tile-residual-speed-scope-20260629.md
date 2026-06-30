# AMD ISA Native Tile Residual Speed Scope — 2026-06-29

## What Is Happening

The native AMD ISA decode-attention route is now correct, route-bound, deterministic, search-owned, and measured. It is
not blocked on correctness anymore. It is blocked on residual speed:

| ctx | native | owned | native % |
|---:|---:|---:|---:|
| 512 | 67.09 | 103.15 | 65.0% |
| 1024 | 66.72 | 101.22 | 65.9% |
| 2048 | 64.46 | 98.92 | 65.2% |
| 4096 | 57.40 | 94.10 | 61.0% |

The attribution stack now has three layers:

| layer | artifact | conclusion |
|---|---|---|
| whole step | `bench/amd-isa-backend-phase-n4/latest.json` | native attention tile is the native-vs-owned delta; shared GEMV/reduce mostly cancel |
| PMC category | `bench/amd-isa-backend-phase-n2b/latest.json` | native burns much more dynamic VALU/LDS work per wave; VMEM and LDS-wait are not primary |
| PC/source trace | `bench/amd-isa-backend-pc-source-trace/latest.json` | hottest estimated source groups are `address_index`, `waitcnt`, and `lds_accum_stage` |

The new PC/source trace is not a hardware per-PC stall trace. It is:

```text
category measured by PMC, PC/source rows estimated by static loop weighting
```

That distinction matters. The tool ranks where final instructions came from, but it does not prove that every high
estimated row is a live speed lever.

## Current Trace Rows

Top rows from `bench/amd-isa-backend-pc-source-trace/summary.md`:

| rank | source group | category | static | est dyn ctx512 | lever named by trace | status |
|---:|---|---|---:|---:|---|---|
| 1 | `address_index` | VALU | 114 | 668928 | scalarize uniform prefix / strength-reduce | scalarization path was already refuted by N1B; strength reduction still needs a live-use audit |
| 2 | `waitcnt` | WAIT | 35 | 503040 | finer waitcnt thresholds | medium confidence; PMC did not name wait as the primary bottleneck |
| 3 | `lds_accum_stage` | LDS | 31 | 463872 | register accumulators / fewer LDS round-trips | high confidence structural issue; direct register-accumulator path was blocked by regalloc in N5A |
| 4 | `other` | OTHER | 49 | 422208 | unknown | classifier gap; must be split before acting |
| 5 | `pv_softmax_arith` | VALU | 16 | 294912 | FMA fuse / fewer rescale ops | plausible local codegen/tile algorithm lever |
| 6 | `mov` | VALU | 42 | 232704 | reduce copies | plausible codegen cleanup if live and not scheduler artifact |

## Interpretation

The trace confirms the cheap work is mostly exhausted:

- `address_index` looks hottest, but the obvious scalarization path (`AMD_ISA_N1B`) was already tested and found dead on
  the live tile path. The live address path goes through vector clamp/predicate logic, so the previous scalar temps were
  not consumed. Do not simply re-run N1B.
- `waitcnt` is large statically/dynamically, but N2B did not show a wait-dominated hardware category. Finer thresholds may
  reduce instruction count, but it is unlikely to close the 35-40% gap alone.
- `lds_accum_stage` is the true structural match to the owned-vs-native gap: native uses LDS-backed loop state where owned
  keeps more state in registers / better hand-scheduled state. N5A already tried the direct path and correctly blocked on
  the current regalloc model.
- `pv_softmax_arith` and `mov` are the most plausible smaller local wins if the structural regalloc feature is too large
  for the next pass.

So the next step should not be "try every row." It should be a disciplined feasibility pass:

1. Split trace false positives from live speed levers.
2. Pick the smallest live lever with a credible W==D path.
3. Implement only that lever.
4. Re-run W==D and PC/source trace.

## Phase R0 — Residual Lever Triage

Create an audit-only triage tool:

```text
extra/amd_isa_residual_lever_triage.py
```

Artifacts:

```text
bench/amd-isa-backend-residual-lever-triage/latest.json
bench/amd-isa-backend-residual-lever-triage/summary.md
```

Inputs:

```text
bench/amd-isa-backend-pc-source-trace/latest.json
bench/amd-isa-backend-pc-source-trace/native_inst_stream.json
bench/amd-isa-backend-phase-n2b/latest.json
bench/amd-isa-backend-phase-n4/latest.json
bench/amd-isa-backend-phase-n5/native_tile_residual/latest.json
```

Required rows:

| row | question |
|---|---|
| `address_index_live` | Are the top `address_index` instructions consumed by live address operands, or are they dead/unconsumed scalar-prefix-like work? |
| `address_index_strength_reduce` | Is there a non-N1B strength-reduction opportunity that removes live VALU ops without needing scalar SGPR regalloc? |
| `waitcnt_binding` | Are the 35 waitcnts dynamically on hot loop paths, and do they serialize useful work or just reflect conservative correctness drains? |
| `lds_accum_stage_roundtrips` | How many LDS load/store round-trips are from DEFINE_REG accumulator state vs K/V staging vs cross-lane reduce? |
| `other_classifier_split` | Split `OTHER` into real op families, synthetic carriers, or classifier misses. |
| `pv_softmax_fusion` | Count live PV/softmax arithmetic patterns that can be FMA-fused or algebraically reduced without regalloc changes. |
| `mov_copy_source` | Count live moves by cause: S2V, type materialization, carrier noops, EXEC/predicate, spill/copy equivalent. |

Verdicts:

```text
AMD_ISA_RESIDUAL_TRIAGE_PASS_LEVER_SELECTED
AMD_ISA_RESIDUAL_TRIAGE_BLOCKED_TRACE_METADATA
AMD_ISA_RESIDUAL_TRIAGE_INCONCLUSIVE_NO_LIVE_LEVER
```

Selection rules:

1. If `lds_accum_stage_roundtrips` is still dominant and removing it requires loop-carried register state, select
   `R1A_REGALLOC_FEATURE` only if the implementation scope includes a real regalloc model change.
2. If `address_index_strength_reduce` has live removable VALU ops and does not require the refuted N1B scalar path, select
   `R1B_ADDRESS_STRENGTH_REDUCE`.
3. If `pv_softmax_fusion` or `mov_copy_source` has a live removable cluster of at least 5 static hot-loop VALU ops, select
   `R1C_LOCAL_CODEGEN_CLEANUP`.
4. If only waitcnt is left, select `R1D_WAITCNT_THRESHOLDS` but cap expected W==D movement and require no correctness
   regressions.

## Phase R1 — Implement One Selected Lever

Phase R1 must implement exactly one selected R0 lever. Do not mix them.

### R1A — Regalloc Feature For Loop-Carried Accumulators

Use only if R0 selects it.

Goal: support a backend-generic or AMD-local representation for stateful loop-carried physical accumulators without
breaking SSA/live-range assumptions.

Required design proof before code:

- cite why N5A blocked;
- identify the exact regalloc invariant being extended;
- show how a physical accumulator is represented across `RANGE/END`;
- show how interference, defs, uses, and clobbers are modeled;
- prove this does not silently corrupt ordinary SSA virtual registers.

Acceptance:

- a minimal accumulator microgate passes without LDS-backed state;
- Phase G/H token gate still passes;
- W==D improves or the artifact records `NO_MOVEMENT`.

### R1B — Live Address Strength Reduction

Use only if R0 proves live address-index VALU ops are removable without resurrecting N1B.

Possible levers:

- hoist invariant multiply/add out of inner loops while still ending in a VGPR address;
- replace repeated `v_mul_lo_u32` chains with incrementing offsets;
- fold byte-scale and split/kvh base terms into one live offset chain;
- remove duplicate clamp-derived address recomputation if two consumers share the same clamped token.

Non-goal: generic SGPR scalarization. N1B already proved the naive scalar path is dead/faulting.

Acceptance:

- static live `address_index` VALU count decreases;
- token match and route binding hold;
- W==D measured at ctx512 and ctx4096.

### R1C — Local Codegen Cleanup

Use only if R0 selects PV/softmax/mov cleanup.

Possible levers:

- FMA-fuse PV rescale arithmetic;
- remove redundant f16/f32 converts;
- remove carrier moves that survive into final Insts;
- share intermediate online-softmax values across score/PV where legal.

Acceptance:

- static hot-loop VALU or MOV count decreases by a named amount;
- PC/source trace confirms the targeted row moved;
- W==D measured.

### R1D — Waitcnt Thresholds

Use only if R0 selects it.

Goal: improve `_insert_waitcnt` from full-drain waits to narrower vmcnt/lgkmcnt thresholds where safe.

Acceptance:

- waitcnt count or wait cost decreases;
- no Phase B/C/F/G/H regressions;
- W==D measured;
- if W==D movement is under 2%, record the lever as refuted and do not continue tuning it.

## Required Regression Ladder

After R1 implementation:

```text
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_pc_source_trace.py
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_phase_n4_whole_step_attribution.py
DEV=AMD PYTHONPATH=. python3 extra/amd_isa_phase_n2b_pmc_attribution.py
```

Also re-run the cheapest route/correctness gate already used by Phase N7 if present.

The final report must include:

- selected lever;
- why other top trace rows were not selected;
- static instruction movement;
- PC/source row movement;
- W==D at ctx512 and ctx4096;
- token match / route bound / no fallback.

## Claude Prompt

Use this prompt verbatim:

```text
You are working in /home/ubuntu/tinygrad-arkey.

Read and follow:

  docs/amd-isa-native-tile-residual-speed-scope-20260629.md

Context:
The native AMD ISA decode-attention route is correct and route-bound but sits at roughly 61-66% of owned. N4 attributes the native-vs-owned delta to native_block_tile. N2B says the tile burns much more dynamic VALU/LDS work per wave, not VMEM or LDS-wait. The new PC/source trace passes and ranks the top estimated source groups:

  1. address_index
  2. waitcnt
  3. lds_accum_stage
  4. other
  5. pv_softmax_arith
  6. mov

Important:
The PC/source trace is not hardware per-PC stall data. It is PMC category truth plus estimated static loop weighting. Do not blindly optimize the #1 row. N1B already refuted naive address scalarization. N5A already blocked direct register accumulators on regalloc.

Task:
Implement Phase R0 first: add an audit-only residual lever triage tool:

  extra/amd_isa_residual_lever_triage.py

Artifacts:

  bench/amd-isa-backend-residual-lever-triage/latest.json
  bench/amd-isa-backend-residual-lever-triage/summary.md

R0 must decide which one R1 lever is actually live and feasible:

  R1A_REGALLOC_FEATURE
  R1B_ADDRESS_STRENGTH_REDUCE
  R1C_LOCAL_CODEGEN_CLEANUP
  R1D_WAITCNT_THRESHOLDS

R0 must explicitly answer:

  address_index_live
  address_index_strength_reduce
  waitcnt_binding
  lds_accum_stage_roundtrips
  other_classifier_split
  pv_softmax_fusion
  mov_copy_source

Stop after R0 if the selected lever is R1A_REGALLOC_FEATURE and the required regalloc design proof is not ready. Do not hand-wave through the regalloc blocker.

If R0 selects R1B/R1C/R1D and the implementation is small and clearly bounded, implement exactly that one lever. Do not mix levers.

Acceptance:

  - token_match remains true
  - route_bound remains true
  - no hidden fallback
  - PC/source trace is rerun and shows the targeted row moved, or the lever is recorded as NO_MOVEMENT/refuted
  - W==D is measured at ctx512 and ctx4096 if any codegen change is made

Required final report:

  - selected R1 lever
  - top 3 trace rows and why each was selected/refuted/deferred
  - exact source files changed
  - static instruction movement
  - PC/source row movement
  - W==D ctx512/ctx4096
  - final verdict

Do not optimize without R0. Do not edit autogen/**. Do not make native attention the shipped default. Stop at the first hard blocker.
```

