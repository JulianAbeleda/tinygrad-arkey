# Prefill Schedule-Diff Oracle And Search Reduction — Exhaustive Scope / Claude Prompt

Date: 2026-06-23

## Mission

Build the missing **schedule-diff oracle** for prefill GEMM.

The current project can already answer:

- whether a candidate transfers to synced whole-prefill;
- which role is behind;
- whether graph route / materialization / correctness is valid;
- whether ISA primitives such as WMMA, LDS, VGPR, scratch, and instruction
  families exist;
- whether high-level tile-config knobs recover the gap.

What it cannot yet answer automatically is:

```text
why exactly does the Tensile schedule still win by ~4-5% on some prefill roles,
and can that difference be reduced to bounded machine-search primitives?
```

This scope turns "hand-asm K-loop scheduling" from a vague diagnosis into a
measurable set of schedule primitives. The output is not a new kernel by
default. The output is a **diff oracle** that says which differences are:

1. already matched;
2. measurable but off the critical path;
3. true critical-path schedule gaps;
4. representable as bounded search knobs;
5. not representable without hand-assembly / renderer work / vendored Tensile.

## Core Thesis

Machine search still applies, but only after the search surface is reduced to
real primitives.

Do not search "GEMM configs" again blindly. The previous search already showed:

- the residual prefill gap is stable under clock-pinned synced whole-prefill;
- `kv_proj` was a workgroup-starvation problem and was fixed;
- `down` and `qo` are well-occupied;
- BK/PAD/DBUF/waves variants did not recover the gap;
- the remaining gap looks like K-loop scheduling / Tensile-class instruction
  scheduling, not tile-config.

Therefore the next step is:

```text
Tensile schedule + tinygrad graph-GEMM schedule
-> instruction/resource/loop diff
-> reduce to primitive rows
-> decide which rows are searchable
-> only then generate candidates
```

## Required Reading

Read these first:

1. `docs/prefill-search-result-20260623.md`
2. `docs/prefill-search-scope-20260623.md`
3. `docs/prefill-post-decode-parity-frontier-result-20260623.md`
4. `docs/prefill-per-role-transfer-attribution-result-20260623.md`
5. `docs/prefill-primitive-pmc-result-20260619.md`
6. `docs/prefill-amd-gemm-leanaddr-result-20260620.md`
7. `docs/prefill-tensile-schedule-template-extraction-result-20260620.md`
8. `docs/prefill-tensile-winning-kernel-transfer-table-20260620.md`
9. `docs/prefill-tensile-lds-tile-map-sketch-20260620.md`
10. `docs/machine-code-translation-roadmap-result-20260623.md`
11. `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
12. `docs/oracle-guided-gpu-primitive-explorer-runner-design-20260623.md`
13. `docs/project-search-ledger-contract-20260623.md`
14. `bench/qk-decode-eval/HARNESS_GUIDE.md`
15. `structure/Development/performance-primitive-research-principles.md`
16. `structure/Development/session-handoff.md`

Inspect relevant tools/artifacts:

- `extra/qk_prefill_search_execute.py`
- `extra/qk_prefill_whole_synced.py`
- `extra/qk_prefill_per_role_time_tax.py`
- `extra/qk_isa_primitive_audit.py`
- `extra/qk_amdgpu_isa_primitive_audit.py`
- `extra/qk_tensile_schedule_template_extract.py`
- `extra/qk_project_search_ledger.py`
- `bench/qk-prefill-search/`
- `bench/qk-prefill-post-decode-parity-frontier/`
- `bench/qk-machine-code-translation/`
- `bench/qk-project-search-ledger/ledger.jsonl`

## Non-Goals

- Do not flip defaults.
- Do not replace graph-GEMM with vendored Tensile.
- Do not run broad GEMM tile-config search again.
- Do not use nosync or raw dispatch as authority.
- Do not claim W==D / whole-prefill improvement from isolated GEMM timing.
- Do not generate a hand-asm kernel until the schedule-diff oracle names a
  bounded primitive and the owner explicitly authorizes implementation.
- Do not rewrite historical docs; add superseding notes only.

## Authority Rules

| question | authority |
|---|---|
| Is the role gap real? | clock-pinned, repeated, synced whole-prefill + per-role attribution |
| Does a schedule change transfer? | synced whole-prefill only |
| Is an instruction primitive present? | code object disassembly / ISA audit |
| Is a counter difference real? | PMC/SQ/GL2 counters, normalized and caveated |
| Is a candidate promotable? | correctness + whole-prefill transfer + fallback/default policy |
| Is a machine-search knob valid? | bounded range + deterministic gate + ledger entry |

PROFILE, DEBUG, raw dispatch, isolated run-linear, or nosync numbers are
diagnostic only.

## Definitions

### Schedule Primitive

A schedule primitive is a measurable, reusable lowering or execution pattern,
for example:

- K-loop depth / `DepthU`;
- LDS staging and double-buffering;
- global load vector width and cadence;
- LDS store/read vector shape;
- wait/barrier placement;
- prefetch distance / pipeline stage;
- workgroup mapping / traversal order;
- scalar versus vector address advancement;
- accumulator/WMMA issue cadence;
- VGPR/SGPR allocation envelope;
- scratch/spill absence;
- load/compute overlap evidence;
- instruction interleave around `v_wmma`;
- cache/L2 behavior if measurable.

### Searchable Primitive

A schedule primitive is searchable only if it has:

- a bounded knob range;
- a correctness harness;
- an ISA/resource check;
- an authority benchmark;
- a stop rule;
- no dependency on broad renderer redesign unless explicitly scoped.

## Phase 0 — Authority Lock

Record the exact baseline before diffing.

Tasks:

1. Capture git SHA, branch, GPU arch, ROCm version, clock mode, model path.
2. Record active prefill route flags and confirm default graph-GEMM route.
3. Run a short synced whole-prefill authority check for graph-GEMM and Tensile.
4. Confirm the stable ~4-5% gap at ctx512/1024 or record drift.
5. Confirm the role(s) to analyze first: likely `ffn_down` and `qo_proj`, not
   `kv_proj`.

Deliverable:

```text
bench/qk-prefill-schedule-diff-oracle/authority.json
```

Verdicts:

- `PREFILL_SCHEDULE_DIFF_AUTHORITY_LOCKED`
- `PREFILL_SCHEDULE_DIFF_AUTHORITY_DRIFT_STOP`

Stop if authority does not reproduce. Do not diff stale or noisy baselines.

## Phase 1 — Select Comparable Shapes And Kernels

Choose one primary role and one secondary role.

Recommended:

1. Primary: `ffn_down` (`4096 x 12288`) because prior search shows below-parity
   and well-occupied.
2. Secondary: `qo_proj` (`4096 x 4096`) because it is also below-parity and
   well-occupied.

Tasks:

1. Identify graph-GEMM code object/symbol for each role.
2. Identify matched Tensile code object/symbol/solution row for each role.
3. Record launch geometry, grid, local size, problem dimensions, dtype, beta,
   layout, and ABI differences.
4. Mark any non-apples-to-apples differences as confounds, not schedule facts.

Deliverable:

```text
bench/qk-prefill-schedule-diff-oracle/kernel_pair_manifest.json
```

Verdicts:

- `PREFILL_KERNEL_PAIRS_SELECTED`
- `PREFILL_KERNEL_PAIR_CONFOUNDED_STOP`

## Phase 2 — Static ISA / Resource Diff

Build a normalized static diff between graph-GEMM and Tensile.

Tasks:

1. Disassemble both code objects.
2. Segment functions into:
   - prologue;
   - steady K-loop / compute-bearing region;
   - epilogue.
3. Count by region:
   - `v_wmma`;
   - global loads by width/type;
   - LDS stores/loads by width;
   - `s_waitcnt`;
   - `s_barrier`;
   - VALU;
   - SALU;
   - vector memory;
   - scalar memory;
   - branches;
   - LDS offset families;
   - VGPR/SGPR;
   - scratch/spill.
4. Produce a side-by-side row for each primitive.
5. Carry caveats for layout/beta/ABI differences.

Possible tool:

```text
extra/qk_prefill_schedule_diff.py
```

Deliverable:

```text
bench/qk-prefill-schedule-diff-oracle/static_isa_diff.json
```

Verdicts:

- `STATIC_SCHEDULE_DIFF_READY`
- `STATIC_SCHEDULE_DIFF_INSUFFICIENT`

## Phase 3 — Dynamic Counter Diff

Use PMC counters to decide which static differences are on the critical path.

Counter families:

- global memory / L2 read traffic;
- LDS activity;
- wait/stall cycles;
- VALU/SALU instruction counts;
- wave count / occupancy proxy;
- active cycles;
- cache hit rate if available.

Tasks:

1. Run graph-GEMM and Tensile under the same clock/shape conditions.
2. Collect counters in bounded passes; respect PMC caveats.
3. Compare ratios, not noisy absolute wall times from PMC.
4. Map counters back to schedule primitives:
   - load traffic;
   - load/compute overlap;
   - LDS reuse;
   - address overhead;
   - occupancy/wave lifetime.
5. Flag primitives that are already ruled out by prior evidence:
   - VALU address count may be neutral if LEANADDR already matched it;
   - occupancy may be ruled out for well-occupied roles.

Deliverable:

```text
bench/qk-prefill-schedule-diff-oracle/dynamic_counter_diff.json
```

Verdicts:

- `DYNAMIC_COUNTER_DIFF_READY`
- `DYNAMIC_COUNTER_DIFF_UNAVAILABLE`
- `DYNAMIC_COUNTER_DIFF_CONFOUNDED`

## Phase 4 — Schedule Primitive Reduction

Reduce the static + dynamic diff into primitive rows.

Each row must include:

- primitive id;
- description;
- tinygrad evidence;
- Tensile evidence;
- critical-path likelihood;
- prior status: matched / ruled out / active / confounded;
- machine-searchability;
- required implementation surface;
- expected whole-prefill upside;
- stop rule.

Example row:

```json
{
  "primitive_id": "kloop_prefetch_distance",
  "description": "Tensile issues next K-tile global/LDS work before current WMMA group completes",
  "tinygrad_evidence": "serial wait-before-compute pattern",
  "tensile_evidence": "buffer_load/ds_store interleaved with v_wmma region",
  "critical_path_likelihood": "high",
  "searchability": "bounded_if_prefetch_knob_exists",
  "knobs": {"prefetch_distance": [0, 1, 2]},
  "authority": "synced_whole_prefill",
  "stop_rule": "if ISA shows no interleaving or whole-prefill <= default, reject"
}
```

Deliverable:

```text
bench/qk-prefill-schedule-diff-oracle/primitive_reduction.json
```

Verdicts:

- `SCHEDULE_PRIMITIVES_REDUCED`
- `NO_SEARCHABLE_SCHEDULE_PRIMITIVE_FOUND`

## Phase 5 — Search Surface Decision

Classify each primitive:

| class | meaning | next action |
|---|---|---|
| `SEARCHABLE_NOW` | bounded knob exists in current graph-GEMM route | generate candidates |
| `SEARCHABLE_AFTER_SMALL_HOOK` | one additive env/codegen hook needed | scope hook |
| `HAND_ASM_OR_RENDERER_REQUIRED` | cannot be reached by config knobs | do not search blindly |
| `VENDORED_TENSILE_ONLY` | best solved by external dependency | record policy choice |
| `RULED_OUT_OFF_CRITICAL_PATH` | measurable but no throughput transfer | close |
| `CONFOUNDED` | not apples-to-apples | do not rank |

Deliverable:

```text
bench/qk-prefill-schedule-diff-oracle/search_surface_decision.json
```

Verdicts:

- `PREFILL_SEARCH_SURFACE_READY`
- `PREFILL_HAND_ASM_SCHEDULING_REQUIRED`
- `PREFILL_NO_SEARCHABLE_PRIMITIVE_REMAINS`

## Phase 6 — Optional Bounded Search Only If Unlocked

Run this only if Phase 5 returns at least one `SEARCHABLE_NOW` or
`SEARCHABLE_AFTER_SMALL_HOOK` primitive.

Rules:

- one primitive family per run;
- bounded knob grid;
- correctness first;
- ISA/resource evidence before whole-prefill;
- clock-pinned repeated synced whole-prefill as authority;
- ledger entries for every candidate;
- no default flip.

Candidate families that may be valid if Phase 4/5 supports them:

- prefetch distance / prefetch stage;
- waitcnt placement;
- LDS double-buffer stage count;
- K-loop unroll group;
- global-load vector grouping;
- workgroup mapping if a bounded traversal knob exists;
- scalar-address advancement if not already ruled out;
- LDS read grouping / bank layout if offset diff supports it.

Do **not** rerun:

- generic BK/PAD/DBUF/waves search already covered by
  `docs/prefill-search-result-20260623.md`;
- occupancy search for well-occupied roles;
- LEANADDR as a speed lever if prior result says throughput neutral;
- isolated GEMM-only winner claims.

Deliverable:

```text
bench/qk-prefill-schedule-diff-oracle/search_runs/<primitive_id>/
```

Verdicts:

- `PREFILL_SCHEDULE_SEARCH_WD_PASS`
- `PREFILL_SCHEDULE_SEARCH_ORACLE_REMAINS_BEST`
- `PREFILL_SCHEDULE_SEARCH_NONTRANSFER`

## Phase 7 — Machine Search Integration

Update the oracle-guided explorer only after Phase 5.

If a searchable primitive exists:

- add a prefill schedule-diff oracle entry;
- add a SearchRow example for the primitive;
- add learned rules to the project ledger;
- add gate requirements:
  - static schedule diff;
  - dynamic counter diff if available;
  - ISA/resource check;
  - synced whole-prefill.

If no searchable primitive exists:

- record `PREFILL_HAND_ASM_SCHEDULING_REQUIRED`;
- keep machine search gated off for prefill speed;
- allow only codegen microprimitive learning rows.

Deliverables:

```text
bench/qk-oracle-gpu-primitive-explorer/spec_prefill_schedule_diff_example.json
bench/qk-project-search-ledger/ledger.jsonl
docs/oracle-guided-gpu-primitive-explorer-result-20260623.md (superseding note only)
```

## Phase 8 — Result Doc

Write:

```text
docs/prefill-schedule-diff-oracle-and-search-reduction-result-20260623.md
```

Required answers:

1. Is the ~4-5% gap still real under authority measurement?
2. Which role(s) carry the gap?
3. What exact schedule primitives differ between graph-GEMM and Tensile?
4. Which differences are on the critical path?
5. Which differences were ruled out?
6. Which differences are searchable now?
7. Which require hand-asm / renderer work?
8. What is the expected upside if solved?
9. Does this reopen prefill machine search?
10. What is the next executable task?

## Expected Outcomes

Possible final outcomes:

### Outcome A — Searchable Primitive Found

```text
PREFILL_SEARCH_SURFACE_READY
```

Meaning:

- the gap reduces to at least one bounded knob;
- run Phase 6 search;
- machine search remains relevant for prefill.

### Outcome B — Hand-Asm / Renderer Required

```text
PREFILL_HAND_ASM_SCHEDULING_REQUIRED
```

Meaning:

- the residual is real but below current search abstraction;
- no more high-level config search;
- next work is deterministic schedule implementation or native renderer work.

### Outcome C — Gap Confounded / Not Stable

```text
PREFILL_SCHEDULE_GAP_NOT_AUTHORITY_STABLE
```

Meaning:

- do not optimize;
- fix measurement authority first.

### Outcome D — No Material Gap Remains

```text
PREFILL_NO_ACTION_PARITY_WITHIN_NOISE
```

Meaning:

- close prefill for speed;
- only maintenance / cross-shape / codegen learning remains.

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch
`qk-prefill-flag-leak-resolution`.

Task: execute the prefill schedule-diff oracle scope. The goal is to reduce the
remaining graph-GEMM vs Tensile prefill gap into measurable schedule primitives
and decide whether machine search can validly operate on any of them.

Read first:

- `docs/prefill-schedule-diff-oracle-and-search-reduction-scope-20260623.md`
- `docs/prefill-search-result-20260623.md`
- `docs/prefill-post-decode-parity-frontier-result-20260623.md`
- `docs/prefill-per-role-transfer-attribution-result-20260623.md`
- `docs/prefill-primitive-pmc-result-20260619.md`
- `docs/prefill-amd-gemm-leanaddr-result-20260620.md`
- `docs/prefill-tensile-schedule-template-extraction-result-20260620.md`
- `docs/prefill-tensile-winning-kernel-transfer-table-20260620.md`
- `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
- `bench/qk-decode-eval/HARNESS_GUIDE.md`
- `structure/Development/performance-primitive-research-principles.md`
- `structure/Development/session-handoff.md`

Execute phases in order:

1. Authority lock: reproduce the clock-pinned synced whole-prefill gap or stop.
2. Select comparable kernel pairs for `ffn_down` and `qo_proj`.
3. Build static ISA/resource schedule diff by region.
4. Build dynamic counter diff if PMC is available and stable.
5. Reduce the diff into primitive rows.
6. Classify each primitive as searchable, hand-asm/renderer, ruled out, or
   confounded.
7. Run bounded search only if a primitive is legitimately searchable.
8. Write the result doc and ledger entries.

Boundaries:

- no default flips;
- no vendored Tensile promotion;
- no broad tile-config search repeat;
- no nosync authority;
- no isolated GEMM promotion claims;
- no hand-asm implementation unless the schedule-diff result explicitly names
  a primitive and the owner authorizes implementation.

Final response must include:

- verdict labels;
- whether the ~4-5% gap reproduced;
- exact role(s) analyzed;
- primitive rows and classifications;
- whether machine search is reopened for prefill;
- artifacts written;
- files changed;
- git status.
