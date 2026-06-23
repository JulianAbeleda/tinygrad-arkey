# Prefill Adversarial Tensile Liveness Audit — Exhaustive Scope / Claude Prompt

Date: 2026-06-23

## Mission

Audit whether the prior verdict:

```text
REGISTER_POOL_INSUFFICIENT_HW_LIMIT
```

is truly justified.

The current result may be directionally useful, but the phrase **hardware
limit** is stronger than the evidence unless it is reconciled against Tensile's
actual behavior on the same RDNA3 GPU.

If Tensile is faster on the same hardware, then the honest possibilities are:

1. current `build_gemm_lds2` representation is exhausted;
2. current static liveness model is too conservative;
3. current tile / fragment layout is inefficient;
4. current instruction grouping extends live ranges unnecessarily;
5. current register allocation lacks a Tensile-like pool;
6. true hardware limit only under our specific tile/occupancy constraints;
7. genuine hardware ceiling, but Tensile's advantage comes from a different
   work/layout/ABI confound.

This scope must distinguish those cases.

## Corrected Framing

Use this as the starting hypothesis:

```text
CURRENT_BUILD_GEMM_LDS2_REPRESENTATION_EXHAUSTED
```

Do **not** start from:

```text
REGISTER_POOL_INSUFFICIENT_HW_LIMIT
```

The audit may end at a hardware/occupancy ceiling, but only after proving why
Tensile does not contradict that conclusion.

## Required Reading

Read these first:

1. `docs/prefill-register-lifetime-pool-representation-result-20260623.md`
2. `docs/prefill-kloop-schedule-template-microkernel-result-20260623.md`
3. `docs/prefill-schedule-diff-oracle-and-search-reduction-result-20260623.md`
4. `docs/machine-search-representation-expansion-decode-prefill-result-20260623.md`
5. `docs/prefill-search-result-20260623.md`
6. `docs/prefill-amd-gemm-leanaddr-result-20260620.md`
7. `docs/prefill-tensile-schedule-template-extraction-result-20260620.md`
8. `docs/prefill-tensile-lds-tile-map-sketch-20260620.md`
9. `docs/prefill-tensile-winning-kernel-transfer-table-20260620.md`
10. `docs/prefill-primitive-pmc-result-20260619.md`
11. `docs/oracle-guided-gpu-primitive-explorer-result-20260623.md`
12. `docs/project-search-ledger-contract-20260623.md`
13. `bench/qk-decode-eval/HARNESS_GUIDE.md`
14. `structure/Development/performance-primitive-research-principles.md`
15. `structure/Development/session-handoff.md`

Inspect:

- `extra/qk_schedule_interleave_detector.py`
- `extra/qk_prefill_kloop_template_microkernel.py`
- `extra/gemm/rdna3_wmma_matmul.py`
- `extra/qk_tensile_schedule_template_extract.py`
- `extra/qk_amdgpu_isa_primitive_audit.py`
- `extra/qk_isa_primitive_audit.py`
- `bench/qk-prefill-register-lifetime/`
- `bench/qk-prefill-kloop-schedule-template/`
- `bench/qk-prefill-schedule-diff-oracle/`
- `bench/qk-tensile-extraction/`
- `bench/qk-project-search-ledger/ledger.jsonl`

## Non-Goals

- Do not implement a new prefill kernel.
- Do not change defaults.
- Do not route anything into the model.
- Do not rerun broad tile/config search.
- Do not promote vendored Tensile.
- Do not use this audit to claim speedup.
- Do not call something a hardware limit unless Tensile is reconciled.

## Authority Rules

This is an understanding audit.

Allowed authorities:

- code-object disassembly;
- readelf/resource metadata;
- Tensile `.dat` solution row;
- static schedule segmentation;
- approximate liveness reconstruction;
- PMC/counters only if needed and stable;
- prior synced whole-prefill only as context.

Not authority:

- isolated timing as promotion;
- no-sync timing;
- unverified assumptions about WMMA operand liveness;
- "would likely collapse occupancy" without resource evidence.

## Phase 0 — Authority Lock And Claim Restatement

Record the exact claim being tested.

Tasks:

1. Copy the prior result's liveness equation:
   - accumulators = 128 regs;
   - current A/B = 64 regs;
   - reserved = 10;
   - reusable pool = 32 regs;
   - A-prefetch reuses 32;
   - B-prefetch requires 32 more;
   - ideal full A+B ≈ 266 VGPR > 256.
2. Record the exact current tile shape:
   - WM/WN;
   - thread tile;
   - macro tile;
   - wave/workgroup;
   - LDS bytes;
   - VGPR counts for DBUF/PLRA/PLRAB/smaller tiles.
3. Record the selected Tensile solution row:
   - macroTile;
   - threadTile;
   - workGroup;
   - depthU;
   - prefetch/global read settings if recoverable;
   - VGPR/SGPR/scratch if recoverable.
4. State the weaker starting verdict:
   `CURRENT_BUILD_GEMM_LDS2_REPRESENTATION_EXHAUSTED_PENDING_TENSILE_RECONCILIATION`.

Deliverable:

```text
bench/qk-prefill-adversarial-tensile-liveness/authority_and_claim.json
```

Verdict:

- `ADVERSARIAL_LIVENESS_CLAIM_LOCKED`

## Phase 1 — Tensile Resource Extraction

Extract Tensile's real resource envelope.

Tasks:

1. Locate selected Tensile code object and symbol.
2. Run readelf/objdump/resource extraction.
3. Record:
   - VGPR;
   - SGPR;
   - LDS/group segment bytes;
   - scratch/spill;
   - wavefront size;
   - local size / workgroup;
   - occupancy proxy if available.
4. Compare to `build_gemm_lds2` variants:
   - default / PLRA;
   - DBUF;
   - PLRAB smaller tiles;
   - static PLRAB 4x4 estimate.

Deliverable:

```text
bench/qk-prefill-adversarial-tensile-liveness/tensile_resource_envelope.json
```

Verdicts:

- `TENSILE_RESOURCE_ENVELOPE_EXTRACTED`
- `TENSILE_RESOURCE_ENVELOPE_UNAVAILABLE`

Critical question:

```text
Does Tensile itself fit a comparable pipelined schedule under <=256 VGPR at useful occupancy?
```

If yes, the "hardware limit" claim weakens.
If no, explain how Tensile's selected work/layout differs.

## Phase 2 — Tensile Schedule Reconstruction Beyond Span Counts

The current detector says Tensile is `PIPELINED`, but not how it fits.

Tasks:

1. Segment Tensile into prologue / steady / epilogue.
2. Within the steady region, identify repeated WMMA groups.
3. For each group, record nearby:
   - global loads;
   - LDS stores;
   - LDS reads;
   - waits;
   - barriers;
   - scalar address updates;
   - vector address updates.
4. Attempt to infer:
   - WMMA group size;
   - prefetch distance;
   - whether A/B are both prefetched;
   - whether global loads feed next tile or current tile;
   - whether ds_loads are current tile only or next tile staging;
   - waitcnt placement relative to consumers.
5. Produce a schedule trace, even if approximate.

Deliverable:

```text
bench/qk-prefill-adversarial-tensile-liveness/tensile_schedule_trace.json
```

Verdicts:

- `TENSILE_SCHEDULE_TRACE_EXTRACTED`
- `TENSILE_SCHEDULE_TRACE_HEURISTIC_ONLY`
- `TENSILE_SCHEDULE_TRACE_BLOCKED`

## Phase 3 — Liveness Model Challenge

Challenge each assumption in the current liveness model.

Assumptions to test:

1. Accumulators must occupy 128 VGPR for the entire K-loop.
2. A/B current fragments remain live until WMMA completes.
3. Current A/B cannot be overwritten immediately after WMMA issue.
4. Only coop-load temp regs are reusable during compute.
5. A-prefetch consumes exactly the full 32-register pool.
6. B-prefetch necessarily needs 32 additional regs.
7. Smaller tile occupancy tradeoff cannot transfer.
8. 1 wave/SIMD necessarily collapses useful performance for this kernel.

For each assumption, classify:

- proven by ISA / architecture / code;
- inferred but not proven;
- contradicted by Tensile;
- unknown.

Deliverable:

```text
bench/qk-prefill-adversarial-tensile-liveness/liveness_assumption_challenge.json
```

Verdicts:

- `LIVENESS_MODEL_VALIDATED`
- `LIVENESS_MODEL_TOO_CONSERVATIVE`
- `LIVENESS_MODEL_INCONCLUSIVE`

## Phase 4 — Compare Fragment Layout / Tile Shape

Determine whether Tensile avoids the VGPR wall by using a different layout.

Tasks:

1. Compare:
   - macro tile;
   - thread tile;
   - workgroup layout;
   - depthU;
   - WMMA issue count;
   - accum fragment layout;
   - LDS read shape;
   - global load shape.
2. Identify whether Tensile has:
   - fewer live accumulators per wave;
   - shorter A/B fragment lifetimes;
   - different operand packing;
   - different LDS layout reducing temp regs;
   - different WMMA grouping.
3. State whether our `4x4` tile comparison is actually apples-to-apples.

Deliverable:

```text
bench/qk-prefill-adversarial-tensile-liveness/layout_tile_comparison.json
```

Verdicts:

- `TENSILE_AVOIDS_WALL_BY_LAYOUT`
- `TENSILE_COMPARABLE_LAYOUT_CONFIRMED`
- `LAYOUT_COMPARISON_INCONCLUSIVE`

## Phase 5 — Alternative Path Searchability

If the hardware-limit claim weakens, name the possible path.

Possible paths:

1. `different_tile_shape_search`
   - not broad BK/PAD/waves;
   - specifically tile/fragment shape that lowers live accumulator pressure
     while preserving useful occupancy.

2. `shorter_wmma_group_lifetime`
   - schedule WMMA groups to shorten A/B liveness.

3. `alternate_fragment_layout`
   - pack/reuse fragments differently.

4. `separate_A_B_prefetch_depth`
   - A-prefetch deep, B-prefetch shallow or vice versa.

5. `occupancy_tradeoff_retest`
   - only if Tensile proves lower occupancy can still win.

6. `assembler_register_allocation`
   - explicit register reassignment / pool beyond current builder.

7. `renderer_liveness_allocator`
   - codegen capability, not immediate search.

For each path:

- required representation;
- bounded knobs;
- gates;
- expected upside;
- implementation cost;
- whether it is machine-searchable now.

Deliverable:

```text
bench/qk-prefill-adversarial-tensile-liveness/alternative_path_matrix.json
```

Verdicts:

- `ALTERNATIVE_PATH_FOUND`
- `NO_ALTERNATIVE_PATH_FOUND`
- `ALTERNATIVE_PATH_REQUIRES_HAND_ASM`

## Phase 6 — Revised Verdict

Replace or confirm the prior verdict.

Allowed final verdicts:

### Strong confirmation

```text
REGISTER_POOL_HW_LIMIT_CONFIRMED_AGAINST_TENSILE
```

Use only if:

- Tensile resource/schedule does not contradict the 256-VGPR/occupancy claim;
- differences are work/layout/ABI confounds or vendor-only tricks not portable
  to our dependency-free kernel.

### Weaker, likely safer verdict

```text
CURRENT_LDS2_REPRESENTATION_EXHAUSTED_TENSILE_PATH_UNRESOLVED
```

Use if:

- current representation is exhausted;
- Tensile appears to do something we have not reconstructed;
- no bounded path is named yet.

### Searchable path reopened

```text
PREFILL_ALTERNATIVE_SCHEDULE_PATH_SCOPED
```

Use if:

- Tensile suggests a concrete bounded representation/knob that current search
  has not tried.

### Hand-asm path only

```text
PREFILL_TENSILE_LIKE_PATH_REQUIRES_ASM_ALLOCATOR
```

Use if:

- a path exists but only with hand assembly/register allocation, not current
  generator/search.

Deliverable:

```text
bench/qk-prefill-adversarial-tensile-liveness/decision.json
```

## Phase 7 — Result Doc

Write:

```text
docs/prefill-adversarial-tensile-liveness-audit-result-20260623.md
```

Required answers:

1. Is "hardware limit" justified?
2. What is Tensile's actual VGPR/SGPR/LDS/scratch envelope?
3. Does Tensile run a comparable pipeline under the same VGPR ceiling?
4. Which liveness assumptions are proven versus inferred?
5. Does Tensile use a different tile/fragment layout?
6. Is there a bounded alternative search path?
7. If not, what exact capability is missing?
8. What should the project verdict be changed to?
9. What should machine search do next?

## Phase 8 — Ledger / Documentation Update

Update project state conservatively.

Tasks:

1. Add one ledger entry.
2. If the verdict weakens the hardware-limit claim, add a superseding note to:
   - `docs/prefill-register-lifetime-pool-representation-result-20260623.md`
     only if appropriate, or a README/handoff pointer instead.
3. Update session handoff.
4. Do not rewrite old benchmark history.

Deliverables:

```text
bench/qk-project-search-ledger/ledger.jsonl
docs/README.md
structure/Development/session-handoff.md
```

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch
`qk-prefill-flag-leak-resolution`.

Task: execute the adversarial Tensile liveness audit. The goal is to test
whether the prior `REGISTER_POOL_INSUFFICIENT_HW_LIMIT` verdict is actually
justified, or whether it should be weakened to "current LDS2 representation
exhausted" with a remaining Tensile path unresolved.

Read first:

- `docs/prefill-adversarial-tensile-liveness-audit-scope-20260623.md`
- `docs/prefill-register-lifetime-pool-representation-result-20260623.md`
- `docs/prefill-kloop-schedule-template-microkernel-result-20260623.md`
- `docs/prefill-schedule-diff-oracle-and-search-reduction-result-20260623.md`
- `docs/prefill-tensile-schedule-template-extraction-result-20260620.md`
- `docs/prefill-tensile-lds-tile-map-sketch-20260620.md`
- `extra/qk_schedule_interleave_detector.py`
- `extra/qk_tensile_schedule_template_extract.py`
- `bench/qk-decode-eval/HARNESS_GUIDE.md`
- `structure/Development/performance-primitive-research-principles.md`
- `structure/Development/session-handoff.md`

Execute phases:

1. Lock and restate the claim.
2. Extract Tensile resource envelope.
3. Reconstruct Tensile schedule beyond span counts.
4. Challenge each liveness assumption.
5. Compare fragment/tile layout.
6. Decide whether an alternative bounded path exists.
7. Write revised verdict.
8. Update ledger/docs.

Boundaries:

- no kernel implementation;
- no default changes;
- no broad search;
- no speed claims;
- no vendored Tensile promotion;
- no rewriting historical docs except superseding notes.

Final response must include:

- verdict labels;
- whether hardware-limit was confirmed or weakened;
- Tensile resource envelope;
- liveness assumptions proven/inferred/refuted;
- alternative path matrix;
- artifacts written;
- files changed;
- git status.
