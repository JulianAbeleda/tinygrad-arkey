# Prefill Post-Decode-Parity Frontier — Exhaustive Audit + Search-Readiness Scope (2026-06-23)

## Mission

After decode reached at/above llama.cpp parity, move the optimization frontier to **prefill**.

Do not restart prefill from scratch. The repo already contains a large prefill corpus. This scope requires a read-first
reconciliation, then a focused audit of the remaining compute-bound/Tensile-class headroom, then a machine-search readiness
decision.

Core question:

```text
What is the next bounded prefill primitive after decode parity, and is it ready for ISA-guarded machine search?
```

Expected current hypothesis from the post-parity synthesis:

```text
Prefill LDS GEMM / Tensile-class compute-bound lane is the next frontier.
```

But the scope must verify that from current artifacts and fresh measurements before recommending search.

## Required Reading

Read these first:

1. `docs/decode-campaign-final-synthesis-20260623.md`
2. `docs/machine-code-translation-roadmap-result-20260623.md`
3. `docs/prefill-RECONCILIATION-source-of-truth-20260619.md`
4. `docs/prefill-TRUE-throughput-and-matmul-penalty-20260620.md`
5. `docs/prefill-amd-LEARNINGS-BANKED-and-prefill-benchmark-20260620.md`
6. `docs/prefill-amd-gemm-PROMOTED-dependency-free-20260620.md`
7. `docs/prefill-amd-gemm-residual-resolved-20260620.md`
8. `docs/prefill-amd-gemm-tensile-traceable-clockmatched-20260620.md`
9. `docs/prefill-amd-gemm-tensile-pmc-hard-audit-20260620.md`
10. `docs/prefill-tensile-winning-kernel-transfer-table-20260620.md`
11. `docs/prefill-tensile-schedule-template-extraction-result-20260620.md`
12. `docs/prefill-graph-gemm-default-on-readiness-result-20260620.md`
13. `docs/prefill-graph-gemm-default-perf-result-20260620.md`
14. `docs/prefill-default-policy-evaluation-result-20260620.md`
15. `docs/amd-gpu-holistic-primitive-model-20260623.md`
16. `docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md`
17. `structure/Development/performance-primitive-research-principles.md`
18. `structure/Development/session-handoff.md`

Inspect relevant code/tools/artifacts:

- `extra/qk_decode_runtime_overhead.py`
- `extra/qk_decode_time_tax_audit.py`
- prefill benchmark tools under `extra/` matching `qk*prefill*`
- prefill graph GEMM tools under `extra/`
- `extra/qk_isa_primitive_audit.py`
- `bench/` prefill and tensile artifacts
- `bench/qk-machine-code-translation/`
- `bench/qk-decode-eval/candidates.json`

## Boundaries

- Do not change decode unless a regression guard requires documentation.
- Do not start machine search until search-readiness gates pass.
- Do not implement new kernels in the audit phase.
- Do not flip defaults.
- Do not do 14B/32B unless explicitly requested.
- Do not trust stale prefill headlines; reconcile source-of-truth first.
- Do not use local GEMM-only speed as final authority; require whole-prefill transfer.
- Do not ignore ISA/resource evidence.

## Required Artifact Directory

```text
bench/qk-prefill-post-decode-parity-frontier/
```

Required artifacts:

- `authority.json`
- `corpus_reconciliation.json`
- `baseline_prefill.json`
- `shape_inventory.json`
- `time_tax.json`
- `isa_audit.json`
- `tensile_gap_attribution.json`
- `search_readiness.json`
- `next_action_decision.json`

Required docs:

- `docs/prefill-post-decode-parity-frontier-result-20260623.md`
- if search is justified:
  - `docs/prefill-lds-gemm-machine-search-scope-20260623.md`
- if not:
  - `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`

## Phase 0 — Authority Lock

Record:

- HEAD;
- git status;
- GPU/arch;
- ROCm/HIP toolchain state;
- model path;
- current decode default state;
- current prefill flags/defaults;
- prefill graph GEMM/default policy state;
- llama/Tensile/rocBLAS references available;
- artifacts reused vs refreshed.

Artifact:

- `bench/qk-prefill-post-decode-parity-frontier/authority.json`

Verdicts:

- `PREFILL_AUTHORITY_LOCKED`
- `PREFILL_AUTHORITY_INCOMPLETE_STOP`

## Phase 1 — Corpus Reconciliation

Purpose:

The repo has many prefill docs with historical corrections. Build one current source-of-truth before measuring.

Required:

1. Read the required prefill docs.
2. Extract:
   - current promoted prefill routes;
   - default-on/default-off status;
   - known good benchmarks;
   - known false starts;
   - known clock/noise traps;
   - known shape coverage;
   - known Tensile-class kernel status;
   - known residual headroom.
3. Identify stale or superseded claims.
4. Produce a compact ledger:

| item | current status | evidence | stale/superseded notes |
|---|---|---|---|

Artifact:

- `bench/qk-prefill-post-decode-parity-frontier/corpus_reconciliation.json`

Verdicts:

- `PREFILL_CORPUS_RECONCILED`
- `PREFILL_CORPUS_CONFLICTS_NEED_MANUAL_RESOLUTION`

Stop if source-of-truth cannot be established.

## Phase 2 — Baseline Prefill Benchmark

Purpose:

Measure the current prefill path after decode changes, with a clean current baseline.

Required contexts / prompt lengths:

- choose from existing canonical prefill harness;
- include at least short, medium, long prefill sizes used by prior docs;
- record exact lengths.

Required configs:

1. current default;
2. any prefill graph GEMM route if default-off but candidate-relevant;
3. llama/rocBLAS/Tensile reference if available from existing artifacts or cheap rerun.

Required metrics:

- tokens/s or ms/prompt;
- GPU time;
- wall time;
- graph/host overhead if measurable;
- correctness / output equivalence where applicable;
- spread/repeats;
- clock control policy.

Artifact:

- `bench/qk-prefill-post-decode-parity-frontier/baseline_prefill.json`

Verdicts:

- `PREFILL_BASELINE_CONFIRMED`
- `PREFILL_BASELINE_UNSTABLE`
- `PREFILL_BASELINE_REF_MISSING`

## Phase 3 — Shape Inventory

Purpose:

Prefill optimization is shape-driven. Identify dominant GEMM/contraction shapes.

Required table:

| role | M | N | K | dtype | quant/layout | calls | time share | route |
|---|---:|---:|---:|---|---|---:|---:|---|

Roles to classify:

- prefill attention QK/PV;
- FFN gate/up/down;
- projections;
- lm_head if present;
- graph GEMM candidates;
- non-matmul overhead.

Artifact:

- `bench/qk-prefill-post-decode-parity-frontier/shape_inventory.json`

Verdicts:

- `PREFILL_SHAPES_INVENTORIED`
- `PREFILL_SHAPE_ATTRIBUTION_INCOMPLETE`

## Phase 4 — Time-Tax / Bottleneck Audit

Purpose:

Find the actual current prefill bottleneck, not assumed from old docs.

Required buckets:

- GEMM / graph GEMM;
- attention;
- FFN;
- non-matmul;
- graph/runtime overhead;
- copies/materialization;
- quant/dequant;
- host overhead.

Required table:

| bucket | wall ms | gpu ms | share | top kernels | evidence | actionable? |
|---|---:|---:|---:|---|---|---|

Artifact:

- `bench/qk-prefill-post-decode-parity-frontier/time_tax.json`

Verdicts:

- `PREFILL_TAX_COMPUTE_BOUND_GEMM`
- `PREFILL_TAX_RUNTIME_BOUND`
- `PREFILL_TAX_NONMATMUL_BOUND`
- `PREFILL_TAX_UNCLEAR`

## Phase 5 — ISA / Resource Audit Of Current Prefill Kernel(s)

Purpose:

Use the now-ready ISA tooling to inspect the current leading prefill kernels.

Minimum targets:

- promoted/current prefill LDS GEMM kernel if present;
- Tensile/rocBLAS oracle code object if accessible;
- top tinygrad-generated prefill GEMM kernel;
- any hand-asm/owned prefill kernel artifact.

Required fields:

- symbol;
- code object;
- arch;
- VGPR/SGPR;
- LDS bytes;
- scratch/spill;
- MFMA/WMMA/dot/FMA instruction flags;
- LDS load/store counts if available;
- vector/global load evidence;
- occupancy estimate;
- resource difference vs Tensile-class reference.

Artifact:

- `bench/qk-prefill-post-decode-parity-frontier/isa_audit.json`

Verdicts:

- `PREFILL_ISA_AUDIT_CONFIRMED`
- `PREFILL_ISA_GAP_FOUND`
- `PREFILL_ISA_TOOLING_LIMITED`

## Phase 6 — Tensile-Class Gap Attribution

Purpose:

Explain the cited remaining headroom.

Candidate causes:

- LDS bank conflicts;
- occupancy / VGPR pressure;
- wave scheduling;
- global load width/coalescing;
- K-loop software pipeline;
- prefetch depth;
- MFMA/FMA issue mix;
- local write/read pattern;
- graph transfer / integration overhead;
- shape mismatch vs tuned Tensile tile.

Required table:

| suspected gap | evidence | estimated ms/% | bounded knob? | search-ready? |
|---|---|---:|---|---|

Artifact:

- `bench/qk-prefill-post-decode-parity-frontier/tensile_gap_attribution.json`

Verdicts:

- `PREFILL_TENSILE_GAP_ATTRIBUTED`
- `PREFILL_TENSILE_GAP_OVERLAPPED`
- `PREFILL_TENSILE_GAP_NOT_REPRODUCED`
- `PREFILL_TENSILE_GAP_UNCLEAR`

## Phase 7 — Search-Readiness Decision

Do not start search here. Decide whether search is justified.

Search is justified only if:

1. bottleneck is compute-bound and material;
2. local kernel timing transfers to whole-prefill;
3. correctness harness exists;
4. ISA audit can reject bad candidates;
5. bounded knobs exist;
6. expected W==P / whole-prefill gain is meaningful.

Required table:

| lane | searchable? | reason | knobs | first gate |
|---|---|---|---|---|

Candidate knobs if GEMM search is justified:

- block tile sizes;
- BK depth;
- LDS layout;
- vector load width;
- prefetch distance;
- unroll;
- wave grouping;
- occupancy/VGPR target;
- FMA/MFMA path;
- transposition/layout choices.

Artifact:

- `bench/qk-prefill-post-decode-parity-frontier/search_readiness.json`

Verdicts:

- `PREFILL_MACHINE_SEARCH_READY_LDS_GEMM`
- `PREFILL_MACHINE_SEARCH_NOT_READY`
- `PREFILL_NEEDS_NONSEARCH_FIX_FIRST`
- `PREFILL_REST_NO_MATERIAL_HEADROOM`

## Phase 8A — If Search Ready: Write Machine Search Scope

Only if verdict:

- `PREFILL_MACHINE_SEARCH_READY_LDS_GEMM`

Write:

- `docs/prefill-lds-gemm-machine-search-scope-20260623.md`

Required sections:

1. search objective;
2. shapes;
3. candidate generator;
4. ISA reject rules;
5. correctness harness;
6. local timing harness;
7. whole-prefill transfer gate;
8. artifact schema;
9. stop rules;
10. promotion/default policy.

Do not run search unless explicitly asked after this scope.

## Phase 8B — If Not Search Ready: Write Rest / Nonsearch Scope

Write:

- `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`

Required:

- explain why search is not ready;
- identify next nonsearch fix if any;
- decide whether prefill is at rest;
- update project roadmap.

## Phase 9 — Result Doc / Synthesis

Write:

- `docs/prefill-post-decode-parity-frontier-result-20260623.md`

Required sections:

1. Verdict.
2. Authority/config.
3. Corpus reconciliation.
4. Current prefill baseline.
5. Shape inventory.
6. Time-tax / bottleneck map.
7. ISA audit.
8. Tensile-class gap attribution.
9. Search-readiness decision.
10. Next scope written.
11. Files changed.
12. Git status.

Update:

- `docs/README.md`
- `structure/Development/session-handoff.md`
- roadmap docs if needed.

## Final Verdict Labels

Allowed:

- `PREFILL_FRONTIER_AUDIT_COMPLETE`
- `PREFILL_MACHINE_SEARCH_READY_LDS_GEMM`
- `PREFILL_MACHINE_SEARCH_NOT_READY`
- `PREFILL_TENSILE_GAP_ATTRIBUTED`
- `PREFILL_REST_NO_MATERIAL_HEADROOM`
- `PREFILL_NEEDS_NONSEARCH_FIX_FIRST`

## Claude Prompt

You are in `/home/ubuntu/tinygrad-arkey` on branch `qk-prefill-flag-leak-resolution`.

Decode has reached at/above llama.cpp parity. The new frontier is prefill.

Read and execute:

```text
docs/prefill-post-decode-parity-frontier-scope-20260623.md
```

Also read the required prefill source-of-truth docs listed in that scope, especially:

```text
docs/prefill-RECONCILIATION-source-of-truth-20260619.md
docs/prefill-TRUE-throughput-and-matmul-penalty-20260620.md
docs/prefill-amd-LEARNINGS-BANKED-and-prefill-benchmark-20260620.md
docs/prefill-amd-gemm-PROMOTED-dependency-free-20260620.md
docs/prefill-amd-gemm-residual-resolved-20260620.md
docs/prefill-tensile-winning-kernel-transfer-table-20260620.md
docs/prefill-graph-gemm-default-on-readiness-result-20260620.md
```

Mission:

1. Reconcile the existing prefill corpus.
2. Establish current prefill baseline after decode parity.
3. Inventory dominant prefill shapes.
4. Audit current bottlenecks.
5. ISA-audit current prefill/Tensile-class kernels.
6. Attribute remaining Tensile-class headroom.
7. Decide whether prefill LDS GEMM is ready for machine search.
8. If ready, write `docs/prefill-lds-gemm-machine-search-scope-20260623.md`.
9. If not, write `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`.
10. Write `docs/prefill-post-decode-parity-frontier-result-20260623.md`.

Do not:

- change decode;
- implement kernels;
- start machine search;
- flip defaults;
- do 14B/32B;
- trust stale prefill headlines;
- accept local GEMM speed without whole-prefill transfer.

Final response must include:

- final verdict;
- current prefill baseline;
- top bottleneck;
- ISA audit result;
- Tensile-class gap attribution;
- search-readiness verdict;
- next scope written;
- files changed;
- git status.
