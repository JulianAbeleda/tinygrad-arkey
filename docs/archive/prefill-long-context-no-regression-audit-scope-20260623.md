# Prefill Long-Context / No-Regression Audit Scope (2026-06-23)

## Mission

Run a focused prefill audit to answer one question:

```text
Does the current prefill path hold parity at longer context/prompt lengths, and if it loses, is the loss searchable or a
known non-search integration/attention limit?
```

This is a long-context prefill hardening audit. It should not restart a broad emit search unless the search-readiness
phase explicitly says the gap is in a bounded, measurable, whole-prefill-transferable candidate space.

## Required context

Read first:

- `docs/prefill-structural-emit-search-result-20260623.md`
- `docs/prefill-structural-emit-search-runbook-20260623.md`
- `docs/prefill-tensile-vs-graphgemm-whole-prefill-validation-20260623.md`
- `docs/prefill-post-decode-parity-frontier-result-20260623.md`
- `docs/prefill-post-decode-parity-frontier-scope-20260623.md`
- `docs/prefill-per-role-transfer-attribution-result-20260623.md`
- `structure/Development/session-handoff.md`

Current known state:

- `eightwave` and `old_plra` were the only strict-filter survivors in the quick strict prefill search.
- Raw top pipeline candidates were large but gated as `needs_review`.
- The completed quick-run artifacts are:
  - `/tmp/prefill-emits/emit-search-20260623-150134.json`
  - `/tmp/prefill-emits/emit-search-20260623-150134.md`
  - `/tmp/prefill-emits/emit-search-20260623-150134.csv`
- The active question is long-context whole-prefill behavior, not isolated GEMM speed.

## Required artifact directory

```text
bench/qk-prefill-long-context-no-regression-audit/
```

Required artifacts:

- `authority.json`
- `artifact_reconciliation.json`
- `baseline_prefill_by_context.json`
- `candidate_prefill_by_context.json`
- `time_tax_by_context.json`
- `shape_inventory_by_context.json`
- `search_readiness.json`
- `decision.json`

Required result doc:

- `docs/prefill-long-context-no-regression-audit-result-20260623.md`

## Measurement authority

Use whole-prefill synced measurements for performance decisions.

Rules:

- Local/isolated GEMM speed is evidence only; it is not final authority.
- Whole-prefill tok/s or ms/prompt decides whether a candidate transfers.
- Correctness/output equivalence must be recorded where harness support exists.
- Record exact command, environment, git hash, GPU, model path, repeats, spread, and artifact paths.
- Do not flip defaults during the audit.

## Contexts / prompt lengths

Required:

- 512
- 1024
- 2048
- 4096
- 8192 or nearest supported long prompt length

Optional if supported safely:

- 6144
- max supported by current harness/model memory

## Config matrix

### Current default

Use the current default prefill path exactly as shipped.

### Prior/default comparator

Use old route toggles only as a comparator when available:

```text
PREFILL_GEMM_DBUF=0
PREFILL_GEMM_PLRA=1
```

### Strict quick-search survivors

Measure only if the baseline pass is stable:

```text
old_plra
eightwave
```

### Needs-review pipeline candidates

Do not promote from these directly. Measure only if the audit explicitly enters the bounded follow-up phase:

```text
pipe_tm2_tn2
pipe_tm4_tn2
pipe_tm2_tn4
```

## Phase 0: authority lock

Record:

- HEAD
- git status
- GPU and arch
- model path
- exact prefill flags/defaults
- harness scripts and commands
- current prefill result artifacts reused
- current decode default state only for context, not as a target of this audit

Verdicts:

- `PREFILL_LONGCTX_AUTHORITY_LOCKED`
- `PREFILL_LONGCTX_AUTHORITY_INCOMPLETE_STOP`

## Phase 1: artifact reconciliation

Reconcile:

- structural emit quick-run result
- full sweep status if available
- graph GEMM vs Tensile whole-prefill validation
- prior long-context or frontier docs

Required output table:

| artifact | command | contexts | candidate/default | tok/s | decision | trusted for current audit? |
|---|---|---:|---|---:|---|---|

Verdicts:

- `PREFILL_LONGCTX_ARTIFACTS_RECONCILED`
- `PREFILL_LONGCTX_ARTIFACTS_CONFLICT_NEED_RERUN`

## Phase 2: baseline by context

Measure current default across required context lengths.

Required output table:

| ctx | default tok/s | ms/prompt | repeats | spread % | correctness | notes |
|---:|---:|---:|---:|---:|---|---|

Verdicts:

- `PREFILL_LONGCTX_BASELINE_CONFIRMED`
- `PREFILL_LONGCTX_BASELINE_UNSTABLE`

## Phase 3: candidate by context

Measure only bounded candidates against the same contexts and repeats.

Required output table:

| ctx | candidate | baseline tok/s | candidate tok/s | delta % | repeats | spread % | decision |
|---:|---|---:|---:|---:|---:|---:|---|

Promotion guard:

- candidate must improve whole-prefill, not just isolated GEMM
- candidate must not regress long context
- candidate must clear spread/noise at the measured contexts

Verdicts:

- `PREFILL_LONGCTX_CANDIDATES_MEASURED`
- `PREFILL_LONGCTX_NO_TRANSFER`
- `PREFILL_LONGCTX_CANDIDATE_NEEDS_CONFIRM`

## Phase 4: time-tax and shape inventory

Attribute context-dependent losses.

Required buckets:

- graph GEMM
- attention QK/PV
- FFN
- projections
- non-matmul/copies/materialization
- runtime/host overhead

Required shape table:

| ctx | role | M | N | K | calls | time share | route | actionable? |
|---:|---|---:|---:|---:|---:|---:|---|---|

Verdicts:

- `PREFILL_LONGCTX_TAX_ATTRIBUTED`
- `PREFILL_LONGCTX_SHAPES_INVENTORIED`
- `PREFILL_LONGCTX_TAX_UNCLEAR`

## Phase 5: search-readiness decision

Decide whether another search is justified.

Search is allowed only if all are true:

- the gap is whole-prefill measurable
- the gap is in a bounded candidate space
- candidate knobs are already represented or cheaply representable
- correctness and resource gates exist
- measured upside exceeds noise/spread

Search is not allowed if the gap is:

- integration/attention dominated
- outside current representation
- isolated-kernel-only with no whole-prefill transfer
- below measurement spread

Decision options:

- `PREFILL_LONGCTX_AT_PARITY_NO_ACTION`
- `PREFILL_LONGCTX_SEARCH_READY`
- `PREFILL_LONGCTX_NON_SEARCH_INTEGRATION_WORK`
- `PREFILL_LONGCTX_ATTENTION_BOUND`
- `PREFILL_LONGCTX_NO_TRANSFER`

