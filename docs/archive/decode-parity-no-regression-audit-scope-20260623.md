# Decode Parity / No-Regression Audit Scope (2026-06-23)

## Mission

Run a focused decode audit to answer one question:

```text
Is the current decode default still at/near llama parity across context, and if not, where exactly is the lost tok/s?
```

This is not a broad prefill or emit search. `eightwave` is a prefill-side flag and is not expected to directly improve
decode. Decode work should stay on decode routes, decode attention, KV/cache materialization, combine tax, small-op tax,
and harness/config reconciliation.

## Required context

Read first:

- `docs/decode-ctx-slope-audit-result-20260623.md`
- `docs/decode-ctx-slope-audit-scope-20260623.md`
- `docs/owned-tile-buffer-identity-kv-read-result-20260623.md`
- `docs/decode-campaign-final-synthesis-20260623.md`
- `docs/post-owned-attention-default-audit-result-20260623.md`
- `bench/qk-decode-eval/HARNESS_GUIDE.md`
- `structure/Development/session-handoff.md`

Historical benchmark caveat:

- Older artifacts show tinygrad slightly ahead of llama at measured decode contexts.
- Newer default decode artifacts show a lower curve and a gap versus llama.
- Treat this as a harness/config/route reconciliation problem until proven otherwise; do not assume a real regression
  without route and command provenance.

## Required artifact directory

```text
bench/qk-decode-parity-no-regression-audit/
```

Required artifacts:

- `authority.json`
- `artifact_reconciliation.json`
- `wd_decode_by_ctx.json`
- `route_fire_by_ctx.json`
- `time_tax_by_ctx.json`
- `llama_vs_tinygrad_table.json`
- `decision.json`

Required result doc:

- `docs/decode-parity-no-regression-audit-result-20260623.md`

## Measurement authority

Use synced whole-decode W==D only for tok/s claims.

Rules:

- `.item()` / synchronization must be inside the timed decode loop.
- Route firing must be recorded for each context/config.
- PROFILE/GPU timestamps are attribution only.
- Raw dispatch or no-sync timings are not promotion authority.
- Record exact command, environment, git hash, GPU, model path, repeats, spread, and artifact paths.

## Contexts

Required:

- 512
- 1024
- 2048
- 4096

Optional if already supported safely:

- 3072
- 6144
- 8192

## Config matrix

### Current default

Expected:

```text
owned AMDGCN decode attention enabled
DECODE_ATTN_KV_IDENTITY=1/default if supported
whole-cache path fires
```

### Old materialized/slice comparator

Expected:

```text
DECODE_ATTN_KV_IDENTITY=0
old slice/materialization path fires
```

### Legacy attention comparator

Only if cheap:

```text
DECODE_ATTN_AMDGCN_TILE=0
```

### Llama reference

Use the existing llama artifacts first. Refresh only if artifact provenance is incompatible with the current harness.

## Phase 0: authority lock

Record:

- HEAD
- git status
- GPU and arch
- ROCm/HIP state if relevant
- model path
- exact default decode flags
- llama reference source
- selected harness scripts
- all reused artifacts and why they are valid

Verdicts:

- `DECODE_PARITY_AUTHORITY_LOCKED`
- `DECODE_PARITY_AUTHORITY_INCOMPLETE_STOP`

## Phase 1: artifact reconciliation

Purpose:

Reconcile the two decode stories:

- older/alternate artifact where tinygrad was approximately 103-106% of llama
- current default artifact where tinygrad is below llama by roughly 11-12%

Required output table:

| artifact | harness | route flags | contexts | tinygrad tok/s | llama tok/s | delta | trusted for current decision? |
|---|---|---|---:|---:|---:|---:|---|

Verdicts:

- `DECODE_ARTIFACTS_RECONCILED`
- `DECODE_ARTIFACTS_CONFLICT_NEED_RERUN`

## Phase 2: W==D decode by context

Measure current default and comparator configs.

Required output table:

| ctx | config | tinygrad tok/s | ms/token | repeats | spread % | tokens match | route |
|---:|---|---:|---:|---:|---:|---|---|

Verdicts:

- `DECODE_WD_MEASURED`
- `DECODE_WD_UNSTABLE`
- `DECODE_CORRECTNESS_FAIL_STOP`

## Phase 3: route and materialization proof

For each context/config, confirm:

- whether owned AMDGCN tile fires
- whether whole-cache path fires
- whether materialization/slice path appears
- whether fallback path appears
- whether expected kernels match actual kernels

Verdicts:

- `DECODE_ROUTE_CONFIRMED`
- `DECODE_ROUTE_MISMATCH_STOP`

## Phase 4: time-tax attribution

Attribute lost time by context.

Required buckets:

- weight GEMV
- decode attention tile
- combine
- KV/cache copy or materialization
- small ops
- host/runtime overhead
- other top kernels

Required output table:

| ctx | bucket | tinygrad ms/token | llama/reference ms/token | gap ms | actionable? |
|---:|---|---:|---:|---:|---|

Verdicts:

- `DECODE_TAX_ATTRIBUTED`
- `DECODE_TAX_LIMITED`

## Phase 5: decision

Required final table:

| ctx | baseline tok/s | llama tok/s | current tinygrad tok/s | delta vs baseline | delta vs llama | main gap |
|---:|---:|---:|---:|---:|---:|---|

Decision options:

- `DECODE_AT_PARITY_NO_ACTION`
- `DECODE_ROUTE_REGRESSION_FOUND`
- `DECODE_HARNESS_RECONCILIATION_ONLY`
- `DECODE_NEXT_LEVER_COMBINE_OR_MATERIALIZATION`
- `DECODE_NEXT_LEVER_SMALL_OPS_OR_RUNTIME`

Stop condition:

- If current default is below old tinygrad artifact, do not call it a kernel regression until route flags and harness
  semantics have been reconciled.

