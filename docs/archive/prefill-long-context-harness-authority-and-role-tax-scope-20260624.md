# Prefill Long-Context Harness Authority + Role-Tax Scope (2026-06-24)

## Objective

Determine whether the remaining long-context prefill gap is:

- a harness/provenance artifact,
- a real whole-prefill multi-chunk integration slope,
- or both.

This is a scoped audit for Spark. It is not a broad machine search, not a kernel implementation task, and not a default-flip task.

## Current state to preserve

- Current branch: `qk-prefill-flag-leak-resolution`.
- Current prefill emit default: `eightwave` is promoted by default in `extra/qk_prefill_graph_gemm_route.py`.
- Explicit emit overrides must remain honored:
  - `PREFILL_GEMM_8WAVE=0` disables eightwave.
  - Explicit `PREFILL_GEMM_CFG_*`, `PREFILL_GEMM_DBUF`, `PREFILL_GEMM_PLRA`, or `PREFILL_GEMM_PLRAB` suppress the default eightwave unless `PREFILL_GEMM_8WAVE=1`.
- Decode defaults are out of scope.

## Read first

Spark should read these before running or changing anything:

- `structure/Development/session-handoff.md`
- `docs/prefill-long-context-no-regression-audit-result-20260623.md`
- `docs/prefill-eightwave-promotion-result-20260624.md`
- `docs/prefill-eightwave-oldplra-interaction-scope-20260624.md`
- `docs/prefill-post-decode-parity-frontier-result-20260623.md`
- `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`
- `docs/prefill-per-role-transfer-attribution-result-20260623.md`
- `bench/qk-prefill-post-decode-parity-frontier/baseline_prefill.json`
- `bench/qk-prefill-long-context-no-regression-audit/time_tax_by_context.json`
- `bench/qk-decode-eval/HARNESS_GUIDE.md`

## Known evidence

### 1. The promoted eightwave default is validated

The completed strict quick run promoted only `eightwave`:

| ctx | eightwave confirm delta |
|---:|---:|
| 512 | +3.10% |
| 1024 | +2.84% |
| 2048 | +2.67% |
| 4096 | +2.28% |
| 8192 | +1.85% |

The combined `eightwave_old_plra` candidate was rejected because it regressed across contexts.

### 2. The old whole-prefill frontier has unresolved authority tension

The older frontier result contains these authority-grade whole-prefill synced numbers:

| route | whole-prefill tok/s | relative to llama |
|---|---:|---:|
| symbolic V2 | ~1236 | ~40% |
| graph-GEMM | ~1983 | ~66% |
| Tensile | ~2673 | ~87% |
| llama.cpp | ~3020-3070 | 100% |

But the same investigation also found a fresh synced concrete chunk around `3436 tok/s`, close to the headline graph-GEMM prefill numbers. That fresh concrete chunk is diagnostic only because it is one `start_pos=0` chunk, not full multi-chunk whole-prefill.

### 3. The existing per-role profiler is useful but incomplete for this question

`extra/qk_prefill_per_role_time_tax.py` captures the prefill graph under `Context(PROFILE=1)` and attributes GPU-busy by role.

It already found on a concrete `start_pos=0` chunk:

| role | shape | finding |
|---|---|---|
| `ffn_gate_up` | `512x12288x4096` | parity-class; graph-GEMM beats Tensile in that lane |
| `ffn_down` | `512x4096x12288` | deep-K gap; Tensile wins locally |
| `qo_proj` | `512x4096x4096` | below parity but not dominant |
| `kv_proj` | `512x1024x4096` | small-N workgroup starvation |

That result explicitly says the whole multi-chunk symbolic-KV axis was not measured. Spark must not use that concrete-chunk profile as the final long-context answer.

## Authority rules

Only these measurements can be headline authority:

- clean synced whole-prefill timing,
- whole multi-chunk prompt path, not one concrete chunk,
- current promoted default unless a comparator is explicitly labeled,
- repeated enough to report median/spread,
- exact flags and git hash recorded.

These are diagnostic only:

- `qk_prefill_v2_measure` nosync numbers,
- raw dispatch timing,
- PROFILE-only GPU-busy without whole-path synced timing,
- single concrete `start_pos=0` chunk throughput,
- isolated GEMM TFLOPS.

## Artifact directory

Use:

```bash
bench/qk-prefill-long-context-harness-authority-role-tax/
```

Required artifacts:

- `authority.json`
- `harness_reconciliation.json`
- `baseline_whole_prefill_by_ctx.json`
- `single_chunk_vs_whole_prefill.json`
- `per_role_time_tax_by_ctx.json`
- `route_coverage_by_role.json`
- `graphgemm_vs_tensile_integration_by_role.json`
- `decision.json`

Final result doc:

- `docs/prefill-long-context-harness-authority-and-role-tax-result-20260624.md`

## Authority lock

Before timing, write `authority.json` with:

- git commit hash,
- dirty status,
- branch,
- GPU name and arch,
- ROCm/runtime version if available,
- model path,
- command environment,
- `DEV`,
- `JIT`,
- `PREFILL_V2`,
- `PREFILL_GRAPH_GEMM`,
- `PREFILL_TENSILE_GEMM`,
- `PREFILL_GEMM_8WAVE`,
- `PREFILL_GEMM_DBUF`,
- `PREFILL_GEMM_PLRA`,
- `PREFILL_GEMM_PLRAB`,
- `PREFILL_CONCRETE_KV`,
- `PREFILL_SERVER_PROFILE`.

Record unset variables as unset. Do not silently omit them.

## Measurement lanes

### Lane A: harness reconciliation

Build a table of all relevant prefill artifacts and classify them.

Required columns:

| artifact | lane | sync? | chunking | contexts | quoted tok/s | trusted? | why |
|---|---|---|---|---|---:|---|---|

At minimum reconcile:

- `docs/prefill-post-decode-parity-frontier-result-20260623.md`
- `docs/prefill-per-role-transfer-attribution-result-20260623.md`
- `docs/prefill-long-context-no-regression-audit-result-20260623.md`
- `bench/qk-prefill-post-decode-parity-frontier/baseline_prefill.json`
- `bench/qk-prefill-long-context-no-regression-audit/time_tax_by_context.json`
- latest `/tmp/prefill-emits/emit-search-*.json` used for eightwave promotion.

If `time_tax_by_context.json` has unit ambiguity such as `best_candidate_gain_ms` values that look like microseconds/token, call that out explicitly and do not use it as headline without resolving the units.

### Lane B: authority whole-prefill by context

Measure current default whole-prefill synced throughput.

Required contexts:

- 512
- 1024
- 2048
- 4096
- 8192

Required output columns:

| ctx | current default tok/s | ms/token | total ms | repeats | spread | start_pos schedule | trusted |
|---:|---:|---:|---:|---:|---:|---|---|

Rules:

- Use current default behavior. Do not force `PREFILL_GEMM_8WAVE=1` unless the command records why.
- Use synced whole-path timing.
- Record exact command.
- Report repeat count and spread.
- If the command only measures one chunk, classify it as diagnostic and stop before making a promotion/gap decision.

### Lane C: single concrete chunk diagnostic

Measure or reuse single concrete chunk throughput only to explain harness disagreement.

Required output columns:

| ctx/start_pos | single chunk tok/s | whole-prefill tok/s | ratio | authority |
|---|---:|---:|---:|---|

Rules:

- Label as diagnostic.
- Include `start_pos`.
- Compare against Lane B to quantify whether single-chunk numbers overstate full prompt throughput.

### Lane D: whole-prefill per-role tax by context

Extend or wrap `extra/qk_prefill_per_role_time_tax.py` so it can attribute roles across the whole multi-chunk path, not only a concrete `start_pos=0` chunk.

Required contexts:

- 512
- 1024
- 2048
- 4096
- 8192

Required role buckets:

- `ffn_gate_up`
- `ffn_down`
- `q_proj`
- `o_proj`
- `k_proj`
- `v_proj`
- `attention_qk`
- `attention_pv`
- `norm_rope`
- `copy_materialization_layout`
- `runtime_host_sync`
- `other`

Required output columns:

| ctx | role | shape | calls | route | ms | share | grows_with_ctx? | actionable |
|---:|---|---|---:|---|---:|---:|---|---|

Rules:

- Capture under `Context(PROFILE=1)` if using ProfileGraphEvent GPU-busy.
- Pair role attribution with authority Lane B timing so the profiler remains attribution, not headline speed.
- Do not infer whole-prefill role share from one concrete chunk.
- If whole-path role attribution is not technically possible with the existing profiler, stop and write an instrumentation scope instead of guessing.

### Lane E: GraphGEMM vs Tensile integration comparator

If available and byte-identical on the same harness, compare GraphGEMM and Tensile in whole-prefill synced mode.

Required output columns:

| ctx | graphgemm tok/s | tensile tok/s | delta | byte-identical? | role causing delta |
|---:|---:|---:|---:|---|---|

Rules:

- Same harness only.
- Same prompt/chunk schedule only.
- No promotion from this lane alone.
- Purpose is attribution, not default selection.

## Command pointers

Start from existing tools instead of writing one-off scripts:

```bash
DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_emit_search.py --quick
```

```bash
DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_per_role_time_tax.py
```

If `extra/qk_prefill_whole_synced.py` is present and still matches the current harness contract, prefer it for Lane B authority. If it is missing or stale, adapt the synced arbiter pattern from `extra/qk_prefill_tc_attn_concrete_gate.py`: burst, timed prompt path, explicit device synchronize, correctness gate where applicable.

Do not use `qk_prefill_v2_measure` as the headline result. If used, label it nosync diagnostic.

## Stop rules

Stop and write a partial result if any of these occurs:

- `PREFILL_HARNESS_AUTHORITY_INCOMPLETE_STOP`: only single-chunk or nosync data exists.
- `PREFILL_HARNESS_TRAP_CONFIRMED`: nosync/raw-dispatch and synced whole-path disagree materially.
- `PREFILL_ROLE_TAX_INSTRUMENTATION_MISSING`: per-role attribution cannot be collected on the whole-prefill path.
- `PREFILL_ROUTE_MISMATCH_STOP`: expected current default route does not fire.
- `PREFILL_CORRECTNESS_MISMATCH_STOP`: comparator output mismatch appears.
- `PREFILL_SPREAD_TOO_HIGH_STOP`: timing spread is too high to make a decision.

## Decision labels

Final `decision.json` must choose one or more:

- `PREFILL_LONGCTX_HARNESS_ARTIFACT_CONFIRMED`
- `PREFILL_LONGCTX_REAL_INTEGRATION_SLOPE_CONFIRMED`
- `PREFILL_LONGCTX_ROLE_TAX_ATTRIBUTED`
- `PREFILL_LONGCTX_ATTENTION_OR_KV_BOUND`
- `PREFILL_LONGCTX_LAYOUT_OR_INTEGRATION_BOUND`
- `PREFILL_LONGCTX_GEMM_ROLE_COVERAGE_BOUND`
- `PREFILL_LONGCTX_NO_SEARCH_NEXT`
- `PREFILL_LONGCTX_INSTRUMENTATION_REQUIRED`

## Required final tables

The result doc must include:

### Harness reconciliation

| artifact | lane | sync? | chunking | contexts | quoted tok/s | trusted? | why |
|---|---|---|---|---|---:|---|---|

### Whole-prefill by context

| ctx | current default tok/s | ms/token | repeats | spread | start_pos schedule | trusted |
|---:|---:|---:|---:|---:|---|---|

### Single chunk vs whole

| ctx/start_pos | single chunk tok/s | whole-prefill tok/s | ratio | authority |
|---|---:|---:|---:|---|

### Per-role tax

| ctx | role | shape | calls | route | ms | share | grows_with_ctx? | actionable |
|---:|---|---|---:|---|---:|---:|---|---|

### Decision

| gap/source | evidence | next lever | search? |
|---|---|---|---|

## Expected interpretation going in

The current evidence confirms harness confusion exists. It does not yet prove that all long-context loss is harness-only.

The likely useful next answer is not "search more emit variants." It is:

- establish synced whole-prefill authority for the current eightwave default,
- quantify how much single-chunk diagnostics diverge from whole multi-chunk prompt timing,
- attribute the remaining long-context slope by role,
- then decide whether the next lever is attention/KV, layout/materialization, per-shape GEMM config, or instrumentation.

