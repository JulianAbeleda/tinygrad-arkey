# Prefill Eightwave / Old PLRA Interaction Scope (2026-06-24)

## Purpose

Run one bounded interaction check before any prefill promotion decision:

- verify whether confirmed `eightwave` composes with `old_plra`
- decide whether promotion should target `eightwave` alone or `eightwave + old_plra`
- avoid reopening broad emit search

## Current evidence

Existing long-context audit result:

- `eightwave` is confirmed positive across `512,1024,2048,4096,8192`
- `old_plra` is positive but smaller and historically close to the strict long-context gate
- no artifact currently proves the combined config

Reference docs and artifacts:

- `docs/prefill-long-context-no-regression-audit-result-20260623.md`
- `bench/qk-prefill-long-context-no-regression-audit/decision.json`
- `bench/qk-prefill-long-context-no-regression-audit/candidate_prefill_by_context.json`
- `/tmp/prefill-emits/emit-search-20260623-212625.json`
- `/tmp/prefill-emits/emit-search-20260624-102656.json`

## Candidate matrix

Use a custom `--spec` with exactly these candidates:

```json
[
  ["baseline_current_default", {}],
  ["eightwave", {"PREFILL_GEMM_8WAVE": "1"}],
  ["eightwave_old_plra", {
    "PREFILL_GEMM_8WAVE": "1",
    "PREFILL_GEMM_DBUF": "0",
    "PREFILL_GEMM_PLRA": "1"
  }]
]
```

## Command

Create the spec under `/tmp` and run one strict long-context sweep:

```bash
cat > /tmp/prefill-eightwave-oldplra-interaction-spec.json <<'JSON'
[
  ["baseline_current_default", {}],
  ["eightwave", {"PREFILL_GEMM_8WAVE": "1"}],
  ["eightwave_old_plra", {
    "PREFILL_GEMM_8WAVE": "1",
    "PREFILL_GEMM_DBUF": "0",
    "PREFILL_GEMM_PLRA": "1"
  }]
]
JSON

env DEV=AMD JIT=1 PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_emit_search.py \
  --spec /tmp/prefill-eightwave-oldplra-interaction-spec.json \
  --strict --corr-mode fdr --repeats 3 \
  --contexts 512,1024,2048,4096,8192 --maxc 8704 \
  --out /tmp/prefill-emits --confirm-k 1 --confirm-repeats 6 --confirm-timeout 1200
```

## Expected outcomes

Possible decisions:

- `PREFILL_PROMOTE_EIGHTWAVE_ONLY`
- `PREFILL_PROMOTE_EIGHTWAVE_OLDPLRA`
- `PREFILL_INTERACTION_INFEASIBLE_IGNORE_COMBO`
- `PREFILL_INTERACTION_UNSTABLE_STOP`

## Decision rules

Promote `eightwave` alone if:

- `eightwave` remains positive and stable against baseline
- `eightwave_old_plra` is infeasible, equal/noisy, or worse than `eightwave`

Promote `eightwave + old_plra` only if:

- combo is feasible
- combo beats `eightwave` by at least `+0.5%` at `4096` and `8192`
- combo has no regression at `512,1024,2048`
- confirm block passes the same strict filters

Stop without promotion if:

- baseline/candidate spread is unstable enough to invalidate ranking
- combo crashes outside known infeasible handling
- route logs show an unexpected path or flag leak

## Boundaries

- Do not run `--candidates grid`
- Do not include pipeline candidates
- Do not change decode
- Do not flip defaults from this run alone unless the decision artifact is updated
- Do not treat local/isolated GEMM timing as authority; whole-prefill synced sweep is the promotion signal

## Required output update

After execution, add a result artifact/doc that records:

- exact command
- generated `/tmp/prefill-emits/emit-search-*.{json,md,csv}` paths
- baseline/eightwave/combo tok/s by context
- delta vs baseline and delta vs eightwave
- final decision
