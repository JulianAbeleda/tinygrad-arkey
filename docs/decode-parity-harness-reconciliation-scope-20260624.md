# Decode Parity Harness Reconciliation Scope (2026-06-24)

## Purpose

Settle the open decode question with a bounded runbook:

- determine whether the current `DECODE_ATTN_KV_IDENTITY=1` default gap is a real decode regression
- or just a flag-stack inconsistency (`Q4K_GEMV_WARP*` on/off)
- and freeze a canonical decode flag contract before any decode optimization.

## Scope

This is a decode-only harness-reconciliation scope. No source or runtime defaults are changed in this pass.

- required contexts: `512,1024,2048,4096`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- device: `DEV=AMD`, `JIT=1`
- base scripts:
  - `extra/qk_decode_runtime_overhead.py`
  - `extra/qk_decode_route_fire_check.py`
  - `extra/qk_decode_materialization_check.py`
  - `extra/qk_decode_time_tax_audit.py`
- artifact directory:
  - `bench/qk-decode-parity-no-regression-audit/`

## Config matrix for this scope

1. Canonical stack (the reference to compare toward)

```bash
DECODE_ATTN_KV_IDENTITY=1 \
Q4K_GEMV_WARP=1 \
Q4K_GEMV_WARP_DOWN=1 \
Q4K_GEMV_WARP_PROJ=1
```

2. Current default snapshot

```bash
DECODE_ATTN_KV_IDENTITY=1
# Q4K_GEMV_WARP*=default-off
```

3. Old comparator path

```bash
DECODE_ATTN_KV_IDENTITY=0
```

## Execution order

1. phase `DECODE_PARITY_AUTHORITY_LOCKED`
   - Record `HEAD`, `branch`, `git status`, GPU, model path, and selected flags in authority.
   - Verify no prefill or decode-default changes during the run.
2. phase `DECODE_ARTIFACTS_RECONCILED`
   - Copy/compare historical canonical (`Q4K_GEMV_WARP*` enabled) artifact rows from
     `bench/qk-decode-ctx-slope-audit/wd_by_ctx.json` against this run’s rows.
3. phase `DECODE_WD_MEASURED`
   - run authoritative W==D tok/s for config A and config B with:

```bash
QK_CKPTS=512,1024,2048,4096 DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py
QK_CKPTS=512,1024,2048,4096 DECODE_ATTN_KV_IDENTITY=1 DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py
QK_CKPTS=512,1024,2048,4096 DECODE_ATTN_KV_IDENTITY=0 DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_runtime_overhead.py
```
4. phase `DECODE_ROUTE_CONFIRMED`
   - run:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_route_fire_check.py
DECODE_ATTN_KV_IDENTITY=0 DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_route_fire_check.py
```
5. phase `DECODE_MATERIALIZATION_CONFIRMED` (if needed)
   - run:

```bash
DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_materialization_check.py
DECODE_ATTN_KV_IDENTITY=0 DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_materialization_check.py
```
6. phase `DECODE_TAX_ATTRIBUTED`
   - run:

```bash
QK_CKPTS=512,1024,2048,4096 DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_time_tax_audit.py
DECODE_ATTN_KV_IDENTITY=0 QK_CKPTS=512,1024,2048,4096 DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_time_tax_audit.py
```

## Required decision artifacts

- `authority.json`
- `artifact_reconciliation.json`
- `wd_decode_by_ctx.json`
- `route_fire_by_ctx.json`
- `time_tax_by_ctx.json`
- `llama_vs_tinygrad_table.json`
- `decision.json`

## Decision logic

- if canonical stack materially outperforms current default: classify as `DECODE_HARNESS_RECONCILIATION_ONLY`
- if canonical still below canonicalized llama baseline: proceed to `DECODE_NEXT_LEVER_COMBINE_OR_MATERIALIZATION`
- if route mismatch appears in A, stop with `DECODE_ROUTE_MISMATCH_STOP`
- if measurements are unstable beyond spread/consistency thresholds, stop with `DECODE_WD_UNSTABLE`

## Boundaries

- do not change defaults while this scope is running
- do not introduce any new prefill runs in this pass
- do not ship new flags until decision is complete and documented

## Source link

- `structure/Development/session-handoff.md`
- `docs/decode-parity-no-regression-audit-scope-20260623.md`
- `docs/decode-parity-no-regression-audit-result-20260623.md`
- `bench/qk-decode-parity-no-regression-audit/`
