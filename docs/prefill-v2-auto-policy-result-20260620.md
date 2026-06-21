# Phase 1 result: VRAM-aware `PREFILL_V2=auto` policy

Date: 2026-06-20. Scope: `docs/prefill-policy-integration-scope-20260620.md` Phase 1. gfx1100 RX 7900 XTX 24GB,
Qwen3-8B-Q4_K_M. Harness: `extra/qk_prefill_v2_auto_policy_probe.py` → `bench/qk-prefill-policy-integration/prefill_v2_auto_policy.json`.

## What shipped (model.py)

- `PREFILL_V2=auto` (new): resolves on/off from detected VRAM in `from_gguf`, **before** `Transformer()` is built
  (so `__init__`'s warmstart construction sees the resolved value). Explicit `PREFILL_V2=0/1` always win; unset =
  off (no default change).
- `prefill_v2_auto_decision(total_vram, est_fp16, q4_bytes, kv_bytes, min_total_gb=23, margin_gb=3)` — conservative:
  enable only when total VRAM ≥ 23GB floor AND `Q4 + fp16_covered + KV + 3GB margin` fits. None VRAM → OFF.
- `_detect_total_vram_bytes()` — one-shot `rocm-smi --showmeminfo vram` parse (None on failure → conservative OFF).
- Estimate inputs (computed in `from_gguf`): fp16 covered = Σ covered-linear `numel×2` from the (fp16) state_dict;
  Q4 = gguf file size; KV from config (`2·n_kv_heads·max_context·head_dim·2·n_layers`). Logs a one-line reason.

## Evidence (all gates PASS)

Decision unit tests (8B-class: Q4 5.0, fp16 14.0, KV 1.2 GB):

| card | decision | reason |
|---|---|---|
| 24GB (25.7GB total) | **ON** | need 20.2GB + 3GB ≤ 25.7GB |
| 16GB (17.2GB total) | OFF | below 23GB floor |
| exactly 22GB | OFF | below 23GB floor |
| VRAM unknown | OFF | rocm-smi unavailable → conservative |

Real loads (Qwen3-8B, ctx2048):

| `PREFILL_V2` | resolved | peak VRAM | note |
|---|---|---:|---|
| unset | False | 5.03 GB | default unchanged |
| `0` | False | 5.03 GB | explicit off |
| `1` | True | 18.92 GB | explicit on |
| `auto` | **True** | 18.92 GB | `need 19.2GB + 3GB margin ≤ 25.8GB total → ON (fp16 covered 13.9GB, Q4 5.0GB, KV 0.3GB @ctx2048)` |

The estimate (fp16 covered 13.9GB) matches the measured +14GB footprint. Detected total VRAM 25.75GB.

Gates: 24GB→ON ✓, 16GB→OFF ✓, unknown→OFF ✓, explicit 0/1 ✓, unset off ✓, auto resolves+logs ✓. `all_gates_pass`.

## Policy

`PREFILL_V2=auto` is **safe to recommend**: it enables the fast prefill path only on cards that clearly fit it,
stays off on ≤16GB and when VRAM can't be read, and never overrides an explicit choice. **Default remains off**
(no behavior change) — flipping the global default to `auto` is an owner call (the conservative floor makes it
low-risk on 24GB+, a no-op elsewhere). Recommended CLI/server wiring: pass `PREFILL_V2=auto` (Phase 4).

Reproduce: `DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_prefill_v2_auto_policy_probe.py`
