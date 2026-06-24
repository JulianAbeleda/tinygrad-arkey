# qk-decode-eval — the decode machine-search evaluator

`extra/qk_decode_eval.py` turns a registered decode candidate into a **machine-readable verdict** by running the
lifecycle ladder (correctness → local A/B → whole-decode W==D → policy) as **isolated subprocesses**, wrapping the
existing benchmark scripts. **Measurement infrastructure only** — it never edits `tinygrad/` or changes model
defaults; it only measures and classifies. It is the first-class form of the ladder the lifecycle-search system
will drive.

## Measurement authority (never mixed)

| class | use | source |
|---|---|---|
| clean W==D, PROFILE off, **auto clock** | **promotion authority** | `extra/qk_decode_runtime_overhead.py` (`tok_s_W`) |
| clock-pinned local A/B | **diagnostic only** (never promotion) | `extra/qk_clock_pin.py` + the tile A/Bs / flash_l child |
| PROFILE GPU timestamps | attribution only | `extra/qk_decode_current_route_attribution.py` |

Promotion is **reported, never applied**. Default decode behavior is never changed by running the evaluator.

## CLI

```bash
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py --list
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py --candidate flash_l_64 [--dry-run] [--repeats N] [--out DIR]
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py --suite historical [--out DIR]
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py --validate runs/<file>.json
```

## Files

- `candidates.json` — declarative registry (id, family, env, rungs, thresholds, historical expected verdict).
- `schema.json` — JSON Schema for every emitted run artifact (`--validate` checks against it).
- `runs/<timestamp>-<candidate>.json` — one verdict artifact per candidate run (full reproducibility metadata).
- `summaries/latest.json` — last suite's verdict table.
- `HARNESS_GUIDE.md` — best-practices checklist for new performance-claiming harnesses, grounded in MLPerf,
  SPEC RG reproducibility methodology, Google Benchmark, GPU profiler docs, FlashAttention, and vLLM/PagedAttention.

## Verdict enum

Single source of truth: **`extra/qk_modes.py` → `Verdict`** (the 10 values `classify()` emits); the schema enum,
`search_policy.json` map, and `evaluator_contract.json` are kept in sync by `test/unit/test_verdict_ssot.py`.

`PASS_PROMOTE` · `PASS_OPT_IN` · `PASS_ORACLE_LOCAL_AB` · `LOCAL_PASS_WD_FAIL` · `FAIL_CORRECTNESS` ·
`FAIL_LOCAL_AB` · `FAIL_ORACLE_LOCAL_AB` · `NEEDS_GPU_STATE_TOOLING` · `SELFTEST_PASS` · `REST`.

## Historical replay (the proof requirement)

The `historical` suite proves the evaluator reproduces the project's existing lifecycle classifications (not exact
numbers — same verdict class):

| candidate | expected | source |
|---|---|---|
| `baseline_default` | `REST` + reproducibility band | the W==D falsifier (auto-clock variance vs the 5% margin) |
| `flash_l_64` | `LOCAL_PASS_WD_FAIL` | `docs/archive/decode-vector-flash-tile-realigned-result-20260621.md` |
| `warp_flash_tile` | `FAIL_LOCAL_AB` | `bench/qk-decode-vector-flash-tile/warp_tile_ab.json` |
| `q8_opt_in` | `PASS_OPT_IN` | `bench/qk-decode-primitive-transfer/decode_q8_model_route_timing_audit_result.json` + historical dNLL |

## Adding a candidate

Append to `candidates.json` with `id, family, description, env, rungs[{rung,runner,...}], contexts,
correctness_req, thresholds (or use defaults), historical_expected_verdict`. Runners: `runtime_overhead` (wd),
`q8_audit` (wd_q8), `ab_script` (wrap a tile A/B), `flash_l_local` (the built-in FLASH_L local A/B),
`q8_dnll_historical` (correctness). The ledger contract is `bench/qk-lifecycle-search/evaluator_contract.json`.

Do not register candidates that reopen closed lanes (PREFILL_V2 default flip, MMVQ, WMMA decode, bounded fusion,
FLASH_L=64 promotion). The evaluator measures; it does not promote.
