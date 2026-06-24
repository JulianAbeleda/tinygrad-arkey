# Decode Evaluation & Benchmark Hardening — Result

Date: 2026-06-21

Scope: `docs/decode-evaluation-harness-hardening-scope-20260621.md`

## Final decision: **`EVALUATOR_READY_FOR_LIFECYCLE_SEARCH`**

A durable, automated, reproducible decode evaluator now exists and **reproduces the project's existing lifecycle
classifications** (a known FAIL, a known LOCAL-PASS/W==D-FAIL, a known PASS). The central falsifier is answered:
**whole-decode W==D under auto clock is reproducible to <0.6%** across N=5 runs — far inside the 5% promotion
margin — so the evaluator is sufficient for promotion decisions **without** GPU-state-control tooling. No model
defaults / kernels / `tinygrad/` changed.

## What was built

`extra/qk_decode_eval.py` — a subprocess-orchestrating evaluator. Per candidate it runs the lifecycle ladder
(correctness → local A/B → whole-decode W==D → policy) as **isolated subprocesses** (env set in the child only;
`getenv` is `@cache`d), wrapping the existing benchmark scripts, and emits a schema'd machine-readable verdict.
It only measures and classifies — it never edits `tinygrad/` or changes a default.

- **CLI:** `--list` · `--candidate <id> [--dry-run] [--repeats N] [--out DIR]` · `--suite historical [--out DIR]`
  · `--validate <run.json>`.
- **Runners:** `runtime_overhead` (clean W==D, N repeats, auto clock = promotion authority), `q8_audit`
  (baseline-vs-q8 W==D), `ab_script` (wrap an existing tile A/B), `flash_l_local` (a built-in FLASH_L attention
  A/B, clock-pinned = diagnostic), `q8_dnll_historical` (correctness from the historical dNLL artifact).
- **Authority discipline:** clean W==D PROFILE-off auto-clock = promotion authority; clock-pinned local and
  PROFILE timings are diagnostic/attribution only and never promotion. Promotion is *reported, never applied*.

## Candidate registry

`bench/qk-decode-eval/candidates.json` (schema `decode_eval_candidate_registry_v1`) — declarative: each candidate
declares `id, family, env, rungs, contexts, correctness_req, thresholds, historical_expected_verdict`. Thresholds:
local ≥1.05× vs `gqa_coop_vec`; W==D ≥5%@1024 / ≥7%@4096, ctx512 regress ≤1%; opt-in ≥1.03×; dNLL ≤0.01; repro
band ≤5%.

## Artifact schema

`bench/qk-decode-eval/schema.json` (JSON Schema draft-07, `decode_eval_run_v1`). Every emitted run carries the full
reproducibility metadata: `git_commit, dirty_tree, hardware, perf_state_before/after, clock_pin_mode, env,
commands[], contexts, repeats, warmups, correctness, local_ab, wd{per-ctx + repro band}, verdict,
verdict_expected, verdict_matches_expected, thresholds, stop_reason, source_files, notes, default_behavior_changed:false`.
`--validate` checks any run against the schema (all 4 emitted runs validate).

## Historical replay results (the proof)

`bench/qk-decode-eval/summaries/latest.json` — **all four verdicts match the documented classification**:

| candidate | verdict | expected | match | key numbers |
|---|---|---|---|---|
| `baseline_default` | `REST` | `REST` | ✓ | tok/s 71.8/68.3/66.6/60.9 @128/512/1024/4096; **repro band 0.56/0.59/0.30/0.33%** |
| `flash_l_64` | **`LOCAL_PASS_WD_FAIL`** | `LOCAL_PASS_WD_FAIL` | ✓ | local 1.082× @ctx1024 (byte-exact, err 3e-4); W==D Δ +2.9%@512 / +2.1%@1024 / **−1.6%@4096** (below ≥5% gate) |
| `warp_flash_tile` | **`FAIL_LOCAL_AB`** | `FAIL_LOCAL_AB` | ✓ | local 0.606× < 1.05 |
| `q8_opt_in` | **`PASS_OPT_IN`** | `PASS_OPT_IN` | ✓ | whole-decode 1.065× + dNLL 0.00289 ≤ 0.01 (opt-in, default-off) |

The flash_l_64 W==D numbers differ slightly from the documented +2.8/+1.8/−1.2% (auto-clock session variance,
within band) but the **classification is identical** — which is the replay requirement (same verdict class, not
exact values).

## Reproducibility band (the decisive measurement)

| ctx | whole-decode W==D band (max−min)/median, N=5 |
|---:|---:|
| 128 | 0.56% |
| 512 | 0.59% |
| 1024 | **0.30%** |
| 4096 | 0.33% |

**Whole-decode W==D auto-clock variance is <0.6% everywhere ≪ the 5% promotion margin.** The earlier "auto-clock
volatility" (~+17% session-to-session) was at the **per-kernel lifecycle** granularity (diagnostic), not at the
whole-decode W==D wall (promotion authority). The flash_l_64 +2.1%@1024 signal sits ~5× above its 0.44% band → a
real, distinguishable, *sub-gate* movement, correctly classified as non-promotion.

## Is W==D stable enough? **Yes.**

For whole-decode promotion decisions, auto-clock W==D is reproducible to <0.6% — comfortably below every promotion
margin. GPU-state-control tooling is **not** a prerequisite. (Per-kernel diagnostic timing still needs the clock
pin, which the diagnostic rungs already apply and restore.)

## Known limitations

- **q8 dNLL is historical.** The q8 audit script measures speed, not quality; the evaluator reads the historical
  `bench/q8-ffn-handwritten-oracle/nll_{baseline,q8_route}.json` (dNLL 0.00289). A live dNLL rung is a future add.
- **Tile A/Bs are whole-script candidates.** The raw-C tile scripts have no argparse/env, so each whole script is
  one fixed candidate (parse `first_gate_pass`); per-candidate parameterization would need an adapter. This did
  **not** block the historical replay (each tile is one candidate), so the verdict is not `NEEDS_BESPOKE_TEMPLATE`.
- **Single GPU / model.** gfx1100 + Qwen3-8B-Q4_K_M (the closed search domain).

## Next project unlocked

**The lifecycle-search loop** (audit #2): the evaluator is the execution substrate. The next project closes the
`generate → evaluate → prune` loop on top of `decode_eval` (route/fusion templates → automated ladder →
machine-readable results → refutation pruning). GPU-state-control tooling is **not** needed (band <0.6%), so
`docs/decode-gpu-state-control-scope-20260621.md` was **not** written. The ledger contract for the loop is
`bench/qk-lifecycle-search/evaluator_contract.json`.

## Exact commands

```bash
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py --list
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py --candidate flash_l_64 --dry-run
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py --suite historical --out bench/qk-decode-eval/runs/
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_eval.py --validate bench/qk-decode-eval/runs/<file>.json
PYTHONPATH=. python3 extra/qk_policy_consistency_check.py
```

## Acceptance gates

| gate | result |
|---|---|
| G1 schema validates a run | PASS (all 4) |
| G2 list/dry-run/single/suite/validate | PASS |
| G3 ≥3 verdicts (fail / local-pass-wd-fail / pass) | PASS (4) |
| G4 flash_l_64 → LOCAL_PASS_WD_FAIL | PASS |
| G5 known fail → non-promotable | PASS (FAIL_LOCAL_AB) |
| G6 artifacts carry command/env/commit/GPU/perf/repeats/thresholds/verdict | PASS |
| G7 policy consistency check passes | PASS (also extended: WMMA-decode-as-llama, MMVQ-as-gap) |
| G8 no model/default/kernel change (`git diff tinygrad/` empty) | PASS |
| G9 tree clean after commit | (commit step) |

## Boundary

No `tinygrad/llm/model.py` or `tinygrad/` change; no default/flag promotion; clock pin only in diagnostic rungs,
restored to `auto` (verified before/after = auto). Closed lanes untouched.
