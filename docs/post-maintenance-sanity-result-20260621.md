# Post-Maintenance Sanity — Result

Date: 2026-06-21

Lightweight readiness pass after the repo-principles / harness-maintenance sequence (verdict SSOT, artifact
portability guard, probe-harness de-clone, child-env centralization, GPU perf-state boundary + command-string
centralization, default-model-path SSOT, harness best-practices guide, optional-gguf test skip). **Guard/sanity only —
no performance benchmark rerun** (the hard rule: benchmark only if a runtime path changed; it did not).

## Decision: **`POST_MAINTENANCE_SANITY_PASS_NO_BENCH_NEEDED`**

No runtime/model/kernel/decode-route code changed in the maintenance sequence, every guard/CLI/test passes, so the
repo is ready to continue without re-running the decode/prefill benchmark matrix.

## Runtime-path verification (the load-bearing check)
Baseline = `7a5dc6381~1` (parent of the first maintenance commit, the verdict-SSOT scope revision).

`git diff <baseline>..HEAD` over `tinygrad/`, `tinygrad/llm/model.py`, `extra/qk_flash_decode.py`,
`extra/q4_k_gemv_primitive.py`, `extra/q6_k_gemv_primitive.py`:
- **Zero runtime `.py` changes.** The only change under `tinygrad/` in range is a new **docs** file
  `tinygrad/llm/FILE_INDEX.md` (+12 lines, the per-folder index). `git diff --name-only … -- tinygrad/ | grep -v '\.md$'`
  → **NONE**.
- The last commits that touched each runtime file all **predate** the maintenance baseline (`model.py` →
  `e1591e09d [nn]`; `qk_flash_decode.py` → `b52e23ac7 [nn]`; gemv primitives → `[nn]`/`[codegen]`).

→ **No runtime-affecting change. Benchmark not required.**

## Command results

| # | command | result |
|---|---|---|
| 1 | `git status --short --branch` | **PASS** — clean; `qk-prefill-flag-leak-resolution`, ahead 1 |
| 2 | `git diff <baseline>..HEAD -- <runtime files>` | **PASS** — no runtime `.py` change (only docs `FILE_INDEX.md`) |
| 3 | `qk_policy_consistency_check.py` | **PASS** — 5 canonical docs clean |
| 4a | `qk_decode_eval.py --list` | **PASS** |
| 4b | `qk_lifecycle_search_loop.py --list` | **PASS** |
| 4c | `qk_candidate_template_gen.py --list-templates` | **PASS** |
| 5 | `pytest` unit guard set (7 files) | **PASS** — 20 passed, 1 skipped (gguf optional-skip by design) |
| 6 | `pytest test/unit --collect-only` | **PASS** — 745 tests collected, no collection errors |
| 7 | `qk_harness_contract.py` selftest | **PASS** — `SELFTEST_PASS` (thin→WEAK, stamped→CONFORMS 13/13) |

## Notes
- **No bench needed:** nothing in the run indicates a performance remeasurement is warranted (no runtime/kernel/route
  change; CLIs + guards green).
- **No runtime/model/kernel files changed** (verified above). The `[runtime] NFC` maintenance commits
  (`5b1a13a1f` qk_paths model-path SSOT, `5bd7e512d` qk_clock_pin command-string centralization) touch **`extra/`
  tooling only**, not `tinygrad/`.
- **System clang:** `clang` is **not on PATH**. The 7-file guard set is pure-Python (light `extra/qk_*` modules, no
  tinygrad compile) and is **clang-independent** — it passed. The **full** `test/unit` suite (745 tests) includes
  backend/codegen tests that need a compiler, so a full run would skip/fail those **for environment reasons, not a
  project regression**. Collection (check 6) succeeds regardless. Not treated as a regression.

## Boundary
Sanity/readiness only. No `tinygrad/` change, no model/default/kernel/route change, no benchmark matrix rerun. Docs
+ this result only.
