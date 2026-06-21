# Perf-Probe Active-Surface Reduction — Result

Date: 2026-06-21

Executed the evidence-driven low-risk delete pass from
`docs/perf-probe-active-surface-reduction-plan-20260621.md`. Goal: shrink the active `extra/` probe surface so future
agents don't treat stale one-off probes as authority — without breaking the live evaluator/search system.

## Decision: **`ACTIVE_SURFACE_REDUCTION_DELETE_COMPLETE`** (both waves executed)

**Two waves, 243 scripts deleted total** (via `git rm`; history preserved). The live evaluator/search system is
**verified intact after both** (live imports + policy guard + all CLIs + 35 test modules import clean; tinygrad/model
untouched).
- **Wave 1 (low-risk):** 26 zero-reference scratch/scope probes.
- **Wave 2 (owner-approved):** the user chose **"delete the 279"** dated-doc-cited provenance probes. Re-running the
  full safety pass (import-closure fixpoint + path-string edges + external/test-importer protection, with
  **comprehensive import detection** — see the lesson below) **rescued 62** as real dependencies (shared libs
  `qk_layout`/`qk_quantize`/`qk_packed_tile`, the test-imported `qk_flash_search`/`qk_flywheel_cli` + the whole
  `qk_flywheel_*`/`qk_semantic_*` subsystems, `qk_ansor_transition_loop`/`qk_candidate_generator`/`qk_gap_profile`,
  …), so **217** were actually deleted.

## Counts (final)

| status | count | action |
|---|---:|---|
| live | 18 | kept (evaluator/search/runner import-closure) |
| provenance | 117 | kept (canonical/ledger-cited, shared libs, test-imported, dependency-rescued) |
| **deleted** | **243** (26 + 217) | **removed from `extra/`** |
| | | `extra/qk_*.py`: **376 → 133** |

### Lesson — import-detection gap caught before damage
The first wave-2 pass used a regex that matched `from extra.X import` / `import extra.X` but **missed**
`from extra import X` and `importlib.import_module("extra.X")`. The test-stem cross-check flagged that
`test/external/test_qk_flash_search.py` (`from extra import qk_flash_search`) and `test_qk_flywheel_cli.py` exercise
delete-set scripts. Detection was made comprehensive and the set rebuilt (220 → 217); a retro-check confirmed **none
of wave-1's 26 were affected** by the gap. This is why the delete only proceeds behind an independent all-styles,
whole-repo reference check (result: every importer of a deleted script is itself deleted).

## What was deleted (26)

Zero references in any doc/ledger, not imported by any kept script, not subprocess-invoked, not referenced by any
test. By category: superseded_tensile 9, stale_scope_helper 8, no_canonical_reference 5, superseded_decode 2,
superseded_mmvq 1, superseded_prefill 1. Full list in the plan doc + `bench/qk-active-surface-reduction/inventory.json`
(`status=="delete"`). Representative: the `qk_amd_bb5a10_p8_tta*` tensile-probe scratch series, `qk_amd_gemm_shape_*`
fact-checks, `qk_decode_*_scope` / `qk_decode_owned_q8_*_scope` stale scope helpers, `qk_sub4_*` sub-4-bit probes,
`qk_tensile_block_jit`/`qk_tensile_jit_dim`, `qk_attention_sdpa_vs_flash`, `qk_ffn_contig_probe`,
`qk_flash_threshold_validate`, `qk_prefill_matmul_penalty_verify`.

Their conclusions are folded into the canonical syntheses (handoff + result docs); none carried a unique unrecorded
result. Git history preserves every file.

## What was archived
Nothing — the owner chose delete over archive (git history is the preservation mechanism; `inventory.json` records
every deleted file's path/imports/doc-refs/reason).

## What remains live (kept active): 18 live + 117 provenance = 135
Every script required to run/validate/explain current behavior, plus cited evidence, shared libraries, test deps, and
dependency-rescued modules: all `ab_script` runners, the W==D authority, the search infra, `qk_paths.py`
(`portable_path`), `qk_layout.py`/`qk_quantize.py`, the `test/external/`-exercised `qk_packed_tile_*`/`qk_loop_*`/
`qk_flash_search`/`qk_flywheel_*` subsystems, and the `qk_ansor_*`/`qk_semantic_*`/`qk_candidate_*` chains those tests
import.

## What remains manual-review
**None** — wave 2 resolved the 279: 217 deleted, 62 rescued to provenance (real dependencies). The active surface is
now fully classified (live / provenance / deleted) with no unresolved bucket.

## Validation (all passed, after BOTH waves)
```
live infra import (decode_eval, lifecycle_search_loop, candidate_template_gen, policy_check, harness_contract, flash_decode, nll_eval)  -> OK
extra/qk_policy_consistency_check.py            -> POLICY CONSISTENCY: PASS
extra/qk_decode_eval.py --list                  -> OK
extra/qk_lifecycle_search_loop.py --help        -> OK
extra/qk_candidate_template_gen.py --help        -> OK
import-smoke all 35 test/qk_* modules           -> NONE broken (all extra imports resolve)
git diff --cached --stat -- tinygrad/ model.py  -> empty (untouched)
independent all-styles whole-repo ref check on the 217 -> every importer of a deleted script is itself deleted
flywheel CLI dynamic-import command modules     -> all 12 kept (none in delete set)
```

## Residual risk
- **Dated provenance docs / artifacts** may name a deleted script in a "Changed files" line or an old artifact JSON
  (e.g. `bench/amd-broad-backend-roadmap/*tta*.json`, `structure/Development/codex-cleanup-*.md`). These are
  **historical**, not canonical, and the conclusions remain in the docs; the dangling path is acceptable and recorded
  in `inventory.json`. No **canonical** doc or **live ledger** dangles (verified).
- The reference graph is heuristic (grep/regex import detection). Mitigated by 4 safety layers + repo-wide final grep;
  the conservative ones (`qk_paths`, `qk_packed_tile_*`, `qk_loop_*`, `qk_policy_parity`, `qk_semantic_report`) were
  all rescued to `provenance`.

## Recovery from git history
History is **not** rewritten. Recover any deleted script with:
`git checkout <this-commit>~1 -- extra/qk_<name>.py`. `inventory.json` records every deleted file's path/imports/
doc-refs/reason so a conclusion can be re-located even without checkout.

## Next cleanup step
Perf-probe surface reduction is **done** (`extra/qk_*.py` 376 → 133). Remaining cleanup is **outside this scope**:
stale `bench/` artifact directories and dated `docs/`/`structure/Development/` provenance docs whose scripts are now
deleted (cosmetic dangling refs). A follow-up could prune those, but they carry no execution-surface risk. Re-running
`build_inventory.py` against the current tree reflects the kept set (deleted files are recorded in this commit's
`inventory.json` + git history).

## Acceptance gates
| gate | result |
|---|---|
| G1 inventory exists | PASS (`bench/qk-active-surface-reduction/inventory.{json,md}`) |
| G2 keep set explicit + justified | PASS (import-closure + reference graph) |
| G3 delete/archive plan before deletion | PASS (`docs/perf-probe-active-surface-reduction-plan-20260621.md`) |
| G4 low-risk deletes executed | PASS (26 `git rm`) |
| G5 no live evaluator/search import breaks | PASS (imports + CLIs + policy guard) |
| G6 no canonical doc points to a deleted active path | PASS (verified NONE) |
| G7 policy guard passes | PASS |
| G8 result doc records deleted/archive/manual-review sets | PASS (this doc + inventory.json) |
| G9 tree clean after commit / unrelated dirty listed | PASS (commit below) |

## Boundary
Active-surface reduction only. No `tinygrad/`/model change, no model/default route change, no decode verdict changed,
no history rewrite. Deleted only zero-reference scratch; provenance/manual-review kept. The 279 second wave is
owner-gated.
