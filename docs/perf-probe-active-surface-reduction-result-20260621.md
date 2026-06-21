# Perf-Probe Active-Surface Reduction — Result

Date: 2026-06-21

Executed the evidence-driven low-risk delete pass from
`docs/perf-probe-active-surface-reduction-plan-20260621.md`. Goal: shrink the active `extra/` probe surface so future
agents don't treat stale one-off probes as authority — without breaking the live evaluator/search system.

## Decision: **`ACTIVE_SURFACE_REDUCTION_DELETE_COMPLETE`** (low-risk tier)

26 zero-reference scratch/scope/superseded probe scripts **deleted** (via `git rm`; history preserved). The live
evaluator/search system is **verified intact**. The 279 dated-doc-cited provenance probes are **kept** this pass and
documented as an owner-gated **second wave** (archive or delete) — per the gate "do not delete manual-review files;
execute only low-risk deletes; if uncertain, archive rather than delete."

## Counts

| status | count | action |
|---|---:|---|
| live | 18 | kept (evaluator/search/runner import-closure) |
| provenance | 55 | kept (canonical/ledger-cited, shared utilities, test-imported) |
| manual_review | 279 | kept this pass (second-wave candidates) |
| **deleted** | **26** | **removed from `extra/` (350 qk_ scripts remain, was 376)** |

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
Nothing this pass (no `archive/` move). The second-wave path offers archive **or** delete.

## What remains live (kept active)
- **18 live** + **55 provenance** = 73 kept perf scripts that are required to run/validate/explain current behavior or
  are cited evidence / shared utilities / test deps. Includes all `ab_script` runners, the W==D authority, the search
  infra, `qk_paths.py` (`portable_path`), and the `test/external/`-imported `qk_packed_tile_*` / `qk_loop_*` scripts.

## What remains manual-review (279, second wave)
Cited only by dated result docs (conclusion captured there): superseded_decode 64, superseded_tensile 53,
superseded_mmvq 48, superseded_prefill 38, no_canonical 41, stale_scope 21, generated_scratch 14. **None are imported
by live code** (verified), so they can be archived/deleted in one command when the owner decides. Recommended:
`archive/perf-probes-20260621/` with a manifest, or `git rm` (history preserves). See `inventory.json` for the
per-file list + reasons.

## Validation (all passed)
```
live infra import (decode_eval, lifecycle_search_loop, candidate_template_gen, policy_check, harness_contract, flash_decode)  -> OK
extra/qk_policy_consistency_check.py            -> POLICY CONSISTENCY: PASS
extra/qk_decode_eval.py --list                  -> OK (candidates listed)
extra/qk_lifecycle_search_loop.py --help        -> OK
extra/qk_candidate_template_gen.py --help        -> OK
git diff --cached --stat -- tinygrad/ model.py  -> empty (untouched)
canonical-doc + live-ledger refs to deleted paths -> NONE
test/ refs to deleted paths                     -> NONE
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
Owner-gated **second wave** on the 279 manual_review provenance probes: archive to `archive/perf-probes-20260621/`
(preserve, browseable) **or** `git rm` (smaller surface, git-recoverable). Re-run `build_inventory.py` after to confirm
counts. The 279 are import-safe for either action.

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
