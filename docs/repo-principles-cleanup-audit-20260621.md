# Whole-Repo Principles Cleanup Audit — 2026-06-21

Evidence-based audit of **every tracked file** against the project's tiny principles
(`structure/Development/coding-principles.md`, `…/performance-primitive-research-principles.md`),
the canonical current state (`docs/current-project-state-handoff-20260621.md`), and the harness
contract (`bench/qk-decode-eval/HARNESS_GUIDE.md`).

We are not cleaning for aesthetics. The goal is to **reduce false authority and sprawl** so future
machine-search/codegen work is supported by — not misled by — the historical probe log. The project
completes only when (1) we beat llama on the target, (2) the evaluator/search lifecycle is closed,
and (3) the live surface is lean enough to reason about end-to-end.

- **Backing inventory:** `bench/qk-repo-principles-cleanup/inventory.json` (one row per file:
  references, importers, LOC, principle_score, risk_if_deleted/kept, recommendation, reason).
- **Re-runnable builder:** `bench/qk-repo-principles-cleanup/build_repo_inventory.py` (read-only;
  extends the prior round's proven import-closure + import-safety-fixpoint engine to the whole repo).
- **Method note:** this generalizes the same evidence graph the active-surface reduction used today
  (`docs/perf-probe-active-surface-reduction-result-20260621.md`, `extra/qk_*.py` 376→133). That round
  covered only `extra/*.py` + `docs/*.md`; **this audit covers bench, tests, structure, root, and the
  non-qk surfaces too.**

## Headline

| measure | value |
|---|---|
| tracked files | 2466 |
| vendor (upstream tinygrad — excluded at dir granularity) | **835 files in 26 dirs** |
| project files (fully classified, 1 row each) | **1611** |
| **DELETE candidates (proven zero-reference)** | **0** |
| live surface that serves the current goal (CORE + LIVE_TOOLING + LIBRARY_HELPER) | **95 files / ~17.0K LOC** |
| provenance held as assets (docs + bench artifacts + spent probes) | **925 files / ~139K LOC** |
| authority docs + tests | 331 docs / 260 tests |

**The single most important finding:** the deletable scratch is already gone. Today's active-surface
reduction removed every zero-reference `extra/` probe; this whole-repo pass finds **no remaining file
with no importer, no doc ref, no test ref, and no ledger ref.** The remaining bulk (925 files) is
**provenance** — the chronological probe log + frozen verdict artifacts. Per the principles,
provenance is an **asset** (refutations prevent reopening dead lanes) and must be **kept but indexed**,
not deleted. So the lever for "lean enough" is **navigation/indexing, not deletion.**

## Recommendation taxonomy — counts

| recommendation | count | meaning |
|---|---:|---|
| `KEEP_CORE` | 27 | runtime/model path + root build config |
| `KEEP_LIVE_TOOLING` | 47 | evaluator/search/runner in the live import closure + live ledgers |
| `KEEP_LIBRARY_HELPER` | 21 | shared libs imported by ≥3 kept scripts (`qk_layout`×17, `qk_paths`×5, …) |
| `KEEP_DOC_AUTHORITY` | 331 | canonical (4) + current (296) docs + structure principles/role layer (31) |
| `KEEP_TEST` | 260 | fork test suite (`test/{external,unit,amd,testextra}`) |
| `ARCHIVE_PROVENANCE` | 925 | historical result docs (357), bench verdict artifacts (427), spent probes (128 qk + others) — keep as asset, index |
| `EXTRACT_HELPER_DELETE_DRIVER` | 0 | (see Wave 2 — no safe collapse; near-dup drivers are *cited provenance*, not duplication) |
| `UPGRADE_TO_PRINCIPLES` | 3 (overlay) | live tooling not yet on an SSOT/boundary contract (see Special Checks) |
| `DELETE` | 0 | proven zero-reference scratch — none remain |
| `IGNORE_EXTERNAL_VENDOR` | 26 dirs | upstream tinygrad framework/examples/tests (not fork work) |

> `UPGRADE_TO_PRINCIPLES` is recorded as an overlay on otherwise-`KEEP` rows (the file stays; its
> contract is tightened in Wave 4), so it is not a separate partition of the 1611.

## By subsystem

| subsystem | files | dominant recommendation | notes |
|---|---:|---|---|
| core_runtime (`tinygrad/llm/`) | 5 | KEEP_CORE | model.py / cli.py / gguf.py — the decode hot path. **Untouched** (acceptance gate). |
| extra_qk_tooling (`extra/*.py`) | 176 | 27 LIVE / 21 HELPER / 128 PROVENANCE | live evaluator+search+runners are the lean core; 128 spent probes kept as cited provenance |
| evaluator_search_ledger (`bench/qk-{decode-eval,lifecycle-search}/`) | 18 | KEEP_LIVE_TOOLING | the closed-loop ledgers/contracts; **0 absolute paths**, HARNESS_GUIDE-compliant |
| audit_tooling (`bench/*/build_*.py`) | 2 | KEEP_LIVE_TOOLING | re-runnable inventory/index builders |
| bench_artifact (`bench/**`) | 427 | ARCHIVE_PROVENANCE | frozen verdict artifacts (rest of `bench/**` is gitignored) |
| docs (`docs/*.md` + artifacts) | 663 | 4 canonical / 296 current / 357 provenance | the chronological log; provenance already indexed |
| structure (`structure/**`) | 43 | 31 KEEP_DOC_AUTHORITY / 12 ARCHIVE | principles+role layer current; 2 caches self-marked STALE |
| test (`test/{external,unit,amd,testextra}`) | 260 | KEEP_TEST | fork boundary/byte-proof/SSOT suite |
| root_config | 24 | KEEP_CORE | README, pyproject, gitignore, spec/ |

### Docs surface (acceptance gate: every `docs/*.md` classified)
657 fork docs classified — **4 canonical / 296 current / 357 provenance** (14 remaining `docs/*.md`
are upstream tinygrad pages → vendor). Provenance is navigable via
`docs/provenance-index-20260621.md` (topic→current-authority map). Largest provenance clusters:
prefill, decode, q8, tensile, mmvq, flash.

### Vendor exclusion (acceptance gate: explicitly excluded, not silently dropped)
835 files in 26 dirs are upstream tinygrad and **out of fork scope** — listed at dir granularity in
the inventory (`type:"vendor_dir"`, with `member_count`). Largest: `tinygrad/` core (175),
`examples/` (243), `extra/thunder/` (88), `test/null/` (54), `test/backend/` (40), `extra/gemm/` (37),
`extra/sqtt/` (30), `extra/amdpci/` (29). **`extra/gemm/` is upstream** (PRs #14310/#1563/#776) but
imported by 3 live qk prefill scripts as reference oracles → kept, never edited. The fork's own
prefill GEMM lives in `extra/qk_prefill_graph_gemm_route.py` (a classified project file).

## Special checks

1. **DELETE proof / SSOT — drivers without refs:** 0. Every project script has ≥1 importer, doc,
   test, or ledger reference (import-safety fixpoint + all-import-styles + external-importer scan, the
   same machinery that protected 62 rescued deps last round).

2. **SSOT drift — test duplicates enum membership (causes the 1 failing test).**
   `test/external/test_qk_search_spec.py:40` hardcodes
   `{"primitive_policy","demotion","flash_threshold","storage","schedule","lds_blocking"}` but the
   `SearchSpace` enum (`extra/qk_search_spec.py:65-70`) also defines `FLASH_VARIANT="flash_variant"`.
   The test asserts a literal set instead of *deriving* it from the enum → **pre-existing failure**
   `test_choices_match_enum_members` (Wave-0 baseline: 1 failed / 298 passed / 57 skipped). Fix =
   derive the expectation from `SearchSpace` (the enum is the single source). **Wave 4 / trivial.**

3. **SSOT drift — model-weights path has no single source.**
   `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf` is hardcoded in **17 project scripts (4 live/helper**,
   incl. `qk_harness_contract.py`**, 13 provenance)**. `extra/qk_paths.py` provides repo-relative
   `portable_path()` but no canonical `MODELS_DIR`/default-model constant. Low blast radius (machine
   fixture; the real CLI takes `-m`), but it is the classic "hidden config spread across the repo"
   anti-pattern. **Wave 4** (add one constant to `qk_paths`, route the 4 live scripts through it).

4. **Contain-dangerous-power — GPU perf-state boundary mostly holds, one documented leak.**
   `extra/qk_clock_pin.py` is the single boundary for `rocm-smi --setperflevel/--resetperfdeterminism`;
   `qk_harness_contract.py` correctly imports `read_perf_state` from it. **`qk_decode_q8_model_route_timing_audit.py:92-113`
   duplicates the rocm-smi pin sequence** with an in-code "DOCUMENTED EXCEPTION" (it emits its own
   provenance dict shape, wrapped in try/finally). It is honest and contained-by-try/finally, but it
   is a second writer of privileged state. **Wave 4** (have the boundary emit the provenance shape, or
   route through it). No other uncontained sysfs/power writers found.

5. **Bench artifacts with absolute paths — expected, not a defect.** 141 tracked bench files contain
   `/home/ubuntu`, but **the live ledgers (`bench/qk-decode-eval/*`, `bench/qk-lifecycle-search/*`)
   have 0**; the rest are **frozen verdict artifacts** where capturing the exact command/env is
   *required* by HARNESS_GUIDE (artifact contract field 4). No action — recorded as a provenance
   property, not portability debt.

6. **Good SSOT (confirmed single-sourced):** the `Verdict` enum (`extra/qk_modes.py`), the decode
   comparator `gqa_coop_vec` (authoritative in `bench/qk-decode-eval/candidates.json`), and the
   `FLASH_L` default (=128, `tinygrad/llm/model.py:924` via `getenv`) each have exactly one source.

7. **Stale docs masquerading as current:** only the 2 cache files
   (`structure/cache/repo-{cache,map}.md`) risk this; both already carry a `⚠ STALE (2026-06-21)`
   banner pointing at the handoff. ~30 docs self-mark stale/superseded. The canonical-doc guard
   (`extra/qk_policy_consistency_check.py`) passes — no canonical doc reopens a closed question.

## Waves — what this pass executed

- **Wave 0 — verification baseline (done).** Clean tree, `git diff tinygrad/` empty, policy guard
  PASS, pytest `1 failed (pre-existing SSOT drift, finding #2) / 298 passed / 57 skipped`.
- **Wave 1 — delete proven zero-reference scratch: NO-OP (0 files).** Honest outcome, not a skip:
  today's active-surface reduction already removed every deletable probe; the whole-repo reference
  graph finds nothing else with zero references. *Untracked* working-dir cruft (root `*.log`,
  `linux-mmhub-*`/`ubuntu-amdgpu-*` snapshot dirs) is **not tracked**, so out of audit scope.
- **Wave 2 — collapse duplicate drivers: EMPTY (with justification).** The near-duplicate
  `qk_*_ab.py` runners are **each the named artifact of a specific result doc/ledger refutation row**
  (cited provenance), so collapsing them would destroy refutation provenance for ~0 LOC of true
  shared knowledge — a wrong-abstraction trade the principles explicitly warn against. The *one* real
  duplication (the rocm-smi pin, finding #4) is a **boundary upgrade (Wave 4), not a driver delete**.
- **Wave 3 — archive/index stale docs (done, no moves/deletes).** Regenerated the docs supersession
  index for the post-reduction tree (`docs/provenance-index-20260621.md` via the existing
  `build_docs_index.py`) and refreshed `bench/qk-active-surface-reduction/docs_index.json`; confirmed
  the 2 STALE cache banners are present. **Docs are never moved/deleted** — that would break the ~251
  canonical→dated-doc pointers (per the index's own warning); classification + banners are the lever.

## Ranked next-action plan

> **Execution status (post-audit follow-up):** items 1–4 (the Wave-4 `UPGRADE_TO_PRINCIPLES` set)
> were executed in follow-up commits after the audit was accepted — each is a small, gate-verified,
> NFC/test-only change. Items 5–6 stand as standing guidance / deferred scope.

1. **✅ DONE — Fix the SSOT-drift test (finding #1, the only failing test).** Derived
   `test_qk_search_spec.py:40`'s expected set from the `SearchSpace` enum. Suite now 299 pass / 0 fail
   / 57 skip. *(commit `[test] derive search_space_choices assertion from SearchSpace enum`)*
2. **✅ DONE — Regenerate the two STALE caches (`structure/cache/repo-{cache,map}.md`).** Rewrote both
   as lean SSOT-pointers (current state → handoff; file ownership → this inventory; live evaluator/
   search lifecycle added; test counts refreshed; no dead refs to deleted scripts). They now classify
   KEEP_DOC_AUTHORITY (de-staled). *(commit `[docs] regenerate stale repo-cache/repo-map`)*
3. **✅ DONE — Centralize the model-weights path (finding #3).** Added `DEFAULT_MODEL_GGUF` to
   `extra/qk_paths.py`; routed the 4 live scripts through it. Value byte-identical.
   *(commit `[runtime] NFC - centralize default model-weights path in qk_paths`)*
4. **✅ DONE — Close the perf-state boundary leak (finding #4).** Hoisted the privileged sysfs/rocm-smi
   command strings into `qk_clock_pin` (`PIN_PEAK_CMD`/`SET_AUTO_CMD`/`RESET_PERF_DETERMINISM`); the
   audit script reuses them and keeps only its own provenance formatting. Commands byte-identical.
   *(commit `[runtime] NFC - centralize GPU perf-state command strings in qk_clock_pin boundary`)*
5. **Keep provenance, lean on the index.** Do **not** mass-delete the 925 provenance files — they are
   refutation assets. The real "lean enough" win is already in hand: 95 live files / ~17K LOC is the
   surface a future agent must reason about; the provenance index lets them ignore the rest safely.
6. **Deferred (Wave 5, separate scope, out of this pass):** any change to `tinygrad/`, model
   defaults, or benchmark routes. None were touched; the acceptance gate (`git diff tinygrad/` empty)
   holds.

## Acceptance gates (this pass)

- `git diff tinygrad/` empty ✓ (no core/model/default change).
- Policy guard `extra/qk_policy_consistency_check.py` PASS ✓.
- pytest `test/external/` == Wave-0 baseline (no NEW failures; the 1 failure is pre-existing finding #2) ✓.
- Inventory covers every tracked file or explicitly excludes vendor dirs ✓ (1611 project rows, 0
  uncovered, 0 duplicates; 835 vendor files in 26 dir rows).
- Every `extra/qk*.py` classified (134/134) ✓; every fork `docs/*.md` = canonical/current/provenance
  (657/657) ✓; every bench/harness file classified vs HARNESS_GUIDE ✓; every DELETE row carries proof
  (0 DELETE) ✓.
