> **⚠ SUPERSEDED (2026-06-21) — historical provenance only.** Current state lives in `docs/current-project-state-handoff-20260621.md` (+ `docs/README.md`). Do NOT treat this as authority. Many scripts/paths it references were removed in the active-surface reduction (`docs/perf-probe-active-surface-reduction-result-20260621.md`, 291 perf files deleted). Kept for history.

# Codex Task Packet — Round 2: finish the sweep + byte-safe consolidations

**Role:** Development Agent (cleanup). **Repo:** `/home/ubuntu/tinygrad-arkey`, `master`.

## Citations (authority — read first)

1. [`structure/Development/repo-audit-2026-06-16.md`](repo-audit-2026-06-16.md) — this packet finishes its **§A medium-confidence** items + the now-dead `block_dot` modules, and executes its **§C byte-safe consolidations**. §D (QKConfig/BEAM globals) and §E (adapter scaffold, dataset/verdict rewrites) remain **out of scope**.
2. [`coding-principles.md`](coding-principles.md) ("Reducing Code The Right Way": *line count is not the metric, knowledge duplication is*; *duplication is cheaper than the wrong abstraction* — do NOT merge divergent extractors) + [`tinygrad-coding-overrides.md`](tinygrad-coding-overrides.md) (prefixes; NFC discipline; anti-re-sprawl).
3. [`session-handoff.md`](session-handoff.md) — the recorded verdict for every probe.
4. Round-1 sweep = commits `cc5f11297`, `f6bf328a9`, `f9161da9e`, `52d088daa`; the fix for the red they shipped = `7014a4a94`. **Learn from that red (below).**

## ⚠️ Lessons from Round 1 — MANDATORY

Round 1 shipped a **red commit** (`52d088daa`): it deleted a module but left a use of a deleted symbol (`build_codegen_verdict`) in a *kept* test → `NameError`. It also shipped a **malformed mixed-subsystem commit** (one `[codegen]` subject covering `tinygrad/uop` + `tinygrad/llm` + `extra/` + tests). Do not repeat either:

- **Before committing a deletion, grep the deleted symbols/module across `extra/ test/ tinygrad/`** and remove every remaining reference (imports AND uses inside kept tests). `git grep -n "<deleted_name>"` must be empty (or only benign strings/comments).
- **Run the full suite after EVERY commit.** Never commit red. Record your baseline pass count first.
- **One owning prefix per commit.** Split by subsystem: `[codegen]` = `tinygrad/uop`+renderer/codegen; `[nn]` = `tinygrad/llm` (model/gguf); `[test]` = `extra/` tooling + `test/`. Never `[test]` for `tinygrad/` core. Never mix subsystems in one commit.

## Environment

`/home/ubuntu/tinygrad-arkey/.venv/bin/python`; tests: `.venv/bin/python -m pytest test/external/ -q 2>&1 | tail -5` (baseline currently **246 passed** on this AMD+model box). Don't run BEAM/risky search. Leave `bench/**` artifacts untouched.

---

## Phase A — medium-confidence probe deletions → `[test]`

For each module: run `git grep -n "<basename>" -- extra/ test/ tinygrad/`. **Delete the module ONLY if** nothing imports it (a test that merely `read_text()`s a committed `bench/*.json` without `import`ing the module is fine — delete the module and that test's module-specific assertions; a test that `import`s the module means LEAVE BOTH).

Candidates (audit §A medium): `extra/qk_batched_b0.py`, `extra/qk_packed_tile_consumption_probe.py`, `extra/qk_packed_tile_lowering_analysis.py`, `extra/qk_packed_tile_closeout_diagnostic.py`. Check `test/external/test_qk_batched_b0.py` and `test/external/test_qk_packed_tile.py` for `import` vs artifact-read. Report which you deleted vs kept and the grep evidence. (~1,030 LOC if all clear.)

## Phase B — now-dead `block_dot` modules → `[test]`

Round 1 removed `Ops.QK_BLOCK_DOT` from tinygrad core, which left two modules neutered (their core path now `raise`s "intentionally disabled"): `extra/qk_block_dot_compile_gate.py` and `extra/qk_block_dot_microbench.py`. They are concluded (`qk_block_dot_microbench_rejected`, handoff). 
1. Find all references: `git grep -n "qk_block_dot_compile_gate\|qk_block_dot_microbench"` — `extra/qk_flywheel_shadow.py` `run_outcomes` subprocess-calls them for the `qk_block_dot` `FRESH_SPECS` mechanism. Un-wire that the same way `codegen_v3` was handled (replace the call with a clear `raise "...removed, replay path no longer runnable"`, matching `shadow.py:278`); the Phase-4 shadow is concluded so this path is dead.
2. Delete both modules + their tests (`test_qk_block_dot_compile_gate.py`, and a microbench test if present). 
3. **Also check `extra/qk_semantic_op.py`**: it defined the `QK_BLOCK_DOT` *contract*. With the op removed it may now be orphaned/non-functional. Grep its importers and whether `test_qk_semantic_op.py` still passes; if it's dead, delete it + its test, else leave it. Report which.
Verify the full suite after un-wiring AND after deletion.

---

## Phase C — byte-safe consolidations (audit §C) → `[test]`/`[nn]`, each **NFC**, each its own commit

These remove genuinely-duplicated KNOWLEDGE (single-source-of-truth). Each must be **byte-proven** (state the proof in the commit body) and tagged `NFC`. Do them one at a time; if you cannot byte-prove one, SKIP it and report — do not guess.

Ordered easiest→hardest:

1. **`_load_json`/`_read_json` + id-validated jsonl readers → `extra/llm_eval_common`.** 3–4 byte-identical copies (`llm_eval_matrix`, `llm_runtime_contract`, `llm_training_data_probe`, `llm_rollout_compare`; the id-jsonl reader in `llm_json_rejection_sample`/`llm_rollout_compare`/`llm_training_data_probe`/`llm_sft_smoke_train`). Route them to the canonical `load_json`/`read_jsonl`. **Proof:** full suite green (these have tests) + `git grep` shows no local redefinition left. ~50 LOC.
2. **matrix/roofline/scorecard/gap helpers → one source.** `_load_json`/`_fmt`/`_decision_path`/`_last_runtime_storage` duped across `qk_experiment_matrix`/`qk_bandwidth_roofline`/`qk_llama_scorecard`/`qk_gap_profile`/`qk_decode_summary`; `LLAMA_REFS` re-typed in `qk_policy_pipeline.py:13` (import it from `qk_experiment_matrix`); `_git_commit` duped `qk_ansor`↔`qk_policy_pipeline`. **Proof:** suite (these are tested via `test_qk_ansor_transition`/`test_qk_policy_pipeline`/`test_qk_decode_summary`). ~70 LOC.
3. **opt-string + load-width parse rule → one helper.** Same regexes in `qk_flywheel_cost_model._opts_features/_parse_load_width`, `qk_flywheel_dataset_v1._opts_features/_load_width_words`, `qk_flywheel_feature_enrich._load_width_words`. Extract `parse_opts()`/`parse_load_width_words()` returning raw ints; each caller casts/renames locally so output is unchanged. **Proof:** the cost-model golden (`test_flywheel_dataset_golden.py::test_cost_model_centroid_output_is_pinned`) + kernel-triage regen must stay byte-identical. ~35 LOC.
4. **`qk_flywheel_shadow` jsonl helpers → `llm_eval_common`** (keep `_jsonl_bytes`/`_sha256` for freeze hashing). **Proof:** the deterministic freeze test (`test_qk_flywheel_phase4.py::test_freeze_is_deterministic...`) + freeze.json hash unchanged. ~10 LOC.
5. **staged-shadow `v0`+`v2` constants → rows in the existing `STAGED_BATCHES` table** in `qk_flywheel_shadow.py`; collapse the 14-way CLI `step` enum to `freeze/run/score --batch <id>`. Output dirs/tensors unchanged → artifacts regenerate identically. **Proof:** `test_qk_flywheel_phase4.py` green + (if you can) a freeze on `v2` produces the same `freeze.json` bytes as before. ~40 LOC.
6. **`q4k_bench` device-metric driver → one helper.** `_classify`/`DEVICE_RE`/`CORRECT_RE`/`_measure` duped across `qk_generation_g0`/`qk_generation_g0prime`/`qk_metric_audit`/`qk_threeway_load_microbench` (all kept/tested). Extract `q4k_bench_driver`. **Proof:** `test_qk_generation_g0*`/`test_qk_metric_audit` green. ~80 LOC.
7. **Q4_K/Q6_K scale-min unpack + dequant → `tinygrad/llm/gguf.py` as authority; `extra/qk_layout.py` imports it.** `[nn]`. **Proof:** add/keep a numeric-equality assert (`qk_layout.q4_k_reference` vs gguf path on a sample block) and full suite green; confirm `model.py` decode still byte-exact via a tiny fixed-seed run if you can. ~6 load-bearing lines (but high care — it's the dequant). If unsure, SKIP and report.
8. **test fixtures → new `test/external/_qk_testutil.py`** (`REPO`, `read_jsonl`, `triage_row(**overrides)`, `triage_prompt(**overrides)`); route the ~6 phase tests + the 36 `parents[2]` copies through it. **Proof:** identical pass count, no behavior change. ~120 LOC net.
9. **`llm_json_rejection_sample` → use `llm_generate`** (`configure_process_env`/`load_model_and_tokenizer`/`generate_one`) instead of re-rolling env+load+loop — the one place the extraction is bypassed. **Proof:** this changes generation plumbing → **fixed-seed token parity required**: before the change, run a tiny rejection-sample on a 2-prompt file + fixed seed and save the accepted tokens; after, rerun and diff — must be identical. If you cannot run generation, SKIP and hand back to the maintainer (do not commit unproven). ~40 LOC.
10. small: `_majority` (`cost_model`→import from `triage_eval`), dead no-op ternary in `targeted_outcomes`, dead branch in `feature_enrich._load_width_report_paths`. **Proof:** suite + relevant golden. ~25 LOC.

## Out of scope (do NOT do)

- audit §D (`QKConfig` env-flag centralization; BEAM/warm-start module-globals) — human decision.
- audit §E (adapter dataset scaffold — needs goldens added first; dataset-chain / verdict-shadow rewrites — change the `plus` corpus = research evidence).
- Do NOT merge the divergent row-builders / different-dataset builders (principles: wrong abstraction).
- Do NOT touch `bench/**` artifacts or the "already good" list (audit §F).

## Verification gate (every commit) & discipline

- Baseline first; after each commit: full suite green (= baseline − intentionally-deleted tests, zero new failures); `git grep` the deleted/renamed symbol across `extra/ test/ tinygrad/` is empty; `py_compile` touched files; `git diff --check` clean.
- NFC commits (Phase C) must be byte-proven — cite the golden/freeze hash or fixed-seed token parity in the body.
- One owning prefix per commit; split tinygrad-core from extra/tests. Cite the audit + handoff in each body; end with the `Co-Authored-By` trailer. **Pull-rebase before push.**

## Handoff artifact

Per phase: deleted/changed files + LOC, grep evidence for each medium-confidence keep/delete decision, the `qk_semantic_op` orphan verdict, which Phase-C consolidations you byte-proved vs skipped (and why), baseline vs final pass count, and commit SHAs.
