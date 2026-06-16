# Flywheel Judging Tooling — Rewrite Scope

Audience: whoever executes the consolidation (human or agent). Goal: shrink the
working-but-unmaintainable judging flywheel from ~7,000 LOC to a clean ~1,800-LOC
core you can build forward on, **without losing the judging capability**, and
with a structure that *prevents re-sprawl*. Written to
`structure/Development/coding-principles.md` (centralize, modularize, abstract,
orthogonalize; encode invariants; NFC discipline).

> Verdict from the analysis: **salvageable. ~4× reduction. The sprawl is ~75%
> accidental** (clone functions, version chains, triplicated staging, repeated
> IO/markdown boilerplate), not inherent. The deterministic-scoring nucleus is
> already clean. Irreducible complexity is only ~300–400 LOC.

## The Problem

The judging loop — **generate → score deterministically → filter → verdict** —
works and is worth keeping. But it is spread across ~27 scripts / ~7,061 LOC
with ~26 near-identical row-builder functions, v0–v4 chains, triplicated shadow
staging, and IO/argparse/markdown reimplemented 3–10×. It is too big to extend
safely, so forward progress stalls. The fix is structural collapse, not a
feature change.

## Essential vs Incidental (what we are protecting vs cutting)

- **Essential nucleus (already clean — keep ~as-is):** the deterministic scorer
  (`extra/llm_json_scorer.py`, ~92% essential — `score_expected_json`,
  `wilson_interval`, axis summary) + `extra/llm_eval_common.py:94-138`
  (`score_prompt`, `quality_summary`); the verdict label rule
  (`qk_flywheel_dataset.py:74-89` `_label_reason_retry`); the cost-model feature
  policy + backends (`qk_flywheel_cost_model_features.py`,
  `qk_flywheel_cost_model_score.py`, ~94–98% essential); the metrics/baselines
  (`qk_flywheel_triage_eval.py`).
- **Incidental bloat (collapse/cut):** ~26 clone row-builders
  (`qk_flywheel_dataset*.py`, `qk_flywheel_targeted_outcomes.py`) all following
  one template; the `dataset_v1` and `semantic_codegen v2/v3/v4` version chains;
  triplicated Phase-4.1/4.2/4.3 staging in `qk_flywheel_shadow.py`; the
  cost-model wrapper module; ~330 LOC of repeated dict→markdown; `read_jsonl`/
  manifest-load/argparse reimplemented across files; one-off probes
  (`protocol_diagnostic`, `coverage_plan`, `nearmiss_audit`, `feature_audit`).
- **Split: ~25% essential / ~75% incidental.**

## Target Architecture (~1,840 LOC, 8 modules)

| Module | Responsibility | Target LOC | Collapses from |
| --- | --- | ---: | --- |
| `flywheel/io.py` | jsonl/json read+write, portable paths, markdown-table helper | ~120 | scattered IO + `qk_paths` + ~330 LOC markdown |
| `flywheel/scorer.py` | deterministic scoring (contains/regex/exact/json axes, wilson, quality summary) | ~220 | `llm_json_scorer` + `llm_eval_common` scorer (**keep as-is**) |
| `flywheel/dataset.py` | **one parameterized `build_row(source_spec)` factory + a declarative source table** + `label_reason_retry` + validation | ~280 | `dataset.py` + `dataset_v1` + `targeted_outcomes` + `feature_enrich` (~2,050 LOC) |
| `flywheel/generate.py` | rollout generation: in-process + subprocess-isolated child, env setup | ~260 | `llm_rollout` + harness child (~480 LOC), de-duplicated |
| `flywheel/filter.py` | rejection sampling / coverage gate | ~200 | `llm_json_rejection_sample` + `rs_coverage_gate` (~420 LOC) |
| `flywheel/cost_model.py` | feature extraction + centroid/xgboost backends + ranking | ~360 | merge `cost_model_features` + `cost_model_score`, drop wrapper (~570 LOC) |
| `flywheel/verdict.py` | baselines, macro-F1/ranking metrics, generic staged-eval (shadow), report rendering | ~280 | `triage_eval` + shadow scoring core + report (~1,500 LOC) |
| `flywheel/cli.py` | one argparse dispatcher over the above | ~120 | ~10 duplicated `main()`s |

(Module path is illustrative — a package dir `extra/flywheel/` or a flat
`extra/qk_flywheel_*` namespace both work; pick one and be consistent.)

## Irreducible Complexity — do NOT try to remove

Total ~300–400 LOC; these are real, not bloat:
- **AMD env-ordering:** `DEV`/`JIT`/`QK_PRIMITIVE_STORAGE` must be set *before*
  `from tinygrad import Tensor/Transformer` (`llm_rollout.py:17-20,110`,
  `llm_eval_harness.py:32-33,186`, `llm_json_rejection_sample.py:201-216`). The
  ordering is inherent.
- **Subprocess isolation for generation:** the harness spawns a child per policy
  mode (`llm_eval_harness.py:65`) for clean per-run AMD/JIT device state and a
  JSON summary over stdout. Keep it.
- **Dual cost-model backends:** centroid + (optional) xgboost are distinct
  algorithms; the dispatch is legitimate (~150 LOC).

## Rewrite Path (sequenced, low-risk first, each step behavior-preserving)

1. **Lock the scorer.** Promote `llm_json_scorer` + `score_prompt`/
   `quality_summary` into `scorer.py` unchanged. Add **golden tests** that assert
   byte-identical scores against committed artifacts. (Safety anchor for all
   later steps.)
2. **Collapse the row-builders** (biggest win, ~2,050 → ~280). Replace the ~26
   clone functions with `build_row(source_spec)` driven by a declarative source
   table; route all through `label_reason_retry`. Mechanical pattern-collapse.
3. **De-split the cost model.** Merge features + score into `cost_model.py`;
   delete the wrapper module + its markdown/re-export overhead.
4. **Unify generation.** Extract shared env-setup + generation loop from
   `llm_rollout` and the harness child into `generate.py`; preserve both the
   in-process and subprocess-isolated entry points (the irreducible bit).
5. **Fold metrics + shadow + reporting into `verdict.py`**, de-triplicating the
   shadow staging into one generic staged-eval driven by a stage list.
6. **Single `cli.py`** replacing the duplicated `main()`s; route IO through
   `io.py`.
7. **Drop the probes** (`protocol_diagnostic`, `coverage_plan`, `nearmiss_audit`,
   `feature_audit`, plus `eval_matrix`/`rollout_compare`/`runtime_contract` if
   regenerate-on-demand). They stay in git history and are reproducible from the
   core when a future phase needs them.

Each step is its own commit; NFC steps tagged `NFC`; never commit red.

## Verification Gate (every step)

- **Golden tests** (step 1) must stay byte-identical — this is what proves the
  collapse changed nothing.
- Full `test/external/` suite green: no previously-passing test regresses.
  (Baseline in this env: ~245 passed / 6 pre-existing failures from missing
  gguf model + no AMD device — those are not in scope.)
- The **reproduce-from-artifact** tests are the real behavior check; they
  regenerate artifacts through the touched helpers.
- `py_compile` all touched files; `git diff --check` clean.

## The Anti-Re-Sprawl Rule (the durable value)

The 7,000 LOC happened because, under speed, **every new experiment got a new
clone function/script.** The factory + source-table structure (step 2) is what
prevents recurrence — but only if the discipline is enforced:

> **A new experiment adds a *row to the source table*, not a new file or a new
> `build_*` function.** New scoring axes extend `scorer.py`; new prediction
> backends extend `cost_model.py`. If you find yourself copy-pasting a `main()`
> or a row-builder, stop — that is the re-sprawl, and it is the thing this
> rewrite exists to kill.

This belongs in `tinygrad-coding-overrides.md` as a standing rule once the
rewrite lands.

## Coupling & Where to Run

- **Mac-safe (do here):** steps 1–3 and 6 — scorer, row-factory, cost-model
  merge, CLI; all verifiable on CPU via golden + reproduce-from-artifact tests.
- **Ubuntu/AMD-coupled (do there):** steps 4–5 — `generate.py` and the shadow
  fold touch the generation path, whose full tests need the gguf model + AMD
  device to prove behavior-preserving (same reason the `shadow.py` split was
  deferred earlier).

## Done Definition

- The judging loop runs end-to-end through the 8 clean modules.
- Total flywheel LOC ≈ 1,800 (±), down from ~7,061 (~4× reduction).
- Golden + reproduce-from-artifact tests green; no previously-passing test
  regressed.
- The ~26 clone row-builders are gone, replaced by one factory + table.
- The probes and version chains are removed (history-preserved).
- The anti-re-sprawl rule is recorded in the overrides.

## Estimated Effort & ROI

- Effort: a multi-step reorganization of ~7K LOC; steps 1–2 are most of the
  reduction and are low-risk pattern-collapse. Realistically a few focused
  sessions, executable module-by-module by an agent with golden-test gating.
- ROI: converts an unmaintainable 7K tool you depend on into a maintainable
  ~1.8K core that you can extend by *config*, not copy-paste. The one-time cut
  matters less than the structure that keeps it small.
