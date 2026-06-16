# Flywheel Rewrite — Exhaustive Status

Date: 2026-06-16. Companion to [`docs/flywheel-judging-rewrite-scope.md`](../../docs/flywheel-judging-rewrite-scope.md)
(the plan) and [`docs/flywheel-rewrite-ubuntu-handoff.md`](../../docs/flywheel-rewrite-ubuntu-handoff.md)
(the Ubuntu task list). Measured against [coding-principles.md](coding-principles.md)
+ [tinygrad-coding-overrides.md](tinygrad-coding-overrides.md).

## Headline

The flywheel/LLM tooling is **~8,656 LOC across 31 files** (`extra/qk_flywheel_*`
4,758 + `extra/llm_*` 3,898). The scope doc's target for the *judging* flywheel
is **~1,840 LOC**. We are far over because **the two largest collapses have not
happened**, and a **second subsystem (the Track-1 adapter/training stack) is not
even counted in the scope doc**.

## Current LOC by subsystem

### A. Judging flywheel (scope doc's domain, target ~1,840) — ~6,600 LOC

| scope-doc target module (LOC target) | current files | current LOC | status |
|---|---|---:|---|
| `dataset.py` (~280) | dataset 528, dataset_v1 350, targeted_outcomes 854, targeted_outcomes_report 145, feature_enrich 292, feature_audit 372, coverage_plan 166 | **2,707** | ❌ not collapsed — **biggest overhang** |
| `verdict.py` (~280) | triage_eval 254, shadow 999 | **1,253** | ❌ not folded (shadow split deferred) |
| `cost_model.py` (~360) | cost_model 553 | 553 | ✅ merged (was 962) |
| `generate.py` (~260) | rollout 201, eval_harness 230, generate 105 | 536 | 🟡 core shared; callers keep manifest/markdown |
| `filter.py` (~200) | rejection_sample 329, rs_coverage_gate 93 | 422 | ❌ not consolidated |
| `scorer.py` (~220) | llm_json_scorer 118, eval_common 148 | 266 | ✅ locked w/ golden tests |
| `cli.py` (~120) | cli 67 | 67 | ✅ done |
| (extras) | triage_sft 178, eval_matrix 163, rollout_compare 268, runtime_contract 214 | 823 | 🟡 step-7 "regenerate-on-demand" candidates, but depended on by kept tooling |

### B. Track-1 adapter/training stack (NOT in scope-doc count) — ~2,029 LOC

`llm_adapter` 160, `llm_adapter_train` 270, `llm_adapter_suffix_train` 365,
`llm_adapter_json_data` 131, `llm_adapter_json_data_v4` 318,
`llm_adapter_json_data_v4_1_compiler` 268, `llm_adapter_signal_data` 138,
`llm_sft_smoke_train` 210, `llm_training_data_probe` 169.

This is the practical SFT/LoRA line (handoff Track 1.x). It has its **own
version-chain sprawl** (`json_data` + `v4` + `v4_1_compiler` + `signal` ≈ 855 LOC
of dataset builders that are the same "new experiment = new file" anti-pattern).

## Why it is still this big (root causes, not symptoms)

1. **Step 2 (dataset/row factory) never executed — ~2,400 LOC overhang.** The
   row *construction* is centralized (`assemble_row`), but the **version chain
   and multi-stage pipeline are not folded**: `dataset` → `dataset_v1` (re-normalizes
   v0) → `feature_enrich` → `feature_audit` → `coverage_plan` → `targeted_outcomes`
   (+ its `_report` split). Each stage is a separate file. Task C found the *row
   builders* are genuinely divergent (correct — don't force-collapse those), but
   the **stage chain and the v0/v1 versioning are the real bloat** and are
   collapsible.
2. **Step 5 (verdict.py fold) deferred — ~970 LOC overhang.** `shadow` (999) +
   `triage_eval` (254) should fold to ~280. Deferred because the shadow
   score/outcome path is **not regeneration-byte-provable on this box** (tests
   read committed `summary.json`/`outcomes.jsonl`; regenerating needs the GPU
   microbench generators). Real, but blocked on a verification story.
3. **A whole second subsystem (B, ~2,029 LOC) was never in scope** and carries
   its own version-chain sprawl (`json_data` v0/v4/v4_1/signal).
4. **Version chains everywhere.** `dataset_v1`, `json_data_v4`,
   `json_data_v4_1_compiler` — each "v-next" became a new file instead of a row
   in a config/spec table. This is exactly the anti-re-sprawl rule's target.

## What landed (this work)

- Cost-model merge (962→553), CLI unify, scorer golden-locked (Mac side).
- `generate.py` extraction (Task A, NFC, byte-proven).
- Shadow freeze de-dup (Task B partial, NFC, byte-proven).
- Row-builder collapse verified already-done for identical-schema sources (Task C).
- Row-id + cost-model golden portability fixes (Task D + step 0).
- Dropped 2 dead probes (`nearmiss`, `protocol_diagnostic`), −472 LOC.

## Prioritized reduction plan (value × verifiability)

Ranked by LOC ÷ risk. Each is its own commit, byte-proven, never red.

| # | action | LOC ↓ (est) | verifiable here? | risk |
|---|---|---:|---|---|
| 1 | **Fold the dataset stage-chain** (v0/v1 versioning + enrich/audit/coverage stages) into one module driven by a stage/spec table | ~1,800 | ✅ CPU-only; golden + kernel-triage regen (proven to work on this box) | medium |
| 2 | **Collapse the adapter `json_data` version chain** (v0/v4/v4_1/signal → one config-driven builder) | ~500 | 🟡 needs its committed-dataset regen tests checked | medium |
| 3 | **Consolidate `filter.py`** (rejection_sample + coverage_gate) | ~150 | ✅ CPU, has tests | low |
| 4 | **`verdict.py` fold** (triage_eval + shadow + report) | ~900 | ❌ shadow score path not GPU-regen-provable here | high |
| 5 | **Retire `eval_matrix`/`rollout_compare`/`runtime_contract`** if their consumers (`qwen_eval_matrix`, `scorecard`) can inline | ~600 | 🟡 must rework consumers | medium |

**Biggest safe win = #1** (dataset stage-chain fold, ~1,800 LOC, byte-provable on
this box). That single step roughly halves the judging flywheel. #4 is the next
largest but needs a verification story (regenerate shadow outcomes, or accept a
non-byte-proven move) before it can be done under the discipline.

## Definition of done (target)

Judging flywheel ≈ 1,840 LOC across the 8 scope-doc modules; adapter stack
collapsed to a builder + config table; version chains gone; golden +
reproduce-from-artifact tests byte-identical; anti-re-sprawl rule enforced.
