> **⚠ SUPERSEDED (2026-06-21) — historical provenance only.** Current state lives in `docs/current-project-state-handoff-20260621.md` (+ `docs/README.md`). Do NOT treat this as authority. Many scripts/paths it references were removed in the active-surface reduction (`docs/perf-probe-active-surface-reduction-result-20260621.md`, 291 perf files deleted). Kept for history.

# Repo Script Map + Relationship Audit — 2026-06-16

All `extra/*.py` scripts ranked, mapped by import relationship, and audited
against [coding-principles.md](coding-principles.md) +
[tinygrad-coding-overrides.md](tinygrad-coding-overrides.md). Graph built by AST
(import edges within `extra/`, plus importer counts from `test/` and `tinygrad/`).
**Post-Round-1 state:** `extra/` is now **96 files / 21,112 LOC** (was 137 / 26.3k).
Companion to [repo-audit-2026-06-16.md](repo-audit-2026-06-16.md).

## 0. The "highest LOC outside autogen" answer (strategic)

`tinygrad/runtime/autogen/` = 118,078 LOC (excluded). The largest hand-written
files outside it are **upstream tinygrad fork baggage the AMD project never uses**:

| area | LOC | origin | used by AMD decode? |
|---|---:|---|---|
| `extra/nv_pma/` (NVIDIA CUPTI bindings, `cupti.py` alone = 14,183) | 14,773 | upstream | no |
| `extra/gemm/` (incl. `cdna_asm_gemm.py` 2,910) | 5,873 | upstream | partial (cdna ref only) |
| `extra/torch_backend/` | 2,199 | upstream | no |
| `extra/qcom_gpu_driver/` + `extra/dsp/` (Qualcomm) | 2,772 | upstream | no |
| 13 upstream example/tool orphans (`export_model`, `thneed`, `hook_cuda`, `archprobe`, `multitensor`, `training`, `gradcheck`, …) | ~2,392 | upstream | no |

**This is ~25k+ LOC — far larger than the entire arkey work product.** It is the
single biggest LOC lever, but **deleting it diverges from upstream** (harder
future merges). This is a fork-strategy decision, **not** a principles cleanup:
- If you will keep re-syncing tinygrad upstream → **leave it** (carrying cost is real but merge-compat matters).
- If this fork is now its own AMD-decode line and won't merge upstream → **prune the non-AMD subsystems** (`nv_pma`, `qcom_gpu_driver`, `dsp`, `torch_backend`, the non-AMD examples) for a one-time ~22k LOC drop.
The principles do not apply to generated/vendored bindings — do not "refactor" them; the only move is keep-or-prune.

## 1. Ranked arkey scripts (top of the work product)

| module | LOC | in | out | tests | cluster |
|---|---:|---:|---:|---:|---|
| qk_flywheel_shadow | 978 | 0 | 7 | 1 | flywheel |
| qk_flywheel_targeted_outcomes | 854 | 1 | 6 | 2 | flywheel |
| q4_k_gemv_primitive | 707 | 4 | 2 | 3 | kernel (hub) |
| qk_policy_pipeline | 674 | 0 | 3 | 1 | policy (big leaf) |
| qk_ansor | 620 | 1 | 2 | 2 | policy |
| qk_flywheel_cost_model | 553 | 2 | 3 | 6 | flywheel (hub) |
| qk_flywheel_dataset | 528 | 5 | 0 | 4 | flywheel (hub) |
| qk_threeway_load_microbench | 374 | 0 | 2 | 1 | kernel |
| qk_flywheel_feature_audit | 372 | 1 | 3 | 1 | flywheel |
| llm_adapter_suffix_train | 365 | 0 | 4 | 2 | adapter |
| qk_flywheel_dataset_v1 | 350 | 0 | 0 | 2 | flywheel |
| … (full table from the AST script) | | | | | |

Cluster LOC rollup (arkey): **flywheel ~4.3k · kernel/q4k ~3.0k · semantic/ansor
~2.6k · policy/scorecard ~2.3k · llm rollout/eval ~2.0k · adapter/SFT ~1.9k ·
loop ~0.8k · shared infra ~0.6k.**

## 2. Relationship map

**Shared-infra hubs (single-source-of-truth — the principles done right; keep):**
- `llm_eval_common` (**in=25**) — IO/scoring/quality SoT for the whole llm + flywheel surface.
- `qk_layout` (in=12) — GGUF/Q4_K layout + reference dequant; imported by 12 extra + 6 tests + `model.py`.
- `qk_paths` (in=7, 9 LOC) — portable-path boundary.
- `qk_modes` (in=6) — mode/format enums (encode-invariants).
- `qk_flywheel_dataset` (in=5), `qk_descriptor_policy` (in=5), `q4_k_safety` (in=5, the risky-search gate), `q4_k_gemv_primitive` (in=4), `llm_adapter` (in=4).

**Clusters (direction = depends-on):**
- **flywheel:** `dataset → dataset_v1 → {feature_enrich, feature_audit, coverage_plan} → targeted_outcomes(+_report) → cost_model → triage_eval → shadow`; `cli`, `triage_sft`. (the chain from repo-audit §A — divergent extractors, do NOT force-merge.)
- **llm rollout/eval:** `eval_common` ← {rollout, eval_harness, generate, json_scorer, json_rejection_sample, rs_coverage_gate, eval_matrix, rollout_compare, runtime_contract, training_data_probe, sft_smoke_train}.
- **adapter/SFT:** `llm_adapter` ← {adapter_train, adapter_suffix_train, json_data v0/v4/v4_1/signal}.
- **kernel:** `qk_layout`+`q4_k_gemv_primitive`+`q4_k_safety`+`q4_k_bench` ← {flash_decode, q6_k_gemv, threeway, generation_g0*, q4_k_profile_report, …}.
- **semantic/ansor:** `qk_descriptor_policy`+`qk_semantic_candidate` ← {descriptor, schedule, schedule_bench(subproc), op, report(subproc), candidate_generator, candidate_static_gate, ansor_transition_loop}.
- **policy/scorecard:** `qk_policy_pipeline` (674) → {ansor, policy_parity, experiment_matrix, decode_summary, llama_scorecard, gap_profile, bandwidth_roofline}.
- **loop:** `qk_beam_log` ← {loop_learnability, loop_live, loop_benchmark, loop_verdict}.

## 3. Audit findings (what the graph adds beyond repo-audit-2026-06-16)

- `[HIGH] 3 NEW arkey dead-probe orphans` (zero importers/tests/subprocess refs, verdicts in handoff) the cleanup packets missed: **`q6_k_policy_sweep` (178)**, **`q8_1_q4k_bench` (117, verdict at handoff:1579)**, **`q4_k_output_ab` (145)**. Delete (≈440 LOC). (The other 13 import-orphans are UPSTREAM — leave per §0.)
- `[MED] Big low-fan-in leaf modules` = where remaining internal sprawl hides: `qk_flywheel_shadow` (978, in=0), `qk_policy_pipeline` (674, in=0), `qk_ansor` (620, in=1). These are CLI orchestrators; they hold the §C duplication (`LLAMA_REFS`/`_git_commit`/`_fmt` re-types) — already in the Round-2 packet. No new abstraction needed, just the consolidations.
- `[INFO] Centralization is healthy.` The fan-in distribution is correct: tiny deep hubs (`qk_paths` 9 LOC/in=7, `q4_k_safety` 26/in=5, `qk_modes` 101/in=6, `llm_eval_common` 148/in=25) carry the cross-cutting policy; big modules are leaves. This is "deep modules, simple surfaces" — the structure is sound, the bloat was probes (now mostly deleted), not tangle.
- `[INFO] No import cycles found within `extra/`.` Orthogonality holds.
- Pending from repo-audit (Round-2 packet): medium-confidence probes (`qk_batched_b0`, 3× `packed_tile`), the neutered `qk_block_dot_*`, and the §C byte-safe consolidations.

## 4. Recommendation (ranked by value × safety)

1. **Decide the upstream-baggage question (§0)** — the only ~22k-LOC lever; a fork-strategy call, not a cleanup. Default: leave unless abandoning upstream sync.
2. **Delete the 3 new dead-probe orphans** (~440 LOC) — fold into the Codex Round-2/3 sweep (same discipline).
3. **Finish Round-2** (medium-confidence probes + block_dot zombies + §C consolidations).
4. **Then re-map** — the hub/leaf structure should be unchanged; convergence target is the arkey work product at its irreducible bespoke-extractor floor (~14–16k LOC), with upstream baggage handled separately by the §0 decision.
