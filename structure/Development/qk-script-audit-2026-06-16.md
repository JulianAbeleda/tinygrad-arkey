# QK Flywheel Script Ownership Audit — 2026-06-16

Read-only audit of the six core QK flywheel scripts against
[coding-principles.md](coding-principles.md) + [tinygrad-coding-overrides.md](tinygrad-coding-overrides.md).
No code changed. Context: the model-to-kernel flywheel itself was **falsified**
(`docs/amd-decode-flywheel-postmortem.md` — "the original framing was a dead end");
the deterministic end-to-end generated-policy wins stand (`docs/amd-decode-current-verdicts.md`).
LOC/importers/tests measured via `git grep`/`wc`.

## Summary table

| file | LOC | purpose | status | recommendation |
|---|--:|---|---|---|
| `qk_flywheel_dataset.py` | 558 | triage-row schema authority: `LABELS`/`REASONS`, `assemble_row`, `parse_opts`/`parse_load_width_words` (SSoT added C3) | **active** | **keep** — schema authority |
| `qk_flywheel_cost_model.py` | 525 | learned feature→policy cost model / triage scoring | **active** | **keep** — feature-policy authority |
| `qk_policy_pipeline.py` | 674 | generated-policy orchestration (manifest-checked pipeline, CLI) | **active** | **keep** — orchestration authority |
| `qk_ansor.py` | 620 | Ansor-transition descriptor→candidate→static-gate loop | **active** | **keep** |
| `qk_flywheel_targeted_outcomes.py` | 854 | replay committed bench artifacts → labeled targeted-family rows + README render | **active** | **keep**; row-builder consolidation only if byte-safe |
| `qk_flywheel_shadow.py` | 968 | Phase-4 v0 live shadow: freeze / run-outcomes / score, staged batches, CLI | **concluded (test-pinned)** | **keep as frozen golden harness; do NOT refactor** |

## Per-file evidence

### `qk_flywheel_dataset.py` — keep (schema authority)
- **Importers (10):** `cost_model`, `dataset_v1`, `feature_audit`, `feature_enrich`, `shadow`,
  `targeted_outcomes`, `triage_eval`, `triage_sft` + tests `test_qk_flywheel_dataset`,
  `test_qk_flywheel_triage_eval`. The most-depended-on QK module.
- **Why keep:** single source for the row schema + the opt-string/load-width parse rule
  (C3 routed three copies here). Exemplary "centralize what defines the system." Leaf
  module (no `extra/` back-imports) → no cycle risk.

### `qk_flywheel_cost_model.py` — keep (feature-policy authority)
- **Importers (7):** `feature_audit`, `shadow` + tests incl. `test_flywheel_dataset_golden`
  (pins `test_cost_model_centroid_output_is_pinned`), `test_qk_flywheel_cost_model`,
  `_phase3d`, `_phase3e`, `_phase4`.
- **Why keep:** owns the dual cost-model backends + `_majority` (C10 routed the dup to
  `triage_eval`). Byte-pinned by the centroid golden — high-value, well-tested.

### `qk_policy_pipeline.py` — keep (orchestration authority)
- **Importers (1):** `test_qk_policy_pipeline`; also a CLI entry point. Imports `qk_ansor`,
  `qk_decode_summary`, `qk_layout`, `qk_experiment_matrix` (LLAMA_REFS via C2).
- **Why keep:** the manifest-checked generated-policy pipeline orchestrator. A large
  (674) but cohesive orchestrator — defensible deep module. Watch for internal sprawl;
  no split warranted now.

### `qk_ansor.py` — keep
- **Importers (2):** `qk_policy_pipeline` (live) + `test_qk_ansor`.
- **Why keep:** the descriptor/candidate generation + static-gate loop is still imported by
  the live pipeline and tested. Active surface, not a dead probe.

### `qk_flywheel_targeted_outcomes.py` — keep; consolidate only if byte-safe
- **Importers (3):** `qk_flywheel_shadow` + tests `test_flywheel_dataset_golden`,
  `test_qk_flywheel_phase3f`.
- **Why keep:** replays committed bench artifacts (e.g. `qk-block-dot-microbench`) into
  labeled rows; golden-tested. C9/C10 already removed the no-op ternary + dead branch here.
- **Refactor note:** the remaining `_*_rows` builders are **divergent** (different family
  schemas) — audit §F lists divergent row-builders as do-NOT-merge. Only consolidate a
  pair if it is byte-proven identical; otherwise leave (duplication < wrong abstraction).

### `qk_flywheel_shadow.py` — concluded, test-pinned; keep frozen, do not refactor
- **Importers (1):** `test_qk_flywheel_phase4` only (11 tests; pins the deterministic
  `freeze.json` + scoring). Imports `cost_model`, `targeted_outcomes`, `dataset`.
- **Status:** the Phase-4 model-to-kernel shadow is **concluded** (flywheel falsified). Round-2
  Phase B already un-wired its replay paths: `run_outcomes` raises past the packed-load gate,
  and the `codegen_v3`/`qk_block_dot` subprocess replays were removed. What remains live is the
  **freeze/score golden-reproduction** path the phase-4 test pins.
- **Recommendation:** **keep as-is.** It is the largest QK script (968) and mixes CLI +
  freeze + outcomes + scoring + the `STAGED_BATCHES` table, so it *looks* like a split
  candidate — but it is concluded code with its run paths already disabled. NFC-splitting a
  frozen module is churn the anti-re-sprawl rule does not ask for, and risks the phase-4
  freeze-hash golden. **Skip the optional shadow split** (recommended-commit-6) unless the
  module returns to active development. A future option, only with a byte-proof that
  `test_qk_flywheel_phase4` still passes, is to archive the dead live-run scaffolding (not
  split it into more files).

## Kernel-research assets (recommended-commit-7 rows)

These are **upstream tinygrad `extra/` assets**, not project-authored QK scripts, and both are
test-covered — out of scope for project cleanup, **keep**:

| asset | size | status | evidence |
|---|--:|---|---|
| `extra/thunder/amd/fa_bwd_causal.cpp` | 206 KB | keep (upstream, test-covered) | `test/testextra/test_hk_fa.py`, `test/testextra/test_tk.py` |
| `extra/gemm/cdna_asm_gemm.py` | 160 KB | keep (upstream, test-covered) | `test/backend/test_asm_gemm.py`, `extra/llama_kernels/cast_amax` |

No README/verdict additions needed: they belong to upstream tinygrad subsystems with their
own test coverage, not the project's QK campaign.

## Verdict

Five of six QK scripts are **active authorities** (schema, cost model, orchestration, ansor
loop, targeted outcomes) and correctly shaped — keep all. `qk_flywheel_shadow.py` is
**concluded but test-pinned**; keep it frozen and do not refactor. No archive/delete
recommended (nothing is a dead, un-tested probe). The standing principle work is the
**functional QK invariants** (env-flag guards, see the QKConfig follow-up), not further
flywheel-script restructuring.
