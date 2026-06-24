# Perf-Probe Active-Surface Reduction — Dry-Run Plan

Date: 2026-06-21

Evidence-driven mass-delete plan (Phase 3). **No deletion happens until this plan exists.** Goal: remove stale
one-off probe scripts from the active `extra/` surface so future agents don't treat them as authority — **without**
breaking the live evaluator/search system or destroying evidence whose conclusion isn't recorded.

Authority for "live": `docs/current-project-state-handoff-20260621.md` (REST_DECODE + v2), `extra/qk_decode_eval.py`,
`qk_lifecycle_search_loop.py`, `qk_candidate_template_gen.py`, `qk_policy_consistency_check.py`, the lifecycle ledgers.

## Method — reference graph (Phase 0)

`bench/qk-active-surface-reduction/build_inventory.py` classifies all **378** perf scripts by a real reference graph:
- **import-closure** from the live roots (evaluator/search/runner + ledger-named `ab_script` scripts) → LIVE;
- **canonical-doc** refs (handoff, READMEs, principles, north-star, headline) + **live-ledger** refs → PROVENANCE;
- **dated-doc-only** refs → MANUAL_REVIEW (conclusion captured in a dated result doc; not live/canonical);
- **zero refs anywhere** → DELETE candidate.

Four safety layers ensure no kept file is left with a broken dependency (all verified empty for the delete set):
1. **import-safety fixpoint** — a script is DELETE only if every importer is also DELETE;
2. **path-string edges** — subprocess invocations (`_run([..., "extra/qk_X.py"])`) count as dependencies;
3. **external-importer protection** — anything imported/invoked by `test/`, `model.py`, or other non-inventory code is kept;
4. **repo-wide final grep** — no remaining `from extra.X import` of any delete-set script from kept code; no `test/` reference.

Artifacts: `bench/qk-active-surface-reduction/inventory.json` (per-file path/type/imports/imported_by/doc-refs/
ledger-refs/status/reason/category) + `inventory.md`.

## Summary counts

| status | count | disposition |
|---|---:|---|
| **live** | 18 | KEEP active (evaluator/search/runner closure) |
| **provenance** | 55 | KEEP active (cited by canonical docs/ledgers, or a shared utility / test-imported) |
| **manual_review** | 279 | KEEP this pass (cited by dated result docs; conclusion captured) — documented second wave |
| **delete** | **26** | **DELETE this pass** (zero refs anywhere, import/test/subprocess-safe) |
| total | 378 | |

## Keep set (Phase 1) — proven by the reference graph

**Live (18):** the import-closure of the evaluator/search roots — `qk_decode_eval.py`, `qk_lifecycle_search_loop.py`,
`qk_candidate_template_gen.py`, `qk_policy_consistency_check.py`, `qk_decode_runtime_overhead.py`, `qk_flash_decode.py`,
`qk_clock_pin.py`, `qk_harness_contract.py`, `qk_nll_eval.py`, and every `ab_script` runner named in
`candidates.json`/`binding_templates.json` (`fused_flash_concrete_gate`, `matmul_pv_diagnostic`, `fused_softmax_v_tail`,
`north_star_flash_attn_tile`, `llama_flash_attn_tile_oracle`, `decode_warp_flash_tile`, `decode_fused_flash_tile`).

**Provenance (55):** scripts cited by a canonical doc/ledger, plus 4 **shared utilities / test-imported** scripts the
fixpoint rescued from delete (`qk_paths.py` [`portable_path`, imported by 4], `qk_semantic_report.py` +
`qk_policy_parity.py` [subprocess-invoked by `qk_policy_pipeline.py`], `qk_packed_tile_{closeout_diagnostic,
lowering_analysis}.py` + `qk_loop_{benchmark,verdict}.py` [imported by `test/external/`]).

## Delete set (Phase 4 — low-risk, this pass): 26 scripts

All 26: **zero references in any doc/ledger, not imported by any kept script, not subprocess-invoked, not referenced by
any test.** Categories: superseded_tensile 9, stale_scope_helper 8, no_canonical_reference 5, superseded_decode 2,
superseded_mmvq 1, superseded_prefill 1.

```
extra/qk_amd_bb5a10_p7e_p8_handoff.py
extra/qk_amd_bb5a10_p8_global_direct_candidate_decision.py
extra/qk_amd_bb5a10_p8_tta2_authority_sample_correctness.py
extra/qk_amd_bb5a10_p8_tta3_macro_candidate.py
extra/qk_amd_bb5a10_p8_tta_completion_scope.py
extra/qk_amd_bb5a10_p8_tta_scope.py
extra/qk_amd_gemm_shape_factcheck.py
extra/qk_amd_gemm_shape_tile_sweep.py
extra/qk_amd_gemm_tensile_vs_ours_probe.py
extra/qk_attention_sdpa_vs_flash.py
extra/qk_decode_dnr4_t3_issue_latency_scope.py
extra/qk_decode_dual_track_next_scope.py
extra/qk_decode_owned_q8_first_build_scope.py
extra/qk_decode_owned_q8_producer_hip_delta_scope.py
extra/qk_decode_owned_q8_producer_target_reconcile.py
extra/qk_ffn_contig_probe.py
extra/qk_flash_threshold_validate.py
extra/qk_gemv_role_efficiency.py
extra/qk_inmodel_integration_penalty_audit_scope.py
extra/qk_prefill_graph_route_transfer_scope.py
extra/qk_prefill_matmul_penalty_verify.py
extra/qk_sub4_byte_census.py
extra/qk_sub4_nll_eval.py
extra/qk_sub4_quant_probe.py
extra/qk_tensile_block_jit.py
extra/qk_tensile_jit_dim.py
```
> Authoritative source: `bench/qk-active-surface-reduction/inventory.json` (`status == "delete"`). The 4 test-imported
> scripts (`qk_loop_benchmark`, `qk_loop_verdict`, `qk_packed_tile_closeout_diagnostic`, `qk_packed_tile_lowering_analysis`)
> were rescued by the external-importer scan and are `provenance` (kept).

## Manual-review / second wave (NOT deleted this pass): 279

Cited only by dated result docs (their conclusion is recorded there): superseded_decode 64, superseded_tensile 53,
superseded_mmvq 48, superseded_prefill 38, no_canonical 41, stale_scope 21, generated_scratch 14. These are
**provenance whose conclusion lives in the cited dated doc**. Per the project gates ("do not delete manual-review
files; execute only low-risk deletes; if uncertain, archive rather than delete") they are **kept this pass** and
recommended for an **owner-gated second wave** (archive to `archive/perf-probes-20260621/` or delete — one command,
since none are imported by live code). Full per-file list + reasons in `inventory.json`.

## Risk table

| risk | likelihood | mitigation |
|---|---|---|
| delete a script imported by live code | none | import-closure + fixpoint; verified no kept code imports the 26 |
| delete a script a test imports | none | external-importer scan of `test/` rescued 4; final grep `test/`→NONE |
| delete a shared utility | none | fixpoint rescued `qk_paths` etc. (imported by ≥1 kept) |
| dangling ref in a **dated** provenance doc/artifact | low/acceptable | conclusion preserved in the doc; `inventory.json` records the file; git history preserves the script |
| dangling ref in a **canonical** doc | none | canonical-doc refs → PROVENANCE (never deleted) |
| lose an undocumented conclusion | low | the 26 are scratch/scope/superseded probes with no unique recorded result; git history recoverable |

## Validation commands (Phase 6)

```
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_policy_consistency_check.py
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_decode_eval.py --list
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_lifecycle_search_loop.py --list      # or --help
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_candidate_template_gen.py --list-templates  # or --help
PYTHONPATH=. .venv/bin/python -c "import extra.qk_decode_eval, extra.qk_lifecycle_search_loop, extra.qk_candidate_template_gen"
rg -l "<deleted stem>" docs/ bench/qk-decode-eval bench/qk-lifecycle-search   # expect: no CANONICAL/ledger hits
git diff --stat ; git status
```

## Rollback plan

History is **not** rewritten. Any deleted script is recoverable with
`git checkout <pre-deletion-commit>~1 -- extra/qk_<name>.py` (the pre-deletion commit is recorded in the result doc).
`inventory.json` preserves every deleted file's path, imports, doc-refs, and reason, so a deleted conclusion can be
re-located even without checking the file out.

## Decision
Execute the **26 low-risk deletes** (Phase 4). Keep live (18) + provenance (55) + manual_review (279). The 279
provenance probes are the documented owner-gated second wave (archive or delete). This plan must exist before any
`git rm`.
