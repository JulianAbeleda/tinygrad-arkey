# QK-CONSOLIDATE-R1: Config/Code Decoupling + Doc-Truth Reconciliation

Date: 2026-06-30

Status: solution scope produced from the principles audit at the 2026-06-30 stopping point (audit workflow
`wf_b9e3896e-e4a`; 6 dimensions × adversarial verification × synthesis). Audit-first, no-default-change, bounded; every
phase gates on a recorded verdict. **NO route default changes anywhere in this scope** — the live decode/prefill defaults
stay exactly as the route manifest declares. This is a refactor + doc-truth pass.

## Why (the principles the audit found drifting)

The pure-search/cache stack (~20 tools) works and is honestly gated, but it has accreted config-in-code, three-way
duplicated truth, and reached-verdict sprawl. Verified scorecard (confirmed/adjusted findings only):

| principle | grade | why |
|---|---|---|
| Centralize / single-source-of-truth | D | tier thresholds, refuted axes (×3 files, 3 dispositions), and quant facts each have ≥2 already-diverged sources |
| Tiny / no copy-pasted main | D | `burst()` byte-identical ×3; hand-rolled latest.json+summary.md IO in ~15 tools; two divergent factorizers |
| Design for replacement | D | `extra/qk_artifact_cache.py` is fully built and wired to **nothing** (dead code) |
| Harnesses-are-primitives | D | the verdict-SSOT test was deleted but 3 live docs still mandate it; evaluator emits 7 verdicts, none in the `Verdict` enum |
| Modularize / Orthogonalize / data-over-code / anti-sprawl | C | two same-named `LaneMapTemplate` classes; policy constants (EXPLOSION_LIMIT, thresholds) in code while the profile already carries an unread `thresholds` block; ~8 reached-verdict one-shots still wired |
| In-model gates | **B (strongest)** | routes are genuinely env-gated in model.py with real rollbacks |

## Confirmed bugs (fix in the phases below)

1. **Tier thresholds diverged + dead profile copy.** profile `qwen3_8b_q4_k_m_gfx1100.json:17-18` (5.0/2.0, noted "mirrors DEFAULT_THRESHOLDS") vs `qk_candidate_evaluator.py:38-39` (2.0/0.5); `evaluate()` never reads the profile block → annotation false, profile copy dead.
2. **Dead per-ctx regression guard.** `qk_candidate_evaluator.py:108` fires only when `med < tier_b_pct` → a +1.5% median with one ctx at −3% skips the guard and promotes. Apply the per-ctx guard unconditionally.
3. **Verdict-SSOT test deleted but still mandated.** `test/unit/test_verdict_ssot.py` gone (only `.pyc`); still cited by `HARNESS_GUIDE.md:115`, `README.md:43`, `qk_modes.py:54`. Evaluator verdicts off the `Verdict` enum.
4. **Inverted default-state comment.** `qk_prefill_graph_gemm_route.py:58` says "opt-in … default-off" but line 61 defaults ON (promoted); header :53-55 still describes global pipe as default.
5. **Ledger-scan KeyError vs safe-get divergence.** `qk_ledger_seed_pms_r4.py:140` (`[...]`) vs `qk_pure_search_next_candidate.py:106` (`.get`).
6. **Manifest attention wall-share prose contradicts the ceiling artifact.** `qk_route_manifest.py:125` says "~0%@ctx4096"; the SSOT bench json measures 0.03 (3%).

## Phase plan

- **Phase 0 — Inventory refresh (audit-only).** Regenerate FILE_INDEX; emit a drift report (3 threshold copies, 3 refuted-axis copies, dual quant sources, reached-verdict audits). Verdict `INVENTORY_REFRESHED`.
- **Phase 1 — Doc-truth (zero code-risk) — DONE in this pass.** All §"doc accuracy" edits (stale prefill role-selective numbers, G3-default caveat, default-off comment #4, attention wall-share #6, deleted-test citations #3). Verdict `DOCS_MATCH_MANIFEST`.
- **Phase 2 — Threshold SSOT + guard fix.** Profile = single threshold source, centralized by `authority_type`; per-ctx guard unconditional; false "mirrors" note deleted; restore an enum-backed `TierVerdict` + SSOT test. Acceptance: the 3 REPLAYS reproduce recorded ledger tiers; SSOT test green. Verdict `THRESHOLDS_SINGLE_SOURCE`.
- **Phase 3 — Refuted-axis generator + drift check.** Dump `do_not_search`/`known_refuted` FROM `qk_route_manifest.REFUTED`; `qk_search_space_manifest_check` asserts the 3 sets agree on (axis, route_id, disposition). Verdict `REFUTED_AXES_GENERATED`.
- **Phase 4 — Artifact emit helper + cache activation (C2/C3).** One `emit_artifact(out_dir, payload, md_lines, *, kind, inputs, code_paths)` routing through `qk_artifact_cache.write_artifact`, stamping `cache_meta`, labeling `input_artifact` vs `derived_artifact`; migrate the ~15 hand-rolled IO copies. Acceptance: inventory reports ≥1 wrapped artifact (no longer "none wrapped"); a threshold/grammar change invalidates derived artifacts. Verdict `IO_HELPER_ADOPTED`. (This is the C2/C3 the cache docstring already anticipates — it makes the dead cache live.)
- **Phase 5 — Shared primitives.** Extract `burst()`, `ledger_candidate_ids`, `coalesced_lane_factorization`, one `GROUPINGS` constant; move `EXPLOSION_LIMIT` → `topology_grammar_v1.json` (`max_candidates`); repoint TG2 `load_profile_facts()` at `quant_spec_fields("Q4_K")`; R5 audit imports `G3_LANE_OWNERSHIP_INDEX`. Acceptance: zero byte-duplicated helper; all tool outputs byte-identical (refactor-only). Verdict `PRIMITIVES_SHARED`.
- **Phase 6 — Search-space completeness.** Add a `deferred` status + a codegen profile (`v_dot2_lowering`, `cross_lane_mixed_reduce`) so the blocked-but-open north-star is a tracked-open row, NOT conflated with refuted; move `baseline_route_id`/`oracle` into manifest ROUTES; tag attention axes; rename attention ledger rows by dataflow. Verdict `SEARCH_SPACE_REPRESENTS_OPEN_FRONTIER`.
- **Phase 7 — Archive (provenance-safe) — DONE in this pass.** Move the superseded non-canonical scope docs to `docs/archive/`; repoint provenance cells + prompt reading-lists; verify no dangling reference. Verdict `ARCHIVE_NO_DANGLE`.

## Do NOT
- Delete fallbacks: `decode_q4k_owned_warp` (rollback_reference) and global-pipe (superseded_rollback) MUST remain — they are the rollback chain.
- Merge TG into PMS — they are a layered pipeline sharing one evaluator, not redundant.
- Collapse TG2's `derive_packed_word_index()` into the constant — its independence IS the anti-hardcode audit.
- Over-abstract (Rule of Three): extract a helper only with ≥3 real call sites.

## Bounding rules
One subsystem-prefix commit per phase; no non-deterministic bench timing committed; abort any phase whose acceptance
regresses a recorded ledger verdict or changes a live route default (none should). Phases 1 and 7 are pure doc/move and
ship independently of the code phases (2–6), which can be deferred.
