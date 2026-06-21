> **⚠ SUPERSEDED (2026-06-21) — historical provenance only.** Current state lives in `docs/current-project-state-handoff-20260621.md` (+ `docs/README.md`). Do NOT treat this as authority. Many scripts/paths it references were removed in the active-surface reduction (`docs/perf-probe-active-surface-reduction-result-20260621.md`, 291 perf files deleted). Kept for history.

# Codex Task Packet — Flywheel/Kernel Dead-Probe Cleanup Sweep

**Role:** Development Agent (cleanup/sweep).
**Repo:** `/home/ubuntu/tinygrad-arkey`, branch `master`.

## Citations (read these first — they are the authority for this task)

1. **The audit that defines this work:** [`structure/Development/repo-audit-2026-06-16.md`](repo-audit-2026-06-16.md) — produced by 8 parallel subsystem auditors. This packet executes its **Section A (dead probes)** and **Section B (`Ops.QK_BLOCK_DOT`)**; Section C (byte-safe consolidations) is a separate later packet; Section E is on hold.
2. **The rules you must obey:** [`structure/Development/coding-principles.md`](coding-principles.md) (esp. "Reducing Code The Right Way": *line count is not the metric, knowledge duplication is*; *duplication is cheaper than the wrong abstraction*) and [`structure/Development/tinygrad-coding-overrides.md`](tinygrad-coding-overrides.md) (commit prefixes; **`tinygrad/` core is `[codegen]`/`[runtime]`/`[nn]`, never `[test]`**; NFC discipline; anti-re-sprawl: *"one-off probes that have reached a verdict are deleted once their conclusion is recorded in the session handoff... do not leave dead probes wired into the CLI"*).
3. **The verdict record (why each probe is dead):** [`structure/Development/session-handoff.md`](session-handoff.md) — every probe below has a recorded conclusion there. Deletion is safe because git history preserves the scripts.

## Context

The repo is large mostly because **~6,800 LOC of concluded one-off probe scripts were never deleted** (audit §Headline). This is the anti-re-sprawl rule's exact target. The divergent extractors and different-dataset builders are **NOT** in scope — do not "DRY" them (audit §F, principles "duplication is cheaper than the wrong abstraction"). This task is **deletion only**, not refactoring.

## Environment

- Python: `/home/ubuntu/tinygrad-arkey/.venv/bin/python` (3.12).
- Run tests: `cd /home/ubuntu/tinygrad-arkey && .venv/bin/python -m pytest test/external/ -q 2>&1 | tail -5`.
- This box has the AMD GPU + the gguf model, so the full external suite runs for real. **Record your own baseline pass count BEFORE any change** — that is your floor; no previously-passing test may regress. (On a Mac ~6 model/AMD-gated tests fail pre-existing; not your concern if you are on the AMD box.)
- Do NOT run BEAM / risky schedule search (overrides §AMD invariants).

## Scope (files you may touch)

`extra/*.py`, `test/external/*.py`, and the four `tinygrad/` core touchpoints of `Ops.QK_BLOCK_DOT` listed in Phase 4. Nothing else.

## Out of scope (do NOT touch)

- The "Already good" list (audit §F): `llm_generate.py`, `qk_modes.py`, `qk_paths.py`, `qk_layout.py`, `qk_quantize.py`, `q4_k_safety.py`, `q4_k_gemv_primitive.py` (live core — only the dead internal variants in Phase 3), `qk_flash_decode.py`, `q4_k_bench.py`, the golden tests, `assemble_row`, the cost-model backends.
- The byte-safe consolidations (audit §C), structural `QKConfig`/BEAM-globals (§D), and the hold items (§E — adapter scaffold, dataset/verdict rewrites). Those are NOT this task.
- Committed `bench/**` artifacts — **leave them** (they are the evidence record; the artifact-tests keep guarding the verdicts after the scripts are gone).

## The deletion manifest

Verify "zero importers" yourself before each delete: `git grep -n "<module_basename>" -- extra/ test/ tinygrad/`. A module is deletable if the only hits are its own file and (optionally) a test that merely reads a committed `bench/*.json` (does NOT `import` the module).

### Phase 1 — bench long-tail probes (high-confidence, zero importers) → `[test]`
Delete these `extra/` files (handoff verdicts cited in the audit §A):
`qk_cold_perlayer.py`, `qk_decode_breakdown.py`, `qk_decode_profile.py`, `qk_decode_verify_loop.py`, `qk_decode_warmstart.py`, `qk_gemm_b1.py`, `qk_integer_vector_load_probe.py`, `qk_memory_access_audit.py`, `qk_partial_schedule_log.py`, `qk_prefetch_gemv.py`, `qk_profile_pmc.py`, `qk_quant_sensitivity.py`, `qk_speculative.py`, `qk_vdot4_builtin_d0.py`, `qk_wmma_w1.py`, `amd_vdot_smoke.py`, `weekly_commits_table.py`, `_flash_bench.py`, `_flash_verify_model.py`, `_prefill_bench.py`, `_s0_safety.py`.
Also delete the paired tests **iff present and they cover only the deleted probe**: `test_qk_wmma_w1.py`, `test_qk_gemm_b1.py`. (~2,026 LOC.)

### Phase 2 — q4_k kernel probes → `[test]`
High-confidence (zero importers or artifact-test-only): `q4_k_opt_sweep.py`, `q4_k_policy_sweep.py`, `q4_k_primitive_probe.py`, `q4_k_beam_containment.py`, `qk_batch_ceiling_probe.py`, `qk_marlin_track0.py`, `qk_marlin_w1b.py`, `qk_marlin_w2.py`, `qk_matmul_decoded.py`, `qk_loop_search.py`, `qk_loop_dataset.py`, `qk_loop_dataset_smalln.py`, `qk_loop_beam_warmstart.py`. Delete paired tests that only read artifacts: `test_qk_marlin_w1b.py`, `test_qk_marlin_w2.py`, `test_qk_matmul_decoded.py`, `test_qk_loop_search.py`. (~1,130 LOC.)
**MEDIUM-confidence — verify the test does not `import` the module before deleting:** `qk_batched_b0.py`, `qk_packed_tile_consumption_probe.py`, `qk_packed_tile_lowering_analysis.py`, `qk_packed_tile_closeout_diagnostic.py` (check `test_qk_batched_b0.py`, `test_qk_packed_tile.py`). If a test imports the module, leave both; if it only reads JSON, delete script + the module-specific test methods. (~1,030 LOC.) **KEEP:** `qk_loop_learnability.py`, `qk_loop_live.py` (live loop substrate, unit-tested) and their tests.

### Phase 3 — semantic codegen v1–v4 chain → `[test]`
**PRECONDITION (must do first):** `qk_flywheel_shadow.py` subprocess-invokes `qk_semantic_codegen_v3.py` (around `extra/qk_flywheel_shadow.py:266-279`, the `run_outcomes` packed-load path). Inspect it. If that shadow path is the concluded Phase-4-v0 path (handoff: shadow-v0 "honest negative"), remove/neutralize that invocation; if you cannot cleanly un-wire it, **keep `qk_semantic_codegen_v3.py` and its verdict** and delete only v1/v2/v4. Conservative > complete (audit §A note).
Then delete: `qk_semantic_codegen.py` (v1, delete last — v2/v3/v4 cross-import it), `qk_semantic_codegen_verdict.py`, `qk_semantic_codegen_v2.py`, `qk_semantic_codegen_v2_verdict.py`, `qk_semantic_codegen_v4.py`, `qk_semantic_codegen_v4_verdict.py`, `qk_semantic_schedule_verdict.py`, and (if un-wired) `qk_semantic_codegen_v3.py` + `qk_semantic_codegen_v3_verdict.py`. In `test/external/test_qk_ansor_transition.py` delete ONLY the codegen v1–v4 test methods; **keep** the schedule/descriptor/candidate/load-width/devectorizer tests (they exercise live infra). (~1,950 + ~230 test LOC.)
**KEEP (live/infra):** `qk_semantic_candidate.py`, `qk_semantic_descriptor.py`, `qk_semantic_schedule.py`, `qk_semantic_schedule_bench.py`, `qk_semantic_op.py` (current frontier), `qk_semantic_report.py`.

### Phase 4 — dead internal kernel variants + tinygrad dead-gated → `[nn]`/`[codegen]`
- In `extra/q4_k_gemv_primitive.py` delete the dead internal kernels once Phase 2 removed their only callers: `q4k_gemv_hoist_partial_kernel` (+ helpers `_q4k_block_dot_hoist`, `_q4k_group_dot_hoist`), `q4k_q8_1_coop_fused_kernel`, `q4k_q8_1_fused_intdot_kernel`. **Verify** `git grep` shows no remaining caller (incl. `model.py`). This is byte-safe for the live decode path. `[test]`. (~95 LOC.)
- **`Ops.QK_BLOCK_DOT` (audit §B, highest-value):** remove the op + its four core touchpoints — `tinygrad/uop/__init__.py`, `tinygrad/uop/ops.py`, `tinygrad/uop/spec.py`, `tinygrad/renderer/cstyle.py`. Verify zero live emit first (only the now-deleted probes referenced it). Commit prefix **`[codegen]`** (never `[test]` for `tinygrad/` core). 
- Dead-gated branches: `GGUF_Q4K_WIDE` (`tinygrad/llm/gguf.py:61-72`), `GQA_ATTN` (`tinygrad/llm/model.py:528-536`), `Q4K_BATCHED` (`tinygrad/llm/model.py:571`) — all concluded-negative (handoff). Remove the gated branch, keep the default path. Commit prefix **`[nn]`**. Leave `Q4K_VDOT`/`Q4K_FUSE` (parked research levers, default-off — keep).
- Leave the BEAM/warm-start module globals (`postrange.py`/`search.py`) for a separate decision (audit §D) unless you also delete `qk_loop_beam_warmstart.py`'s last references cleanly.

## Verification gate (after EVERY commit — non-negotiable)

1. `.venv/bin/python -m pytest test/external/ -q` — pass count must be `baseline − (tests you intentionally deleted)`, with **zero** new failures/errors. If any kept test errors on a missing import, you deleted something live — revert that delete.
2. `git grep -n "<deleted_module_basename>" -- extra/ test/ tinygrad/` returns nothing (no dangling import/subprocess reference).
3. `.venv/bin/python -m py_compile` on any non-deleted file you edited; `git diff --check` clean.

## Commit discipline (overrides §Commit Prefixes)

- One owning prefix per commit. Group deletions by phase/area. Suggested commits:
  `[test] drop dead bench-probe scripts (decode/memory/prefetch/...)`,
  `[test] drop dead q4_k kernel probes (marlin/matmul/loop/sweeps)`,
  `[test] drop concluded semantic-codegen v1-v4 chain`,
  `[test] remove dead internal q4_k kernel variants`,
  `[codegen] remove Ops.QK_BLOCK_DOT (rejected probe op)`,
  `[nn] remove concluded-negative decode flags (GGUF_Q4K_WIDE/GQA_ATTN/Q4K_BATCHED)`.
- Each commit body: cite `repo-audit-2026-06-16.md` §A/§B and the handoff verdict; state LOC removed and that the suite stayed green.
- **Never commit red.** Never mix a deletion with a refactor. End commit messages with the repo's `Co-Authored-By` trailer.
- **Pull-rebase before push.** Push only after the full suite is green.

## Success criteria

- ~6,000–6,800 LOC of dead concluded-probe code removed (depends on how many medium-confidence + the v3 precondition resolve).
- `Ops.QK_BLOCK_DOT` and the three concluded-negative decode flags gone from `tinygrad/` core.
- Full `test/external/` suite green at your baseline minus intentionally-deleted probe tests; zero dangling references; `bench/**` artifacts and the golden tests untouched.
- No item from audit §C/§D/§E touched (those are out of scope).

## Handoff artifact (what to report back)

Per phase: files deleted + LOC, the v3-precondition decision (un-wired vs kept-conservative) and any medium-confidence probe you left (with the importer evidence that made you keep it), baseline vs final pass count, and the commit SHAs. Note anything that turned out to be live despite the audit (so the audit can be corrected).
