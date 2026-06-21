# Lifecycle-Search Loop v0 — Result

Date: 2026-06-21

Scope: build the first closed `generate → evaluate → prune` loop on top of the decode evaluator
(`docs/decode-evaluation-harness-hardening-result-20260621.md`).

## Final decision: **`LIFECYCLE_SEARCH_V0_READY`**

The loop runs valid candidates through `decode_eval` and **refuses invalid ones before benchmarking** — proving
both halves of a search loop. It executed 4 candidates (verdicts matching their historical classifications, all
linked artifacts valid) and pruned 2 (a closed-lane reopen + a default-promotion attempt) without running a
benchmark. It builds no kernels, changes no defaults, and only *proposes* (dedup'd) ledger updates.

## Phase 0 — design note

- **Why v0 is narrow.** v0 proves the *loop machinery* (load → prune → evaluate → classify → propose), not a new
  search space. It replays known candidates plus two intentionally-invalid ones; it does **not** auto-generate
  kernel code or sweep a large space (that is the next layer).
- **Why it consumes `decode_eval` instead of benchmarking directly.** The evaluator is the single trusted
  measurement authority (clean W==D, band <0.6%, the gate thresholds). The loop **duplicates no benchmark logic**
  — separation of concerns: the loop owns *orchestration + policy*; the evaluator owns *measurement + verdicts*.
- **What "generation" is in v0.** Loading candidate specs from `bench/qk-lifecycle-search/search_candidates.json`
  (a static registry) + two synthetic invalid candidates. Automatic candidate generation is deferred.
- **What "pruning" is in v0.** Independent closed-lane + forbidden-promotion detection from each candidate's
  intent/env/text (`search_policy.json`), applied **before** any benchmark — not merely trusting a self-declared
  `allowed_by_policy` flag (a candidate cannot lie its way past the guard).
- **Not included yet.** Auto candidate generation, new kernel code, ledger auto-mutation (the loop is
  *propose-only*), multi-GPU/model, the full vLLM-style search space.

## What was built

- `extra/qk_lifecycle_search_loop.py` — the loop (distinct from `extra/qk_lifecycle_search.py`, the read-only
  seed-ledger generator; the loop *runs* candidates, the seed *records* schemas). CLI:
  `--list` · `--dry-run --suite <s>` · `--suite <s> [--repeats N] [--out DIR]` · `--candidate <id>` ·
  `--validate <run.json>`. For EXECUTE candidates it subprocesses `decode_eval --candidate <id>`, locates +
  validates the emitted artifact, and maps the verdict to a lifecycle decision.
- `bench/qk-lifecycle-search/search_candidates.json` (registry, schema `decode_lifecycle_search_candidates_v1`),
  `search_policy.json` (closed lanes + forbidden promotions + verdict→decision map),
  `search_schema.json` (run-artifact JSON Schema).

## Verdict → lifecycle-decision map (`search_policy.json`)

`PASS_PROMOTE`→`candidate_promotable_owner_decision` · `PASS_OPT_IN`→`opt_in_candidate_banked` ·
`LOCAL_PASS_WD_FAIL`→`refute_for_promotion_bank_learning` · `FAIL_*`→`refute_candidate` ·
`FAIL_REPRODUCIBILITY`→`stop_search_needs_measurement` · `REST`→`bank_baseline_or_rest` ·
`NEEDS_GPU_STATE_TOOLING`→`stop_search_needs_gpu_state` · `NEEDS_BESPOKE_TEMPLATE`→`stop_search_needs_template`.

## v0 replay result (`bench/qk-lifecycle-search/runs/decode_v0-*.json`)

| candidate | decision | decode_eval verdict | lifecycle decision | expected | match | artifact valid |
|---|---|---|---|---|---|---|
| baseline_default | EXECUTE | REST | bank_baseline_or_rest | REST | ✓ | ✓ |
| flash_l_64 | EXECUTE | LOCAL_PASS_WD_FAIL | refute_for_promotion_bank_learning | LOCAL_PASS_WD_FAIL | ✓ | ✓ |
| q8_opt_in | EXECUTE | PASS_OPT_IN | opt_in_candidate_banked | PASS_OPT_IN | ✓ | ✓ |
| warp_flash_tile | EXECUTE | FAIL_LOCAL_AB | refute_candidate | FAIL_LOCAL_AB | ✓ | ✓ |
| wmma_decode_reopen_attempt | **PRUNE_CLOSED_LANE** | — (not benchmarked) | — | PRUNE_CLOSED_LANE | ✓ | n/a |
| promote_flash_l_64_attempt | **PRUNE_POLICY_VIOLATION** | — (not benchmarked) | — | PRUNE_POLICY_VIOLATION | ✓ | n/a |

**4 executed (verdicts match), 2 pruned (no benchmark), policy guard PASS.** The two pruned candidates prove the
loop refuses (a) a closed-lane reopen (WMMA decode) and (b) a default-promotion attempt (FLASH_L=64) — the latter
binds to a valid evaluator candidate yet is pruned on *intent* (`promote_default`), so a real candidate cannot be
default-promoted by the loop.

## What the loop surfaced (and the fix it drove)

The first v0 run **caught a real measurement fragility**: `q8_opt_in` returned `REST` instead of `PASS_OPT_IN`
because the q8 audit's **auto lane** measures baseline and q8 in *separate child processes*, so at ctx512 q8 read
0.99× purely from per-process auto-clock variance — dragging the 2-ctx median to 1.026, just under the 1.03 opt-in
threshold. The q8 effect is real (~1.06× at ctx1024) but the *auto-lane separate-process comparison is
clock-confounded at that signal level*. **Fix:** `decode_eval`'s q8 runner now reads the audit's **clock-controlled
`manual_peak` lane** (the lane designed to isolate the q8 effect; the historical artifact is PASS there) →
`q8_opt_in` → `PASS_OPT_IN` reliably. This is exactly the loop doing its job — exposing a measurement issue an
ad-hoc run would have shipped — and the fix is to the *measurement authority*, not a default or kernel.

## Phase 4 — ledger integration (propose-only, dedup'd)

The loop **proposes** ledger updates and **does not blindly append**. The dedup recognizes
*conceptually-equivalent* existing refutations: each candidate carries a `maps_to_ledger_candidate` link, and the
loop checks the existing `refutations.json` `applies_to` for that ledger-candidate id (the existing refutations are
keyed by ledger-candidate ids, not decode_eval ids). So both refuting candidates (flash_l_64, warp_flash_tile) map
to `decode_vector_flash_tile_high_kvsplit` and are detected as **`already_present_skip`** (covered by an existing
refutation — e.g. `gqa_coop_vec_more_splits_passes_local_misses_wd` /
`vector_tile_bounded_levers_miss_needs_full_llama_engineering`); q8 → `bank_opt_in` already banked (default-off);
baseline → no ledger change. **Net: zero new rows proposed, zero defaults changed.** Applying ledger writes is
deferred to a future loop version; v0 surfaces the (dedup'd) proposals in the run artifact for review.

## Acceptance gates

| gate | result |
|---|---|
| G1 `--list` | PASS |
| G2 `--dry-run --suite decode_v0` shows execute/prune | PASS (4 would-execute, 2 pruned) |
| G3 `--suite decode_v0` runs accepted through decode_eval | PASS |
| G4 `--validate` validates search + linked evaluator artifacts | PASS |
| G5 ≥4 executed + ≥2 pruned | PASS (4 + 2) |
| G6 expected classifications match | PASS (all 6) |
| G7 no closed lane benchmarked | PASS (pruned before eval) |
| G8 no model/default/kernel change | PASS (`git diff tinygrad/` empty) |
| G9 policy consistency check passes | PASS |
| G10 tree clean after commit | PASS (commit below) |

Final run artifact: `bench/qk-lifecycle-search/runs/decode_v0-20260621T020358.json` (validates, + all 4 linked
decode_eval artifacts validate).

## Next unlocked project

**Candidate-template generation layer** — turn route/fusion/layout templates into *auto-generated* `decode_eval`
candidates (closing the "generate" half with real generation, not a static registry). After that, north-star
`flash_attn_tile` templates (many-KV-split / stream-k combine) become candidates the loop can evaluate.

## Falsifiers checked (none tripped)

- `decode_eval` is reliably callable as a subprocess (4 runs, artifacts located + validated) → not
  `NEEDS_EVALUATOR_API_CLEANUP`.
- the existing candidates are expressible as specs without bespoke code → not `NEEDS_CANDIDATE_TEMPLATE_LAYER`
  (for v0; templating is the *next* feature, not a blocker).
- policy pruning produced no false positives (the 4 valid candidates executed; only the 2 intended were pruned)
  → not `NEEDS_POLICY_LANGUAGE_CLEANUP`.
- ledger proposals dedup'd cleanly (no duplicate-heavy output) → not `NEEDS_LEDGER_NORMALIZATION`.

## Boundary

No `tinygrad/llm/model.py` or `tinygrad/` change; no default/flag promotion; no kernels; closed lanes pruned, never
benchmarked. The loop is propose-only on the ledger. Commands:

```bash
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_lifecycle_search_loop.py --list
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_lifecycle_search_loop.py --dry-run --suite decode_v0
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_lifecycle_search_loop.py --suite decode_v0 --repeats 2 --out bench/qk-lifecycle-search/runs/
DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_lifecycle_search_loop.py --validate bench/qk-lifecycle-search/runs/<run>.json
```
