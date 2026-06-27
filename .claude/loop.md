# Project `/loop` task — owned-oracle PARITY CLOSURE (ONE step per fire)

Default task for a bare `/loop`. The owned hand-coded AMDGCN tile (`extra/qk_owned_flash_decode.hip`,
`DECODE_ATTN_AMDGCN_TILE=1`) is the **ORACLE**. The loop closes a 6+1-layer **parity matrix** against it — it is NOT
open-ended search and NOT primitive discovery (the primitives are known from the owned ASM). Spec:
`docs/pure-search-loop-owned-oracle-reconstruction-20260627.md`; matrix tool:
`extra/qk_owned_oracle_parity_audit.py` → `bench/qk-owned-oracle-parity/latest.json`.

**Hard rules (do not drift):**
- **Candidate generation is DRIVEN BY failed parity rows.** No candidate may run unless it targets a FAILED row
  (`status` MISMATCH/MISSING with a `candidate_axis`). Enforced mechanically:
  `qk_pure_search_next_candidate.py --failed-rows <rows>`.
- **An UNKNOWN row → improve that instrument FIRST; do not search it.**
- **A candidate that does not MOVE its target row → `SEARCH_SPACE_BUG`** (or `TOOLING_BUG` if the move is
  unobservable). Deep looping without row movement is a bug.
- **`PROMOTABLE` requires W==D + token-match** (`extra/qk_decode_runtime_overhead.py` +
  `extra/qk_decode_token_match_check.py`); isolated timing is diagnostic only.
- Generator-only candidate authority; append-only JSONL ledger; **do not push**.

## STATE  `bench/qk-pure-search-loop/state.json`  `{"meta":0,"max_meta":3,"status":"running"}` (ceiling 6)

## STEP 0 — escape hatch
file missing → create. `status!="running"` → print summary, stop. `meta>=min(max_meta,6)` → `META_CAP_REACHED`.
`run_gap_audit().degraded` → `DEGRADED` (fix harness first).

## STEP 1 — one parity-closure step
1. **parity = run `extra/qk_owned_oracle_parity_audit.py`** (refresh its inputs first if stale: the isa_diff /
   isa_vectorization / hotloop / occupancy / economics / transfer artifacts for the CURRENT generated candidate).
2. If `parity.verdict` is `PARITY_CLOSED...` AND W==D+token clears → stop `PROMOTABLE`.
3. **If any `unknown_rows`** (e.g. `load_vectorization`, `reduce_placement`): this is `INSTRUMENTATION_GAP`. Improve
   the row's `responsible_tool` until it emits a comparable owned-vs-generated datum (e.g. capture block-tile
   load-WIDTH markers; normalize cross-lane per logical reduction; capture owned VGPR). **Do not search.** Commit the
   instrument change. End fire.
4. **Else** drive candidate gen off the failed rows:
   `cand = qk_pure_search_next_candidate.py --failed-rows <parity.searchable_failed_rows>`.
   - `NO_UNTRIED_CANDIDATE_TARGETS_A_FAILED_ROW` → `SEARCH_SPACE_BUG`: the failed row has **no searchable axis** (e.g.
     `vgpr` 88 vs 64 needs a *work-removal* capability, not a knob) or its axes are exhausted. Classify + add the
     missing capability/axis (a `MISSING_PRIMITIVE`/topology/work-removal axis) OR escalate to an instrumentation
     gap. **Do not loosen the parity gate.** End fire.
   - Else implement `cand.env_flags` (default-off, cache-keyed), GATE: microgate fail → `FAIL_CORRECTNESS` revert;
     occupancy fail → `REFUTED_OCCUPANCY` revert. Then **re-run the target row's `responsible_tool` and VERIFY the
     row's counter moved toward owned.** If it did NOT move (here or across prior candidates for this row) →
     `--record` `SEARCH_SPACE_BUG` for that row, stop searching it, fix the generator/metric. If `requires_wd` → skip
     isolated, run W==D; else isolated-then-W==D. `token_match and pct_of_owned>=90` → `PROMOTABLE` stop; else
     `REFUTED_WD`/`REFUTED_NO_SLOPE`. `--record` the outcome. End fire.

## STEP 2 — commit/revert + escape
`meta += 1` on a capability/instrument improvement that reopens parity progress. Commit
(`[codegen]`/`[nn]`/`[test]`/`[docs]`, no Co-Authored-By) or revert clean — never leave the tree dirty. Surface SHA.
**Do not push.** On stop, print the parity summary (MATCH/MISMATCH/UNKNOWN), failed+unknown rows, ledger counts, the
audit score, commits, and the single recommended next step.

*Built-in: `Esc` cancels a pending fire; recurring tasks expire after 7 days. One-turn bounded run: `/pure-search-loop [max_meta]`.*
