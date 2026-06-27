# Project `/loop` task — pure-machine-search X→Y→Z state machine (ONE step per fire)

Default task for a bare `/loop` in this repo. A **strict state machine**, not a prose cycle. Each fire advances one
step and persists `phase`, so the escalation survives across fires. Prefer dynamic `/loop` (no interval) so you
self-end. Full spec: `docs/pure-machine-search-xyz-loop-codex-handoff-20260627.md`.

**Hard rules (do not drift):**
- **Candidate authority = the generator ONLY.** `extra/qk_pure_search_next_candidate.py` (pairs ON by default).
  Never hand-pick a lever; the audit's `next_actions` are **advisory only**.
- **`PROMOTABLE` is reserved for W==D.** Microgate + occupancy + isolated-slope passing is only
  `LOCAL_PASS_WD_REQUIRED`. Only `wd.token_match and wd.pct_of_owned >= 90` yields `PROMOTABLE`; a W==D miss is `REFUTED_WD`.
- **Ledger writes go through the generator** `--record '<json>'` (append-only JSONL). Outcomes ∈
  {FAIL_CORRECTNESS, REFUTED_OCCUPANCY, REFUTED_NO_SLOPE, LOCAL_PASS_WD_REQUIRED, REFUTED_WD, PROMOTABLE}.

## STATE  `bench/qk-pure-search-loop/state.json`
`{"meta":0,"max_meta":3,"phase":"X","status":"running","stop_reason":null}`  (phase ∈ X|Y|Z; absolute meta ceiling 6)

## STEP 0 — escape hatch (check FIRST, every fire)
- file missing → create as above. `status!="running"` → print last summary, **stop, no reschedule** (CronDelete if cron).
- `run_gap_audit().degraded` → set stopped `stop_reason=DEGRADED`, stop (fix the harness first; never loop on bad data).
- `meta >= min(max_meta, 6)` → stopped `stop_reason=META_CAP_REACHED`, stop.

## STEP 1 — act on `phase`

### phase == "X"  (solve the declared space)
1. `audit = run extra/qk_pure_search_gap_audit.py` (DIAGNOSE; **advisory** — do not pick from it).
2. `cand = run extra/qk_pure_search_next_candidate.py` (AUTHORITY, pairs on).
   - `cand.verdict == "SEARCH_SPACE_EXHAUSTED"` → set `phase="Y"`, save, end fire.
3. Else implement `cand.env_flags` as a **default-off, cache-keyed** change, then GATE (timeout every GPU call):
   - microgate fail → `--record {outcome:"FAIL_CORRECTNESS"}`, **revert clean**, end fire.
   - occupancy guardrail fail → `--record {outcome:"REFUTED_OCCUPANCY"}`, **revert clean**, end fire.
   - isolated slope does NOT bend → `--record {outcome:"REFUTED_NO_SLOPE"}`, commit-or-revert, end fire.
   - isolated slope bends → `--record {outcome:"LOCAL_PASS_WD_REQUIRED"}`, then run **W==D**
     (`QK_CKPTS=512,4096 extra/qk_decode_runtime_overhead.py` + `extra/qk_decode_token_match_check.py`):
     - `token_match and pct_of_owned >= 90` → `--record {outcome:"PROMOTABLE"}`, set stopped `stop_reason=PROMOTABLE`, stop (hand to promotion review; do NOT auto-default).
     - else → `--record {outcome:"REFUTED_WD"}`, commit-or-revert, end fire.

### phase == "Y"  (interpret exhaustion — ask the instruments what layer blocks)
Run `qk_decode_hotloop_schedule_diff.py`, `qk_decode_occupancy_guardrail.py`, `qk_split_kv_economics_audit.py`,
`qk_pure_machine_search_gap_audit.py`. Synthesize the wall + next layer. If an answer implies a **new searchable
lever not in the ledger and addable as an axis** → add the axis to
`bench/qk-search-spaces/decode_attention_loop_search_space.json`, set `phase="X"`, save, end fire. Else set
`phase="Z"`, save, end fire.

### phase == "Z"  (improve the auditor/searcher itself)
Pick the highest-value tool capability gap (Level-2 topology axes; W==D-in-loop; harness-measured snapshot;
BubbleBeam flag-ownership; fused combine — see the handoff doc §5/§8). If none → set stopped
`stop_reason=GENUINE_EXHAUSTION`, stop. Else implement the gap (default-off/new axes/new gate), then **verify it
reopens search** (the generator emits a new candidate OR Y now has an answer): if yes → `meta += 1`, set `phase="X"`,
save, end fire; if no → set stopped `stop_reason=TOOL_IMPROVEMENT_DID_NOT_REOPEN_SEARCH`, stop.

## Every fire
Commit (`[codegen]`/`[nn]`/`[test]`/`[docs]`, no Co-Authored-By) or revert clean — **never leave the tree dirty**.
Surface SHA + title. **Do not push.** On stop, print: phase reached, ledger counts
(`tried_in_space/remaining_in_space/historical_refutations` from the generator), audit score+verdict, commits, and
the single recommended next step.

*Built-in safety also applies: `Esc` cancels a pending fire; recurring tasks expire after 7 days; session-scoped.
For an immediate bounded run in ONE turn, use `/pure-search-loop [max_meta]`.*
