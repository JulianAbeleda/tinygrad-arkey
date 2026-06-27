# Project `/loop` task — pure-machine-search decode loop (ONE iteration per fire)

This is the default task a **bare `/loop`** (or `/loop <interval>`) runs in this repo. Each fire executes
**exactly one** diagnose→solve→gate→record→decide iteration and updates a persistent state file, so the loop is
**self-bounding across fires**. Prefer **dynamic `/loop`** (no interval) so you can self-end by not scheduling the
next wakeup. Full protocol + exact gate commands: `.claude/commands/pure-search-loop.md`.

## STATE FILE = the escape hatch (read/write every fire)
`bench/qk-pure-search-loop/state.json`, shape:
`{"iteration":0,"max_iterations":3,"status":"running","stop_reason":null,"ledger":[]}`

## STEP 0 — check the escape hatch FIRST, before any work
1. If the file is **missing** → create it with the shape above (`max_iterations`=`$1` if you passed one, else 3).
2. If `status == "stopped"` → **the loop is already complete. STOP NOW**: print the last summary, do **not** run an
   iteration, do **not** reschedule, and if running under a fixed interval, `CronDelete` this task. (To start a new
   campaign, the user deletes `bench/qk-pure-search-loop/state.json` first.)
3. If `iteration >= min(max_iterations, 6)` (absolute ceiling 6) → set `status="stopped"`,
   `stop_reason="iteration_cap"`, write the final summary, **stop + do not reschedule + CronDelete if cron**.
4. If the last `ledger` outcome ∈ {`PROMOTABLE`,`HARD_WALL`,`DEGRADED`,`NO_NEW_LEVER`} → same: stop, summary, no
   reschedule, CronDelete if cron.

Only if **none** of the above fired do you proceed to run one iteration.

## STEP 1 — run ONE iteration (the `.claude/commands/pure-search-loop.md` protocol, single pass)
- **DIAGNOSE**: `qk_pure_search_gap_audit.py` (+ `qk_decode_hotloop_schedule_diff.py`, `qk_decode_occupancy_guardrail.py`).
- **GENERATE LEVER** (do NOT hand-pick — this is the machine-search "generate + prune" step):
  ```
  PYTHONPATH=. .venv/bin/python extra/qk_pure_search_next_candidate.py
  ```
  If it prints `SEARCH_SPACE_EXHAUSTED` → outcome `NO_NEW_LEVER`, stop. Otherwise take its `candidate` + `env_flags`
  (the next untried one-factor delta over the declared space, with refuted candidates already pruned). The audit's
  prose `next_actions` are only a hint; the **generator over the manifest is the source of truth** so the loop
  covers the space instead of repeating one static suggestion.
- **SOLVE**: default-off + cache-keyed change.
- **GATE** (authority order, timeout every GPU call ≈540s, a hang = failed gate):
  microgate correctness (`BLOCK_TILE_MICROGATE_PASS`, max_abs ≤ 5e-3) → occupancy guardrail (no regress) →
  hotloop/isolated-timing (did it move the right loop / bend the slope, diagnostic only).
- **RECORD (remember)**: **append `{candidate, outcome, gate, why}` to the manifest ledger**
  (`bench/qk-search-spaces/decode_attention_loop_search_space.json` → `ledger`; `git add -f` it — `bench/**` is
  ignored) so the generator prunes it next time. Update any relevant contract artifact, re-run the audit, **commit**
  (`[codegen]`/`[nn]`/`[test]`/`[docs]`, no Co-Authored-By) **or revert clean**. Surface SHA + title. Never leave the
  tree dirty. (A win — bends slope + gates pass — is a PROMOTABLE outcome, not just a ledger row.)
- **CLASSIFY OUTCOME**: `PROMOTABLE` | `REFUTED` | `HARD_WALL` | `DEGRADED` | `NO_NEW_LEVER`.

## STEP 2 — update state + decide whether to continue
- `iteration += 1`; append `{lever, outcome, sha}` to `ledger`; save.
- If outcome is terminal (`PROMOTABLE`/`HARD_WALL`/`DEGRADED`/`NO_NEW_LEVER`) → set `status="stopped"` +
  `stop_reason`, print summary, **do not reschedule** (CronDelete if cron).
- Else (outcome `REFUTED`, under cap) → schedule/allow the next fire to take the **next-layer** lever.

## On STOP — always print
iterations run, levers + outcomes (the ledger), current audit score + verdict, commits (SHA + title), and the
single recommended next lever. A refutation is a result. **Do not push** unless explicitly asked.

---
*Built-in safety also applies: press `Esc` to cancel a pending fire; recurring tasks auto-expire after 7 days;
session-scoped (stops when the terminal closes). For an immediate bounded run in ONE turn (no scheduling), use
`/pure-search-loop [max_iterations]`.*
