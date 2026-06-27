# Pure-search loop — strict state-machine correction (2026-06-27)

Follow-up that removes drift from the X→Y→Z loop so it is hard to fool. Spec:
`docs/pure-machine-search-xyz-loop-codex-handoff-20260627.md`.

## What was wrong → fixed

1. **Candidate authority was ambiguous** (loop could hand-pick / follow the audit's prose `next_actions`).
   → **The generator `extra/qk_pure_search_next_candidate.py` is now the SINGLE source of truth** for candidate
   selection in both `.claude/loop.md` and `.claude/commands/pure-search-loop.md`. Audit `next_actions` are advisory only.

2. **Pair-search policy was implicit** ("then pairs"). → **Pairs are in scope and ON by default** (`--no-pairs` to
   disable). `SEARCH_SPACE_EXHAUSTED` now means the declared active space (singles **and** pairs) is actually exhausted.

3. **`PROMOTABLE` could be claimed before W==D.** → Local gates (microgate + occupancy + isolated-slope) now yield
   only **`LOCAL_PASS_WD_REQUIRED`**. Only `token_match and pct_of_owned >= 90` → **`PROMOTABLE`**; a W==D miss → **`REFUTED_WD`**.
   Isolated timing is explicitly never promotion authority.

4. **Immutable space and mutable ledger were fused** (the manifest carried a `ledger` array). → **Split:**
   - declared axes/baseline only: `bench/qk-search-spaces/decode_attention_loop_search_space.json` (schema v2);
   - append-only outcomes: `bench/qk-pure-search-loop/decode_attention_loop_ledger.jsonl` (JSONL; written via the
     generator `--record`).
   - The generator reads both and reports `tried_in_space_count`, `remaining_in_space_count`, and separately
     `historical_refutations_count` (refutations recorded that are not in the current declared space).

## State machine (now executable, not prose)

`X` solves the declared space (generator-driven, W==D-gated). On `SEARCH_SPACE_EXHAUSTED` → `Y` interprets the
exhaustion from the instruments and may add a new searchable axis (reopens X). If Y is dry → `Z` improves the
auditor/searcher itself (e.g. Level-2 topology axes, W==D-in-loop) and must verify it reopens search. Stop is valid
only when **X, Y, and Z all fail explicitly** (`GENUINE_EXHAUSTION`) — or `META_CAP_REACHED` / `DEGRADED` /
`PROMOTABLE`. Outcome vocabulary:
`{FAIL_CORRECTNESS, REFUTED_OCCUPANCY, REFUTED_NO_SLOPE, LOCAL_PASS_WD_REQUIRED, REFUTED_WD, PROMOTABLE}`.

## Verified

Generator with the split space/ledger: next = `DECODE_STAGE_COALESCE=2`; counts
`tried_in_space=1, remaining_in_space=13, active_space_total=14, historical_refutations=4`; `--record` append +
re-prune round-trips. The point: human lever-picking is off the hot path, exhaustion is precise, and "promotable"
cannot be claimed before W==D authority.
