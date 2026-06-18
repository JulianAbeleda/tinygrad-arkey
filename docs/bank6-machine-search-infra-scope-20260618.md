# Bank 6 — machine-search infrastructure (scope) 2026-06-18

Hypothesis: tooling that auto-generates the primitive ledger / search rows / verdicts avoids repeated manual
table work and false starts. Compounding strategic value, not immediate tok/s.

## What already exists (partial)
- `extra/qk_search_spec.py` — the **search-row schema SSOT**: `assemble_search_row(**validated)`, enums
  `Phase/Model/OpScope/SearchSpace/Objective`, AMD-only backend invariant. The schema models reality (built
  against a real AcceptedPolicy shape).
- `extra/qk_demote_search.py` — a working bounded search orchestrator (dogfooded on the ffn_down demotion).
- `extra/qk_nll_eval.py` — the decode-path dNLL gate runner.
- `bench/**/*.json` — many durable result artifacts; `bench/README.md` — the benchmark index.

So the **schema + one orchestrator + the gate runner exist**. The gap is the *ledger/ingest/auto-next-target*
layer.

## Remaining scope (the build)
1. **Ledger collector:** ingest `bench/**/result.json` + the doc verdicts into one primitive ledger
   (role × variant × %peak × correct? × shipped/refuted × reason).
2. **llama/tinygrad comparison rows** auto-emitted from the ledger.
3. **Auto search-row generator:** rank candidate (role, transform) rows by gap-to-llama × tractability, skipping
   refuted rows (so we stop re-deriving the campaign map by hand).
4. **Artifact validator + gate runner** wired to the existing dNLL/W==D gates.

## Verdict: small, high-leverage, do AFTER a live target is chosen
- Not an immediate tok/s win, but it would have prevented several false starts this campaign (e.g. the unsigned-
  `_sdot4` mislabel, re-deriving the q8-pack wall thrice).
- **Recommend a *small* investment** (the ledger collector + validator, ~1-2 files reusing the existing schema),
  run *alongside* whatever live bank is funded — so the live arc's results auto-populate the ledger.
- Don't build the full auto-search before targets are clearer (it would emit generic suggestions).

## Files
`[docs]` this. Reuses `extra/qk_search_spec.py`, `extra/qk_nll_eval.py`. No code/model changes this task.
