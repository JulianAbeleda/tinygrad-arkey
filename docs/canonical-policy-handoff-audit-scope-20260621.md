# Canonical Policy / Handoff Audit Scope

Date: 2026-06-21

Owner: Claude 2

Status: scope only

## Context

Claude 2 resolved the project headline and prefill-default policy:

- `87.6` is a numeric coincidence:
  - real ctx≈0 decode `tok/s`;
  - separate real ctx4096 `ms/token`;
  - never quote it bare.
- steady-context decode headline remains `68.1 / 66.4 / 60.7 tok/s @ctx512/1024/4096`, about `~67%` llama.
- q8 remains default-off opt-in at about `72.8 / 70.9 / 64.3 tok/s`.
- prefill is kernel-solved and policy-shipped.
- global `PREFILL_V2` default stays **OFF**.
- `PREFILL_V2=auto` and `PREFILL_SERVER_PROFILE=1` stay opt-in because they keep about `+14GB` fp16 prefill state
  resident during decode for no decode benefit.
- decode remains the frontier.

Claude 1 owns the next decode roadmap scope:

- `docs/decode-fused-coop-primitive-roadmap-scope-20260621.md`

Claude 2 should not start decode implementation. Claude 2's job is to make the canonical state durable, searchable,
and hard to accidentally contradict.

## Objective

Produce a clean canonical-policy handoff package that:

1. audits the repo for stale or contradictory headline/policy text;
2. confirms recent commits did not accidentally bank irrelevant scratch artifacts;
3. records the current project state in one short handoff;
4. adds lightweight guardrails so future docs/bench rows do not re-open closed policy questions.

The output should make it impossible for the next session to ask again:

- is `87.6` the decode headline?
- should global `PREFILL_V2` become `auto`?
- is prefill still open?
- is bounded decode fusion still open?

## Non-Goals

Do not:

- benchmark new kernels;
- build decode fused+coop primitives;
- change `tinygrad/llm/model.py` defaults;
- re-open `PREFILL_V2=auto` as a global default;
- re-open bounded attention/FFN micro-fusion;
- modify Claude 1's decode roadmap except to link it correctly.

## Phase 1 — Commit Hygiene Audit

Claude 2 noted that `git add -A` swept in a few untracked files from the other session. Audit the last two policy
commits:

- `57dcaf45e`
- `88139b47f`

Required commands / checks:

```bash
git show --stat 57dcaf45e
git show --stat 88139b47f
git show --name-status 57dcaf45e
git show --name-status 88139b47f
```

Classify files:

| file | commit | category | keep? | reason |
|---|---|---|---|---|

Categories:

- canonical doc;
- benchmark artifact;
- research harness;
- lifecycle-search ledger;
- accidental scratch;
- generated junk.

Gate:

- if accidental scratch/generated junk was committed, propose a cleanup commit;
- if all swept files are relevant, record that explicitly in the result doc.

Do not rewrite history unless the owner explicitly asks. Prefer a forward cleanup commit if needed.

## Phase 2 — Stale Reference Sweep

Search for contradictory references to:

- bare `87.6`;
- `PREFILL_V2=auto` as an open global default call;
- "remaining owner call" for prefill default;
- prefill being unsolved;
- decode fusion as current tactical work;
- `~86 tok/s` quoted without ctx≈0;
- `~67% llama` missing context.

Suggested search:

```bash
rg -n "87\\.6|owner call|PREFILL_V2.*default|default.*PREFILL_V2|flip.*auto|~86|67% llama|bounded.*fusion|decode.*frontier" docs bench README.md
```

Required output:

| reference | file | current text | action |
|---|---|---|---|

Allowed actions:

- keep as canonical;
- update wording;
- mark historical/provenance;
- link to canonical reconciliation;
- delete only if clearly duplicate junk.

Gate:

- no current/canonical doc may present global `PREFILL_V2=auto` as still undecided;
- no current/canonical doc may quote `87.6` without context;
- any historical doc that contains stale claims must be visibly provenance/historical or superseded.

## Phase 3 — Canonical Handoff Doc

Write:

- `docs/current-project-state-handoff-20260621.md`

It should be short and high-signal.

Required sections:

1. **Canonical Numbers**
   - decode ctx≈0;
   - decode ctx512/1024/4096;
   - q8 opt-in;
   - prefill opt-in/server profile;
   - VRAM policy.
2. **Decided Policies**
   - global `PREFILL_V2` off;
   - `PREFILL_V2=auto` opt-in;
   - server profile opt-in;
   - q8 opt-in.
3. **Closed Lanes**
   - prefill kernels;
   - prefill default-owner call;
   - bounded decode fusion;
   - `87.6` ambiguity.
4. **Open Frontier**
   - Claude 1: fused+coop primitive roadmap;
   - no tactical decode patch until that roadmap returns `BRIDGE_FIRST` or `LINEARIZER_FIRST`.
5. **Where To Start**
   - `docs/README.md`;
   - `bench/README.md`;
   - `docs/decode-prefill-headline-reconciliation-result-20260621.md`;
   - `docs/decode-fused-coop-primitive-roadmap-scope-20260621.md`.

## Phase 4 — Lightweight Guardrail

Add a small checker script or doc-only checklist, whichever is least invasive.

Preferred if cheap:

- `extra/qk_policy_consistency_check.py`

It should scan canonical docs for banned ambiguous/current-state phrases and exit nonzero if a current doc says:

- bare `87.6` without `ctx`;
- `remaining owner call` near `PREFILL_V2`;
- `flip global PREFILL_V2=auto`;
- `decode headline 87`;
- bounded decode fusion presented as current implementation work.

If a script is too much for the session, add a checklist section to `docs/current-project-state-handoff-20260621.md`
with exact `rg` commands.

Gate:

- guardrail catches at least the exact stale phrases found in Phase 2;
- guardrail must not fail on historical/provenance docs unless they are in the canonical start-here set.

## Phase 5 — README / Bench Index Final Pass

Ensure these files point to the handoff and do not conflict:

- `docs/README.md`
- `bench/README.md`
- `docs/prefill-policy-integration-result-20260620.md`
- `docs/decode-prefill-headline-reconciliation-result-20260621.md`

Required canonical wording:

```text
Global PREFILL_V2 default stays OFF.
PREFILL_V2=auto is opt-in.
87.6 is contextual; never quote it bare.
Decode headline is the curve, not ctx0.
Bounded decode fusion is closed.
The only live decode lever is fused+coop in one primitive.
```

## Result Doc

Write:

- `docs/canonical-policy-handoff-audit-result-20260621.md`

Minimum sections:

1. commit hygiene verdict;
2. stale-reference sweep table;
3. files updated;
4. guardrail added or exact checklist;
5. final canonical state;
6. remaining work for Claude 1 only.

## Stop Conditions

Stop and ask before proceeding if:

- cleanup would require rewriting published history;
- a canonical doc conflicts with code behavior;
- a committed artifact is very large or clearly unrelated and cannot be safely cleaned forward;
- any benchmark number differs from the committed reconciliation without a new clean rerun.

## Expected Final State

After this scope:

- Claude 2's policy/headline lane is closed and banked.
- Claude 1 has a clean decode roadmap lane.
- `docs/README.md` and `bench/README.md` are enough to orient a new session.
- The project cannot accidentally drift back to quoting `87.6` bare or treating global `PREFILL_V2=auto` as open.
