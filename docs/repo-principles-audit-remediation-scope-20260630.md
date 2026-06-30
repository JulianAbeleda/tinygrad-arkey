# Repo Principles Audit Remediation Scope

Date: 2026-06-30

Status: scope-only. This converts the independent principles audit into an executable cleanup/hardening plan. Do not change model defaults or performance routes as part of this scope.

## Goal

Bring the repo back into alignment with the working principles:

- one source of truth for current state and policy;
- tiny, bounded changes with explicit gates;
- runtime/API errors are structured and predictable;
- search/evaluator logic is data-driven where the current abstractions support it;
- stale docs, flags, and helper comments do not mislead future work.

## Current Findings To Close

1. Runtime request validation is partial. `tinygrad/llm/cli.py:452` indexes `body["messages"]`, `msg["role"]`, and `msg["content"]` directly. `tinygrad/llm/cli.py:465-466` accepts unvalidated `max_tokens` and casts `temperature` directly. Bad provider payloads can become `internal_runtime_error` instead of `invalid_request`.
2. Current docs still misstate the prefill promotion tier. `docs/current-project-state-handoff-20260624.md:52-53` and `docs/pure-machine-search-remaining-hot-kernels-scope-20260630.md:19` call role-selective prefill `TIER_A`; the actual consolidated state is: global pipe is TIER_A vs old lds2, role-selective is a promoted TIER_B residual over global.
3. Runtime status docs have count drift after R8 closure. `docs/tinygrad-runtime-client-separation-implementation-status-20260630.md:47` still says R0 is `19/19`; `:76` still says the provider gate is an `11-check gate`, while current local code has R8 compile counters and R9 is 12/12.
4. `/runtime/load` unloads the old model before the new one has successfully loaded (`tinygrad/llm/cli.py:263`). This is either a deliberate destructive single-model policy or a two-phase load gap; make it explicit.
5. `extra/qk_policy_consistency_check.py` guards older closed questions but does not catch current-state drift around role-selective TIER_B, G3 as the Q4_K current route, or owned warp as rollback/reference.
6. `extra/qk_candidate_evaluator.py:101-108` still has a fixed `REPLAYS` table and `:135-136` rejects unknown routes. That is acceptable for replaying known decisions, but it is not yet the more agnostic evaluator shape implied by the PMS/TG direction.
7. `extra/qk_artifact_cache.py:5-6` still says the cache is not wired into generators/evaluator, stale after QK-CONSOLIDATE-R1.

## Phase A0 - Doc Truth Reconciliation

Scope:

- Update current-state docs to reflect actual tiering:
  - global `pipe_tm2_tn2` is TIER_A vs old lds2;
  - role-selective pipe is promoted TIER_B residual over global;
  - rollback chain is `PREFILL_PIPE_ROLE_SELECTIVE=0` -> global pipe, then `PREFILL_GEMM_PIPELINE=0` -> old lds2.
- Update runtime implementation status counts to match R8/R9 reality:
  - R0 count if the boundary audit now reports 20/20;
  - provider gate count as 12/12;
  - `/runtime/cache` compile counters are closed, not a caveat.
- Clarify Q4_K decode docs:
  - G3 is the current generated speed-equivalent route;
  - owned warp remains fallback/reference, not the current optimization frontier.
- Update `extra/qk_artifact_cache.py` docstring so it no longer contradicts the live cache wiring.

Acceptance:

- `rg "role-selective.*TIER_A|11-check gate|19/19|not wire into" docs extra/qk_artifact_cache.py` has no current-state false positives.
- Historical/archive mentions are allowed only if clearly marked historical.

## Phase A1 - Policy Guard Expansion

Scope:

- Extend `extra/qk_policy_consistency_check.py` to catch:
  - role-selective prefill incorrectly called TIER_A;
  - rollback chain missing `PREFILL_PIPE_ROLE_SELECTIVE=0`;
  - G3 omitted as the current Q4_K generated route;
  - owned warp described as current frontier instead of fallback/reference.
- Add/extend a lightweight unit/static test if there is an existing pattern; otherwise keep the checker self-contained.

Acceptance:

- `PYTHONPATH=. python3 extra/qk_policy_consistency_check.py` passes on corrected docs.
- A temporary local bad line containing "role-selective pipe is TIER_A" would fail the checker. Do not commit the temporary bad line.

## Phase A2 - Provider Request Validation Hardening

Scope:

- Add small validation helpers in `tinygrad/llm/cli.py` rather than scattering checks:
  - chat `messages` must be a non-empty list of objects;
  - `role` must be supported by the tokenizer or rejected as `invalid_request`;
  - `content` must be string or a list of supported text parts;
  - `max_tokens` / `max_completion_tokens` must be non-negative integers when present;
  - `temperature` must parse to float and be finite;
  - completions `prompt` must be string or list of strings.
- Preserve the existing structured runtime error envelope.
- Do not add OpenAI surface area outside the existing `/v1/chat/completions` and `/v1/completions` contracts.

Acceptance:

- Existing `extra/tinygrad_provider_compat_gate.py` still passes.
- Add negative provider checks for malformed chat body, invalid role, non-integer `max_tokens`, bad temperature, and unsupported prompt type.
- Each negative case returns a 400-style JSON error with `code=invalid_request`, not `internal_runtime_error`.

## Phase A3 - Load Lifecycle Semantics

Scope:

- Decide and encode the policy for `POST /runtime/load`:
  - Option 1: keep destructive single-model load and document it in R10/status docs;
  - Option 2: make load failure-safe where feasible by resolving/fetching before unload, while still freeing the old model before heavyweight allocation.
- If Option 2 cannot be made memory-safe for 8B, choose Option 1 and make the destructive behavior explicit in API docs.

Acceptance:

- A failed unknown-model load does not silently pretend the old model is still loaded.
- `/runtime/status` reports a clear `last_error` after a failed load.
- R10 documents whether load is destructive or best-effort two-phase.

## Phase A4 - Evaluator Adapter Decoupling

Scope:

- Keep current replay behavior, but reduce hardcoding:
  - move route -> artifact/authority wiring into the manifest or a small adapter registry keyed by `authority_type`;
  - leave route-specific artifact adapters only where artifact schemas genuinely differ;
  - keep current three reproduced decisions byte-for-byte equivalent.
- Do not start new GPU measurements or new candidate searches.

Acceptance:

- `PYTHONPATH=. python3 extra/qk_candidate_evaluator.py` still reproduces:
  - Q4_K G3 speed-equivalent promote;
  - Q6_K direct-route refuted regression;
  - prefill role-selective promoted TIER_B.
- Adding a manifest route with a known authority type should not require editing a top-level closed `REPLAYS` table unless its artifact schema is new.

## Phase A5 - Remaining Tiny Cleanup

Scope:

- Use the QK-CONSOLIDATE-R1 drift report as the source for optional helper migration.
- Migrate remaining hand-rolled result writers only when there are at least 3 similar call sites or the change removes a real inconsistency.
- Avoid cosmetic churn in bench artifacts.

Acceptance:

- No broad mechanical rewrite without a before/after inconsistency.
- `git diff --stat` stays small enough to audit manually.

## Phase A6 - Regression Gate

Run:

```bash
PYTHONPATH=. python3 extra/qk_search_space_manifest_check.py
PYTHONPATH=. python3 extra/qk_candidate_evaluator.py
PYTHONPATH=. python3 extra/qk_policy_consistency_check.py
PYTHONPATH=. .venv/bin/python -m pytest test/unit/test_verdict_ssot.py -q
```

If runtime validation changes were made, also run the live provider gate against a small model:

```bash
python -m tinygrad.llm.cli --serve 8000 -m qwen3:0.6b --max_context 1024
PYTHONPATH=. python3 extra/tinygrad_provider_compat_gate.py --base-url http://127.0.0.1:8000 --model qwen3:0.6b
```

## Non-Goals

- No decode/prefill performance route changes.
- No default flag flips.
- No GPU search.
- No new client/TUI implementation.
- No broad archive rewrite.

## Expected Verdict

`REPO_PRINCIPLES_REMEDIATION_PASS` when docs, policy guards, provider errors, and evaluator/cache comments are aligned and the regression gates pass.

