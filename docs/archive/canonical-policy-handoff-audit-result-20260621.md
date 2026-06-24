# Canonical Policy / Handoff Audit — RESULT (Claude 2)

Date: 2026-06-21. Scope: `docs/canonical-policy-handoff-audit-scope-20260621.md`. Lane: policy/headline durability
(no decode implementation; no `model.py` default changes).

## 1. Commit hygiene verdict

| commit | files | verdict |
|---|---|---|
| `88139b47f` | 3 docs (README, reconciliation-result, prefill-policy-integration-result) | **clean** — all mine, all canonical docs |
| `57dcaf45e` | see below | **swept 3 of Claude-1's decode files** + 1 transient bench artifact — **NOT junk; no cleanup** |

`57dcaf45e` name-status: `bench/README.md` (M, mine), `docs/README.md` (M, mine),
`docs/decode-prefill-headline-reconciliation-{result,scope}.md` (A, mine), `bench/qk-decode-runtime-overhead/result.json`
(M, benchmark artifact — regeneratable, holds my last clean rerun), and three **Claude-1 decode-lane files** swept
by a `git add -A`:
`docs/decode-latency-hiding-lifecycle-codegen-{result,scope}.md`, `extra/qk_decode_fused_flash_tile_ab.py`.

**Verdict: no accidental scratch / generated junk.** The three swept files are legitimate Claude-1 decode-roadmap
work (Claude 1 is actively editing them in the working tree). They were banked under a `[docs]` reconciliation
message — a provenance mislabel, not junk. **No forward cleanup and no history rewrite** (per scope; deleting them
would destroy Claude-1's committed work). Recorded here. **Lesson applied this session: stage files explicitly,
never `git add -A` in this shared two-agent tree.**

## 2. Stale-reference sweep

| reference | file | was | action |
|---|---|---|---|
| "Recommendation: do NOT flip global PREFILL_V2=auto" | `docs/README.md` (reconciliation entry) | recommendation | → **DECIDED: stays OFF** + handoff link |
| "First reconcile the `87.6 tok/s` headline" (pending) | `docs/README.md` (Claude-1 latency-hiding entry) | implied pending | → marked **RECONCILED**, linked result (index-link only) |
| "flipping the global default to `auto` is an owner call" | `docs/prefill-v2-auto-policy-result-20260620.md:45` | open call | → **DECIDED off** |
| "remaining owner call … flip global `PREFILL_V2=auto`" | `docs/decode-prefill-headline-reconciliation-scope-20260621.md` | scope precondition | → **RESOLVED banner** at top (provenance) |
| `87.6` mentions in canonical docs | README, handoff, reconciliation-result, bench/README | — | all carry context (ctx≈0 / ms / coincidence); **never bare** |
| "current state: amd-decode-banked-20260616" | `bench/README.md` | stale pointer | → repointed to the **handoff** + canonical-policy block |
| `87.6` in `decode-latency-hiding-*` (Claude-1) | Claude-1 lane | reconciled in their result; scope is provenance | left to Claude-1 (not my lane) |

No current/canonical doc now presents `PREFILL_V2=auto` as undecided, quotes `87.6` bare, or shows bounded decode
fusion as current work.

## 3. Files updated (mine only)

- `docs/current-project-state-handoff-20260621.md` (**new** — canonical state)
- `extra/qk_policy_consistency_check.py` (**new** — guardrail)
- `docs/README.md` (handoff as #1 start-here; reconciliation entry → DECIDED; Claude-1 latency entry → reconciled; canonical-policy-handoff entry → result)
- `bench/README.md` (canonical-policy block; current-state pointer → handoff)
- `docs/prefill-v2-auto-policy-result-20260620.md` (owner-call line → DECIDED off)
- `docs/decode-prefill-headline-reconciliation-scope-20260621.md` (RESOLVED banner)

Not touched: Claude-1's `decode-latency-hiding-*`, `decode-fused-coop-primitive-roadmap-scope`, `qk_decode_fused_flash_tile_ab.py`, `qk-lifecycle-search/*`.

## 4. Guardrail

`extra/qk_policy_consistency_check.py` (no GPU). Scans ONLY the canonical start-here set
(README, bench/README, handoff, reconciliation-result, prefill-policy-integration-result) and exits 1 if a current
doc re-opens a closed question: bare `87.6` without context, an open `PREFILL_V2=auto` owner call, an affirmative
"flip global PREFILL_V2=auto", `87` as the decode headline, or bounded decode fusion as current work. Uses a ±1-line
context window + self-referential-meta skip so explanatory/provenance docs don't false-positive. **PASS on all 5
canonical docs; verified it CATCHES an injected "Decode is 87.6 tok/s … flip global PREFILL_V2=auto" line.** Run:
`PYTHONPATH=. python3 extra/qk_policy_consistency_check.py`.

## 5. Final canonical state

Per `docs/current-project-state-handoff-20260621.md`: decode **68.1/66.4/60.7 @ctx512/1024/4096 (~67% llama)**,
~85–86 @ctx≈0; q8 opt-in 72.8/70.9/64.3 (default-off); prefill kernel-solved, opt-in fast paths shipped
(`PREFILL_V2=auto` / `PREFILL_SERVER_PROFILE=1`); **global `PREFILL_V2` default OFF (decided)**; bounded decode
fusion closed; `87.6` reconciled (never quote bare). The four "is X open?" questions are closed and guardrailed.

## 6. Remaining work (Claude 1 only)

Decode is the frontier; the one live lever is **fused + coop in one primitive** —
`docs/decode-fused-coop-primitive-roadmap-scope-20260621.md`. No tactical decode patch until it returns
`BRIDGE_FIRST` or `LINEARIZER_FIRST`. Claude 2's policy/headline lane is **closed and banked**.
