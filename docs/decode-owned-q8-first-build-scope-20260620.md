# Decode Owned q8 First Build Scope - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_FIRST_BUILD_SCOPE_READY`

The parity harness says the owned successor needs two implementation rows:

- owned producer/cache;
- owned `ffn_gate`/`ffn_up` packed q4/q8 consumers.

Do the producer/cache first.

## Why Producer First

The consumer route is still blocked by DNR4-T3: local native schedule edits are correct but not material, and PMC did
not create a search objective. Reopening the consumer without ATT would repeat the same loop.

The producer/cache is different. It owns the route-level lifecycle contract:

- q8 activation format;
- reuse count `2`;
- quality/dNLL policy;
- fallback behavior;
- coverage boundary.

That can be scoped and validated before any consumer schedule work resumes.

## Decision

| track | decision | reopen gate |
|---|---|---|
| owned q8 producer/cache | do first | byte/scale semantics, reuse contract, dNLL/fallback gates |
| owned gate/up consumer | park | ATT PC timeline or lowerable objective |

Next executable probe:

```text
extra/qk_decode_owned_q8_producer_cache_scope.py
```
