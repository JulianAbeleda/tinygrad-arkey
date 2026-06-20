# Decode q8 NT1024 Reconciliation Scope - 2026-06-20

Verdict target: `PASS_DECODE_Q8_NT1024_RECONCILED`

The `NT=1024` producer closes the producer-only delta, and a single lifecycle run passed in a fast timing band. This
scope reconciles whether that producer fix survives repeated fresh sessions.

## Tool

`extra/qk_decode_q8_nt1024_reconciliation.py`

## Method

Run the `NT=1024` lifecycle gate repeatedly in fresh Python sessions:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_nt1024_reconciliation.py --sessions 5 --rounds 24
```

Each child session preserves its own artifact under:

`bench/qk-decode-primitive-transfer/decode_q8_nt1024_reconciliation_session_*.json`

The reconciler reports:

| metric | reason |
|---|---|
| full median of session medians | original lifecycle policy |
| steady median of session medians | drops first four lifecycle rows |
| producer median | confirms the `NT=1024` fix survives lifecycle context |
| consumer median | checks whether remaining debt is shared consumer/session timing |
| best observed row | distinguishes physical capability from repeated policy |

## Decision Policy

| verdict | condition |
|---|---|
| `PASS_DECODE_Q8_NT1024_RECONCILED` | full or steady repeated median clears `115.24us` |
| `BLOCKED_DECODE_Q8_NT1024_THRESHOLD_VARIANCE` | miss is `<= 1us` |
| `BLOCKED_DECODE_Q8_NT1024_CONSUMER_SESSION_DEBT` | producer is recovered but repeated lifecycle remains above target |
| `BLOCKED_DECODE_Q8_NT1024_PRODUCER_NOT_RECOVERED` | producer no longer matches the `NT=1024` target band |
| `BLOCKED_DECODE_Q8_NT1024_INCORRECT` | correctness fails |

This is the promotion-policy reconciliation for the owned native producer successor path, not a default-on change.
