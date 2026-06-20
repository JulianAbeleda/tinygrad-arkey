# Decode q8 NT1024 Reconciliation Result - 2026-06-20

Verdict: `BLOCKED_DECODE_Q8_NT1024_CONSUMER_SESSION_DEBT`

Command:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_nt1024_reconciliation.py --sessions 5 --rounds 24
```

## Result

| metric | us |
|---|---:|
| target | 115.24 |
| full median of session medians | 122.04 |
| steady median of session medians | 122.00 |
| best policy delta | +6.76 |
| best observed row | 107.84 |
| producer median of session medians | 20.46 |
| consumer median of session medians | 101.50 |

Correctness passed in every child session.

## Session Table

| session | producer us | consumer us | total median us | steady median us | min total us | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 17.44 | 90.80 | 108.22 | 108.16 | 107.84 | pass |
| 1 | 17.46 | 90.84 | 108.34 | 108.30 | 108.00 | pass |
| 2 | 20.58 | 101.60 | 122.22 | 122.18 | 121.40 | blocked |
| 3 | 20.46 | 101.50 | 122.04 | 122.00 | 121.48 | blocked |
| 4 | 20.54 | 101.64 | 122.26 | 122.14 | 121.56 | blocked |

## Interpretation

The `NT=1024` producer fix survives repeated lifecycle context:

| route state | old repeated `NT=256` | new repeated `NT=1024` |
|---|---:|---:|
| producer median | ~30.18us | 20.46us |
| consumer median | ~101.6us | 101.50us |
| total median | ~131.84us | 122.04us |

So the producer change recovered roughly `9.8us`, matching the producer-only variant probe and the oracle comparator.

The route still does not reconcile to the `115.24us` target under median-of-sessions policy. The remaining miss is not
producer correctness, q8 addressing, q4 residency, or producer thread shape. It is the shared consumer/session band:
consumer is `~90.8us` in passing sessions and `~101.5-101.6us` in blocked sessions.

## Decision

`NT=1024` should be treated as the owned native producer candidate. Promotion is still blocked by consumer/session
timing, not by the producer.

The next bounded action is to reconcile the hipcc/LLD fused gate/up consumer band directly:

1. Run consumer-only repeated sessions with a prebuilt q8 buffer and resident q4 buffers.
2. Compare the `~90.8us` and `~101.6us` bands against clock/session provenance.
3. If the slow band is stable, reopen consumer issue/resource attribution; if it tracks session state, make timing policy
   explicit before promotion.
