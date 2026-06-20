# Decode q8 Lifecycle Target Reconciliation Result - 2026-06-20

Verdict: `BLOCKED_DECODE_Q8_LIFECYCLE_SCHEDULE_DEBT`

Command:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_lifecycle_target_reconciliation.py --sessions 5 --rounds 24
```

## Result

| metric | us |
|---|---:|
| target | 115.24 |
| full median of session medians | 131.92 |
| steady median of session medians | 131.82 |
| best policy delta | +16.58 |
| best observed row | 115.44 |
| best observed row delta | +0.20 |

Correctness passed in all five child sessions. The route is therefore not blocked by q8 producer correctness, q4
consumer correctness, or artifact generation.

## Session Table

| session | producer us | consumer us | total median us | steady median us | min total us |
|---:|---:|---:|---:|---:|---:|
| 0 | 25.32 | 90.86 | 116.18 | 115.96 | 115.44 |
| 1 | 30.40 | 101.56 | 132.02 | 131.82 | 131.16 |
| 2 | 30.62 | 101.60 | 132.16 | 132.12 | 131.16 |
| 3 | 30.26 | 101.60 | 131.84 | 131.76 | 131.28 |
| 4 | 30.32 | 101.54 | 131.92 | 131.92 | 130.84 |

## Gates

| gate | result |
|---|---|
| all artifacts present | pass |
| all producer correctness | pass |
| all consumer correctness | pass |
| full reconciled <= target | fail |
| steady reconciled <= target | fail |
| delta <= 1us threshold variance | fail |

## Interpretation

The previous single-session result was not enough to call the miss threshold variance. Repeated fresh sessions produced
one near-target session and four stable `~132us` sessions. The repeated steady median is `16.58us` above target, so the
blocker is real schedule/lifecycle debt under this harness.

The per-session split shows both components slow down in the repeated bad band:

| component | near-target session | repeated slow band |
|---|---:|---:|
| producer | ~25us | ~30us |
| consumer | ~91us | ~102us |
| total | ~116us | ~132us |

That pattern means the next route should not only tune the q8 producer. The larger repeated gap sits in the consumer
path and whole lifecycle timing. The useful objective is now a repeated paired-session target, not a one-off best row.

## Next Step

Reopen decode schedule attribution with this objective:

1. Compare the repeated `~132us` band against the hipcc/LLD oracle under the same repeated-session wrapper.
2. Attribute whether the slow band follows clock state, launch lifecycle, consumer issue behavior, or producer+consumer
   serialization.
3. Only then choose between native consumer schedule work, launch/lifecycle changes, or target policy changes.

Until that is done, q8 decode remains blocked on repeated lifecycle schedule debt.
