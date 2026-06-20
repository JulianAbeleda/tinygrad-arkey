# Decode q8 Consumer-Only Reconciliation Result - 2026-06-20

Verdict: `BLOCKED_DECODE_Q8_CONSUMER_ONLY_SLOW_BAND`

Classification: `SLOW_BAND`

Command:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_consumer_only_reconciliation.py --sessions 5 --rounds 32
```

## Result

| metric | us |
|---|---:|
| consumer-only median of session medians | 101.22 |
| consumer-after-dummy median of session medians | 101.36 |
| best observed session median | 92.80 |
| worst observed session median | 101.44 |
| fast sessions (`<=91us`) | 0 |
| slow sessions (`>=100us`) | 3 |

Correctness passed in every child session.

## Session Table

| session | consumer only us | consumer after dummy us | min us | max us |
|---:|---:|---:|---:|---:|
| 0 | 101.22 | 92.78 | 90.60 | 910.60 |
| 1 | 101.44 | 101.54 | 90.76 | 214.96 |
| 2 | 92.80 | 101.36 | 90.64 | 212.76 |
| 3 | 101.34 | 92.96 | 88.88 | 136.32 |
| 4 | 97.04 | 101.50 | 90.64 | 214.56 |

## Interpretation

Consumer-only timing reproduces the repeated lifecycle slow band. The median-of-session-medians is `101.22us`, matching
the `~101.5us` consumer component seen in the `NT=1024` lifecycle reconciliation.

This rules out the producer as the remaining blocker:

| stage | status |
|---|---|
| q8 producer correctness | cleared |
| q8 producer thread shape | cleared by `NT=1024` |
| mixed lifecycle producer | recovered at `~20.5us` in slow sessions |
| fused gate/up consumer | still `~101us` repeated median |

There is still intra-session variability: several sessions have `~90us` rows and one session has a `92.80us` median.
But the repeated policy lands on the slow band, and the dummy row also reconciles to `101.36us`. The next problem is
consumer/session attribution, not lifecycle composition.

## Next Step

Reopen fused gate/up consumer attribution:

1. Capture static resource/ISA metadata for the hipcc/LLD `q8_mmvq_gateup` artifact next to the current timing bands.
2. Split consumer timing into order/band probes: first-N rows, alternating dummy rows, repeated same dispatch, and fresh
   process sessions.
3. If the `~101us` band stays stable, use PMC/SQTT or static issue analysis to target consumer issue/resource behavior.

Until this is done, decode q8 remains blocked on the fused gate/up consumer band.
