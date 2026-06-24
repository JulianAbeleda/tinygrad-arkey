# Decode q8 Oracle Repeated Comparator Result - 2026-06-20

Verdict: `PASS_DECODE_Q8_ORACLE_REPEATED_COMPARATOR_ATTRIBUTED`

Classification: `OWNED_PRODUCER_DEBT`

Command:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_oracle_repeated_comparator.py --sessions 5 --rounds 24
```

## Result

| metric | us |
|---|---:|
| target | 115.24 |
| owned route total, median of session medians | 131.84 |
| oracle route total, median of session medians | 122.68 |
| owned minus oracle total | +9.16 |
| owned producer, median of session medians | 30.18 |
| oracle producer, median of session medians | 21.06 |
| owned-route consumer, median of session medians | 101.64 |
| oracle-route consumer, median of session medians | 101.62 |

Correctness passed for both routes in all five sessions.

## Session Table

| session | owned producer | owned consumer | owned total | oracle producer | oracle consumer | oracle total |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30.18 | 101.64 | 131.74 | 20.96 | 101.64 | 122.68 |
| 1 | 30.24 | 101.52 | 131.72 | 21.20 | 101.46 | 122.68 |
| 2 | 30.36 | 101.70 | 131.96 | 21.06 | 101.74 | 122.80 |
| 3 | 30.14 | 101.64 | 131.86 | 21.20 | 101.56 | 122.74 |
| 4 | 30.14 | 101.64 | 131.84 | 21.04 | 101.62 | 122.66 |

## Interpretation

This attributes the repeated `~132us` mixed-route band:

1. The hipcc/LLD consumer is effectively identical in both routes: `101.64us` vs `101.62us`.
2. The tinygrad-owned producer is materially slower than the hipcc/LLD producer: `30.18us` vs `21.06us`.
3. Replacing only the producer recovers `~9.16us`, moving total lifecycle from `~131.84us` to `~122.68us`.
4. The oracle route still does not clear the `115.24us` target in this repeated-session band.

So there are two separate debts:

| debt | size | evidence |
|---|---:|---|
| owned producer debt | ~9us | owned producer slower than hipcc/LLD producer with same consumer |
| shared consumer/session debt | ~7.4us vs target | oracle route still `122.68us` vs `115.24us` |

The next decode step should not be a blind consumer rewrite. The first bounded native target is the q8 producer: match the
hipcc/LLD producer's `~21us` repeated-session behavior or import that artifact for the research route. After that, the
remaining target miss is a consumer/session attribution problem.

## Next Step

Scope the q8 producer delta:

1. Compare tinygrad `NORM_SOURCE` against hipcc/LLD `q8_rmsnorm_side` at the ISA/resource level.
2. Identify whether the `~9us` delta is thread shape, reduction structure, q8 pack vectorization, instruction selection,
   or launch metadata.
3. Build the smallest producer-only candidate or decide to keep the hipcc/LLD producer artifact for the research route.

The consumer remains a later problem: it is shared by both routes in this comparator and does not explain the owned-vs-
oracle delta.
