# Decode q8 Producer Resource Context Audit Result - 2026-06-20

Verdict: `PASS_DECODE_Q8_PRODUCER_RESOURCE_CONTEXT_NOT_REPRODUCED`

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_resource_context_audit.py --warmups 8 --iters 20
```

## Result

| row | median us | delta vs baseline |
|---|---:|---:|
| baseline | `13.16` | `0.00` |
| real q4 alloc-only | `13.28` | `+0.12` |
| real q4 copied | `13.38` | `+0.22` |
| dummy same alloc-only | `13.12` | `-0.04` |
| dummy same copied | `15.48` | `+2.32` |
| dummy half copied | `25.20` | `+12.04` |
| dummy double copied | `15.56` | `+2.40` |

Correctness passed on every row.

The real q4 gate/up bytes were:

| item | bytes |
|---|---:|
| gate q4 bytes | `28,311,552` |
| up q4 bytes | `28,311,552` |
| total q4 bytes | `56,623,104` |

## Interpretation

This audit did **not** reproduce the previous context-isolation slowdown:

```text
previous: producer-only 21.60us -> after q4 buffers 30.94us
this run: baseline 13.16us -> real q4 copied 13.38us
```

So the simple explanation is refuted:

```text
resident real q4 buffers alone do not deterministically slow the producer by ~9us
```

The producer timing is order/session/state sensitive. The dummy copied rows show outliers and median movement, but
not a clean monotonic size relationship. The half-size copied row was slower than same/double-size copied rows,
which points away from a simple resident-byte pressure model.

## Decision

Do not claim q4 buffer residency is causal yet. The next decode gate must measure order and provenance directly:

- interleave producer-only and producer-with-context rows in one run,
- randomize or repeat row order,
- attach clock/provenance if available,
- keep producer and context buffers alive consistently,
- report whether the slowdown follows row order, buffer residency, previous copy, previous dispatch, or session.

Until that passes, decode remains blocked at:

```text
BLOCKED_DECODE_Q8_PRODUCER_CONTEXT_ORDER_OR_SESSION_SENSITIVE
```
