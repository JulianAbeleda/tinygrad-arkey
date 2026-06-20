# Decode q8 Oracle Repeated Comparator Scope - 2026-06-20

Verdict target: `PASS_DECODE_Q8_ORACLE_REPEATED_COMPARATOR_ATTRIBUTED`

The lifecycle target reconciliation found one near-target session and four stable `~132us` sessions. The next question is
whether the slow band follows the current mixed route specifically or also appears when the producer is the hipcc/LLD
oracle.

## Tool

`extra/qk_decode_q8_oracle_repeated_comparator.py`

## Routes

| route | producer | consumer |
|---|---|---|
| `owned_lifecycle` | tinygrad COMGR `NORM_SOURCE`, 256 threads | hipcc/LLD fused gate/up |
| `oracle_lifecycle` | hipcc/LLD `q8_rmsnorm_side`, 1024 threads | hipcc/LLD fused gate/up |

Both routes use the same resident q4 gate/up buffers, q8 buffer, destination buffers, random activation, and correctness
reference per child session.

## Method

Run fresh child sessions:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_oracle_repeated_comparator.py --sessions 5 --rounds 24
```

Each child randomizes route order per round and records producer, consumer, and total lifecycle time.

## Gates

| gate | threshold |
|---|---:|
| all child artifacts present | pass |
| owned route correctness | pass |
| oracle route correctness | pass |
| route medians available | pass |

## Decision Policy

| verdict | condition |
|---|---|
| `PASS_DECODE_Q8_ORACLE_REPEATED_COMPARATOR_ATTRIBUTED` | comparator runs and correctness passes |
| `BLOCKED_DECODE_Q8_ORACLE_COMPARATOR_INCORRECT` | any route fails correctness |

The result classifies the blocker:

| classification | meaning |
|---|---|
| `OWNED_PRODUCER_DEBT` | oracle route clears target or is materially faster |
| `SHARED_CONSUMER_OR_SESSION_DEBT` | both routes sit in the same slow band |
| `MIXED_SESSION_VARIANCE` | results are inconsistent and need clock control |
