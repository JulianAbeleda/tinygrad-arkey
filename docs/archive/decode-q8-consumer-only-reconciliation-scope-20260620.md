# Decode q8 Consumer-Only Reconciliation Scope - 2026-06-20

Verdict target: `PASS_DECODE_Q8_CONSUMER_ONLY_RECONCILED`

The `NT=1024` lifecycle reconciliation proved the producer fix survives repeated sessions. The remaining miss is the
hipcc/LLD fused gate/up consumer band:

| band | producer us | consumer us | total us |
|---|---:|---:|---:|
| passing sessions | ~17.45 | ~90.8 | ~108.3 |
| blocked sessions | ~20.5 | ~101.6 | ~122.0 |

This scope isolates the consumer by keeping q8 and q4 buffers resident and timing only `q8_mmvq_gateup`.

## Tool

`extra/qk_decode_q8_consumer_only_reconciliation.py`

## Method

Each fresh child session:

1. Builds one correct q8 activation with the `NT=1024` producer.
2. Keeps q8, q4 gate/up, and output buffers resident.
3. Warms and measures only the hipcc/LLD fused gate/up consumer.
4. Checks correctness before and after measurement.

Run:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_consumer_only_reconciliation.py --sessions 5 --rounds 32
```

## Gates

| gate | threshold |
|---|---:|
| all artifacts present | pass |
| producer setup correctness | pass |
| consumer correctness | pass |
| repeated consumer median | reported |
| band classification | required |

## Decision Policy

| verdict | meaning |
|---|---|
| `PASS_DECODE_Q8_CONSUMER_ONLY_RECONCILED_FAST` | repeated consumer median is `<= 91us` |
| `BLOCKED_DECODE_Q8_CONSUMER_ONLY_MID_BAND` | repeated median is `> 91us` but below the `~101us` lifecycle slow band |
| `BLOCKED_DECODE_Q8_CONSUMER_ONLY_SLOW_BAND` | repeated median is stable near `101us` |
| `BLOCKED_DECODE_Q8_CONSUMER_ONLY_BIMODAL` | both fast and slow bands appear across sessions |
| `BLOCKED_DECODE_Q8_CONSUMER_ONLY_INCORRECT` | correctness fails |

If the consumer-only probe is slow/bimodal, the remaining decode work is consumer/session attribution, not lifecycle
composition.
