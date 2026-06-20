# Decode q8 Producer Order Provenance Audit Result - 2026-06-20

Verdict: `PASS_DECODE_Q8_PRODUCER_ORDER_PROVENANCE_NO_CONTEXT_SLOWDOWN`

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_producer_order_provenance_audit.py --rounds 24
```

## Result

The audit randomized four producer rows in each round while keeping all context buffers alive.

| row | median us | delta vs producer-only |
|---|---:|---:|
| producer-only | `25.68` | `0.00` |
| producer with real q4 resident | `25.86` | `+0.18` |
| producer after gate/up dispatch | `25.18` | `-0.50` |
| producer with dummy resident | `25.60` | `-0.08` |

Gates:

| gate | result |
|---|---:|
| all producer correctness | pass |
| rows per label | `24` |
| post-gateup delta < `5us` | pass |
| resident q4 delta < `5us` | pass |

Clock provenance was captured but not used as timing authority:

| point | `rocm-smi --showgpuclocks` |
|---|---|
| before | `sclk level 1 (587Mhz)` |
| after | `sclk level 1 (1202Mhz)` |

## Interpretation

The previous context slowdown is not stable under interleaving. In a randomized same-run comparison:

- real q4 residency does not slow the producer,
- dummy residency does not slow the producer,
- an immediately prior gate/up dispatch does not slow the producer.

This means the prior `30.94us` producer row should not drive code changes by itself. The absolute producer timing is
session/clock/order sensitive, but the interleaved deltas are small.

## Decision

Rerun the lifecycle gate with interleaved/paired ordering before changing producer or consumer code. Decode now moves
from:

```text
BLOCKED_DECODE_Q8_PRODUCER_CONTEXT_ORDER_OR_SESSION_SENSITIVE
```

to:

```text
READY_DECODE_Q8_RERUN_INTERLEAVED_LIFECYCLE_GATE
```
