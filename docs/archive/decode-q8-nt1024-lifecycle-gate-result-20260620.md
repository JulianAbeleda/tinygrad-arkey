# Decode q8 NT1024 Lifecycle Gate Result - 2026-06-20

Verdict: `PASS_DECODE_Q8_NT1024_LIFECYCLE_GATE`

Command:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_nt1024_lifecycle_gate.py --rounds 24
```

## Result

| row | median us |
|---|---:|
| producer only | 9.22 |
| lifecycle producer | 9.16 |
| lifecycle consumer | 47.84 |
| lifecycle total | 57.00 |
| target lifecycle | 115.24 |

Correctness passed before and after timing:

| check | max abs |
|---|---:|
| producer fp | 0 |
| q8 dequant | 0.010329 |
| gate output | 9.537e-7 |
| up output | 1.431e-6 |

## Same-Band Sanity Check

Immediately after this run, the old `NT=256` interleaved lifecycle gate was rerun with the same target:

```sh
PYTHONPATH=. python3 extra/qk_decode_owned_q8_interleaved_lifecycle_gate.py --rounds 24 \
  --out bench/qk-decode-primitive-transfer/decode_owned_q8_interleaved_lifecycle_gate_current_compare.json
```

| route | producer us | consumer us | total us |
|---|---:|---:|---:|
| old `NT=256` | 13.12 | 47.34 | 60.56 |
| new `NT=1024` | 9.16 | 47.84 | 57.00 |

This confirms the producer fix in the same timing band: `NT=1024` improves producer time by `~3.96us` here and total
lifecycle by `~3.56us`.

## Interpretation

The absolute lifecycle pass is clock/session-band sensitive. This run landed in a much faster band than the previous
repeated-session `~132us` results; in the same band, even the old `NT=256` route passed the `115.24us` target.

The stable conclusion is narrower and stronger:

1. `NT=1024` is a valid owned producer fix.
2. It preserves correctness.
3. It improves the owned producer in both producer-only and lifecycle context.
4. It removes the need to import the hipcc/LLD producer artifact for producer parity.

This does not by itself prove the decode q8 route is promotion-ready. Promotion still needs a repeated-session lifecycle
policy that controls or explicitly classifies the fast/slow timing bands.

## Next Step

Apply the `NT=1024` producer as the research-route candidate, then rerun the repeated target reconciliation with this
producer. If the slow band moves from `~131.8us` toward `~122.7us`, the remaining blocker is shared consumer/session
timing. If it stays near `~131.8us`, the producer fix is not surviving repeated lifecycle context.
