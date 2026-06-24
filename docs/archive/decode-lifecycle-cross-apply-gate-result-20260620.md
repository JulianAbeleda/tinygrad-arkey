# Decode Lifecycle Cross-Apply Gate Result - 2026-06-20

Verdict: `BLOCKED_DECODE_Q8_PROMOTION_ON_PRODUCER_CONTEXT`

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_lifecycle_cross_apply_gate.py
```

## Result

| item | value |
|---|---:|
| mixed q8 lifecycle | `123.64us` |
| target lifecycle | `115.24us` |
| gap | `8.40us` |
| producer in lifecycle | `30.54us` |
| consumer in lifecycle | `93.10us` |

The consumer is near expected. The producer is the current blocker.

Decision: apply the prefill audit rule directly. Do not promote q8 from isolated producer speed. Next decode work is
producer-only batch/context isolation, then a lifecycle rerun.
