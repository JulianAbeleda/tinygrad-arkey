# Decode Owned q8 Mixed Lifecycle Result - 2026-06-20

Verdict: `BLOCKED_DECODE_OWNED_Q8_MIXED_LIFECYCLE_NOT_MATERIAL`

This measures:

```text
owned COMGR producer/cache + existing hipcc/LLD HCQ gate/up consumer
```

It is a mixed-ownership lifecycle row, not a fully owned successor.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_mixed_lifecycle_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_mixed_lifecycle_result.json
```

## Result

| row | median us |
|---|---:|
| owned COMGR producer in mixed harness | `30.44` |
| hipcc/LLD HCQ gate/up consumer | `101.66` |
| mixed lifecycle | `132.10` |
| artifact lifecycle target | `115.24` |

Correctness passes for producer, gate, and up. The lifecycle gate fails by `16.86us`.

## Decision

The owned producer row remains valid as an HCQ-parity standalone producer row, but it does not compose into a faster
mixed lifecycle with the artifact gate/up consumer in this harness. Do not promote the mixed route.

Next work must explain the non-composition or move to a fully owned consumer/lifecycle plan; simply swapping the
producer into the artifact consumer path is not enough.
