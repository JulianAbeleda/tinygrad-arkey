# Decode Owned q8 Mixed Lifecycle Scope - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_MIXED_LIFECYCLE_SCOPE_READY`

The next executable row is mixed ownership:

```text
owned COMGR producer/cache + existing hipcc/LLD HCQ gate/up consumer
```

This does not make the successor fully owned. It tests whether the accepted owned producer row improves the measured
artifact lifecycle before consumer ownership resumes.

Projected from existing rows:

| component | us |
|---|---:|
| owned COMGR producer | `15.70` |
| HCQ artifact gate/up consumer | `~93.54` |
| projected mixed lifecycle | `~109.24` |
| artifact lifecycle target | `115.24` |

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_mixed_lifecycle_scope.py
```

Next executable probe:

```text
extra/qk_decode_owned_q8_mixed_lifecycle_probe.py
```
