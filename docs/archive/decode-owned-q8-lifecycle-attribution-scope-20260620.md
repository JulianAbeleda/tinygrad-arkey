# Decode Owned q8 Lifecycle Attribution Scope - 2026-06-20

Verdict: `PASS_DECODE_OWNED_Q8_LIFECYCLE_ATTRIBUTION_SCOPE_READY`

The mixed lifecycle row failed even though the component projection looked favorable:

| row | median us |
|---|---:|
| owned producer standalone | `15.70` |
| projected artifact gate/up consumer | `93.54` |
| projected mixed lifecycle | `109.24` |
| mixed producer measured | `30.44` |
| mixed gate/up consumer measured | `101.66` |
| mixed lifecycle measured | `132.10` |
| lifecycle target | `115.24` |

The next scope is attribution, not a new schedule search. The required first pass is a same-process timing ladder:

1. owned producer before loading the gate/up artifact;
2. owned producer after loading the gate/up artifact;
3. gate/up consumer after an owned producer fill;
4. full mixed lifecycle through the existing `perf_gateup` helper.

This answers whether the regression is producer inflation, consumer inflation, or composition/runtime state. ATT/PC
timeline should stay parked until this timing ladder cannot explain the row.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_lifecycle_attribution_scope.py
PYTHONPATH=. python3 extra/qk_decode_owned_q8_lifecycle_attribution_probe.py --warmups 8 --iters 20
```
