# Decode Owned q8 Lifecycle Attribution Result - 2026-06-20

Verdict: `BLOCKED_DECODE_OWNED_Q8_LIFECYCLE_ATTRIBUTION_COMPOSITION_SLOW`

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_owned_q8_lifecycle_attribution_probe.py --warmups 8 --iters 20
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_attribution_result.json
```

## Result

| row | median us |
|---|---:|
| producer before gate/up artifact load | `30.92` |
| producer after gate/up artifact load | `30.90` |
| gate/up consumer after owned producer | `93.22` |
| controlled lifecycle | `124.12` |
| full `perf_gateup` mixed lifecycle | `123.64` |
| target lifecycle | `115.24` |

Correctness passes for all measured rows.

## Interpretation

The previous `132.10us` mixed row overstated the consumer problem. In this controlled pass, the gate/up consumer is back
near the projected `~93.54us` row.

The blocker is the producer in lifecycle context:

- loading the hipcc/LLD gate/up artifact does not perturb producer timing (`30.92us` -> `30.90us`);
- the producer is already slow before artifact load when the full lifecycle fixture is resident;
- a same-session standalone rerun of `extra/qk_decode_owned_q8_producer_cache_lowering_candidate.py` measured
  `21.62us`, so the old `15.70us` producer row is not stable enough to use as the lifecycle projection basis;
- the remaining controlled lifecycle gap is `124.12us - 115.24us = 8.88us`.

Next scope should isolate why the producer changes across contexts:

1. standalone producer with only producer buffers;
2. producer after allocating q4 gate/up buffers but before gate/up program load;
3. producer after gate/up program load;
4. producer after gate/up execution;
5. clock/provenance attached to each row.

ATT/PC timeline is not the next tool yet; the lifecycle miss is visible at the timing/context level.
