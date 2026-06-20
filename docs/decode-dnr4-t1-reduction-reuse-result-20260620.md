# Decode DNR-4 T1 Reduction Reuse Result - 2026-06-20

Verdict: `PASS_DNR4_T1_REDUCTION_REUSE_STRUCTURAL_CORRECT`

DNR4-T1 implements the first resource/liveness target: remove the high native reduction/tail band `v50-v54` by reusing low dead temporaries after the dot/scale body.

## What Changed

Only the reduction/tail temporary registers changed:

- old reduction lane/temp band: `v50-v54`;
- new reduction lane/temp band: `v1-v6`;
- `v10` remains the accumulated partial;
- dot4 body, global load count, DS topology, barrier, and store shape are unchanged.

## Gates

| gate | result |
| --- | --- |
| launches | pass |
| correctness | pass |
| dot4 count preserved | 16 |
| `ds_bpermute` topology preserved | 5 |
| barrier preserved | 1 |
| grouped global loads unchanged | 22 |
| static max VGPR index | `41` |

This meets the structural goal from DNR-4: the candidate no longer requires the high `v50-v54` band and stays at `v0-v41` statically. This is not a performance claim.

## Decision

Next step is timing against native/best-static/C7C in the same harness. If timing does not move materially, DNR4-T1 is only a resource cleanup and the next meaningful target is DNR4-T2 dot-body vector-band compression.

Probe: `extra/qk_decode_dnr4_t1_reduction_reuse_probe.py`

