# Decode attention control-plane closure result

## Verdict

`CONTROL_PLANE_PRESENT__OUTER_B_LOWERING_AND_SEARCH_BINDING_REMAIN`

This closes the missing audit/control-plane pieces for the generated decode-attention ctx regression path. It does not claim the speed-producing outer-`b` split-combine lowering is implemented.

## Built

| Piece | Artifact | Status |
|---|---|---|
| Occupancy guardrail | `extra/qk_decode_occupancy_guardrail.py`, `bench/qk-decode-occupancy-guardrail/latest.json` | Present |
| Outer-`b` split-combine search contract | `extra/qk_decode_outer_b_split_contract.py`, `bench/qk-decode-outer-b-split-combine/latest.json` | Search vocab present, lowering not built |
| Pressure/search ownership audit | `extra/qk_decode_pressure_search_ownership_audit.py`, `bench/qk-decode-pressure-search-ownership/latest.json` | Present |
| Split-aware hotloop oracle | `extra/qk_decode_hotloop_schedule_diff.py`, `bench/qk-decode-hotloop-schedule-diff/latest.json` | Present |

## Current measured diagnosis

| Counter | Owned | Generated |
|---|---:|---:|
| Selected loop class | outer b/main ctx | outer b/main ctx |
| `ds_bpermute` | 5 | 40 |
| `s_waitcnt` | 21 | 50 |
| Global loads | 22 | 10 |

The generated ctx slope is now diagnosed as cross-lane/waitcnt pressure in the selected outer ctx loop.

## Guardrail result

`OCCUPANCY_GUARDRAIL_PASS` for the current generated best-stack artifact.

The guardrail policy rejects candidates that exceed:

| Resource | Limit |
|---|---:|
| VGPR | 88 |
| Scratch | 0 |
| LDS | 8192 bytes |
| wg/CU | at least 4.0 |
| Cross-lane marker | 40 |
| Selected-loop waitcnt | 50 |

## Remaining real implementation work

| Item | Status | Meaning |
|---|---|---|
| `OuterBlockLoop.lds_staged_split_combine.lowering` | Not built | Need a generated candidate that splits the outer `b` online-softmax carry, stores partial state in LDS, and combines once. |
| `Scheduler.pressure_aware_latency_hiding.search_binding` | Partial | Guardrails exist, but winning flags are still manually selected rather than BubbleBeam-owned. |

## Next executable step

Implement the outer-`b` split-combine lowering behind the search contract, with this gate order:

1. Standalone numeric/microgate.
2. `OCCUPANCY_GUARDRAIL_PASS`.
3. Split-aware hotloop counters improve: generated `ds_bpermute` and/or `s_waitcnt` must move down from 40/50.
4. Route clean: no owned attention, no materialization.
5. W==D: ctx4096 improves without ctx512 regression.
