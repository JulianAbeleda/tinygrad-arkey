# Decode DNR-4 Live-Range Scope - 2026-06-20

Verdict: `PASS_DNR4_RESOURCE_LIVERANGE_SCOPE_READY`

The oracle profiling surface reopened native decode only in a narrow way: target resource/liveness, not another issue-order guess. DNR-4 starts from the concrete VGPR envelope difference.

## Register Envelope

Oracle static stage scan:

| stage | unique VGPR | max VGPR index |
| --- | ---: | ---: |
| S0 setup/bounds/addresses | 10 | 9 |
| S1 scale/min byte select | 7 | 9 |
| S2 q4 vector load prefetch | 22 | 25 |
| S3 interleaved unpack/dot4/scale | 23 | 25 |
| S4 cross-lane partial reduce | 6 | 6 |
| S5 final writeback | 5 | 4 |

Oracle stays within `v0-v25`; kernel trace reports `32` VGPR. Native allocates `56` VGPR/workitem and has a separate high `v50-v54` reduction/tail band.

## Targets

| target | evidence | gate |
| --- | --- | --- |
| DNR4-T1 reduction-band reuse | Oracle reuses low registers for S4/S5; native reserves `v50-v54`. | Allocated VGPR decreases from `56` toward `<=40` with correctness intact. |
| DNR4-T2 dot-body vector-band compression | Oracle S2/S3 stays within `v0-v25`; native dot body spans disjoint bands through `v37`. | S3-equivalent native body max VGPR index `<=31` and 16 dot4 preserved. |
| DNR4-T3 live-interval expiry check | Native accumulator/reduction spans cross into tail; oracle restarts low registers for reduction/writeback. | Probe proves dead registers at S3->S4 and S4->S5 before timing. |

## Decision

Next executable step is DNR4-T1: build a structural/live-range candidate that reuses reduction/tail registers instead of reserving the high `v50-v54` band.

Stop condition: no promotion from static similarity. A candidate must be correct and move same-run timing materially; `<=40` allocated VGPR is only a structural gate.

Probe: `extra/qk_decode_dnr4_liverange_scope.py`

