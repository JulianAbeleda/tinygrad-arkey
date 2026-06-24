# Decode DNR-4 T1 Timing Result - 2026-06-20

Verdict: `BLOCKED_DNR4_T1_STRUCTURAL_ONLY_TIMING_NOT_MATERIAL`

DNR4-T1 is correct and structurally useful, but timing does not move materially in the same-harness comparison. This means the high `v50-v54` reduction band was not the main reason native trails the oracle.

## Timing

Same-process interleaved timing, 4 warmups, 12 iterations:

| row | median us |
| --- | ---: |
| native DNR-2 | `409.826` |
| best static DNR-3C6 | `394.963` |
| C7C best | `380.095` |
| DNR4-T1 reduction reuse | `407.992` |

The decision gate is material movement: `>=30us` vs native, `>=15us` vs best static, or `>=10us` vs C7C. DNR4-T1 does not meet that gate.

DNR4-T1 moves only `+1.833us` vs native and is slower than both best static and C7C in this run.

## Decision

DNR4-T1 should be kept as a structural cleanup candidate only. The next meaningful target is DNR4-T2 dot-body vector-band compression: reduce the S2/S3 live range and load/unpack vector footprint, because the reduction-tail register reuse alone did not explain the gap.

Probe: `extra/qk_decode_dnr4_t1_timing_probe.py`
