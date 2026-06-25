# Coalesced dequant M-E result 20260625-155214

Verdict: `PROCEED_P3_SEARCH_GENERALIZATION`

## Throughput

| ctx | owned tok/s | sched_packed | sched_wordlane | lane_partition | best scheduler | best/owned | tokens match |
|---:|---:|---:|---:|---:|---|---:|---|
| 512 | 103.4 | 22.5 | 14.2 | 103.2 | lane_partition | 0.998 | True |
| 1024 | 101.3 | 22.4 | 14.2 | 101.3 | lane_partition | 1.000 | True |
| 2048 | 99.1 | 22.3 | 14.1 | 98.8 | lane_partition | 0.997 | True |
| 4096 | 94.3 | 22.0 | 14.0 | 94.2 | lane_partition | 0.999 | True |

## Interpretation

Best scheduler/lane-partition arm reached the >=90% owned threshold at every ctx with matching tokens. Proceed to P3.

## Artifact

- `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-155214.json`
