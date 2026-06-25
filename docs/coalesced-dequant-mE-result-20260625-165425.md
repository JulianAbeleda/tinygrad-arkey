# Coalesced dequant M-E result 20260625-165425

Verdict: `PROCEED_P3_SEARCH_GENERALIZATION`

## Throughput

| ctx | owned tok/s | sched_packed | generated_skeleton | sched_wordlane | lane_partition | bubblebeam_futuresight | best scheduler | best/owned | tokens match |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 512 | 103.4 | 22.5 | 22.5 | 14.2 | 103.7 | 103.7 | lane_partition | 1.003 | True |
| 1024 | 101.8 | 22.4 | 22.4 | 14.2 | 101.8 | 101.7 | lane_partition | 1.000 | True |
| 2048 | 99.0 | 22.3 | 22.2 | 14.1 | 99.3 | 99.4 | bubblebeam_futuresight | 1.004 | True |
| 4096 | 94.5 | 22.0 | 22.0 | 14.0 | 94.6 | 94.5 | lane_partition | 1.001 | True |

## Interpretation

Best scheduler/lane-partition arm reached the >=90% owned threshold at every ctx with matching tokens. Proceed to P3.

## Artifact

- `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-165425.json`
