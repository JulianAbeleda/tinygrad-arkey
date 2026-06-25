# Coalesced dequant M-E result 20260625-162422

Verdict: `PROCEED_P3_SEARCH_GENERALIZATION`

## Throughput

| ctx | owned tok/s | sched_packed | sched_wordlane | lane_partition | bubblebeam_futuresight | best scheduler | best/owned | tokens match |
|---:|---:|---:|---:|---:|---:|---|---:|---|
| 512 | 103.3 | 22.5 | 14.2 | 103.5 | 103.5 | lane_partition | 1.002 | True |
| 1024 | 101.5 | 22.4 | 14.2 | 101.6 | 101.6 | lane_partition | 1.001 | True |
| 2048 | 98.9 | 22.2 | 14.1 | 99.0 | 99.1 | bubblebeam_futuresight | 1.002 | True |
| 4096 | 94.2 | 22.0 | 14.0 | 94.4 | 94.4 | lane_partition | 1.002 | True |

## Interpretation

Best scheduler/lane-partition arm reached the >=90% owned threshold at every ctx with matching tokens. Proceed to P3.

## Artifact

- `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-162422.json`
