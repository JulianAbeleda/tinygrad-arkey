# Coalesced dequant M-E result 20260625-185351

Verdict: `PROCEED_P3_SEARCH_GENERALIZATION`

## Throughput

| ctx | owned tok/s | sched_packed | generated_skeleton | sched_wordlane | g2_lanemap | g3_lanemap_codegen | lane_partition | bubblebeam_futuresight | best scheduler | best/owned | tokens match |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 512 | 103.5 | 22.5 | 22.5 | 14.2 | 14.2 | 103.7 | 103.5 | 103.6 | g3_lanemap_codegen | 1.002 | True |
| 1024 | 101.7 | 22.4 | 22.4 | 14.2 | 14.2 | 101.6 | 101.6 | 101.5 | g3_lanemap_codegen | 0.999 | True |
| 2048 | 99.1 | 22.2 | 22.3 | 14.1 | 14.1 | 99.3 | 99.0 | 99.1 | g3_lanemap_codegen | 1.002 | True |
| 4096 | 94.4 | 22.0 | 22.0 | 14.0 | 14.0 | 94.2 | 94.2 | 94.2 | g3_lanemap_codegen | 0.998 | True |

## Interpretation

Best scheduler/lane-partition arm reached the >=90% owned threshold at every ctx with matching tokens. Proceed to P3.

## Artifact

- `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-185351.json`
