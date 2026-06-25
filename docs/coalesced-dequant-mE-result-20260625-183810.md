# Coalesced dequant M-E result 20260625-183810

Verdict: `STOP_CUSTOM_NEEDED_FOR_GEMV_TARGET`

## Throughput

| ctx | owned tok/s | sched_packed | generated_skeleton | sched_wordlane | g2_lanemap | g3_lanemap_codegen | lane_partition | bubblebeam_futuresight | best scheduler | best/owned | tokens match |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 512 | 103.4 | 22.5 | 22.5 | 14.2 | 14.2 | 103.7 | 103.4 | 103.5 | g3_lanemap_codegen | 1.003 | True |
| 1024 | 101.9 | 22.4 | 22.4 | 14.2 | 14.2 | 101.8 | 101.8 | 101.6 | g3_lanemap_codegen | 0.999 | True |
| 2048 | 99.2 | 22.3 | 22.3 | 14.1 | 14.2 | 99.4 | 99.1 | 99.3 | g3_lanemap_codegen | 1.002 | True |
| 4096 | 94.8 | 22.0 | 22.0 | 14.0 | 14.1 | 94.6 | 94.5 | 94.5 | g3_lanemap_codegen | 0.998 | True |

## Interpretation

Best scheduler/lane-partition arm did not reach the >=90% owned threshold at every ctx. Do not fund P3; record CUSTOM as needed for this GEMV performance target.

## Artifact

- `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-183810.json`
