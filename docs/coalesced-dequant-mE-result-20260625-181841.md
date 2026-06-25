# Coalesced dequant M-E result 20260625-181841

Verdict: `PROCEED_P3_SEARCH_GENERALIZATION`

## Throughput

| ctx | owned tok/s | sched_packed | generated_skeleton | sched_wordlane | g2_lanemap | g3_lanemap_codegen | lane_partition | bubblebeam_futuresight | best scheduler | best/owned | tokens match |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 512 | 103.5 | 22.5 | 22.5 | 14.2 | 14.2 | 103.7 | 103.3 | 103.7 | g3_lanemap_codegen | 1.002 | True |
| 1024 | 101.8 | 22.4 | 22.4 | 14.2 | 14.2 | 101.7 | 101.6 | 101.5 | g3_lanemap_codegen | 0.999 | True |
| 2048 | 99.2 | 22.3 | 22.3 | 14.1 | 14.1 | 99.4 | 99.0 | 99.4 | g3_lanemap_codegen | 1.002 | True |
| 4096 | 94.8 | 22.0 | 22.0 | 14.0 | 14.0 | 94.5 | 94.4 | 94.4 | g3_lanemap_codegen | 0.997 | True |

## Interpretation

Best scheduler/lane-partition arm reached the >=90% owned threshold at every ctx with matching tokens. Proceed to P3.

## G3.1 LaneMap codegen result

`g3_lanemap_codegen` verdict: `G3_LANEMAP_PROMOTABLE`. It is token-correct and route-clean, with tok/s 512: 103.7 (1.002x owned), 1024: 101.7 (0.999x owned), 2048: 99.4 (1.002x owned), 4096: 94.5 (0.997x owned). It emits `q4k_g3_lanemap_gemv_12288_4096` instead of owned warp or lane-partition bridge programs.

## Artifact

- `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-181841.json`
