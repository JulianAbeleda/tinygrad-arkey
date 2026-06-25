# Coalesced dequant M-E result 20260625-171554

Verdict: `PROCEED_P3_SEARCH_GENERALIZATION`

## Throughput

| ctx | owned tok/s | sched_packed | generated_skeleton | sched_wordlane | g2_lanemap | lane_partition | bubblebeam_futuresight | best scheduler | best/owned | tokens match |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| 512 | 103.4 | 22.5 | 22.5 | 14.2 | 14.2 | 103.4 | 103.2 | lane_partition | 1.000 | True |
| 1024 | 101.5 | 22.4 | 22.4 | 14.2 | 14.2 | 101.3 | 101.1 | lane_partition | 0.998 | True |
| 2048 | 98.8 | 22.2 | 22.3 | 14.1 | 14.1 | 98.8 | 98.6 | lane_partition | 1.000 | True |
| 4096 | 94.2 | 22.0 | 22.0 | 14.0 | 14.0 | 94.0 | 93.8 | lane_partition | 0.998 | True |

## Interpretation

Best scheduler/lane-partition arm reached the >=90% owned threshold at every ctx with matching tokens. Proceed to P3.

## G2.3 LaneMap result

`g2_lanemap` verdict: `SEARCH_GENERATED_WD_FAIL`. It is token-correct and route-clean but measures 512: 14.2 tok/s (0.137x owned), 1024: 14.2 tok/s (0.140x owned), 2048: 14.1 tok/s (0.143x owned), 4096: 14.0 tok/s (0.149x owned). This proves the remaining blocker is runtime/codegen lowering, not LaneMap/address representation.

## Artifact

- `bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_20260625-171554.json`
