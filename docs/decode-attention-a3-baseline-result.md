# Decode Attention A3 Baseline Result

## Verdict

`DECODE_ATTENTION_A3_BASELINE_CAPTURED`

A2 is lifecycle-clean but not speed-competitive yet.

## Artifact

- `bench/qk-decode-attention-a3-baseline/latest.json`
- Tool: `extra/qk_decode_attention_a3_baseline.py`

## W==D result

| ctx | owned tok/s | A2 whole-cache tok/s | A2 / owned | delta tok/s |
|---:|---:|---:|---:|---:|
| 512 | 105.1 | 78.2 | 74.4% | -26.9 |
| 1024 | 103.4 | 75.6 | 73.1% | -27.8 |
| 2048 | 101.0 | 69.7 | 69.0% | -31.3 |
| 4096 | 96.1 | 60.6 | 63.1% | -35.5 |

## Route result

| Check | A2 result |
|---|---:|
| route clean | yes |
| token byte-identical | yes |
| owned tile fires | 0 |
| owned combine fires | 0 |
| generated attention programs | 7 |
| `E_49152` present | no |
| selected-route buffer identity | yes |

## Interpretation

A3 baseline confirms the correct next problem:

- Lifecycle purity is solved for the generated skeleton.
- The remaining gap is performance primitive/codegen quality.
- The gap grows with context, so the high-ROI path is the long-context attention work, not dispatch-only cleanup.

A2 generated programs:

- `flash_score_whole_cache_32_128`
- `flash_max_32`
- `flash_prob_32`
- `flash_gmax_32`
- `flash_partial_coop_vec_whole_cache_32_128`
- `flash_den_32`
- `flash_combine_32_128`

## Next step

Proceed to A3.1:

`whole-cache score v_dot2 lowering`

Reason:

- A2 now has a stable whole-cache score program.
- It is the cleanest place to attach a dot-product primitive without touching the owned route.
- Any `v_dot2` attempt must keep the same lifecycle gates: generated route, no owned flash, no `E_49152`, tokens match.
