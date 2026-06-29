# Phase N4 whole-step attribution

**Verdict:** AMD_ISA_PHASE_N4_PASS_WHOLE_STEP_ATTRIBUTION_PINNED  
**Selected N5 branch:** N5A  
**Top:** largest native-vs-owned GPU-compute delta at ctx512: native_attn_tile (native 5791.68 vs owned 0, delta 5791.68). Native top single owner: q4k_gemv_ffn 41.7% of native GPU-compute.


## ctx512 native dynamic-S (by owner)
| owner | dur/step | % | n |
|---|---|---|---|
| q4k_gemv_ffn | 7108.87 | 41.7% | 6 |
| native_attn_tile | 5791.68 | 34.0% | 1 |
| generated_reduce | 2811.66 | 16.5% | 15 |
| generated_elementwise | 893.3 | 5.2% | 11 |
| attn_combine | 228.67 | 1.3% | 1 |
| attn_gmax | 155.15 | 0.9% | 1 |
| other | 38.3075 | 0.2% | 1 |
| **total** | **17027.6375** | | |

## ctx4096 native dynamic-S (by owner)
| owner | dur/step | % | n |
|---|---|---|---|
| native_attn_tile | 7722.9 | 39.3% | 1 |
| q4k_gemv_ffn | 7100.16 | 36.1% | 6 |
| generated_reduce | 2807.23 | 14.3% | 15 |
| generated_elementwise | 891.38 | 4.5% | 11 |
| attn_combine | 715.26 | 3.6% | 1 |
| attn_gmax | 368.7 | 1.9% | 1 |
| other | 36.2475 | 0.2% | 1 |
| **total** | **19641.8775** | | |