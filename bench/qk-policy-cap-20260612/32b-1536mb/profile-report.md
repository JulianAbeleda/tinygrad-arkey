# Quant Residual Decode Profile

Scope: Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode.

Classifier rules are calibrated to Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode logs. Use this report outside that scope only after adding boundary tests for the new kernel signatures.

Steady-state rows drop the first 1 benchmark token(s). `batched` rows are the throughput truth. `named` rows are attribution-only: they disable graph batching via JIT_BATCH_SIZE=1, so use AMD-kernel percentages rather than named wall time for bottleneck decisions.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32B | baseline batched | 7 | 3.46 | 288.77 | 287.90 | 0.87 | 0.30 |
| 32B | QK_GENERATED_POLICY batched | 7 | 4.09 | 249.11 | 248.21 | 0.90 | 0.36 |
| 32B | baseline named | 7 | 1.87 | 535.51 | 287.26 | 248.25 | 46.36 |
| 32B | QK_GENERATED_POLICY named | 7 | 1.79 | 557.62 | 288.34 | 269.28 | 48.29 |

## Parse Health

| model | mode | lines | tokens | AMD lines | ignored lines | non-AMD DEBUG lines | trailing AMD lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32B | baseline batched | 3989 | 8 | 3687 | 294 | 0 | 0 |
| 32B | QK_GENERATED_POLICY batched | 4631 | 8 | 4039 | 440 | 144 | 0 |
| 32B | baseline named | 15931 | 8 | 15631 | 292 | 0 | 0 |
| 32B | QK_GENERATED_POLICY named | 18237 | 8 | 17647 | 438 | 144 | 0 |

## Buckets

| model | mode | bucket | ms/tok | % wall | % AMD kernel | top kernels |
| --- | --- | --- | --- | --- | --- | --- |
| 32B | baseline batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | baseline batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | baseline batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 32B | baseline batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 32B | baseline batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 32B | baseline batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 32B | baseline batched | copy | 0.04 | 0.01 | 0.01 | copy        4 B,     AMD <- AMD |
| 32B | baseline batched | other_amd | 287.86 | 99.68 | 99.99 | batched 512, batched 256, batched 128 |
| 32B | baseline batched | residual_overhead | 0.87 | 0.30 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | copy | 0.04 | 0.02 | 0.02 | copy        4 B,     AMD <- AMD |
| 32B | QK_GENERATED_POLICY batched | other_amd | 248.17 | 99.62 | 99.98 | batched 512, batched 314, batched 256 |
| 32B | QK_GENERATED_POLICY batched | residual_overhead | 0.90 | 0.36 | 0.00 |  |
| 32B | baseline named | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | baseline named | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | baseline named | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 32B | baseline named | fallback_quant_fused | 275.42 | 51.43 | 95.88 | r_40_32_4_100_2_2_2_32, r_40_32_4_100_4_2_32, r_40_32_4_32_4_2_32 |
| 32B | baseline named | attention_misc | 1.50 | 0.28 | 0.52 | r_4_(start_pos+1)_8_4_4_16, r_8_2_8_16_4_(start_pos+1), r_16_4_(start_pos+1)n1 |
| 32B | baseline named | norm_sampling_misc | 0.89 | 0.17 | 0.31 | E_2_2_8_16_4_4 |
| 32B | baseline named | copy | 0.06 | 0.01 | 0.02 |  |
| 32B | baseline named | other_amd | 9.39 | 1.75 | 3.27 | r_8_8_16_2_20_4_2_32, r_8_8_16_2_20_2_2_2_32, r_16_320n1 |
| 32B | baseline named | residual_overhead | 248.25 | 46.36 | 0.00 |  |
| 32B | QK_GENERATED_POLICY named | q4k_primitive_gemv | 4.78 | 0.86 | 1.66 | q4k_gemv_partial_5120_25600_2, q4k_gemv_partial_1024_5120_4 |
| 32B | QK_GENERATED_POLICY named | q6k_primitive_gemv | 1.62 | 0.29 | 0.56 | q6k_gemv_partial_1024_5120_2 |
| 32B | QK_GENERATED_POLICY named | q4k_primitive_reduction | 9.54 | 1.71 | 3.31 |  |
| 32B | QK_GENERATED_POLICY named | fallback_quant_fused | 267.94 | 48.05 | 92.92 | r_40_32_4_100_2_2_2_32, r_40_32_4_32_4_2_32, r_200_32_4_20_4_2_32 |
| 32B | QK_GENERATED_POLICY named | attention_misc | 1.87 | 0.34 | 0.65 | r_4_(start_pos+1)_8_4_4_16, r_16_4_(start_pos+1)n1, r_8_2_8_16_4_(start_pos+1) |
| 32B | QK_GENERATED_POLICY named | norm_sampling_misc | 1.22 | 0.22 | 0.42 | E_2_2_8_16_4_4 |
| 32B | QK_GENERATED_POLICY named | copy | 0.06 | 0.01 | 0.02 |  |
| 32B | QK_GENERATED_POLICY named | other_amd | 1.32 | 0.24 | 0.46 | r_16_320n1, r_16_320, r_4_8_4_4_16 |
| 32B | QK_GENERATED_POLICY named | residual_overhead | 269.28 | 48.29 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 32B | baseline batched | 0 | 284.63 | 0.00 |
| 32B | QK_GENERATED_POLICY batched | 0 | 233.55 | 0.00 |
| 32B | baseline named | 0 | 516.34 | 0.00 |
| 32B | QK_GENERATED_POLICY named | 0 | 560.01 | 0.00 |

## Top Kernels

### 32B baseline batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 512 | 131.95 | 923.65 |
| batched 256 | 66.72 | 467.03 |
| batched 128 | 32.51 | 227.58 |
| batched 106 | 31.49 | 220.44 |
| batched 64 | 19.05 | 133.36 |
| batched 32 | 6.14 | 42.97 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.28 |

### 32B QK_GENERATED_POLICY batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 512 | 93.07 | 651.50 |
| batched 314 | 72.60 | 508.18 |
| batched 256 | 40.84 | 285.88 |
| batched 128 | 23.86 | 167.02 |
| batched 64 | 12.83 | 89.81 |
| batched 32 | 4.97 | 34.79 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.26 |

### 32B baseline named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_40_32_4_100_2_2_2_32 | 71.58 | 501.08 |
| r_40_32_4_100_4_2_32 | 54.51 | 381.55 |
| r_40_32_4_32_4_2_32 | 35.03 | 245.23 |
| r_200_32_4_20_4_2_32 | 32.04 | 224.26 |
| r_200_32_4_20_4_2_32n1 | 30.22 | 211.53 |
| r_64_32_4_20_4_2_32 | 25.36 | 177.52 |
| r_8_32_4_20_4_2_32 | 23.37 | 163.59 |
| r_8_8_16_2_20_4_2_32 | 4.00 | 27.98 |
| r_8_8_16_2_20_2_2_2_32 | 3.95 | 27.68 |
| r_1187_32_4_20_2_2_2_32n1 | 3.31 | 23.20 |
| E_2_2_8_16_4_4 | 0.54 | 3.75 |
| r_16_320n1 | 0.44 | 3.10 |
| r_16_320 | 0.44 | 3.06 |
| r_4_(start_pos+1)_8_4_4_16 | 0.42 | 2.92 |
| r_4_8_4_4_16 | 0.32 | 2.26 |
| r_8_2_8_16_4_(start_pos+1) | 0.30 | 2.10 |
| r_16_4_(start_pos+1)n1 | 0.30 | 2.09 |
| r_16_4_(start_pos+1) | 0.24 | 1.70 |
| E_(start_pos+1)_16_4 | 0.24 | 1.67 |
| r_8_16_8 | 0.23 | 1.63 |

### 32B QK_GENERATED_POLICY named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_40_32_4_100_2_2_2_32 | 86.11 | 602.77 |
| r_40_32_4_32_4_2_32 | 46.17 | 323.19 |
| r_200_32_4_20_4_2_32 | 39.57 | 276.99 |
| r_40_32_4_100_4_2_32 | 33.78 | 236.43 |
| r_64_32_4_20_4_2_32 | 29.76 | 208.29 |
| r_200_32_4_20_4_2_32n1 | 27.71 | 193.95 |
| r_200_32_4_20_4_2_32n2 | 9.90 | 69.29 |
| r_1187_32_4_20_2_2_2_32n1 | 3.33 | 23.30 |
| q4k_gemv_partial_5120_25600_2 | 3.13 | 21.89 |
| q4k_gemv_partial_1024_5120_4 | 1.65 | 11.55 |
| q6k_gemv_partial_1024_5120_2 | 1.62 | 11.32 |
| E_2_2_8_16_4_4 | 0.62 | 4.31 |
| r_16_320n1 | 0.57 | 3.99 |
| r_16_320 | 0.52 | 3.63 |
| r_4_(start_pos+1)_8_4_4_16 | 0.50 | 3.52 |
| r_16_4_(start_pos+1)n1 | 0.37 | 2.62 |
| r_8_2_8_16_4_(start_pos+1) | 0.37 | 2.62 |
| r_4_8_4_4_16 | 0.37 | 2.60 |
| E_(start_pos+1)_16_4 | 0.31 | 2.20 |
| r_16_4_(start_pos+1) | 0.31 | 2.14 |

## Decision Gates

- **32B baseline batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **32B QK_GENERATED_POLICY batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **32B baseline named** (AMD-kernel basis): fallback/generic dense quant remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **32B QK_GENERATED_POLICY named** (AMD-kernel basis): fallback/generic dense quant remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
