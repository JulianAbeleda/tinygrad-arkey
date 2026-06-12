# Q4_K Residual Decode Profile

Steady-state rows drop the first 1 benchmark token(s). `batched` logs use normal graph batching and are the real runtime profile. `named` logs set `JIT_BATCH_SIZE=1`; they keep the rollout JIT but avoid graph batching so DEBUG=2 exposes kernel names for attribution.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8B | baseline batched | 31 | 15.69 | 63.75 | 63.07 | 0.67 | 1.06 |
| 8B | Q4K_PRIMITIVE=1 batched | 31 | 29.06 | 34.46 | 33.76 | 0.70 | 2.02 |
| 8B | baseline named | 31 | 4.18 | 239.54 | 119.98 | 119.56 | 49.91 |
| 8B | Q4K_PRIMITIVE=1 named | 31 | 4.38 | 228.29 | 78.68 | 149.60 | 65.53 |
| 14B | baseline batched | 31 | 9.09 | 110.03 | 109.31 | 0.72 | 0.65 |
| 14B | Q4K_PRIMITIVE=1 batched | 31 | 15.77 | 63.59 | 62.89 | 0.71 | 1.11 |
| 14B | baseline named | 31 | 3.59 | 278.76 | 144.10 | 134.66 | 48.31 |
| 14B | Q4K_PRIMITIVE=1 named | 31 | 3.16 | 316.80 | 143.72 | 173.07 | 54.63 |

## Parse Health

| model | mode | lines | tokens | AMD lines | ignored lines | non-AMD DEBUG lines | trailing AMD lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8B | baseline batched | 2465 | 32 | 2253 | 180 | 0 | 0 |
| 8B | Q4K_PRIMITIVE=1 batched | 3095 | 32 | 2559 | 342 | 162 | 0 |
| 8B | baseline named | 24043 | 32 | 23807 | 204 | 0 | 0 |
| 8B | Q4K_PRIMITIVE=1 named | 29281 | 32 | 28721 | 366 | 162 | 0 |
| 14B | baseline batched | 2708 | 32 | 2479 | 197 | 0 | 0 |
| 14B | Q4K_PRIMITIVE=1 batched | 3408 | 32 | 2819 | 377 | 180 | 0 |
| 14B | baseline named | 26659 | 32 | 26407 | 220 | 0 | 0 |
| 14B | Q4K_PRIMITIVE=1 named | 32479 | 32 | 31867 | 400 | 180 | 0 |

## Buckets

| model | mode | bucket | ms/tok | % wall | % AMD kernel | top kernels |
| --- | --- | --- | --- | --- | --- | --- |
| 8B | baseline batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 8B | baseline batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 8B | baseline batched | fallback_q4k_fused | 0.00 | 0.00 | 0.00 |  |
| 8B | baseline batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | baseline batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | baseline batched | copy | 0.04 | 0.07 | 0.07 | copy        4 B,     AMD <- AMD |
| 8B | baseline batched | other_amd | 63.03 | 98.87 | 99.93 | batched 256, batched 142, batched 128 |
| 8B | baseline batched | residual_overhead | 0.67 | 1.06 | 0.00 |  |
| 8B | Q4K_PRIMITIVE=1 batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K_PRIMITIVE=1 batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K_PRIMITIVE=1 batched | fallback_q4k_fused | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K_PRIMITIVE=1 batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K_PRIMITIVE=1 batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K_PRIMITIVE=1 batched | copy | 0.04 | 0.13 | 0.13 | copy        4 B,     AMD <- AMD |
| 8B | Q4K_PRIMITIVE=1 batched | other_amd | 33.72 | 97.85 | 99.87 | batched 286, batched 256, batched 64 |
| 8B | Q4K_PRIMITIVE=1 batched | residual_overhead | 0.70 | 2.02 | 0.00 |  |
| 8B | baseline named | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 8B | baseline named | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 8B | baseline named | fallback_q4k_fused | 112.06 | 46.78 | 93.40 | r_32_32_4_48_2_2_2_32, r_32_32_4_48_4_2_32, r_32_32_4_16_4_2_32 |
| 8B | baseline named | attention_misc | 1.45 | 0.61 | 1.21 | r_4_2_8_16_4_(start_pos+1), r_2_(start_pos+1)_8_4_4_16, r_8_4_(start_pos+1)n1 |
| 8B | baseline named | norm_sampling_misc | 1.43 | 0.60 | 1.19 | E_2_8_16_4_4, r_16_256n1, r_16_256 |
| 8B | baseline named | copy | 0.05 | 0.02 | 0.04 |  |
| 8B | baseline named | other_amd | 4.99 | 2.08 | 4.16 | r_2_8_128_16_2_2_2_32, r_2_8_128_16_4_2_32, r_2_8_4_4_16 |
| 8B | baseline named | residual_overhead | 119.56 | 49.91 | 0.00 |  |
| 8B | Q4K_PRIMITIVE=1 named | q4k_primitive_gemv | 11.42 | 5.00 | 14.51 | q4k_gemv_partial_12288_4096_1, q4k_gemv_partial_4096_4096_1, q4k_gemv_partial_4096_12288_4 |
| 8B | Q4K_PRIMITIVE=1 named | q4k_primitive_reduction | 0.87 | 0.38 | 1.10 |  |
| 8B | Q4K_PRIMITIVE=1 named | fallback_q4k_fused | 55.52 | 24.32 | 70.56 | r_32_32_4_48_2_2_2_32, r_1187_32_4_16_2_2_2_32n1, r_1024_16_4_2_32 |
| 8B | Q4K_PRIMITIVE=1 named | attention_misc | 1.84 | 0.81 | 2.34 | r_4_2_8_16_4_(start_pos+1), r_2_(start_pos+1)_8_4_4_16, r_8_4_(start_pos+1)n1 |
| 8B | Q4K_PRIMITIVE=1 named | norm_sampling_misc | 2.49 | 1.09 | 3.16 | E_2_8_16_4_4, r_16_256n1, r_16_256 |
| 8B | Q4K_PRIMITIVE=1 named | copy | 0.05 | 0.02 | 0.07 |  |
| 8B | Q4K_PRIMITIVE=1 named | other_amd | 6.50 | 2.85 | 8.26 | r_2_8_128_16_2_2_2_32, r_2_8_128_16_4_2_32, r_2_8_4_4_16 |
| 8B | Q4K_PRIMITIVE=1 named | residual_overhead | 149.60 | 65.53 | 0.00 |  |
| 14B | baseline batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | baseline batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | baseline batched | fallback_q4k_fused | 0.00 | 0.00 | 0.00 |  |
| 14B | baseline batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | baseline batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | baseline batched | copy | 0.05 | 0.04 | 0.04 | copy        4 B,     AMD <- AMD |
| 14B | baseline batched | other_amd | 109.26 | 99.30 | 99.96 | batched 256, batched 210, batched 128 |
| 14B | baseline batched | residual_overhead | 0.72 | 0.65 | 0.00 |  |
| 14B | Q4K_PRIMITIVE=1 batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K_PRIMITIVE=1 batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K_PRIMITIVE=1 batched | fallback_q4k_fused | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K_PRIMITIVE=1 batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K_PRIMITIVE=1 batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K_PRIMITIVE=1 batched | copy | 0.05 | 0.08 | 0.08 | copy        4 B,     AMD <- AMD |
| 14B | Q4K_PRIMITIVE=1 batched | other_amd | 62.84 | 98.81 | 99.92 | batched 370, batched 256, batched 128 |
| 14B | Q4K_PRIMITIVE=1 batched | residual_overhead | 0.71 | 1.11 | 0.00 |  |
| 14B | baseline named | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | baseline named | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | baseline named | fallback_q4k_fused | 136.44 | 48.95 | 94.69 | r_40_32_4_68_2_2_2_32, r_40_32_4_68_4_2_32, r_136_32_4_20_4_2_32 |
| 14B | baseline named | attention_misc | 1.12 | 0.40 | 0.78 | r_5_2_8_16_4_(start_pos+1), r_5_2_4_(start_pos+1)n1, r_5_2_4_(start_pos+1) |
| 14B | baseline named | norm_sampling_misc | 0.65 | 0.23 | 0.45 | E_5_2_2_16_4_4, r_32_4_1187 |
| 14B | baseline named | copy | 0.05 | 0.02 | 0.04 |  |
| 14B | baseline named | other_amd | 5.83 | 2.09 | 4.04 | r_8_8_16_2_20_4_2_32, r_8_8_16_2_20_2_2_2_32, r_16_320n1 |
| 14B | baseline named | residual_overhead | 134.66 | 48.31 | 0.00 |  |
| 14B | Q4K_PRIMITIVE=1 named | q4k_primitive_gemv | 20.21 | 6.38 | 14.06 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_4 |
| 14B | Q4K_PRIMITIVE=1 named | q4k_primitive_reduction | 13.74 | 4.34 | 9.56 |  |
| 14B | Q4K_PRIMITIVE=1 named | fallback_q4k_fused | 95.99 | 30.30 | 66.79 | r_40_32_4_68_2_2_2_32, r_8_32_4_20_4_2_32, r_1187_32_4_20_2_2_2_32n1 |
| 14B | Q4K_PRIMITIVE=1 named | attention_misc | 1.91 | 0.60 | 1.33 | r_5_2_8_16_4_(start_pos+1), r_5_2_4_(start_pos+1)n1, r_5_2_4_(start_pos+1) |
| 14B | Q4K_PRIMITIVE=1 named | norm_sampling_misc | 1.85 | 0.58 | 1.29 | E_5_2_2_16_4_4, E_136_32_4, E_40_32_4n1 |
| 14B | Q4K_PRIMITIVE=1 named | copy | 0.05 | 0.02 | 0.04 |  |
| 14B | Q4K_PRIMITIVE=1 named | other_amd | 9.97 | 3.15 | 6.94 | r_8_8_16_2_20_4_2_32, r_8_8_16_2_20_2_2_2_32, r_16_320n1 |
| 14B | Q4K_PRIMITIVE=1 named | residual_overhead | 173.07 | 54.63 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 8B | baseline batched | 0 | 63.67 | 0.00 |
| 8B | Q4K_PRIMITIVE=1 batched | 0 | 34.05 | 0.00 |
| 8B | baseline named | 0 | 238.28 | 0.00 |
| 8B | Q4K_PRIMITIVE=1 named | 0 | 226.66 | 0.00 |
| 14B | baseline batched | 0 | 109.78 | 0.00 |
| 14B | Q4K_PRIMITIVE=1 batched | 0 | 62.30 | 0.00 |
| 14B | baseline named | 0 | 274.83 | 0.00 |
| 14B | Q4K_PRIMITIVE=1 named | 0 | 311.48 | 0.00 |

## Top Kernels

### 8B baseline batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 256 | 25.15 | 779.68 |
| batched 142 | 16.64 | 515.90 |
| batched 128 | 12.02 | 372.67 |
| batched 64 | 6.87 | 213.10 |
| batched 32 | 2.34 | 72.57 |
| copy        4 B,     AMD <- AMD | 0.04 | 1.32 |

### 8B Q4K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 286 | 15.85 | 491.23 |
| batched 256 | 7.95 | 246.36 |
| batched 64 | 4.33 | 134.10 |
| batched 128 | 3.93 | 121.83 |
| batched 32 | 1.67 | 51.81 |
| copy        4 B,     AMD <- AMD | 0.04 | 1.36 |

### 8B baseline named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_32_32_4_48_2_2_2_32 | 28.61 | 886.87 |
| r_32_32_4_48_4_2_32 | 21.72 | 673.34 |
| r_32_32_4_16_4_2_32 | 15.29 | 474.07 |
| r_32_32_4_16_4_2_32n1 | 14.77 | 458.00 |
| r_128_32_3_16_4_2_32 | 13.93 | 431.75 |
| r_128_32_3_16_4_2_32n1 | 13.74 | 425.96 |
| r_2_8_128_16_2_2_2_32 | 3.28 | 101.76 |
| r_1187_32_4_16_2_2_2_32n1 | 3.16 | 98.06 |
| r_2_8_128_16_4_2_32 | 1.29 | 39.84 |
| r_1024_16_4_2_32 | 0.83 | 25.87 |
| r_4_2_8_16_4_(start_pos+1) | 0.40 | 12.29 |
| E_2_8_16_4_4 | 0.37 | 11.40 |
| r_16_256n1 | 0.34 | 10.61 |
| r_2_(start_pos+1)_8_4_4_16 | 0.32 | 9.93 |
| r_16_256 | 0.32 | 9.85 |
| r_8_4_(start_pos+1)n1 | 0.32 | 9.83 |
| r_2_8_4_4_16 | 0.24 | 7.39 |
| r_8_4_(start_pos+1) | 0.23 | 7.09 |
| r_32_4_1187 | 0.21 | 6.40 |
| E_(start_pos+1)_8_4 | 0.19 | 5.94 |

### 8B Q4K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_32_32_4_48_2_2_2_32 | 50.61 | 1568.95 |
| q4k_gemv_partial_12288_4096_1 | 5.86 | 181.76 |
| r_1187_32_4_16_2_2_2_32n1 | 4.36 | 135.29 |
| r_2_8_128_16_2_2_2_32 | 4.28 | 132.79 |
| q4k_gemv_partial_4096_4096_1 | 3.99 | 123.65 |
| r_2_8_128_16_4_2_32 | 1.66 | 51.49 |
| q4k_gemv_partial_4096_12288_4 | 1.56 | 48.49 |
| r_1024_16_4_2_32 | 1.09 | 33.80 |
| r_4_2_8_16_4_(start_pos+1) | 0.49 | 15.07 |
| E_2_8_16_4_4 | 0.47 | 14.65 |
| r_16_256n1 | 0.43 | 13.22 |
| r_16_256 | 0.41 | 12.81 |
| r_2_(start_pos+1)_8_4_4_16 | 0.41 | 12.75 |
| r_8_4_(start_pos+1)n1 | 0.41 | 12.66 |
| r_2_8_4_4_16 | 0.31 | 9.46 |
| r_8_4_(start_pos+1) | 0.30 | 9.16 |
| r_32_4_1187 | 0.28 | 8.67 |
| E_(start_pos+1)_8_4 | 0.24 | 7.51 |
| r_8_16_8 | 0.24 | 7.41 |
| E_128_32_3 | 0.24 | 7.36 |

### 14B baseline batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 256 | 40.00 | 1240.11 |
| batched 210 | 35.44 | 1098.65 |
| batched 128 | 18.94 | 587.23 |
| batched 64 | 11.13 | 345.00 |
| batched 32 | 3.75 | 116.18 |
| copy        4 B,     AMD <- AMD | 0.05 | 1.52 |

### 14B Q4K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 370 | 30.82 | 955.44 |
| batched 256 | 14.61 | 452.97 |
| batched 128 | 7.52 | 233.08 |
| batched 64 | 6.80 | 210.69 |
| batched 32 | 3.09 | 95.78 |
| copy        4 B,     AMD <- AMD | 0.05 | 1.54 |

### 14B baseline named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_40_32_4_68_2_2_2_32 | 30.97 | 960.16 |
| r_40_32_4_68_4_2_32 | 23.82 | 738.30 |
| r_136_32_4_20_4_2_32 | 17.19 | 532.79 |
| r_136_32_4_20_4_2_32n1 | 17.01 | 527.20 |
| r_40_32_4_20_4_2_32 | 15.46 | 479.19 |
| r_8_32_4_20_4_2_32 | 14.47 | 448.67 |
| r_40_32_4_20_4_2_32n1 | 14.28 | 442.76 |
| r_1187_32_4_20_2_2_2_32n1 | 3.25 | 100.64 |
| r_8_8_16_2_20_4_2_32 | 2.53 | 78.43 |
| r_8_8_16_2_20_2_2_2_32 | 2.44 | 75.71 |
| E_5_2_2_16_4_4 | 0.33 | 10.21 |
| r_5_2_8_16_4_(start_pos+1) | 0.33 | 10.14 |
| r_16_320n1 | 0.29 | 8.93 |
| r_16_320 | 0.27 | 8.51 |
| r_5_2_4_(start_pos+1)n1 | 0.25 | 7.81 |
| r_5_2_4_(start_pos+1) | 0.20 | 6.12 |
| r_40_(start_pos+1)_16_8 | 0.20 | 6.08 |
| r_32_4_1187 | 0.17 | 5.40 |
| E_5_(start_pos+1)_2_4 | 0.15 | 4.62 |
| r_40_16_8 | 0.14 | 4.46 |

### 14B Q4K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_40_32_4_68_2_2_2_32 | 78.68 | 2439.11 |
| r_8_32_4_20_4_2_32 | 26.07 | 808.08 |
| q4k_gemv_partial_17408_5120_1 | 11.49 | 356.33 |
| q4k_gemv_partial_5120_5120_1 | 5.20 | 161.17 |
| r_1187_32_4_20_2_2_2_32n1 | 4.61 | 142.93 |
| r_8_8_16_2_20_4_2_32 | 4.35 | 134.95 |
| r_8_8_16_2_20_2_2_2_32 | 4.35 | 134.74 |
| q4k_gemv_partial_5120_17408_4 | 3.52 | 109.07 |
| r_5_2_8_16_4_(start_pos+1) | 0.52 | 16.04 |
| r_16_320n1 | 0.51 | 15.87 |
| E_5_2_2_16_4_4 | 0.51 | 15.79 |
| r_16_320 | 0.47 | 14.67 |
| r_5_2_4_(start_pos+1)n1 | 0.45 | 13.89 |
| r_5_2_4_(start_pos+1) | 0.35 | 10.74 |
| r_40_(start_pos+1)_16_8 | 0.33 | 10.32 |
| E_136_32_4 | 0.27 | 8.37 |
| E_5_(start_pos+1)_2_4 | 0.27 | 8.33 |
| r_8_16_8 | 0.26 | 7.97 |
| r_40_16_8 | 0.25 | 7.84 |
| E_40_32_4n1 | 0.25 | 7.71 |

## Decision Gates

- **8B baseline batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **8B Q4K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **8B baseline named** (AMD-kernel basis): fallback/generic Q4_K remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **8B Q4K_PRIMITIVE=1 named** (AMD-kernel basis): fallback/generic Q4_K remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **14B baseline batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B Q4K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B baseline named** (AMD-kernel basis): fallback/generic Q4_K remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **14B Q4K_PRIMITIVE=1 named** (AMD-kernel basis): fallback/generic Q4_K remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
