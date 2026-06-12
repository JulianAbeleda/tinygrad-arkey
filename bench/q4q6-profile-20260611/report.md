# Quant Residual Decode Profile

Scope: Qwen3 8B/14B Q4_K_M AMD DEBUG=2 decode.

Classifier rules are calibrated to Qwen3 8B/14B Q4_K_M AMD DEBUG=2 decode logs. Use this report outside that scope only after adding boundary tests for the new kernel signatures.

Steady-state rows drop the first 1 benchmark token(s). `batched` rows are the throughput truth. `named` rows are attribution-only: they disable graph batching via JIT_BATCH_SIZE=1, so use AMD-kernel percentages rather than named wall time for bottleneck decisions.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | 31 | 58.77 | 17.04 | 16.32 | 0.73 | 4.28 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | 31 | 5.42 | 184.69 | 33.94 | 150.75 | 81.63 |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 31 | 28.79 | 34.74 | 34.06 | 0.69 | 1.98 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 31 | 4.01 | 249.60 | 78.11 | 171.49 | 68.71 |

## Parse Health

| model | mode | lines | tokens | AMD lines | ignored lines | non-AMD DEBUG lines | trailing AMD lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | 3185 | 32 | 2613 | 360 | 180 | 0 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | 30523 | 32 | 29927 | 384 | 180 | 0 |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 3508 | 32 | 2879 | 397 | 200 | 0 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 33859 | 32 | 33207 | 420 | 200 | 0 |

## Buckets

| model | mode | bucket | ms/tok | % wall | % AMD kernel | top kernels |
| --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | copy | 0.04 | 0.26 | 0.28 | copy        4 B,     AMD <- AMD |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | other_amd | 16.27 | 95.46 | 99.72 | batched 322, batched 256, batched 128 |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | residual_overhead | 0.73 | 4.28 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 11.21 | 6.07 | 33.02 | q4k_gemv_partial_12288_4096_1, q4k_gemv_partial_4096_4096_1, q4k_gemv_partial_4096_12288_4 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 5.94 | 3.21 | 17.50 | q6k_gemv_partial_4096_12288_1 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 0.85 | 0.46 | 2.52 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 4.87 | 2.64 | 14.34 | r_1187_32_4_16_2_2_2_32n1, r_1024_16_4_2_32 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 1.83 | 0.99 | 5.40 | r_4_2_8_16_4_(start_pos+1), r_2_(start_pos+1)_8_4_4_16, r_8_4_(start_pos+1)n1 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 2.69 | 1.45 | 7.91 | E_2_8_16_4_4, r_16_256n1, r_16_256 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.05 | 0.02 | 0.13 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 6.51 | 3.52 | 19.18 | r_2_8_128_16_2_2_2_32, r_2_8_128_16_4_2_32, r_2_8_4_4_16 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 150.75 | 81.63 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | copy | 0.04 | 0.11 | 0.12 | copy        4 B,     AMD <- AMD |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | other_amd | 34.02 | 97.91 | 99.88 | batched 410, batched 256, batched 128 |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | residual_overhead | 0.69 | 1.98 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 20.72 | 8.30 | 26.53 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 9.94 | 3.98 | 12.73 | q6k_gemv_partial_5120_17408_1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 13.90 | 5.57 | 17.80 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 18.85 | 7.55 | 24.13 | r_8_32_4_20_4_2_32, r_1187_32_4_20_2_2_2_32n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 1.99 | 0.80 | 2.55 | r_5_2_8_16_4_(start_pos+1), r_5_2_4_(start_pos+1)n1, r_5_2_4_(start_pos+1) |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 2.27 | 0.91 | 2.90 | E_5_2_2_16_4_4, r_32_4_1187, E_136_32_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.05 | 0.02 | 0.07 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 10.38 | 4.16 | 13.29 | r_8_8_16_2_20_2_2_2_32, r_8_8_16_2_20_4_2_32, r_16_320n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 171.49 | 68.71 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | 0 | 16.88 | 0.00 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 184.33 | 0.00 |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 0 | 34.73 | 0.00 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 248.97 | 0.00 |

## Top Kernels

### 8B Q4K+Q6K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 322 | 8.45 | 261.86 |
| batched 256 | 3.94 | 122.25 |
| batched 128 | 1.99 | 61.66 |
| batched 64 | 1.35 | 41.89 |
| batched 32 | 0.54 | 16.73 |
| copy        4 B,     AMD <- AMD | 0.04 | 1.39 |

### 8B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q6k_gemv_partial_4096_12288_1 | 5.94 | 184.06 |
| q4k_gemv_partial_12288_4096_1 | 5.71 | 177.16 |
| r_1187_32_4_16_2_2_2_32n1 | 4.33 | 134.25 |
| r_2_8_128_16_2_2_2_32 | 4.29 | 133.14 |
| q4k_gemv_partial_4096_4096_1 | 3.95 | 122.30 |
| r_2_8_128_16_4_2_32 | 1.66 | 51.47 |
| q4k_gemv_partial_4096_12288_4 | 1.55 | 47.90 |
| r_1024_16_4_2_32 | 1.07 | 33.31 |
| r_4_2_8_16_4_(start_pos+1) | 0.48 | 14.99 |
| E_2_8_16_4_4 | 0.46 | 14.39 |
| r_16_256n1 | 0.42 | 13.15 |
| r_16_256 | 0.41 | 12.75 |
| r_2_(start_pos+1)_8_4_4_16 | 0.41 | 12.61 |
| r_8_4_(start_pos+1)n1 | 0.41 | 12.60 |
| r_2_8_4_4_16 | 0.30 | 9.44 |
| r_8_4_(start_pos+1) | 0.29 | 9.14 |
| r_32_4_1187 | 0.28 | 8.54 |
| E_(start_pos+1)_8_4 | 0.24 | 7.42 |
| r_8_16_8 | 0.24 | 7.42 |
| E_128_32_3 | 0.24 | 7.34 |

### 14B Q4K+Q6K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 410 | 17.41 | 539.66 |
| batched 256 | 8.60 | 266.72 |
| batched 128 | 4.17 | 129.24 |
| batched 64 | 2.60 | 80.51 |
| batched 32 | 1.24 | 38.40 |
| copy        4 B,     AMD <- AMD | 0.04 | 1.22 |

### 14B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_8_32_4_20_4_2_32 | 27.05 | 838.57 |
| q4k_gemv_partial_17408_5120_1 | 11.76 | 364.62 |
| q6k_gemv_partial_5120_17408_1 | 9.94 | 308.24 |
| q4k_gemv_partial_5120_5120_1 | 5.42 | 167.87 |
| r_1187_32_4_20_2_2_2_32n1 | 5.33 | 165.25 |
| r_8_8_16_2_20_2_2_2_32 | 4.56 | 141.49 |
| r_8_8_16_2_20_4_2_32 | 4.49 | 139.21 |
| q4k_gemv_partial_5120_17408_4 | 3.55 | 109.93 |
| E_5_2_2_16_4_4 | 0.54 | 16.70 |
| r_5_2_8_16_4_(start_pos+1) | 0.54 | 16.69 |
| r_16_320n1 | 0.53 | 16.35 |
| r_16_320 | 0.49 | 15.29 |
| r_5_2_4_(start_pos+1)n1 | 0.46 | 14.32 |
| r_5_2_4_(start_pos+1) | 0.36 | 11.12 |
| r_40_(start_pos+1)_16_8 | 0.35 | 10.81 |
| r_32_4_1187 | 0.29 | 8.84 |
| E_5_(start_pos+1)_2_4 | 0.28 | 8.74 |
| E_136_32_4 | 0.28 | 8.55 |
| r_8_16_8 | 0.27 | 8.27 |
| r_40_16_8 | 0.26 | 8.18 |

## Decision Gates

- **8B Q4K+Q6K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **8B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **14B Q4K+Q6K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive reductions >15% of profile basis: fuse/avoid partial reduction; fallback/generic dense quant remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
