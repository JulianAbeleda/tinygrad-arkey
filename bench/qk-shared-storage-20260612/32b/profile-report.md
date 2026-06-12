# Quant Residual Decode Profile

Scope: Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode.

Classifier rules are calibrated to Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode logs. Use this report outside that scope only after adding boundary tests for the new kernel signatures.

Steady-state rows drop the first 1 benchmark token(s). `batched` rows are the throughput truth. `named` rows are attribution-only: they disable graph batching via JIT_BATCH_SIZE=1, so use AMD-kernel percentages rather than named wall time for bottleneck decisions.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | 7 | 11.76 | 85.01 | 84.12 | 0.89 | 1.05 |
| 32B | QK_GENERATED_POLICY batched | 7 | 17.99 | 55.63 | 54.71 | 0.92 | 1.66 |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | 7 | 2.25 | 444.25 | 157.39 | 286.85 | 64.57 |
| 32B | QK_GENERATED_POLICY named | 7 | 2.48 | 403.58 | 97.45 | 306.12 | 75.85 |

## Parse Health

| model | mode | lines | tokens | AMD lines | ignored lines | non-AMD DEBUG lines | trailing AMD lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | 4309 | 8 | 4007 | 294 | 0 | 0 |
| 32B | QK_GENERATED_POLICY batched | 4437 | 8 | 4135 | 294 | 0 | 0 |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | 18811 | 8 | 18511 | 292 | 0 | 0 |
| 32B | QK_GENERATED_POLICY named | 19963 | 8 | 19663 | 292 | 0 | 0 |

## Buckets

| model | mode | bucket | ms/tok | % wall | % AMD kernel | top kernels |
| --- | --- | --- | --- | --- | --- | --- |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | copy | 0.04 | 0.04 | 0.04 | copy        4 B,     AMD <- AMD |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | other_amd | 84.08 | 98.91 | 99.96 | batched 512, batched 426, batched 256 |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | residual_overhead | 0.89 | 1.05 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 32B | QK_GENERATED_POLICY batched | copy | 0.04 | 0.07 | 0.07 | copy        4 B,     AMD <- AMD |
| 32B | QK_GENERATED_POLICY batched | other_amd | 54.67 | 98.27 | 99.93 | batched 554, batched 512, batched 256 |
| 32B | QK_GENERATED_POLICY batched | residual_overhead | 0.92 | 1.66 | 0.00 |  |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 63.11 | 14.21 | 40.09 | q4k_gemv_partial_25600_5120_1, q4k_gemv_partial_5120_25600_4, q4k_gemv_partial_5120_8192_1 |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 22.47 | 5.06 | 14.28 | q6k_gemv_partial_5120_25600_1 |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 22.26 | 5.01 | 14.14 |  |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 26.87 | 6.05 | 17.07 | r_8_32_4_20_4_2_32, r_1187_32_4_20_2_2_2_32n1 |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 2.66 | 0.60 | 1.69 | r_4_(start_pos+1)_8_4_4_16, r_16_4_(start_pos+1)n1, r_8_2_8_16_4_(start_pos+1) |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 3.30 | 0.74 | 2.10 | E_2_2_8_16_4_4, E_200_32_4 |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.06 | 0.01 | 0.04 |  |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 16.67 | 3.75 | 10.59 | r_8_8_16_2_20_2_2_2_32, r_8_8_16_2_20_4_2_32, r_16_320n1 |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 286.85 | 64.57 | 0.00 |  |
| 32B | QK_GENERATED_POLICY named | q4k_primitive_gemv | 64.15 | 15.90 | 65.83 | q4k_gemv_partial_25600_5120_1, q4k_gemv_partial_5120_25600_2, q4k_gemv_partial_5120_8192_1 |
| 32B | QK_GENERATED_POLICY named | q6k_primitive_gemv | 18.29 | 4.53 | 18.77 | q6k_gemv_partial_5120_25600_2, q6k_gemv_partial_1024_5120_2 |
| 32B | QK_GENERATED_POLICY named | q4k_primitive_reduction | 1.97 | 0.49 | 2.02 |  |
| 32B | QK_GENERATED_POLICY named | fallback_quant_fused | 5.33 | 1.32 | 5.46 | r_1187_32_4_20_2_2_2_32n1 |
| 32B | QK_GENERATED_POLICY named | attention_misc | 2.62 | 0.65 | 2.68 | r_4_(start_pos+1)_8_4_4_16, r_16_4_(start_pos+1)n1, r_8_2_8_16_4_(start_pos+1) |
| 32B | QK_GENERATED_POLICY named | norm_sampling_misc | 3.03 | 0.75 | 3.11 | E_2_2_8_16_4_4, E_200_32_4, E_40_32_4n1 |
| 32B | QK_GENERATED_POLICY named | copy | 0.06 | 0.01 | 0.06 |  |
| 32B | QK_GENERATED_POLICY named | other_amd | 2.01 | 0.50 | 2.06 | r_16_320n1, r_16_320, r_4_8_4_4_16 |
| 32B | QK_GENERATED_POLICY named | residual_overhead | 306.12 | 75.85 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 32B | Q4K+Q6K_PRIMITIVE=1 batched | 0 | 84.82 | 0.00 |
| 32B | QK_GENERATED_POLICY batched | 0 | 55.01 | 0.00 |
| 32B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 444.03 | 0.00 |
| 32B | QK_GENERATED_POLICY named | 0 | 403.22 | 0.00 |

## Top Kernels

### 32B Q4K+Q6K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 512 | 28.28 | 197.93 |
| batched 426 | 28.18 | 197.26 |
| batched 256 | 14.25 | 99.74 |
| batched 128 | 7.38 | 51.68 |
| batched 64 | 4.15 | 29.05 |
| batched 32 | 1.84 | 12.89 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.26 |

### 32B QK_GENERATED_POLICY batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 554 | 21.88 | 153.17 |
| batched 512 | 16.51 | 115.58 |
| batched 256 | 8.51 | 59.55 |
| batched 128 | 4.53 | 31.68 |
| batched 64 | 2.26 | 15.81 |
| batched 32 | 0.98 | 6.87 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.28 |

### 32B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_8_32_4_20_4_2_32 | 43.21 | 302.49 |
| q4k_gemv_partial_25600_5120_1 | 42.19 | 295.30 |
| q6k_gemv_partial_5120_25600_1 | 22.47 | 157.29 |
| q4k_gemv_partial_5120_25600_4 | 8.01 | 56.06 |
| r_8_8_16_2_20_2_2_2_32 | 7.25 | 50.76 |
| r_8_8_16_2_20_4_2_32 | 7.20 | 50.38 |
| q4k_gemv_partial_5120_8192_1 | 6.87 | 48.10 |
| q4k_gemv_partial_8192_5120_1 | 6.04 | 42.29 |
| r_1187_32_4_20_2_2_2_32n1 | 5.32 | 37.25 |
| E_2_2_8_16_4_4 | 0.87 | 6.12 |
| r_16_320n1 | 0.84 | 5.86 |
| r_16_320 | 0.78 | 5.47 |
| r_4_(start_pos+1)_8_4_4_16 | 0.72 | 5.07 |
| r_4_8_4_4_16 | 0.55 | 3.86 |
| r_16_4_(start_pos+1)n1 | 0.54 | 3.78 |
| r_8_2_8_16_4_(start_pos+1) | 0.52 | 3.66 |
| E_200_32_4 | 0.44 | 3.10 |
| r_16_4_(start_pos+1) | 0.44 | 3.08 |
| E_(start_pos+1)_16_4 | 0.43 | 3.02 |
| r_8_16_8 | 0.43 | 2.99 |

### 32B QK_GENERATED_POLICY named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q4k_gemv_partial_25600_5120_1 | 41.83 | 292.83 |
| q6k_gemv_partial_5120_25600_2 | 15.79 | 110.54 |
| q4k_gemv_partial_5120_25600_2 | 7.93 | 55.51 |
| q4k_gemv_partial_5120_8192_1 | 6.65 | 46.55 |
| q4k_gemv_partial_8192_5120_1 | 5.40 | 37.81 |
| r_1187_32_4_20_2_2_2_32n1 | 5.33 | 37.28 |
| q6k_gemv_partial_1024_5120_2 | 2.50 | 17.51 |
| q4k_gemv_partial_1024_5120_4 | 2.34 | 16.35 |
| E_2_2_8_16_4_4 | 0.85 | 5.94 |
| r_16_320n1 | 0.83 | 5.82 |
| r_16_320 | 0.77 | 5.40 |
| r_4_(start_pos+1)_8_4_4_16 | 0.71 | 4.96 |
| r_4_8_4_4_16 | 0.57 | 3.98 |
| r_16_4_(start_pos+1)n1 | 0.53 | 3.73 |
| r_8_2_8_16_4_(start_pos+1) | 0.51 | 3.60 |
| E_200_32_4 | 0.44 | 3.08 |
| r_16_4_(start_pos+1) | 0.43 | 3.04 |
| E_(start_pos+1)_16_4 | 0.42 | 2.97 |
| r_8_16_8 | 0.42 | 2.97 |
| E_40_32_4n1 | 0.40 | 2.82 |

## Decision Gates

- **32B Q4K+Q6K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **32B QK_GENERATED_POLICY batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **32B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **32B QK_GENERATED_POLICY named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
