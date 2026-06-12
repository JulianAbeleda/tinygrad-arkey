# Quant Residual Decode Profile

Scope: Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode.

Classifier rules are calibrated to Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode logs. Use this report outside that scope only after adding boundary tests for the new kernel signatures.

Steady-state rows drop the first 1 benchmark token(s). `batched` rows are the throughput truth. `named` rows are attribution-only: they disable graph batching via JIT_BATCH_SIZE=1, so use AMD-kernel percentages rather than named wall time for bottleneck decisions.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | 7 | 52.18 | 19.17 | 18.41 | 0.76 | 3.94 |
| 8B | QK_GENERATED_POLICY batched | 7 | 55.50 | 18.02 | 17.32 | 0.69 | 3.84 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | 7 | 5.40 | 185.33 | 33.40 | 151.93 | 81.98 |
| 8B | QK_GENERATED_POLICY named | 7 | 5.58 | 179.31 | 31.19 | 148.12 | 82.61 |

## Parse Health

| model | mode | lines | tokens | AMD lines | ignored lines | non-AMD DEBUG lines | trailing AMD lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | 2477 | 8 | 2289 | 180 | 0 | 0 |
| 8B | QK_GENERATED_POLICY batched | 2477 | 8 | 2289 | 180 | 0 | 0 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | 10663 | 8 | 10475 | 180 | 0 | 0 |
| 8B | QK_GENERATED_POLICY named | 10663 | 8 | 10475 | 180 | 0 | 0 |

## Buckets

| model | mode | bucket | ms/tok | % wall | % AMD kernel | top kernels |
| --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | copy | 0.04 | 0.20 | 0.21 | copy        4 B,     AMD <- AMD |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | other_amd | 18.37 | 95.86 | 99.79 | batched 322, batched 256, batched 128 |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | residual_overhead | 0.76 | 3.94 | 0.00 |  |
| 8B | QK_GENERATED_POLICY batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 8B | QK_GENERATED_POLICY batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 8B | QK_GENERATED_POLICY batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 8B | QK_GENERATED_POLICY batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 8B | QK_GENERATED_POLICY batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | QK_GENERATED_POLICY batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 8B | QK_GENERATED_POLICY batched | copy | 0.04 | 0.20 | 0.21 | copy        4 B,     AMD <- AMD |
| 8B | QK_GENERATED_POLICY batched | other_amd | 17.29 | 95.95 | 99.79 | batched 322, batched 256, batched 128 |
| 8B | QK_GENERATED_POLICY batched | residual_overhead | 0.69 | 3.84 | 0.00 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 11.15 | 6.02 | 33.38 | q4k_gemv_partial_12288_4096_1, q4k_gemv_partial_4096_4096_1, q4k_gemv_partial_4096_12288_4 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 5.92 | 3.19 | 17.73 | q6k_gemv_partial_4096_12288_1 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 0.84 | 0.46 | 2.53 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 4.82 | 2.60 | 14.44 | r_1187_32_4_16_2_2_2_32n1, r_1024_16_4_2_32 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 1.46 | 0.79 | 4.37 | r_2_(start_pos+1)_8_4_4_16, r_8_4_(start_pos+1)n1, r_4_2_8_16_4_(start_pos+1) |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 2.65 | 1.43 | 7.93 | E_2_8_16_4_4, r_16_256n1, r_16_256 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.06 | 0.03 | 0.17 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 6.50 | 3.51 | 19.47 | r_2_8_128_16_2_2_2_32, r_2_8_128_16_4_2_32, r_2_8_4_4_16 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 151.93 | 81.98 | 0.00 |  |
| 8B | QK_GENERATED_POLICY named | q4k_primitive_gemv | 11.11 | 6.20 | 35.63 | q4k_gemv_partial_12288_4096_1, q4k_gemv_partial_4096_4096_1, q4k_gemv_partial_4096_12288_4 |
| 8B | QK_GENERATED_POLICY named | q6k_primitive_gemv | 3.80 | 2.12 | 12.18 | q6k_gemv_partial_4096_12288_2 |
| 8B | QK_GENERATED_POLICY named | q4k_primitive_reduction | 0.84 | 0.47 | 2.71 |  |
| 8B | QK_GENERATED_POLICY named | fallback_quant_fused | 4.78 | 2.67 | 15.33 | r_1187_32_4_16_2_2_2_32n1, r_1024_16_4_2_32 |
| 8B | QK_GENERATED_POLICY named | attention_misc | 1.46 | 0.81 | 4.68 | r_2_(start_pos+1)_8_4_4_16, r_8_4_(start_pos+1)n1, r_4_2_8_16_4_(start_pos+1) |
| 8B | QK_GENERATED_POLICY named | norm_sampling_misc | 2.54 | 1.42 | 8.14 | E_2_8_16_4_4, r_16_256n1, r_16_256 |
| 8B | QK_GENERATED_POLICY named | copy | 0.06 | 0.03 | 0.18 |  |
| 8B | QK_GENERATED_POLICY named | other_amd | 6.60 | 3.68 | 21.15 | r_2_8_128_16_2_2_2_32, r_2_8_128_16_4_2_32, r_2_8_4_4_16 |
| 8B | QK_GENERATED_POLICY named | residual_overhead | 148.12 | 82.61 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 batched | 0 | 19.18 | 0.00 |
| 8B | QK_GENERATED_POLICY batched | 0 | 18.01 | 0.00 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 184.45 | 0.00 |
| 8B | QK_GENERATED_POLICY named | 0 | 179.06 | 0.00 |

## Top Kernels

### 8B Q4K+Q6K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 322 | 9.53 | 66.72 |
| batched 256 | 4.34 | 30.40 |
| batched 128 | 2.29 | 16.06 |
| batched 64 | 1.58 | 11.08 |
| batched 32 | 0.62 | 4.34 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.26 |

### 8B QK_GENERATED_POLICY batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 322 | 9.01 | 63.07 |
| batched 256 | 4.16 | 29.10 |
| batched 128 | 2.16 | 15.13 |
| batched 64 | 1.40 | 9.81 |
| batched 32 | 0.56 | 3.91 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.26 |

### 8B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q6k_gemv_partial_4096_12288_1 | 5.92 | 41.44 |
| q4k_gemv_partial_12288_4096_1 | 5.69 | 39.85 |
| r_1187_32_4_16_2_2_2_32n1 | 4.29 | 30.02 |
| r_2_8_128_16_2_2_2_32 | 4.28 | 29.98 |
| q4k_gemv_partial_4096_4096_1 | 3.93 | 27.51 |
| r_2_8_128_16_4_2_32 | 1.67 | 11.70 |
| q4k_gemv_partial_4096_12288_4 | 1.52 | 10.67 |
| r_1024_16_4_2_32 | 1.06 | 7.45 |
| E_2_8_16_4_4 | 0.45 | 3.17 |
| r_16_256n1 | 0.42 | 2.92 |
| r_16_256 | 0.40 | 2.83 |
| r_2_(start_pos+1)_8_4_4_16 | 0.40 | 2.80 |
| r_2_8_4_4_16 | 0.30 | 2.10 |
| r_8_4_(start_pos+1)n1 | 0.29 | 2.06 |
| r_4_2_8_16_4_(start_pos+1) | 0.29 | 2.06 |
| r_32_4_1187 | 0.27 | 1.91 |
| r_8_16_8 | 0.24 | 1.67 |
| r_8_4_(start_pos+1) | 0.24 | 1.67 |
| E_128_32_3 | 0.24 | 1.65 |
| E_(start_pos+1)_8_4 | 0.23 | 1.63 |

### 8B QK_GENERATED_POLICY named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q4k_gemv_partial_12288_4096_1 | 5.69 | 39.82 |
| r_2_8_128_16_2_2_2_32 | 4.28 | 29.96 |
| r_1187_32_4_16_2_2_2_32n1 | 4.25 | 29.73 |
| q4k_gemv_partial_4096_4096_1 | 3.91 | 27.36 |
| q6k_gemv_partial_4096_12288_2 | 3.80 | 26.60 |
| r_2_8_128_16_4_2_32 | 1.66 | 11.61 |
| q4k_gemv_partial_4096_12288_4 | 1.52 | 10.61 |
| r_1024_16_4_2_32 | 1.07 | 7.46 |
| E_2_8_16_4_4 | 0.45 | 3.17 |
| r_16_256n1 | 0.42 | 2.93 |
| r_16_256 | 0.40 | 2.83 |
| r_2_(start_pos+1)_8_4_4_16 | 0.40 | 2.79 |
| r_2_8_4_4_16 | 0.30 | 2.10 |
| r_8_4_(start_pos+1)n1 | 0.29 | 2.06 |
| r_4_2_8_16_4_(start_pos+1) | 0.29 | 2.06 |
| r_32_4_1187 | 0.27 | 1.90 |
| r_8_16_8 | 0.24 | 1.67 |
| r_8_4_(start_pos+1) | 0.24 | 1.67 |
| E_128_32_3 | 0.24 | 1.65 |
| r_32_4_1187n1 | 0.23 | 1.64 |

## Decision Gates

- **8B Q4K+Q6K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **8B QK_GENERATED_POLICY batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **8B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **8B QK_GENERATED_POLICY named** (AMD-kernel basis): named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
