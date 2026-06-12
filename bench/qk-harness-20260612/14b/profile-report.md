# Quant Residual Decode Profile

Scope: Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode.

Classifier rules are calibrated to Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode logs. Use this report outside that scope only after adding boundary tests for the new kernel signatures.

Steady-state rows drop the first 1 benchmark token(s). `batched` rows are the throughput truth. `named` rows are attribution-only: they disable graph batching via JIT_BATCH_SIZE=1, so use AMD-kernel percentages rather than named wall time for bottleneck decisions.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 7 | 23.28 | 43.05 | 42.28 | 0.77 | 1.79 |
| 14B | QK_GENERATED_POLICY batched | 7 | 41.85 | 23.91 | 23.19 | 0.72 | 3.02 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 7 | 4.09 | 244.51 | 77.16 | 167.35 | 68.44 |
| 14B | QK_GENERATED_POLICY named | 7 | 4.40 | 227.09 | 41.66 | 185.44 | 81.66 |

## Parse Health

| model | mode | lines | tokens | AMD lines | ignored lines | non-AMD DEBUG lines | trailing AMD lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 3340 | 8 | 2735 | 397 | 200 | 0 |
| 14B | QK_GENERATED_POLICY batched | 3660 | 8 | 2895 | 477 | 280 | 0 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 12427 | 8 | 11823 | 396 | 200 | 0 |
| 14B | QK_GENERATED_POLICY named | 13387 | 8 | 12623 | 476 | 280 | 0 |

## Buckets

| model | mode | bucket | ms/tok | % wall | % AMD kernel | top kernels |
| --- | --- | --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | copy | 0.04 | 0.09 | 0.09 | copy        4 B,     AMD <- AMD |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | other_amd | 42.24 | 98.12 | 99.91 | batched 410, batched 256, batched 128 |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | residual_overhead | 0.77 | 1.79 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | copy | 0.04 | 0.16 | 0.16 | copy        4 B,     AMD <- AMD |
| 14B | QK_GENERATED_POLICY batched | other_amd | 23.15 | 96.82 | 99.84 | batched 490, batched 256, batched 128 |
| 14B | QK_GENERATED_POLICY batched | residual_overhead | 0.72 | 3.02 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 20.56 | 8.41 | 26.65 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 9.89 | 4.04 | 12.81 | q6k_gemv_partial_5120_17408_1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 13.84 | 5.66 | 17.93 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 18.72 | 7.66 | 24.26 | r_8_32_4_20_4_2_32, r_1187_32_4_20_2_2_2_32n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 1.56 | 0.64 | 2.02 | r_40_(start_pos+1)_16_8, r_5_2_4_(start_pos+1)n1, r_5_2_8_16_4_(start_pos+1) |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 2.24 | 0.92 | 2.91 | E_5_2_2_16_4_4, r_32_4_1187, E_136_32_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.06 | 0.02 | 0.07 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 10.30 | 4.21 | 13.35 | r_8_8_16_2_20_2_2_2_32, r_8_8_16_2_20_4_2_32, r_16_320n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 167.35 | 68.44 | 0.00 |  |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_gemv | 21.64 | 9.53 | 51.95 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_2 |
| 14B | QK_GENERATED_POLICY named | q6k_primitive_gemv | 8.61 | 3.79 | 20.66 | q6k_gemv_partial_5120_17408_2, q6k_gemv_partial_1024_5120_2 |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_reduction | 1.16 | 0.51 | 2.80 |  |
| 14B | QK_GENERATED_POLICY named | fallback_quant_fused | 5.21 | 2.29 | 12.51 | r_1187_32_4_20_2_2_2_32n1 |
| 14B | QK_GENERATED_POLICY named | attention_misc | 1.56 | 0.69 | 3.74 | r_40_(start_pos+1)_16_8, r_5_2_4_(start_pos+1)n1, r_5_2_8_16_4_(start_pos+1) |
| 14B | QK_GENERATED_POLICY named | norm_sampling_misc | 2.13 | 0.94 | 5.11 | E_5_2_2_16_4_4, r_32_4_1187, E_136_32_4 |
| 14B | QK_GENERATED_POLICY named | copy | 0.06 | 0.03 | 0.15 |  |
| 14B | QK_GENERATED_POLICY named | other_amd | 1.28 | 0.57 | 3.08 | r_16_320n1, r_16_320, r_40_32_4_2 |
| 14B | QK_GENERATED_POLICY named | residual_overhead | 185.44 | 81.66 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 0 | 41.67 | 0.00 |
| 14B | QK_GENERATED_POLICY batched | 0 | 23.63 | 0.00 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 244.53 | 0.00 |
| 14B | QK_GENERATED_POLICY named | 0 | 226.77 | 0.00 |

## Top Kernels

### 14B Q4K+Q6K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 410 | 20.96 | 146.74 |
| batched 256 | 10.77 | 75.40 |
| batched 128 | 5.24 | 36.67 |
| batched 64 | 3.80 | 26.63 |
| batched 32 | 1.46 | 10.23 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.27 |

### 14B QK_GENERATED_POLICY batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 490 | 13.38 | 93.64 |
| batched 256 | 4.86 | 33.99 |
| batched 128 | 2.63 | 18.38 |
| batched 64 | 1.64 | 11.49 |
| batched 32 | 0.65 | 4.56 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.26 |

### 14B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_8_32_4_20_4_2_32 | 26.89 | 188.23 |
| q4k_gemv_partial_17408_5120_1 | 11.67 | 81.72 |
| q6k_gemv_partial_5120_17408_1 | 9.89 | 69.21 |
| q4k_gemv_partial_5120_5120_1 | 5.35 | 37.47 |
| r_1187_32_4_20_2_2_2_32n1 | 5.30 | 37.07 |
| r_8_8_16_2_20_2_2_2_32 | 4.54 | 31.75 |
| r_8_8_16_2_20_4_2_32 | 4.45 | 31.12 |
| q4k_gemv_partial_5120_17408_4 | 3.54 | 24.76 |
| E_5_2_2_16_4_4 | 0.53 | 3.70 |
| r_16_320n1 | 0.53 | 3.68 |
| r_16_320 | 0.49 | 3.42 |
| r_40_(start_pos+1)_16_8 | 0.34 | 2.40 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.31 |
| r_5_2_8_16_4_(start_pos+1) | 0.33 | 2.28 |
| r_5_2_4_(start_pos+1) | 0.29 | 2.01 |
| r_32_4_1187 | 0.28 | 1.97 |
| E_136_32_4 | 0.27 | 1.92 |
| E_5_(start_pos+1)_2_4 | 0.27 | 1.91 |
| r_8_16_8 | 0.27 | 1.86 |
| r_40_16_8 | 0.26 | 1.83 |

### 14B QK_GENERATED_POLICY named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q4k_gemv_partial_17408_5120_1 | 11.38 | 79.65 |
| q6k_gemv_partial_5120_17408_2 | 7.03 | 49.21 |
| q4k_gemv_partial_5120_5120_1 | 5.34 | 37.39 |
| r_1187_32_4_20_2_2_2_32n1 | 5.21 | 36.48 |
| q4k_gemv_partial_5120_17408_2 | 3.42 | 23.96 |
| q6k_gemv_partial_1024_5120_2 | 1.58 | 11.03 |
| q4k_gemv_partial_1024_5120_4 | 1.50 | 10.50 |
| E_5_2_2_16_4_4 | 0.54 | 3.76 |
| r_16_320n1 | 0.53 | 3.68 |
| r_16_320 | 0.49 | 3.43 |
| r_40_(start_pos+1)_16_8 | 0.34 | 2.39 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.31 |
| r_5_2_8_16_4_(start_pos+1) | 0.32 | 2.27 |
| r_5_2_4_(start_pos+1) | 0.29 | 2.01 |
| r_32_4_1187 | 0.28 | 1.98 |
| E_136_32_4 | 0.27 | 1.92 |
| E_5_(start_pos+1)_2_4 | 0.27 | 1.91 |
| r_40_32_4_2 | 0.27 | 1.88 |
| r_8_16_8 | 0.27 | 1.86 |
| r_40_16_8 | 0.26 | 1.83 |

## Decision Gates

- **14B Q4K+Q6K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B QK_GENERATED_POLICY batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive reductions >15% of profile basis: fuse/avoid partial reduction; fallback/generic dense quant remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **14B QK_GENERATED_POLICY named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
