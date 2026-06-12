# Quant Residual Decode Profile

Scope: Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode.

Classifier rules are calibrated to Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode logs. Use this report outside that scope only after adding boundary tests for the new kernel signatures.

Steady-state rows drop the first 1 benchmark token(s). `batched` rows are the throughput truth. `named` rows are attribution-only: they disable graph batching via JIT_BATCH_SIZE=1, so use AMD-kernel percentages rather than named wall time for bottleneck decisions.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 7 | 24.08 | 41.53 | 40.82 | 0.71 | 1.70 |
| 14B | QK_GENERATED_POLICY batched | 7 | 42.29 | 23.65 | 22.94 | 0.71 | 3.00 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 7 | 4.08 | 245.05 | 76.75 | 168.30 | 68.68 |
| 14B | QK_GENERATED_POLICY named | 7 | 2.67 | 532.49 | 42.82 | 489.68 | 91.96 |

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
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | other_amd | 40.79 | 98.21 | 99.91 | batched 410, batched 256, batched 128 |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | residual_overhead | 0.71 | 1.70 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | copy | 0.04 | 0.16 | 0.16 | copy        4 B,     AMD <- AMD |
| 14B | QK_GENERATED_POLICY batched | other_amd | 22.90 | 96.84 | 99.84 | batched 490, batched 256, batched 128 |
| 14B | QK_GENERATED_POLICY batched | residual_overhead | 0.71 | 3.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 20.46 | 8.35 | 26.66 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 9.85 | 4.02 | 12.83 | q6k_gemv_partial_5120_17408_1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 13.77 | 5.62 | 17.94 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 18.62 | 7.60 | 24.26 | r_8_32_4_20_4_2_32, r_1187_32_4_20_2_2_2_32n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 1.54 | 0.63 | 2.01 | r_40_(start_pos+1)_16_8, r_5_2_4_(start_pos+1)n1, r_5_2_8_16_4_(start_pos+1) |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 2.21 | 0.90 | 2.88 | E_5_2_2_16_4_4, E_136_32_4, r_32_4_1187 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.07 | 0.03 | 0.09 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 10.23 | 4.18 | 13.33 | r_8_8_16_2_20_2_2_2_32, r_8_8_16_2_20_4_2_32, r_16_320n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 168.30 | 68.68 | 0.00 |  |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_gemv | 22.10 | 4.15 | 51.61 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_2 |
| 14B | QK_GENERATED_POLICY named | q6k_primitive_gemv | 8.72 | 1.64 | 20.37 | q6k_gemv_partial_5120_17408_2, q6k_gemv_partial_1024_5120_2 |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_reduction | 1.16 | 0.22 | 2.70 |  |
| 14B | QK_GENERATED_POLICY named | fallback_quant_fused | 5.40 | 1.01 | 12.62 | r_1187_32_4_20_2_2_2_32n1 |
| 14B | QK_GENERATED_POLICY named | attention_misc | 1.58 | 0.30 | 3.70 | r_40_(start_pos+1)_16_8, r_5_2_4_(start_pos+1)n1, r_5_2_8_16_4_(start_pos+1) |
| 14B | QK_GENERATED_POLICY named | norm_sampling_misc | 2.20 | 0.41 | 5.14 | E_5_2_2_16_4_4, r_32_4_1187, E_40_32_4n1 |
| 14B | QK_GENERATED_POLICY named | copy | 0.34 | 0.06 | 0.80 | copy        4 B,     AMD <- AMD |
| 14B | QK_GENERATED_POLICY named | other_amd | 1.31 | 0.25 | 3.06 | r_16_320n1, r_16_320, r_8_16_8 |
| 14B | QK_GENERATED_POLICY named | residual_overhead | 489.68 | 91.96 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 0 | 41.53 | 0.00 |
| 14B | QK_GENERATED_POLICY batched | 0 | 23.65 | 0.00 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 244.71 | 0.00 |
| 14B | QK_GENERATED_POLICY named | 0 | 623.18 | 0.00 |

## Top Kernels

### 14B Q4K+Q6K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 410 | 20.71 | 144.99 |
| batched 256 | 10.52 | 73.64 |
| batched 128 | 5.04 | 35.25 |
| batched 64 | 3.06 | 21.43 |
| batched 32 | 1.46 | 10.21 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.26 |

### 14B QK_GENERATED_POLICY batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 490 | 13.39 | 93.71 |
| batched 256 | 4.85 | 33.96 |
| batched 128 | 2.62 | 18.35 |
| batched 64 | 1.39 | 9.75 |
| batched 32 | 0.65 | 4.56 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.26 |

### 14B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_8_32_4_20_4_2_32 | 26.76 | 187.29 |
| q4k_gemv_partial_17408_5120_1 | 11.62 | 81.37 |
| q6k_gemv_partial_5120_17408_1 | 9.85 | 68.95 |
| q4k_gemv_partial_5120_5120_1 | 5.30 | 37.08 |
| r_1187_32_4_20_2_2_2_32n1 | 5.26 | 36.85 |
| r_8_8_16_2_20_2_2_2_32 | 4.52 | 31.64 |
| r_8_8_16_2_20_4_2_32 | 4.41 | 30.84 |
| q4k_gemv_partial_5120_17408_4 | 3.54 | 24.77 |
| r_16_320n1 | 0.52 | 3.65 |
| E_5_2_2_16_4_4 | 0.52 | 3.63 |
| r_16_320 | 0.49 | 3.40 |
| r_40_(start_pos+1)_16_8 | 0.34 | 2.36 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.30 |
| r_5_2_8_16_4_(start_pos+1) | 0.32 | 2.26 |
| r_5_2_4_(start_pos+1) | 0.29 | 2.00 |
| E_136_32_4 | 0.27 | 1.91 |
| r_32_4_1187 | 0.27 | 1.90 |
| E_5_(start_pos+1)_2_4 | 0.27 | 1.90 |
| r_8_16_8 | 0.26 | 1.85 |
| r_40_16_8 | 0.26 | 1.82 |

### 14B QK_GENERATED_POLICY named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q4k_gemv_partial_17408_5120_1 | 11.59 | 81.16 |
| q6k_gemv_partial_5120_17408_2 | 7.14 | 49.96 |
| q4k_gemv_partial_5120_5120_1 | 5.46 | 38.25 |
| r_1187_32_4_20_2_2_2_32n1 | 5.40 | 37.82 |
| q4k_gemv_partial_5120_17408_2 | 3.51 | 24.54 |
| q6k_gemv_partial_1024_5120_2 | 1.58 | 11.09 |
| q4k_gemv_partial_1024_5120_4 | 1.53 | 10.74 |
| E_5_2_2_16_4_4 | 0.56 | 3.95 |
| r_16_320n1 | 0.53 | 3.74 |
| r_16_320 | 0.50 | 3.49 |
| r_40_(start_pos+1)_16_8 | 0.35 | 2.44 |
| copy        4 B,     AMD <- AMD | 0.34 | 2.41 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.34 |
| r_5_2_8_16_4_(start_pos+1) | 0.33 | 2.30 |
| r_5_2_4_(start_pos+1) | 0.29 | 2.04 |
| r_32_4_1187 | 0.29 | 2.03 |
| E_5_(start_pos+1)_2_4 | 0.28 | 1.96 |
| E_40_32_4n1 | 0.28 | 1.94 |
| E_136_32_4 | 0.28 | 1.94 |
| r_8_16_8 | 0.27 | 1.87 |

## Decision Gates

- **14B Q4K+Q6K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B QK_GENERATED_POLICY batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive reductions >15% of profile basis: fuse/avoid partial reduction; fallback/generic dense quant remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **14B QK_GENERATED_POLICY named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
