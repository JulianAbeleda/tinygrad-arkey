# Quant Residual Decode Profile

Scope: Qwen3 8B/14B Q4_K_M AMD DEBUG=2 decode.

Classifier rules are calibrated to Qwen3 8B/14B Q4_K_M AMD DEBUG=2 decode logs. Use this report outside that scope only after adding boundary tests for the new kernel signatures.

Steady-state rows drop the first 1 benchmark token(s). `batched` rows are the throughput truth. `named` rows are attribution-only: they disable graph batching via JIT_BATCH_SIZE=1, so use AMD-kernel percentages rather than named wall time for bottleneck decisions.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 7 | 23.84 | 41.96 | 41.22 | 0.75 | 1.78 |
| 14B | QK_GENERATED_POLICY batched | 7 | 42.29 | 23.65 | 22.95 | 0.69 | 2.93 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 7 | 4.03 | 248.32 | 76.96 | 171.36 | 69.01 |
| 14B | QK_GENERATED_POLICY named | 7 | 4.30 | 233.46 | 41.77 | 191.68 | 82.11 |

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
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | copy | 0.04 | 0.10 | 0.10 | copy        4 B,     AMD <- AMD |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | other_amd | 41.18 | 98.12 | 99.90 | batched 410, batched 256, batched 128 |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | residual_overhead | 0.75 | 1.78 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | copy | 0.04 | 0.16 | 0.16 | copy        4 B,     AMD <- AMD |
| 14B | QK_GENERATED_POLICY batched | other_amd | 22.92 | 96.91 | 99.84 | batched 490, batched 256, batched 128 |
| 14B | QK_GENERATED_POLICY batched | residual_overhead | 0.69 | 2.93 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 20.50 | 8.26 | 26.64 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 9.87 | 3.98 | 12.83 | q6k_gemv_partial_5120_17408_1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 13.81 | 5.56 | 17.94 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 18.70 | 7.53 | 24.29 | r_8_32_4_20_4_2_32, r_1187_32_4_20_2_2_2_32n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 1.55 | 0.62 | 2.01 | r_40_(start_pos+1)_16_8, r_5_2_4_(start_pos+1)n1, r_5_2_8_16_4_(start_pos+1) |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 2.22 | 0.89 | 2.88 | E_5_2_2_16_4_4, E_136_32_4, r_32_4_1187 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.06 | 0.03 | 0.08 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 10.26 | 4.13 | 13.33 | r_8_8_16_2_20_2_2_2_32, r_8_8_16_2_20_4_2_32, r_16_320n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 171.36 | 69.01 | 0.00 |  |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_gemv | 21.67 | 9.28 | 51.87 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_2 |
| 14B | QK_GENERATED_POLICY named | q6k_primitive_gemv | 8.67 | 3.71 | 20.74 | q6k_gemv_partial_5120_17408_2, q6k_gemv_partial_1024_5120_2 |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_reduction | 1.14 | 0.49 | 2.73 |  |
| 14B | QK_GENERATED_POLICY named | fallback_quant_fused | 5.27 | 2.26 | 12.60 | r_1187_32_4_20_2_2_2_32n1 |
| 14B | QK_GENERATED_POLICY named | attention_misc | 1.56 | 0.67 | 3.73 | r_40_(start_pos+1)_16_8, r_5_2_4_(start_pos+1)n1, r_5_2_8_16_4_(start_pos+1) |
| 14B | QK_GENERATED_POLICY named | norm_sampling_misc | 2.12 | 0.91 | 5.09 | E_5_2_2_16_4_4, r_32_4_1187, E_136_32_4 |
| 14B | QK_GENERATED_POLICY named | copy | 0.06 | 0.03 | 0.15 |  |
| 14B | QK_GENERATED_POLICY named | other_amd | 1.29 | 0.55 | 3.08 | r_16_320n1, r_16_320, r_8_16_8 |
| 14B | QK_GENERATED_POLICY named | residual_overhead | 191.68 | 82.11 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 0 | 41.50 | 0.00 |
| 14B | QK_GENERATED_POLICY batched | 0 | 23.64 | 0.00 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 247.65 | 0.00 |
| 14B | QK_GENERATED_POLICY named | 0 | 224.80 | 0.00 |

## Top Kernels

### 14B Q4K+Q6K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 410 | 20.68 | 144.75 |
| batched 256 | 10.70 | 74.88 |
| batched 128 | 5.28 | 36.95 |
| batched 64 | 3.06 | 21.42 |
| batched 32 | 1.46 | 10.23 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.30 |

### 14B QK_GENERATED_POLICY batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 490 | 13.39 | 93.75 |
| batched 256 | 4.86 | 34.01 |
| batched 128 | 2.62 | 18.36 |
| batched 64 | 1.40 | 9.77 |
| batched 32 | 0.65 | 4.53 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.26 |

### 14B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_8_32_4_20_4_2_32 | 26.84 | 187.88 |
| q4k_gemv_partial_17408_5120_1 | 11.64 | 81.49 |
| q6k_gemv_partial_5120_17408_1 | 9.87 | 69.11 |
| q4k_gemv_partial_5120_5120_1 | 5.32 | 37.23 |
| r_1187_32_4_20_2_2_2_32n1 | 5.30 | 37.08 |
| r_8_8_16_2_20_2_2_2_32 | 4.53 | 31.70 |
| r_8_8_16_2_20_4_2_32 | 4.42 | 30.91 |
| q4k_gemv_partial_5120_17408_4 | 3.54 | 24.79 |
| E_5_2_2_16_4_4 | 0.52 | 3.65 |
| r_16_320n1 | 0.52 | 3.64 |
| r_16_320 | 0.49 | 3.40 |
| r_40_(start_pos+1)_16_8 | 0.34 | 2.37 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.31 |
| r_5_2_8_16_4_(start_pos+1) | 0.32 | 2.26 |
| r_5_2_4_(start_pos+1) | 0.29 | 2.00 |
| E_136_32_4 | 0.27 | 1.91 |
| r_32_4_1187 | 0.27 | 1.91 |
| E_5_(start_pos+1)_2_4 | 0.27 | 1.90 |
| r_8_16_8 | 0.27 | 1.86 |
| r_40_16_8 | 0.26 | 1.82 |

### 14B QK_GENERATED_POLICY named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q4k_gemv_partial_17408_5120_1 | 11.39 | 79.76 |
| q6k_gemv_partial_5120_17408_2 | 7.06 | 49.40 |
| q4k_gemv_partial_5120_5120_1 | 5.35 | 37.44 |
| r_1187_32_4_20_2_2_2_32n1 | 5.27 | 36.86 |
| q4k_gemv_partial_5120_17408_2 | 3.43 | 23.98 |
| q6k_gemv_partial_1024_5120_2 | 1.61 | 11.26 |
| q4k_gemv_partial_1024_5120_4 | 1.50 | 10.51 |
| E_5_2_2_16_4_4 | 0.54 | 3.77 |
| r_16_320n1 | 0.53 | 3.69 |
| r_16_320 | 0.49 | 3.43 |
| r_40_(start_pos+1)_16_8 | 0.34 | 2.39 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.31 |
| r_5_2_8_16_4_(start_pos+1) | 0.33 | 2.28 |
| r_5_2_4_(start_pos+1) | 0.29 | 2.01 |
| r_32_4_1187 | 0.28 | 1.94 |
| E_5_(start_pos+1)_2_4 | 0.27 | 1.92 |
| E_136_32_4 | 0.27 | 1.91 |
| r_8_16_8 | 0.27 | 1.86 |
| r_40_16_8 | 0.26 | 1.83 |
| E_40_32_4n1 | 0.26 | 1.80 |

## Decision Gates

- **14B Q4K+Q6K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B QK_GENERATED_POLICY batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive reductions >15% of profile basis: fuse/avoid partial reduction; fallback/generic dense quant remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **14B QK_GENERATED_POLICY named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
