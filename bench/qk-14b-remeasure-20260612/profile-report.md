# Quant Residual Decode Profile

Scope: Qwen3 8B/14B Q4_K_M AMD DEBUG=2 decode.

Classifier rules are calibrated to Qwen3 8B/14B Q4_K_M AMD DEBUG=2 decode logs. Use this report outside that scope only after adding boundary tests for the new kernel signatures.

Steady-state rows drop the first 1 benchmark token(s). `batched` rows are the throughput truth. `named` rows are attribution-only: they disable graph batching via JIT_BATCH_SIZE=1, so use AMD-kernel percentages rather than named wall time for bottleneck decisions.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 7 | 24.07 | 41.55 | 40.84 | 0.71 | 1.71 |
| 14B | QK_GENERATED_POLICY batched | 7 | 42.22 | 23.69 | 22.95 | 0.74 | 3.11 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 7 | 3.97 | 251.74 | 77.35 | 174.39 | 69.27 |
| 14B | QK_GENERATED_POLICY named | 7 | 4.45 | 224.72 | 41.75 | 182.97 | 81.42 |

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
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | other_amd | 40.79 | 98.19 | 99.90 | batched 410, batched 256, batched 128 |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | residual_overhead | 0.71 | 1.71 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | copy | 0.04 | 0.16 | 0.17 | copy        4 B,     AMD <- AMD |
| 14B | QK_GENERATED_POLICY batched | other_amd | 22.91 | 96.73 | 99.83 | batched 490, batched 256, batched 128 |
| 14B | QK_GENERATED_POLICY batched | residual_overhead | 0.74 | 3.11 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 20.63 | 8.19 | 26.67 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 9.91 | 3.94 | 12.81 | q6k_gemv_partial_5120_17408_1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 13.86 | 5.50 | 17.91 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 18.75 | 7.45 | 24.24 | r_8_32_4_20_4_2_32, r_1187_32_4_20_2_2_2_32n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 1.56 | 0.62 | 2.02 | r_40_(start_pos+1)_16_8, r_5_2_4_(start_pos+1)n1, r_5_2_8_16_4_(start_pos+1) |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 2.25 | 0.89 | 2.91 | E_5_2_2_16_4_4, r_32_4_1187, E_136_32_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.06 | 0.02 | 0.08 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 10.34 | 4.11 | 13.36 | r_8_8_16_2_20_2_2_2_32, r_8_8_16_2_20_4_2_32, r_16_320n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 174.39 | 69.27 | 0.00 |  |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_gemv | 21.64 | 9.63 | 51.84 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_2 |
| 14B | QK_GENERATED_POLICY named | q6k_primitive_gemv | 8.60 | 3.83 | 20.61 | q6k_gemv_partial_5120_17408_2, q6k_gemv_partial_1024_5120_2 |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_reduction | 1.14 | 0.51 | 2.73 |  |
| 14B | QK_GENERATED_POLICY named | fallback_quant_fused | 5.34 | 2.38 | 12.79 | r_1187_32_4_20_2_2_2_32n1 |
| 14B | QK_GENERATED_POLICY named | attention_misc | 1.56 | 0.69 | 3.73 | r_40_(start_pos+1)_16_8, r_5_2_4_(start_pos+1)n1, r_5_2_8_16_4_(start_pos+1) |
| 14B | QK_GENERATED_POLICY named | norm_sampling_misc | 2.12 | 0.94 | 5.08 | E_5_2_2_16_4_4, r_32_4_1187, E_136_32_4 |
| 14B | QK_GENERATED_POLICY named | copy | 0.07 | 0.03 | 0.17 |  |
| 14B | QK_GENERATED_POLICY named | other_amd | 1.28 | 0.57 | 3.07 | r_16_320n1, r_16_320, r_8_16_8 |
| 14B | QK_GENERATED_POLICY named | residual_overhead | 182.97 | 81.42 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 0 | 41.47 | 0.00 |
| 14B | QK_GENERATED_POLICY batched | 0 | 23.69 | 0.00 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 251.65 | 0.00 |
| 14B | QK_GENERATED_POLICY named | 0 | 223.54 | 0.00 |

## Top Kernels

### 14B Q4K+Q6K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 410 | 20.75 | 145.28 |
| batched 256 | 10.50 | 73.48 |
| batched 128 | 5.02 | 35.17 |
| batched 64 | 3.06 | 21.42 |
| batched 32 | 1.46 | 10.21 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.28 |

### 14B QK_GENERATED_POLICY batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 490 | 13.39 | 93.71 |
| batched 256 | 4.86 | 34.00 |
| batched 128 | 2.62 | 18.37 |
| batched 64 | 1.39 | 9.75 |
| batched 32 | 0.65 | 4.55 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.27 |

### 14B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_8_32_4_20_4_2_32 | 26.94 | 188.55 |
| q4k_gemv_partial_17408_5120_1 | 11.71 | 81.97 |
| q6k_gemv_partial_5120_17408_1 | 9.91 | 69.34 |
| q4k_gemv_partial_5120_5120_1 | 5.37 | 37.59 |
| r_1187_32_4_20_2_2_2_32n1 | 5.30 | 37.08 |
| r_8_8_16_2_20_2_2_2_32 | 4.54 | 31.81 |
| r_8_8_16_2_20_4_2_32 | 4.47 | 31.30 |
| q4k_gemv_partial_5120_17408_4 | 3.55 | 24.86 |
| E_5_2_2_16_4_4 | 0.53 | 3.72 |
| r_16_320n1 | 0.52 | 3.67 |
| r_16_320 | 0.49 | 3.44 |
| r_40_(start_pos+1)_16_8 | 0.34 | 2.41 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.32 |
| r_5_2_8_16_4_(start_pos+1) | 0.33 | 2.29 |
| r_5_2_4_(start_pos+1) | 0.29 | 2.02 |
| r_32_4_1187 | 0.28 | 1.98 |
| E_136_32_4 | 0.28 | 1.93 |
| E_5_(start_pos+1)_2_4 | 0.27 | 1.91 |
| r_8_16_8 | 0.27 | 1.86 |
| r_40_16_8 | 0.26 | 1.84 |

### 14B QK_GENERATED_POLICY named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q4k_gemv_partial_17408_5120_1 | 11.38 | 79.67 |
| q6k_gemv_partial_5120_17408_2 | 7.03 | 49.20 |
| q4k_gemv_partial_5120_5120_1 | 5.34 | 37.38 |
| r_1187_32_4_20_2_2_2_32n1 | 5.34 | 37.36 |
| q4k_gemv_partial_5120_17408_2 | 3.42 | 23.96 |
| q6k_gemv_partial_1024_5120_2 | 1.57 | 11.02 |
| q4k_gemv_partial_1024_5120_4 | 1.50 | 10.49 |
| E_5_2_2_16_4_4 | 0.54 | 3.75 |
| r_16_320n1 | 0.52 | 3.67 |
| r_16_320 | 0.49 | 3.42 |
| r_40_(start_pos+1)_16_8 | 0.34 | 2.38 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.32 |
| r_5_2_8_16_4_(start_pos+1) | 0.32 | 2.27 |
| r_5_2_4_(start_pos+1) | 0.29 | 2.00 |
| r_32_4_1187 | 0.28 | 1.95 |
| E_136_32_4 | 0.27 | 1.91 |
| E_5_(start_pos+1)_2_4 | 0.27 | 1.91 |
| r_8_16_8 | 0.26 | 1.85 |
| r_40_16_8 | 0.26 | 1.83 |
| E_40_32_4n1 | 0.26 | 1.79 |

## Decision Gates

- **14B Q4K+Q6K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B QK_GENERATED_POLICY batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive reductions >15% of profile basis: fuse/avoid partial reduction; fallback/generic dense quant remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **14B QK_GENERATED_POLICY named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
