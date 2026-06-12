# Quant Residual Decode Profile

Scope: Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode.

Classifier rules are calibrated to Qwen3 8B/14B/32B Q4_K_M AMD DEBUG=2 decode logs. Use this report outside that scope only after adding boundary tests for the new kernel signatures.

Steady-state rows drop the first 1 benchmark token(s). `batched` rows are the throughput truth. `named` rows are attribution-only: they disable graph batching via JIT_BATCH_SIZE=1, so use AMD-kernel percentages rather than named wall time for bottleneck decisions.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 7 | 24.15 | 41.40 | 40.70 | 0.70 | 1.69 |
| 14B | QK_GENERATED_POLICY batched | 7 | 42.19 | 23.71 | 22.99 | 0.71 | 3.00 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 7 | 4.10 | 244.11 | 76.82 | 167.29 | 68.53 |
| 14B | QK_GENERATED_POLICY named | 7 | 4.46 | 224.20 | 41.41 | 182.78 | 81.53 |

## Parse Health

| model | mode | lines | tokens | AMD lines | ignored lines | non-AMD DEBUG lines | trailing AMD lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 2740 | 8 | 2535 | 197 | 0 | 0 |
| 14B | QK_GENERATED_POLICY batched | 2820 | 8 | 2615 | 197 | 0 | 0 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 11827 | 8 | 11623 | 196 | 0 | 0 |
| 14B | QK_GENERATED_POLICY named | 12547 | 8 | 12343 | 196 | 0 | 0 |

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
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | other_amd | 40.67 | 98.22 | 99.91 | batched 410, batched 256, batched 128 |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | residual_overhead | 0.70 | 1.69 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q6k_primitive_gemv | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | q4k_primitive_reduction | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | fallback_quant_fused | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | attention_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | norm_sampling_misc | 0.00 | 0.00 | 0.00 |  |
| 14B | QK_GENERATED_POLICY batched | copy | 0.04 | 0.16 | 0.16 | copy        4 B,     AMD <- AMD |
| 14B | QK_GENERATED_POLICY batched | other_amd | 22.96 | 96.85 | 99.84 | batched 490, batched 256, batched 128 |
| 14B | QK_GENERATED_POLICY batched | residual_overhead | 0.71 | 3.00 | 0.00 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 20.50 | 8.40 | 26.68 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 9.86 | 4.04 | 12.83 | q6k_gemv_partial_5120_17408_1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 13.77 | 5.64 | 17.93 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 18.70 | 7.66 | 24.34 | r_8_32_4_20_4_2_32, r_1187_32_4_20_2_2_2_32n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 1.52 | 0.62 | 1.98 | r_5_2_4_(start_pos+1)n1, r_40_(start_pos+1)_16_8, r_5_2_8_16_4_(start_pos+1) |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 2.20 | 0.90 | 2.86 | E_5_2_2_16_4_4, r_32_4_1187, E_136_32_4 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.07 | 0.03 | 0.08 |  |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 10.22 | 4.18 | 13.30 | r_8_8_16_2_20_2_2_2_32, r_8_8_16_2_20_4_2_32, r_16_320n1 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 167.29 | 68.53 | 0.00 |  |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_gemv | 21.57 | 9.62 | 52.09 | q4k_gemv_partial_17408_5120_1, q4k_gemv_partial_5120_5120_1, q4k_gemv_partial_5120_17408_2 |
| 14B | QK_GENERATED_POLICY named | q6k_primitive_gemv | 8.50 | 3.79 | 20.54 | q6k_gemv_partial_5120_17408_2, q6k_gemv_partial_1024_5120_2 |
| 14B | QK_GENERATED_POLICY named | q4k_primitive_reduction | 1.13 | 0.51 | 2.73 |  |
| 14B | QK_GENERATED_POLICY named | fallback_quant_fused | 5.27 | 2.35 | 12.72 | r_1187_32_4_20_2_2_2_32n1 |
| 14B | QK_GENERATED_POLICY named | attention_misc | 1.52 | 0.68 | 3.67 | r_5_2_4_(start_pos+1)n1, r_40_(start_pos+1)_16_8, r_5_2_8_16_4_(start_pos+1) |
| 14B | QK_GENERATED_POLICY named | norm_sampling_misc | 2.09 | 0.93 | 5.05 | E_5_2_2_16_4_4, r_32_4_1187, E_136_32_4 |
| 14B | QK_GENERATED_POLICY named | copy | 0.06 | 0.03 | 0.15 |  |
| 14B | QK_GENERATED_POLICY named | other_amd | 1.26 | 0.56 | 3.04 | r_16_320n1, r_16_320, r_8_16_8 |
| 14B | QK_GENERATED_POLICY named | residual_overhead | 182.78 | 81.53 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 14B | Q4K+Q6K_PRIMITIVE=1 batched | 0 | 41.41 | 0.00 |
| 14B | QK_GENERATED_POLICY batched | 0 | 23.71 | 0.00 |
| 14B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 244.00 | 0.00 |
| 14B | QK_GENERATED_POLICY named | 0 | 223.82 | 0.00 |

## Top Kernels

### 14B Q4K+Q6K_PRIMITIVE=1 batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 410 | 20.63 | 144.43 |
| batched 256 | 10.49 | 73.44 |
| batched 128 | 5.02 | 35.16 |
| batched 64 | 3.06 | 21.43 |
| batched 32 | 1.46 | 10.21 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.25 |

### 14B QK_GENERATED_POLICY batched

| kernel | ms/tok | total ms |
| --- | --- | --- |
| batched 490 | 13.48 | 94.34 |
| batched 256 | 4.83 | 33.81 |
| batched 128 | 2.62 | 18.31 |
| batched 64 | 1.39 | 9.74 |
| batched 32 | 0.64 | 4.50 |
| copy        4 B,     AMD <- AMD | 0.04 | 0.26 |

### 14B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| r_8_32_4_20_4_2_32 | 26.78 | 187.43 |
| q4k_gemv_partial_17408_5120_1 | 11.66 | 81.59 |
| q6k_gemv_partial_5120_17408_1 | 9.86 | 68.99 |
| r_1187_32_4_20_2_2_2_32n1 | 5.33 | 37.31 |
| q4k_gemv_partial_5120_5120_1 | 5.31 | 37.17 |
| r_8_8_16_2_20_2_2_2_32 | 4.51 | 31.60 |
| r_8_8_16_2_20_4_2_32 | 4.40 | 30.80 |
| q4k_gemv_partial_5120_17408_4 | 3.53 | 24.71 |
| E_5_2_2_16_4_4 | 0.52 | 3.64 |
| r_16_320n1 | 0.52 | 3.61 |
| r_16_320 | 0.48 | 3.36 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.30 |
| r_40_(start_pos+1)_16_8 | 0.33 | 2.29 |
| r_5_2_8_16_4_(start_pos+1) | 0.32 | 2.25 |
| r_5_2_4_(start_pos+1) | 0.28 | 1.99 |
| r_32_4_1187 | 0.28 | 1.93 |
| E_136_32_4 | 0.27 | 1.91 |
| r_8_16_8 | 0.26 | 1.85 |
| r_40_16_8 | 0.26 | 1.82 |
| E_5_(start_pos+1)_2_4 | 0.26 | 1.80 |

### 14B QK_GENERATED_POLICY named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q4k_gemv_partial_17408_5120_1 | 11.39 | 79.74 |
| q6k_gemv_partial_5120_17408_2 | 6.95 | 48.68 |
| q4k_gemv_partial_5120_5120_1 | 5.31 | 37.19 |
| r_1187_32_4_20_2_2_2_32n1 | 5.27 | 36.88 |
| q4k_gemv_partial_5120_17408_2 | 3.41 | 23.88 |
| q6k_gemv_partial_1024_5120_2 | 1.55 | 10.85 |
| q4k_gemv_partial_1024_5120_4 | 1.46 | 10.20 |
| E_5_2_2_16_4_4 | 0.52 | 3.67 |
| r_16_320n1 | 0.51 | 3.60 |
| r_16_320 | 0.48 | 3.36 |
| r_5_2_4_(start_pos+1)n1 | 0.33 | 2.30 |
| r_40_(start_pos+1)_16_8 | 0.33 | 2.29 |
| r_5_2_8_16_4_(start_pos+1) | 0.32 | 2.26 |
| r_5_2_4_(start_pos+1) | 0.28 | 1.99 |
| r_32_4_1187 | 0.28 | 1.96 |
| E_136_32_4 | 0.27 | 1.90 |
| r_8_16_8 | 0.26 | 1.85 |
| r_40_16_8 | 0.26 | 1.81 |
| E_5_(start_pos+1)_2_4 | 0.26 | 1.80 |
| E_40_32_4n1 | 0.25 | 1.75 |

## Decision Gates

- **14B Q4K+Q6K_PRIMITIVE=1 batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B QK_GENERATED_POLICY batched** (wall-time basis): real graph-batched runtime has low residual and no outliers; use named logs for inner-kernel ownership
- **14B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive reductions >15% of profile basis: fuse/avoid partial reduction; fallback/generic dense quant remains large: extend primitive coverage or revise policy; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
- **14B QK_GENERATED_POLICY named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
