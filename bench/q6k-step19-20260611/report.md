# Q4_K Residual Decode Profile

Steady-state rows drop the first 1 benchmark token(s). `batched` logs use normal graph batching and are the real runtime profile. `named` logs set `JIT_BATCH_SIZE=1`; they keep the rollout JIT but avoid graph batching so DEBUG=2 exposes kernel names for attribution.

## Summary

| model | mode | samples | tok/s | wall ms/tok | AMD kernel ms/tok | residual ms/tok | residual % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | 31 | 5.30 | 190.74 | 34.23 | 156.51 | 82.05 |

## Parse Health

| model | mode | lines | tokens | AMD lines | ignored lines | non-AMD DEBUG lines | trailing AMD lines |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | 30523 | 32 | 29927 | 384 | 180 | 0 |

## Buckets

| model | mode | bucket | ms/tok | % wall | % AMD kernel | top kernels |
| --- | --- | --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_gemv | 11.36 | 5.95 | 33.18 | q4k_gemv_partial_12288_4096_1, q4k_gemv_partial_4096_4096_1, q4k_gemv_partial_4096_12288_4 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | q6k_primitive_gemv | 5.96 | 3.12 | 17.41 | q6k_gemv_partial_4096_12288_1 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | q4k_primitive_reduction | 0.86 | 0.45 | 2.50 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | fallback_quant_fused | 4.92 | 2.58 | 14.37 | r_1187_32_4_16_2_2_2_32n1, r_1024_16_4_2_32 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | attention_misc | 1.84 | 0.96 | 5.36 | r_4_2_8_16_4_(start_pos+1), r_2_(start_pos+1)_8_4_4_16, r_8_4_(start_pos+1)n1 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | norm_sampling_misc | 2.70 | 1.41 | 7.87 | E_2_8_16_4_4, r_16_256n1, r_16_256 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | copy | 0.11 | 0.06 | 0.32 |  |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | other_amd | 6.50 | 3.41 | 18.99 | r_2_8_128_16_2_2_2_32, r_2_8_128_16_4_2_32, r_2_8_4_4_16 |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | residual_overhead | 156.51 | 82.05 | 0.00 |  |

## Outliers

| model | mode | outliers | median ms | max outlier ms |
| --- | --- | --- | --- | --- |
| 8B | Q4K+Q6K_PRIMITIVE=1 named | 0 | 181.45 | 0.00 |

## Top Kernels

### 8B Q4K+Q6K_PRIMITIVE=1 named

| kernel | ms/tok | total ms |
| --- | --- | --- |
| q6k_gemv_partial_4096_12288_1 | 5.96 | 184.70 |
| q4k_gemv_partial_12288_4096_1 | 5.82 | 180.49 |
| r_1187_32_4_16_2_2_2_32n1 | 4.38 | 135.66 |
| r_2_8_128_16_2_2_2_32 | 4.28 | 132.74 |
| q4k_gemv_partial_4096_4096_1 | 3.99 | 123.81 |
| r_2_8_128_16_4_2_32 | 1.66 | 51.55 |
| q4k_gemv_partial_4096_12288_4 | 1.54 | 47.78 |
| r_1024_16_4_2_32 | 1.08 | 33.50 |
| r_4_2_8_16_4_(start_pos+1) | 0.48 | 14.99 |
| E_2_8_16_4_4 | 0.47 | 14.44 |
| r_16_256n1 | 0.42 | 13.09 |
| r_16_256 | 0.41 | 12.71 |
| r_2_(start_pos+1)_8_4_4_16 | 0.41 | 12.71 |
| r_8_4_(start_pos+1)n1 | 0.41 | 12.63 |
| r_2_8_4_4_16 | 0.31 | 9.48 |
| r_8_4_(start_pos+1) | 0.30 | 9.16 |
| r_32_4_1187 | 0.28 | 8.59 |
| E_(start_pos+1)_8_4 | 0.24 | 7.44 |
| r_8_16_8 | 0.24 | 7.42 |
| E_128_32_3 | 0.24 | 7.35 |

## Decision Gates

- **8B Q4K+Q6K_PRIMITIVE=1 named** (AMD-kernel basis): primitive GEMV >50% of profile basis: build primitive v2; named-log residual is expected launch overhead from disabled graph batching; ignore for runtime decisions
