# QK Gap Profile

Bottleneck attribution for the Ansor-transition loop. `batched` rows are
the throughput truth; `named` rows are attribution-only because graph
batching is disabled for readable kernel names.

## Summary

- profiled models: `2/3`
- missing profiles: `8B`

| model | status | generated tok/s | batched wall ms/tok | named QK GEMV ms/tok | named reduction ms/tok | fallback ms/tok | next decision |
|---|---|---:|---:|---:|---:|---:|---|
| `8B` | `profile_missing` | n/a | n/a | n/a | n/a | n/a | `bench/qk-shared-storage-20260612/8b/profile-report.json is missing; run DEBUG=2 generated/explicit profile before optimizing this model.` |
| `14B` | `profiled` | 42.19 | 23.71 | 30.08 | 1.13 | 5.27 | `qk_semantic_schedule_or_codegen` |
| `32B` | `profiled` | 17.99 | 55.63 | 82.44 | 1.97 | 5.33 | `qk_semantic_schedule_or_codegen` |

## Dominant Named AMD Buckets

### 14B

| bucket | ms/tok | % named AMD |
|---|---:|---:|
| `q4k_primitive_gemv` | 21.57 | 52.09 |
| `q6k_primitive_gemv` | 8.50 | 20.54 |
| `fallback_quant_fused` | 5.27 | 12.72 |
| `norm_sampling_misc` | 2.09 | 5.05 |
| `attention_misc` | 1.52 | 3.67 |
| `other_amd` | 1.26 | 3.04 |
| `q4k_primitive_reduction` | 1.13 | 2.73 |
| `copy` | 0.06 | 0.15 |

### 32B

| bucket | ms/tok | % named AMD |
|---|---:|---:|
| `q4k_primitive_gemv` | 64.15 | 65.83 |
| `q6k_primitive_gemv` | 18.29 | 18.77 |
| `fallback_quant_fused` | 5.33 | 5.46 |
| `norm_sampling_misc` | 3.03 | 3.11 |
| `attention_misc` | 2.62 | 2.68 |
| `other_amd` | 2.01 | 2.06 |
| `q4k_primitive_reduction` | 1.97 | 2.02 |
| `copy` | 0.06 | 0.06 |
