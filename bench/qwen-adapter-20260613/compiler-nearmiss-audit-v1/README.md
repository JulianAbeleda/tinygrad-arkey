# Compiler Near-Miss Audit

- category: `compiler`
- intervention: `prompt_data_fix`

## Rationale

- Compiler failures are mostly valid JSON with wrong values, not form failures.
- The generated values often drop the row-specific numeric suffix or collapse to a broad prefix.
- Accepting those by normalization would change the task contract, so the data/prompt target should be redesigned before training V7.

## Artifact

- path: `bench/qwen-adapter-20260613/training-data-v4-rs-v5-k4`
- attempts: `272`
- accepted attempts: `0`
- selected train rows: `0`
- near misses: `79`
- unique near-miss sources: `68`

### Miss Classification

| classification | count |
|---|---|
| prefix | 54 |
| empty_string | 16 |
| stem_without_index | 8 |
| substring | 1 |

### Top Actual Answers

| actual | count |
|---|---|
| "train_qk" | 33 |
| "train" | 18 |
| "" | 16 |
| "train_qk_gemv" | 8 |
| "train_qk_coalesced" | 2 |
| "train_qk_json" | 1 |
| "qk" | 1 |

### Near Misses By Template

| template | count |
|---|---|
| compiler_glossary_wide_load | 11 |
| compiler_glossary_coalesced_read | 10 |
| compiler_glossary_gemv | 8 |
| compiler_glossary_wavefront | 6 |
| compiler_glossary_dequant | 6 |
| compiler_glossary_q4_block | 6 |
| compiler_glossary_q6_block | 6 |
| compiler_glossary_uop | 6 |
| compiler_glossary_beam | 5 |
| compiler_glossary_policy | 5 |
| compiler_glossary_suffix_cache | 5 |
| compiler_glossary_json_axis | 5 |

## Artifact

- path: `bench/qwen-adapter-20260613/training-data-v4-rs-v5-stratified-v1`
- attempts: `544`
- accepted attempts: `0`
- selected train rows: `0`
- near misses: `158`
- unique near-miss sources: `68`

### Miss Classification

| classification | count |
|---|---|
| prefix | 111 |
| empty_string | 32 |
| stem_without_index | 14 |
| substring | 1 |

### Top Actual Answers

| actual | count |
|---|---|
| "train_qk" | 65 |
| "train" | 41 |
| "" | 32 |
| "train_qk_gemv" | 14 |
| "train_qk_coalesced" | 3 |
| "train_qk_json" | 1 |
| "qk" | 1 |
| "train_qk_wide" | 1 |

### Near Misses By Template

| template | count |
|---|---|
| compiler_glossary_wide_load | 23 |
| compiler_glossary_coalesced_read | 21 |
| compiler_glossary_gemv | 14 |
| compiler_glossary_wavefront | 12 |
| compiler_glossary_dequant | 12 |
| compiler_glossary_q4_block | 12 |
| compiler_glossary_q6_block | 12 |
| compiler_glossary_uop | 12 |
| compiler_glossary_beam | 10 |
| compiler_glossary_policy | 10 |
| compiler_glossary_suffix_cache | 10 |
| compiler_glossary_json_axis | 10 |

## V6 Eval Rollout Reference

- path: `bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v6-gold-v4-rollout`
- passed: `14/34`

| template | passed | scored |
|---|---:|---:|
| `compiler_glossary_beam` | 3 | 3 |
| `compiler_glossary_coalesced_read` | 0 | 3 |
| `compiler_glossary_dequant` | 3 | 3 |
| `compiler_glossary_gemv` | 0 | 3 |
| `compiler_glossary_json_axis` | 2 | 2 |
| `compiler_glossary_policy` | 3 | 3 |
| `compiler_glossary_q4_block` | 0 | 3 |
| `compiler_glossary_q6_block` | 0 | 3 |
| `compiler_glossary_suffix_cache` | 2 | 2 |
| `compiler_glossary_uop` | 0 | 3 |
| `compiler_glossary_wavefront` | 0 | 3 |
| `compiler_glossary_wide_load` | 1 | 3 |
