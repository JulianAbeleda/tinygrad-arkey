# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter: `bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5`
- dataset: `bench/qwen-adapter-20260613/training-data-v4_1-compiler/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `34`
- generated tokens: `276`
- quality: `fail` (30/34)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `json_v4_1_eval_compiler_001` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 9 | 0.33 | {"answer":"qk_wide_load"} |
| `json_v4_1_eval_compiler_002` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 10 | 15.59 | {"answer":"qk_coalesced_read"} |
| `json_v4_1_eval_compiler_003` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 13.13 | {"answer":"qk_wavefront"} |
| `json_v4_1_eval_compiler_004` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 12.72 | {"answer":"qk_dequant"} |
| `json_v4_1_eval_compiler_005` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 13.62 | {"answer":"qk_gemv"} |
| `json_v4_1_eval_compiler_006` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 9 | 13.54 | {"answer":"qk_q4_block"} |
| `json_v4_1_eval_compiler_007` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 9 | 13.85 | {"answer":"qk_q6_block"} |
| `json_v4_1_eval_compiler_008` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 13.39 | {"answer":"qk_uop"} |
| `json_v4_1_eval_compiler_009` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 7 | 13.38 | {"answer":"qk_beam"} |
| `json_v4_1_eval_compiler_010` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 7 | 11.18 | {"answer":"qk_policy"} |
| `json_v4_1_eval_compiler_011` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 12.53 | {"answer":"qk_suffix_cache"} |
| `json_v4_1_eval_compiler_012` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 12.26 | {"answer":"qk_json_axis"} |
| `json_v4_1_eval_compiler_013` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 9 | 11.28 | {"answer":"qk_wide_load"} |
| `json_v4_1_eval_compiler_014` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 10 | 17.75 | {"answer":"qk_coalesced_read"} |
| `json_v4_1_eval_compiler_015` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 15.09 | {"answer":"qk_wavefront"} |
| `json_v4_1_eval_compiler_016` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 14.66 | {"answer":"qk_dequant"} |
| `json_v4_1_eval_compiler_017` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 15.50 | {"answer":"qk_gemv"} |
| `json_v4_1_eval_compiler_018` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 9 | 15.22 | {"answer":"qk_q4_block"} |
| `json_v4_1_eval_compiler_019` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 9 | 17.49 | {"answer":"qk_q6_block"} |
| `json_v4_1_eval_compiler_020` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 15.45 | {"answer":"qk_uop"} |
| `json_v4_1_eval_compiler_021` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 7 | 15.16 | {"answer":"qk_beam"} |
| `json_v4_1_eval_compiler_022` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 7 | 13.32 | {"answer":"qk_policy"} |
| `json_v4_1_eval_compiler_023` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 14.63 | {"answer":"qk_suffix_cache"} |
| `json_v4_1_eval_compiler_024` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 14.02 | {"answer":"qk_json_axis"} |
| `json_v4_1_eval_compiler_025` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `fail` | 8 | 10.38 | {"answer":"qk_wide"} |
| `json_v4_1_eval_compiler_026` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `fail` | 9 | 13.11 | {"answer":"qk_coalesced"} |
| `json_v4_1_eval_compiler_027` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 12.30 | {"answer":"qk_wavefront"} |
| `json_v4_1_eval_compiler_028` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 11.78 | {"answer":"qk_dequant"} |
| `json_v4_1_eval_compiler_029` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `fail` | 6 | 9.80 | {"answer":"gemv"} |
| `json_v4_1_eval_compiler_030` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `fail` | 8 | 11.37 | {"answer":"qk_q4"} |
| `json_v4_1_eval_compiler_031` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 9 | 12.99 | {"answer":"qk_q6_block"} |
| `json_v4_1_eval_compiler_032` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 8 | 12.38 | {"answer":"qk_uop"} |
| `json_v4_1_eval_compiler_033` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 7 | 11.86 | {"answer":"qk_beam"} |
| `json_v4_1_eval_compiler_034` | `json_answer,compiler,eval,v4_1_compiler,stable_key` | `pass` | 7 | 10.63 | {"answer":"qk_policy"} |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `compiler` | 30 | 34 | 0.88 |
| `eval` | 30 | 34 | 0.88 |
| `json_answer` | 30 | 34 | 0.88 |
| `stable_key` | 30 | 34 | 0.88 |
| `v4_1_compiler` | 30 | 34 | 0.88 |

## JSON Quality Axes

| axis | passed | scored | pass rate [95% CI] |
|---|---:|---:|---:|
| `parse_valid` | 34 | 34 | 1.00 [0.90, 1.00] |
| `no_extra_text` | 34 | 34 | 1.00 [0.90, 1.00] |
| `schema_ok` | 34 | 34 | 1.00 [0.90, 1.00] |
| `type_ok` | 34 | 34 | 1.00 [0.90, 1.00] |
| `value_correct` | 30 | 34 | 0.88 [0.73, 0.95] |
| `strict_pass` | 30 | 34 | 0.88 [0.73, 0.95] |
