# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter: `bench/qwen-adapter-20260613/8b-output-lora-r16-v3`
- dataset: `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `204`
- generated tokens: `2298`
- quality: `fail` (69/204)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `json_v4_eval_arithmetic_001` | `json_answer,arithmetic,eval` | `fail` | 24 | 0.89 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_002` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.27 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_003` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.32 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_004` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.56 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_005` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.13 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_006` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.29 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_007` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.54 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_008` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.11 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_009` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.46 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_010` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.44 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_011` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.26 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_012` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.11 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_013` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.53 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_014` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.24 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_015` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.45 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_016` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.56 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_017` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.04 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_018` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.35 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_019` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.31 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_020` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.11 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_021` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.27 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_022` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.58 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_023` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.27 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_024` | `json_answer,arithmetic,eval` | `fail` | 24 | 9.49 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_025` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.34 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_026` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.03 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_027` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.43 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_028` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.47 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_029` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.16 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_030` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.08 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_031` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.53 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_032` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.18 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_033` | `json_answer,arithmetic,eval` | `fail` | 24 | 11.35 | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_034` | `json_answer,arithmetic,eval` | `fail` | 9 | 9.02 | {"answer":"11115"} |
| `json_v4_eval_fact_001` | `json_answer,fact,eval` | `pass` | 5 | 7.52 | {"answer":"At"} |
| `json_v4_eval_fact_002` | `json_answer,fact,eval` | `pass` | 6 | 8.61 | {"answer":"Rn"} |
| `json_v4_eval_fact_003` | `json_answer,fact,eval` | `pass` | 5 | 8.11 | {"answer":"Fr"} |
| `json_v4_eval_fact_004` | `json_answer,fact,eval` | `pass` | 5 | 8.14 | {"answer":"Ra"} |
| `json_v4_eval_fact_005` | `json_answer,fact,eval` | `pass` | 5 | 8.18 | {"answer":"Ac"} |
| `json_v4_eval_fact_006` | `json_answer,fact,eval` | `pass` | 5 | 8.10 | {"answer":"Th"} |
| `json_v4_eval_fact_007` | `json_answer,fact,eval` | `pass` | 5 | 8.10 | {"answer":"Pa"} |
| `json_v4_eval_fact_008` | `json_answer,fact,eval` | `pass` | 5 | 7.95 | {"answer":"U"} |
| `json_v4_eval_fact_009` | `json_answer,fact,eval` | `pass` | 6 | 8.34 | {"answer":"Np"} |
| `json_v4_eval_fact_010` | `json_answer,fact,eval` | `pass` | 5 | 7.65 | {"answer":"Pu"} |
| `json_v4_eval_fact_011` | `json_answer,fact,eval` | `pass` | 5 | 7.76 | {"answer":"Am"} |
| `json_v4_eval_fact_012` | `json_answer,fact,eval` | `pass` | 6 | 8.58 | {"answer":"Cm"} |
| `json_v4_eval_fact_013` | `json_answer,fact,eval` | `pass` | 6 | 8.66 | {"answer":"Bk"} |
| `json_v4_eval_fact_014` | `json_answer,fact,eval` | `pass` | 6 | 8.73 | {"answer":"Cf"} |
| `json_v4_eval_fact_015` | `json_answer,fact,eval` | `pass` | 5 | 8.20 | {"answer":"Es"} |
| `json_v4_eval_fact_016` | `json_answer,fact,eval` | `pass` | 6 | 8.50 | {"answer":"Fm"} |
| `json_v4_eval_fact_017` | `json_answer,fact,eval` | `pass` | 5 | 8.02 | {"answer":"Md"} |
| `json_v4_eval_fact_018` | `json_answer,fact,eval` | `pass` | 5 | 7.85 | {"answer":"No"} |
| `json_v4_eval_fact_019` | `json_answer,fact,eval` | `pass` | 6 | 8.22 | {"answer":"Lr"} |
| `json_v4_eval_fact_020` | `json_answer,fact,eval` | `pass` | 6 | 8.35 | {"answer":"Rf"} |
| `json_v4_eval_fact_021` | `json_answer,fact,eval` | `pass` | 5 | 7.41 | {"answer":"Db"} |
| `json_v4_eval_fact_022` | `json_answer,fact,eval` | `pass` | 6 | 8.28 | {"answer":"Sg"} |
| `json_v4_eval_fact_023` | `json_answer,fact,eval` | `pass` | 6 | 8.56 | {"answer":"Bh"} |
| `json_v4_eval_fact_024` | `json_answer,fact,eval` | `pass` | 6 | 8.78 | {"answer":"Hs"} |
| `json_v4_eval_fact_025` | `json_answer,fact,eval` | `pass` | 5 | 7.20 | {"answer":"Mt"} |
| `json_v4_eval_fact_026` | `json_answer,fact,eval` | `pass` | 5 | 7.80 | {"answer":"Ds"} |
| `json_v4_eval_fact_027` | `json_answer,fact,eval` | `pass` | 6 | 8.31 | {"answer":"Rg"} |
| `json_v4_eval_fact_028` | `json_answer,fact,eval` | `pass` | 6 | 8.61 | {"answer":"Cn"} |
| `json_v4_eval_fact_029` | `json_answer,fact,eval` | `pass` | 5 | 7.98 | {"answer":"Nh"} |
| `json_v4_eval_fact_030` | `json_answer,fact,eval` | `pass` | 5 | 7.48 | {"answer":"Fl"} |
| `json_v4_eval_fact_031` | `json_answer,fact,eval` | `pass` | 5 | 8.15 | {"answer":"Mc"} |
| `json_v4_eval_fact_032` | `json_answer,fact,eval` | `pass` | 5 | 8.02 | {"answer":"Lv"} |
| `json_v4_eval_fact_033` | `json_answer,fact,eval` | `pass` | 5 | 8.11 | {"answer":"Ts"} |
| `json_v4_eval_fact_034` | `json_answer,fact,eval` | `fail` | 5 | 7.85 | {"answer":"O"} |
| `json_v4_eval_code_001` | `json_answer,code,eval` | `fail` | 7 | 7.22 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_002` | `json_answer,code,eval` | `fail` | 7 | 8.98 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_003` | `json_answer,code,eval` | `fail` | 7 | 9.09 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_004` | `json_answer,code,eval` | `fail` | 7 | 8.68 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_005` | `json_answer,code,eval` | `fail` | 7 | 9.11 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_006` | `json_answer,code,eval` | `fail` | 7 | 9.07 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_007` | `json_answer,code,eval` | `fail` | 7 | 8.83 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_008` | `json_answer,code,eval` | `fail` | 7 | 9.10 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_009` | `json_answer,code,eval` | `fail` | 7 | 8.84 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_010` | `json_answer,code,eval` | `fail` | 7 | 8.85 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_011` | `json_answer,code,eval` | `fail` | 7 | 8.79 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_012` | `json_answer,code,eval` | `fail` | 7 | 9.19 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_013` | `json_answer,code,eval` | `fail` | 7 | 8.08 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_014` | `json_answer,code,eval` | `fail` | 7 | 8.91 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_015` | `json_answer,code,eval` | `fail` | 7 | 9.18 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_016` | `json_answer,code,eval` | `fail` | 7 | 9.08 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_017` | `json_answer,code,eval` | `fail` | 7 | 8.97 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_018` | `json_answer,code,eval` | `fail` | 7 | 9.20 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_019` | `json_answer,code,eval` | `fail` | 7 | 9.10 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_020` | `json_answer,code,eval` | `fail` | 7 | 9.08 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_021` | `json_answer,code,eval` | `fail` | 7 | 9.13 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_022` | `json_answer,code,eval` | `fail` | 7 | 8.95 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_023` | `json_answer,code,eval` | `fail` | 7 | 9.00 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_024` | `json_answer,code,eval` | `fail` | 7 | 9.13 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_025` | `json_answer,code,eval` | `fail` | 7 | 8.77 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_026` | `json_answer,code,eval` | `fail` | 7 | 9.00 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_027` | `json_answer,code,eval` | `fail` | 7 | 8.42 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_028` | `json_answer,code,eval` | `fail` | 7 | 8.13 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_029` | `json_answer,code,eval` | `fail` | 7 | 9.16 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_030` | `json_answer,code,eval` | `fail` | 7 | 8.83 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_031` | `json_answer,code,eval` | `fail` | 7 | 9.17 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_032` | `json_answer,code,eval` | `fail` | 7 | 9.14 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_033` | `json_answer,code,eval` | `fail` | 7 | 9.03 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_034` | `json_answer,code,eval` | `fail` | 7 | 9.14 | {"answer":"eval_json_handler"} |
| `json_v4_eval_compiler_001` | `json_answer,compiler,eval` | `fail` | 10 | 4.21 | {"answer":"eval_qk_wide_load"} |
| `json_v4_eval_compiler_002` | `json_answer,compiler,eval` | `fail` | 8 | 6.97 | {"answer":"coalesced_read"} |
| `json_v4_eval_compiler_003` | `json_answer,compiler,eval` | `fail` | 9 | 7.33 | {"answer":"eval_qk_wavefront"} |
| `json_v4_eval_compiler_004` | `json_answer,compiler,eval` | `fail` | 9 | 7.30 | {"answer":"eval_qk_dequant"} |
| `json_v4_eval_compiler_005` | `json_answer,compiler,eval` | `fail` | 6 | 6.18 | {"answer":"gemv"} |
| `json_v4_eval_compiler_006` | `json_answer,compiler,eval` | `fail` | 9 | 6.58 | {"answer":"eval_qk_q1"} |
| `json_v4_eval_compiler_007` | `json_answer,compiler,eval` | `fail` | 10 | 7.62 | {"answer":"eval_qk_q6_block"} |
| `json_v4_eval_compiler_008` | `json_answer,compiler,eval` | `fail` | 9 | 6.08 | {"answer":"eval_qk_uop"} |
| `json_v4_eval_compiler_009` | `json_answer,compiler,eval` | `fail` | 5 | 5.40 | {"answer":"schedule"} |
| `json_v4_eval_compiler_010` | `json_answer,compiler,eval` | `fail` | 8 | 6.92 | {"answer":"eval_qk_policy"} |
| `json_v4_eval_compiler_011` | `json_answer,compiler,eval` | `fail` | 9 | 7.31 | {"answer":"eval_qk_suffix_cache"} |
| `json_v4_eval_compiler_012` | `json_answer,compiler,eval` | `fail` | 9 | 6.65 | {"answer":"eval_qk_json_axis"} |
| `json_v4_eval_compiler_013` | `json_answer,compiler,eval` | `fail` | 10 | 6.26 | {"answer":"eval_qk_wide_load"} |
| `json_v4_eval_compiler_014` | `json_answer,compiler,eval` | `fail` | 11 | 7.94 | {"answer":"eval_qk_coalesced_read"} |
| `json_v4_eval_compiler_015` | `json_answer,compiler,eval` | `fail` | 9 | 7.17 | {"answer":"eval_qk_wavefront"} |
| `json_v4_eval_compiler_016` | `json_answer,compiler,eval` | `fail` | 9 | 7.27 | {"answer":"eval_qk_dequant"} |
| `json_v4_eval_compiler_017` | `json_answer,compiler,eval` | `fail` | 6 | 6.15 | {"answer":"gemv"} |
| `json_v4_eval_compiler_018` | `json_answer,compiler,eval` | `fail` | 9 | 6.66 | {"answer":"eval_qk_q1"} |
| `json_v4_eval_compiler_019` | `json_answer,compiler,eval` | `fail` | 10 | 7.51 | {"answer":"eval_qk_q6_block"} |
| `json_v4_eval_compiler_020` | `json_answer,compiler,eval` | `fail` | 9 | 7.43 | {"answer":"eval_qk_uop"} |
| `json_v4_eval_compiler_021` | `json_answer,compiler,eval` | `fail` | 5 | 5.66 | {"answer":"schedule"} |
| `json_v4_eval_compiler_022` | `json_answer,compiler,eval` | `fail` | 9 | 7.26 | {"answer":"tensor_family_lowering_choice"} |
| `json_v4_eval_compiler_023` | `json_answer,compiler,eval` | `fail` | 7 | 5.99 | {"answer":"prefix_hidden_state"} |
| `json_v4_eval_compiler_024` | `json_answer,compiler,eval` | `fail` | 9 | 6.22 | {"answer":"eval_qk_json_axis"} |
| `json_v4_eval_compiler_025` | `json_answer,compiler,eval` | `fail` | 10 | 7.22 | {"answer":"eval_qk_wide_load"} |
| `json_v4_eval_compiler_026` | `json_answer,compiler,eval` | `fail` | 8 | 5.53 | {"answer":"coalesced_read"} |
| `json_v4_eval_compiler_027` | `json_answer,compiler,eval` | `fail` | 9 | 7.21 | {"answer":"eval_qk_wavefront"} |
| `json_v4_eval_compiler_028` | `json_answer,compiler,eval` | `fail` | 9 | 7.27 | {"answer":"eval_qk_dequant"} |
| `json_v4_eval_compiler_029` | `json_answer,compiler,eval` | `fail` | 6 | 4.72 | {"answer":"gemv"} |
| `json_v4_eval_compiler_030` | `json_answer,compiler,eval` | `fail` | 9 | 6.22 | {"answer":"eval_qk_q1"} |
| `json_v4_eval_compiler_031` | `json_answer,compiler,eval` | `fail` | 10 | 7.64 | {"answer":"eval_qk_q6_block"} |
| `json_v4_eval_compiler_032` | `json_answer,compiler,eval` | `fail` | 9 | 7.00 | {"answer":"eval_qk_uop"} |
| `json_v4_eval_compiler_033` | `json_answer,compiler,eval` | `fail` | 5 | 5.47 | {"answer":"schedule"} |
| `json_v4_eval_compiler_034` | `json_answer,compiler,eval` | `fail` | 9 | 7.03 | {"answer":"tensor_family_lowering_choice"} |
| `json_v4_eval_string_001` | `json_answer,string,eval` | `pass` | 12 | 9.24 | {"answer":"EVALQK001AB"} |
| `json_v4_eval_string_002` | `json_answer,string,eval` | `fail` | 5 | 6.04 | {"answer":"EA"} |
| `json_v4_eval_string_003` | `json_answer,string,eval` | `fail` | 24 | 11.30 | {"answer":"koob1111111111111111111 |
| `json_v4_eval_string_004` | `json_answer,string,eval` | `pass` | 12 | 9.25 | {"answer":"EVALQK004AB"} |
| `json_v4_eval_string_005` | `json_answer,string,eval` | `fail` | 5 | 6.14 | {"answer":"EA"} |
| `json_v4_eval_string_006` | `json_answer,string,eval` | `fail` | 12 | 9.06 | {"answer":"kcab006kqle"} |
| `json_v4_eval_string_007` | `json_answer,string,eval` | `pass` | 12 | 9.16 | {"answer":"EVALQK007AB"} |
| `json_v4_eval_string_008` | `json_answer,string,eval` | `fail` | 5 | 6.12 | {"answer":"EA"} |
| `json_v4_eval_string_009` | `json_answer,string,eval` | `fail` | 24 | 11.32 | {"answer":"koob1111111111111111111 |
| `json_v4_eval_string_010` | `json_answer,string,eval` | `pass` | 12 | 9.39 | {"answer":"EVALQK010AB"} |
| `json_v4_eval_string_011` | `json_answer,string,eval` | `fail` | 5 | 6.18 | {"answer":"EA"} |
| `json_v4_eval_string_012` | `json_answer,string,eval` | `fail` | 24 | 11.15 | {"answer":"ko12ab11111111111111111 |
| `json_v4_eval_string_013` | `json_answer,string,eval` | `pass` | 12 | 9.13 | {"answer":"EVALQK013AB"} |
| `json_v4_eval_string_014` | `json_answer,string,eval` | `fail` | 5 | 6.15 | {"answer":"EA"} |
| `json_v4_eval_string_015` | `json_answer,string,eval` | `fail` | 24 | 11.30 | {"answer":"ko15ab11111111111111111 |
| `json_v4_eval_string_016` | `json_answer,string,eval` | `pass` | 12 | 8.97 | {"answer":"EVALQK016AB"} |
| `json_v4_eval_string_017` | `json_answer,string,eval` | `fail` | 5 | 6.18 | {"answer":"EA"} |
| `json_v4_eval_string_018` | `json_answer,string,eval` | `fail` | 24 | 10.82 | {"answer":"kc81111111111111111111 |
| `json_v4_eval_string_019` | `json_answer,string,eval` | `pass` | 12 | 9.36 | {"answer":"EVALQK019AB"} |
| `json_v4_eval_string_020` | `json_answer,string,eval` | `fail` | 5 | 6.07 | {"answer":"EA"} |
| `json_v4_eval_string_021` | `json_answer,string,eval` | `fail` | 12 | 9.48 | {"answer":"kcab210kqve"} |
| `json_v4_eval_string_022` | `json_answer,string,eval` | `pass` | 12 | 9.18 | {"answer":"EVALQK022AB"} |
| `json_v4_eval_string_023` | `json_answer,string,eval` | `fail` | 5 | 6.21 | {"answer":"EA"} |
| `json_v4_eval_string_024` | `json_answer,string,eval` | `fail` | 12 | 9.57 | {"answer":"kcab420kqave"} |
| `json_v4_eval_string_025` | `json_answer,string,eval` | `pass` | 12 | 7.88 | {"answer":"EVALQK025AB"} |
| `json_v4_eval_string_026` | `json_answer,string,eval` | `fail` | 5 | 5.74 | {"answer":"EA"} |
| `json_v4_eval_string_027` | `json_answer,string,eval` | `fail` | 11 | 9.37 | {"answer":"bk720kqale"} |
| `json_v4_eval_string_028` | `json_answer,string,eval` | `pass` | 12 | 9.20 | {"answer":"EVALQK028AB"} |
| `json_v4_eval_string_029` | `json_answer,string,eval` | `fail` | 5 | 5.55 | {"answer":"EA"} |
| `json_v4_eval_string_030` | `json_answer,string,eval` | `fail` | 24 | 11.34 | {"answer":"ko11111111111111111111 |
| `json_v4_eval_string_031` | `json_answer,string,eval` | `pass` | 12 | 9.00 | {"answer":"EVALQK031AB"} |
| `json_v4_eval_string_032` | `json_answer,string,eval` | `fail` | 5 | 6.16 | {"answer":"EA"} |
| `json_v4_eval_string_033` | `json_answer,string,eval` | `fail` | 24 | 11.45 | {"answer":"ko11111111111111111111 |
| `json_v4_eval_string_034` | `json_answer,string,eval` | `pass` | 12 | 9.29 | {"answer":"EVALQK034AB"} |
| `json_v4_eval_categorization_001` | `json_answer,categorization,eval` | `fail` | 11 | 6.84 | {"answer":"eval_even_bucket_001"} |
| `json_v4_eval_categorization_002` | `json_answer,categorization,eval` | `pass` | 11 | 6.78 | {"answer":"eval_even_bucket_002"} |
| `json_v4_eval_categorization_003` | `json_answer,categorization,eval` | `fail` | 11 | 6.96 | {"answer":"eval_even_bucket_001"} |
| `json_v4_eval_categorization_004` | `json_answer,categorization,eval` | `fail` | 11 | 6.92 | {"answer":"eval_even_bucket_001"} |
| `json_v4_eval_categorization_005` | `json_answer,categorization,eval` | `fail` | 11 | 6.82 | {"answer":"eval_even_bucket_005"} |
| `json_v4_eval_categorization_006` | `json_answer,categorization,eval` | `pass` | 11 | 7.11 | {"answer":"eval_even_bucket_006"} |
| `json_v4_eval_categorization_007` | `json_answer,categorization,eval` | `fail` | 11 | 7.08 | {"answer":"eval_even_bucket_007"} |
| `json_v4_eval_categorization_008` | `json_answer,categorization,eval` | `pass` | 11 | 6.93 | {"answer":"eval_even_bucket_008"} |
| `json_v4_eval_categorization_009` | `json_answer,categorization,eval` | `pass` | 11 | 7.08 | {"answer":"eval_odd_bucket_009"} |
| `json_v4_eval_categorization_010` | `json_answer,categorization,eval` | `fail` | 11 | 7.06 | {"answer":"eval_even_bucket_011"} |
| `json_v4_eval_categorization_011` | `json_answer,categorization,eval` | `pass` | 11 | 6.85 | {"answer":"eval_odd_bucket_011"} |
| `json_v4_eval_categorization_012` | `json_answer,categorization,eval` | `pass` | 11 | 5.87 | {"answer":"eval_even_bucket_012"} |
| `json_v4_eval_categorization_013` | `json_answer,categorization,eval` | `fail` | 11 | 7.14 | {"answer":"eval_odd_bucket_011"} |
| `json_v4_eval_categorization_014` | `json_answer,categorization,eval` | `pass` | 11 | 7.15 | {"answer":"eval_even_bucket_014"} |
| `json_v4_eval_categorization_015` | `json_answer,categorization,eval` | `pass` | 11 | 7.12 | {"answer":"eval_odd_bucket_015"} |
| `json_v4_eval_categorization_016` | `json_answer,categorization,eval` | `pass` | 11 | 7.06 | {"answer":"eval_even_bucket_016"} |
| `json_v4_eval_categorization_017` | `json_answer,categorization,eval` | `pass` | 11 | 7.03 | {"answer":"eval_odd_bucket_017"} |
| `json_v4_eval_categorization_018` | `json_answer,categorization,eval` | `pass` | 11 | 5.45 | {"answer":"eval_even_bucket_018"} |
| `json_v4_eval_categorization_019` | `json_answer,categorization,eval` | `pass` | 11 | 7.14 | {"answer":"eval_odd_bucket_019"} |
| `json_v4_eval_categorization_020` | `json_answer,categorization,eval` | `pass` | 11 | 6.84 | {"answer":"eval_even_bucket_020"} |
| `json_v4_eval_categorization_021` | `json_answer,categorization,eval` | `fail` | 11 | 6.87 | {"answer":"eval_even_bucket_021"} |
| `json_v4_eval_categorization_022` | `json_answer,categorization,eval` | `pass` | 11 | 7.10 | {"answer":"eval_even_bucket_022"} |
| `json_v4_eval_categorization_023` | `json_answer,categorization,eval` | `pass` | 11 | 7.06 | {"answer":"eval_odd_bucket_023"} |
| `json_v4_eval_categorization_024` | `json_answer,categorization,eval` | `pass` | 11 | 5.49 | {"answer":"eval_even_bucket_024"} |
| `json_v4_eval_categorization_025` | `json_answer,categorization,eval` | `pass` | 11 | 7.04 | {"answer":"eval_odd_bucket_025"} |
| `json_v4_eval_categorization_026` | `json_answer,categorization,eval` | `pass` | 11 | 7.16 | {"answer":"eval_even_bucket_026"} |
| `json_v4_eval_categorization_027` | `json_answer,categorization,eval` | `fail` | 11 | 6.33 | {"answer":"eval_even_bucket_027"} |
| `json_v4_eval_categorization_028` | `json_answer,categorization,eval` | `pass` | 11 | 7.13 | {"answer":"eval_even_bucket_028"} |
| `json_v4_eval_categorization_029` | `json_answer,categorization,eval` | `pass` | 11 | 7.16 | {"answer":"eval_odd_bucket_029"} |
| `json_v4_eval_categorization_030` | `json_answer,categorization,eval` | `pass` | 11 | 7.03 | {"answer":"eval_even_bucket_030"} |
| `json_v4_eval_categorization_031` | `json_answer,categorization,eval` | `pass` | 11 | 7.15 | {"answer":"eval_odd_bucket_031"} |
| `json_v4_eval_categorization_032` | `json_answer,categorization,eval` | `pass` | 11 | 6.36 | {"answer":"eval_even_bucket_032"} |
| `json_v4_eval_categorization_033` | `json_answer,categorization,eval` | `fail` | 11 | 6.42 | {"answer":"eval_even_bucket_033"} |
| `json_v4_eval_categorization_034` | `json_answer,categorization,eval` | `pass` | 11 | 6.80 | {"answer":"eval_even_bucket_034"} |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `arithmetic` | 0 | 34 | 0.00 |
| `categorization` | 24 | 34 | 0.71 |
| `code` | 0 | 34 | 0.00 |
| `compiler` | 0 | 34 | 0.00 |
| `eval` | 69 | 204 | 0.34 |
| `fact` | 33 | 34 | 0.97 |
| `json_answer` | 69 | 204 | 0.34 |
| `string` | 12 | 34 | 0.35 |

## JSON Quality Axes

| axis | passed | scored | pass rate [95% CI] |
|---|---:|---:|---:|
| `parse_valid` | 164 | 204 | 0.80 [0.74, 0.85] |
| `no_extra_text` | 164 | 204 | 0.80 [0.74, 0.85] |
| `schema_ok` | 164 | 204 | 0.80 [0.74, 0.85] |
| `type_ok` | 163 | 204 | 0.80 [0.74, 0.85] |
| `value_correct` | 69 | 204 | 0.34 [0.28, 0.41] |
| `strict_pass` | 69 | 204 | 0.34 [0.28, 0.41] |
