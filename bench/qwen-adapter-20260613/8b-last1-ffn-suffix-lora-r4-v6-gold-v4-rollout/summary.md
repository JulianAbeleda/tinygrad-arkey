# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter: `bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v6-gold-v4`
- dataset: `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `204`
- generated tokens: `2017`
- quality: `fail` (162/204)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `json_v4_eval_arithmetic_001` | `json_answer,arithmetic,eval` | `pass` | 9 | 0.33 | {"answer":10521} |
| `json_v4_eval_arithmetic_002` | `json_answer,arithmetic,eval` | `pass` | 9 | 19.15 | {"answer": 5110} |
| `json_v4_eval_arithmetic_003` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.76 | {"answer":3509} |
| `json_v4_eval_arithmetic_004` | `json_answer,arithmetic,eval` | `pass` | 9 | 22.67 | {"answer":10575} |
| `json_v4_eval_arithmetic_005` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.94 | {"answer": 5200} |
| `json_v4_eval_arithmetic_006` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.91 | {"answer":3521} |
| `json_v4_eval_arithmetic_007` | `json_answer,arithmetic,eval` | `pass` | 9 | 22.79 | {"answer":10629} |
| `json_v4_eval_arithmetic_008` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.22 | {"answer": 52800000000000000000 |
| `json_v4_eval_arithmetic_009` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.87 | {"answer":3533} |
| `json_v4_eval_arithmetic_010` | `json_answer,arithmetic,eval` | `pass` | 9 | 21.81 | {"answer":10683} |
| `json_v4_eval_arithmetic_011` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.76 | {"answer":155500000000000000000 |
| `json_v4_eval_arithmetic_012` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.48 | {"answer":3545} |
| `json_v4_eval_arithmetic_013` | `json_answer,arithmetic,eval` | `pass` | 9 | 22.71 | {"answer":10737} |
| `json_v4_eval_arithmetic_014` | `json_answer,arithmetic,eval` | `pass` | 9 | 19.08 | {"answer": 5272} |
| `json_v4_eval_arithmetic_015` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.80 | {"answer":3557} |
| `json_v4_eval_arithmetic_016` | `json_answer,arithmetic,eval` | `pass` | 9 | 22.82 | {"answer":10791} |
| `json_v4_eval_arithmetic_017` | `json_answer,arithmetic,eval` | `fail` | 14 | 24.45 | {"answer":37*11+5000} |
| `json_v4_eval_arithmetic_018` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.85 | {"answer":3569} |
| `json_v4_eval_arithmetic_019` | `json_answer,arithmetic,eval` | `pass` | 9 | 21.93 | {"answer":10845} |
| `json_v4_eval_arithmetic_020` | `json_answer,arithmetic,eval` | `fail` | 24 | 27.95 | {"answer": 51000000000000000000 |
| `json_v4_eval_arithmetic_021` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.86 | {"answer":3581} |
| `json_v4_eval_arithmetic_022` | `json_answer,arithmetic,eval` | `pass` | 9 | 22.62 | {"answer":10899} |
| `json_v4_eval_arithmetic_023` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.18 | {"answer":346400000000000000000 |
| `json_v4_eval_arithmetic_024` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.83 | {"answer":3593} |
| `json_v4_eval_arithmetic_025` | `json_answer,arithmetic,eval` | `pass` | 9 | 20.64 | {"answer":10953} |
| `json_v4_eval_arithmetic_026` | `json_answer,arithmetic,eval` | `fail` | 9 | 18.11 | {"answer": 5466} |
| `json_v4_eval_arithmetic_027` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.88 | {"answer":3605} |
| `json_v4_eval_arithmetic_028` | `json_answer,arithmetic,eval` | `pass` | 9 | 21.91 | {"answer":11007} |
| `json_v4_eval_arithmetic_029` | `json_answer,arithmetic,eval` | `pass` | 9 | 19.06 | {"answer": 5245} |
| `json_v4_eval_arithmetic_030` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.76 | {"answer":3617} |
| `json_v4_eval_arithmetic_031` | `json_answer,arithmetic,eval` | `pass` | 9 | 22.61 | {"answer":11061} |
| `json_v4_eval_arithmetic_032` | `json_answer,arithmetic,eval` | `fail` | 9 | 19.03 | {"answer": 5136} |
| `json_v4_eval_arithmetic_033` | `json_answer,arithmetic,eval` | `pass` | 8 | 18.78 | {"answer":3629} |
| `json_v4_eval_arithmetic_034` | `json_answer,arithmetic,eval` | `pass` | 9 | 22.74 | {"answer":11115} |
| `json_v4_eval_fact_001` | `json_answer,fact,eval` | `pass` | 5 | 16.33 | {"answer":"At"} |
| `json_v4_eval_fact_002` | `json_answer,fact,eval` | `pass` | 6 | 22.98 | {"answer":"Rn"} |
| `json_v4_eval_fact_003` | `json_answer,fact,eval` | `pass` | 5 | 20.47 | {"answer":"Fr"} |
| `json_v4_eval_fact_004` | `json_answer,fact,eval` | `pass` | 5 | 20.47 | {"answer":"Ra"} |
| `json_v4_eval_fact_005` | `json_answer,fact,eval` | `pass` | 5 | 20.52 | {"answer":"Ac"} |
| `json_v4_eval_fact_006` | `json_answer,fact,eval` | `pass` | 5 | 20.34 | {"answer":"Th"} |
| `json_v4_eval_fact_007` | `json_answer,fact,eval` | `pass` | 5 | 19.83 | {"answer":"Pa"} |
| `json_v4_eval_fact_008` | `json_answer,fact,eval` | `pass` | 5 | 20.49 | {"answer":"U"} |
| `json_v4_eval_fact_009` | `json_answer,fact,eval` | `pass` | 6 | 21.68 | {"answer":"Np"} |
| `json_v4_eval_fact_010` | `json_answer,fact,eval` | `pass` | 5 | 19.77 | {"answer":"Pu"} |
| `json_v4_eval_fact_011` | `json_answer,fact,eval` | `pass` | 5 | 20.45 | {"answer":"Am"} |
| `json_v4_eval_fact_012` | `json_answer,fact,eval` | `pass` | 6 | 23.00 | {"answer":"Cm"} |
| `json_v4_eval_fact_013` | `json_answer,fact,eval` | `pass` | 6 | 22.30 | {"answer":"Bk"} |
| `json_v4_eval_fact_014` | `json_answer,fact,eval` | `pass` | 6 | 23.01 | {"answer":"Cf"} |
| `json_v4_eval_fact_015` | `json_answer,fact,eval` | `pass` | 5 | 20.39 | {"answer":"Es"} |
| `json_v4_eval_fact_016` | `json_answer,fact,eval` | `pass` | 6 | 23.00 | {"answer":"Fm"} |
| `json_v4_eval_fact_017` | `json_answer,fact,eval` | `pass` | 5 | 19.17 | {"answer":"Md"} |
| `json_v4_eval_fact_018` | `json_answer,fact,eval` | `pass` | 5 | 20.45 | {"answer":"No"} |
| `json_v4_eval_fact_019` | `json_answer,fact,eval` | `pass` | 6 | 20.08 | {"answer":"Lr"} |
| `json_v4_eval_fact_020` | `json_answer,fact,eval` | `pass` | 6 | 22.29 | {"answer":"Rf"} |
| `json_v4_eval_fact_021` | `json_answer,fact,eval` | `pass` | 5 | 19.84 | {"answer":"Db"} |
| `json_v4_eval_fact_022` | `json_answer,fact,eval` | `pass` | 6 | 21.67 | {"answer":"Sg"} |
| `json_v4_eval_fact_023` | `json_answer,fact,eval` | `pass` | 6 | 21.61 | {"answer":"Bh"} |
| `json_v4_eval_fact_024` | `json_answer,fact,eval` | `pass` | 6 | 23.02 | {"answer":"Hs"} |
| `json_v4_eval_fact_025` | `json_answer,fact,eval` | `pass` | 5 | 19.18 | {"answer":"Mt"} |
| `json_v4_eval_fact_026` | `json_answer,fact,eval` | `pass` | 5 | 19.16 | {"answer":"Ds"} |
| `json_v4_eval_fact_027` | `json_answer,fact,eval` | `pass` | 6 | 21.61 | {"answer":"Rg"} |
| `json_v4_eval_fact_028` | `json_answer,fact,eval` | `pass` | 6 | 21.66 | {"answer":"Cn"} |
| `json_v4_eval_fact_029` | `json_answer,fact,eval` | `pass` | 5 | 19.73 | {"answer":"Nh"} |
| `json_v4_eval_fact_030` | `json_answer,fact,eval` | `pass` | 5 | 19.14 | {"answer":"Fl"} |
| `json_v4_eval_fact_031` | `json_answer,fact,eval` | `pass` | 5 | 19.76 | {"answer":"Mc"} |
| `json_v4_eval_fact_032` | `json_answer,fact,eval` | `pass` | 5 | 19.78 | {"answer":"Lv"} |
| `json_v4_eval_fact_033` | `json_answer,fact,eval` | `pass` | 5 | 19.76 | {"answer":"Ts"} |
| `json_v4_eval_fact_034` | `json_answer,fact,eval` | `pass` | 6 | 20.69 | {"answer":"Og"} |
| `json_v4_eval_code_001` | `json_answer,code,eval` | `pass` | 11 | 18.67 | {"answer":"eval_json_handler_001"} |
| `json_v4_eval_code_002` | `json_answer,code,eval` | `pass` | 11 | 31.88 | {"answer":"eval_json_handler_002"} |
| `json_v4_eval_code_003` | `json_answer,code,eval` | `pass` | 11 | 31.86 | {"answer":"eval_json_handler_003"} |
| `json_v4_eval_code_004` | `json_answer,code,eval` | `pass` | 11 | 31.79 | {"answer":"eval_json_handler_004"} |
| `json_v4_eval_code_005` | `json_answer,code,eval` | `pass` | 11 | 31.83 | {"answer":"eval_json_handler_005"} |
| `json_v4_eval_code_006` | `json_answer,code,eval` | `pass` | 11 | 31.83 | {"answer":"eval_json_handler_006"} |
| `json_v4_eval_code_007` | `json_answer,code,eval` | `pass` | 11 | 31.80 | {"answer":"eval_json_handler_007"} |
| `json_v4_eval_code_008` | `json_answer,code,eval` | `pass` | 11 | 31.89 | {"answer":"eval_json_handler_008"} |
| `json_v4_eval_code_009` | `json_answer,code,eval` | `pass` | 11 | 31.83 | {"answer":"eval_json_handler_009"} |
| `json_v4_eval_code_010` | `json_answer,code,eval` | `pass` | 11 | 31.11 | {"answer":"eval_json_handler_010"} |
| `json_v4_eval_code_011` | `json_answer,code,eval` | `pass` | 11 | 31.73 | {"answer":"eval_json_handler_011"} |
| `json_v4_eval_code_012` | `json_answer,code,eval` | `pass` | 11 | 31.80 | {"answer":"eval_json_handler_012"} |
| `json_v4_eval_code_013` | `json_answer,code,eval` | `pass` | 11 | 31.78 | {"answer":"eval_json_handler_013"} |
| `json_v4_eval_code_014` | `json_answer,code,eval` | `pass` | 11 | 31.86 | {"answer":"eval_json_handler_014"} |
| `json_v4_eval_code_015` | `json_answer,code,eval` | `pass` | 11 | 31.79 | {"answer":"eval_json_handler_015"} |
| `json_v4_eval_code_016` | `json_answer,code,eval` | `pass` | 11 | 31.50 | {"answer":"eval_json_handler_016"} |
| `json_v4_eval_code_017` | `json_answer,code,eval` | `pass` | 11 | 31.73 | {"answer":"eval_json_handler_017"} |
| `json_v4_eval_code_018` | `json_answer,code,eval` | `pass` | 11 | 30.31 | {"answer":"eval_json_handler_018"} |
| `json_v4_eval_code_019` | `json_answer,code,eval` | `pass` | 11 | 28.23 | {"answer":"eval_json_handler_019"} |
| `json_v4_eval_code_020` | `json_answer,code,eval` | `pass` | 11 | 28.55 | {"answer":"eval_json_handler_020"} |
| `json_v4_eval_code_021` | `json_answer,code,eval` | `pass` | 11 | 31.68 | {"answer":"eval_json_handler_021"} |
| `json_v4_eval_code_022` | `json_answer,code,eval` | `pass` | 11 | 31.62 | {"answer":"eval_json_handler_022"} |
| `json_v4_eval_code_023` | `json_answer,code,eval` | `pass` | 11 | 31.81 | {"answer":"eval_json_handler_023"} |
| `json_v4_eval_code_024` | `json_answer,code,eval` | `pass` | 11 | 31.68 | {"answer":"eval_json_handler_024"} |
| `json_v4_eval_code_025` | `json_answer,code,eval` | `pass` | 11 | 31.77 | {"answer":"eval_json_handler_025"} |
| `json_v4_eval_code_026` | `json_answer,code,eval` | `pass` | 11 | 31.71 | {"answer":"eval_json_handler_026"} |
| `json_v4_eval_code_027` | `json_answer,code,eval` | `pass` | 11 | 31.82 | {"answer":"eval_json_handler_027"} |
| `json_v4_eval_code_028` | `json_answer,code,eval` | `pass` | 11 | 28.35 | {"answer":"eval_json_handler_028"} |
| `json_v4_eval_code_029` | `json_answer,code,eval` | `pass` | 11 | 31.53 | {"answer":"eval_json_handler_029"} |
| `json_v4_eval_code_030` | `json_answer,code,eval` | `pass` | 11 | 30.63 | {"answer":"eval_json_handler_030"} |
| `json_v4_eval_code_031` | `json_answer,code,eval` | `pass` | 11 | 31.30 | {"answer":"eval_json_handler_031"} |
| `json_v4_eval_code_032` | `json_answer,code,eval` | `pass` | 11 | 31.69 | {"answer":"eval_json_handler_032"} |
| `json_v4_eval_code_033` | `json_answer,code,eval` | `pass` | 11 | 31.61 | {"answer":"eval_json_handler_033"} |
| `json_v4_eval_code_034` | `json_answer,code,eval` | `pass` | 11 | 31.61 | {"answer":"eval_json_handler_034"} |
| `json_v4_eval_compiler_001` | `json_answer,compiler,eval` | `fail` | 10 | 11.15 | {"answer":"eval_qk_wide_load"} |
| `json_v4_eval_compiler_002` | `json_answer,compiler,eval` | `fail` | 9 | 12.88 | {"answer":"qk_coalesced"} |
| `json_v4_eval_compiler_003` | `json_answer,compiler,eval` | `fail` | 9 | 12.87 | {"answer":"eval_qk_wavefront"} |
| `json_v4_eval_compiler_004` | `json_answer,compiler,eval` | `pass` | 13 | 16.57 | {"answer":"eval_qk_dequant_004"} |
| `json_v4_eval_compiler_005` | `json_answer,compiler,eval` | `fail` | 6 | 9.58 | {"answer":"gemv"} |
| `json_v4_eval_compiler_006` | `json_answer,compiler,eval` | `fail` | 10 | 12.69 | {"answer":"eval_qk_q4_block"} |
| `json_v4_eval_compiler_007` | `json_answer,compiler,eval` | `fail` | 10 | 13.67 | {"answer":"eval_qk_q6_block"} |
| `json_v4_eval_compiler_008` | `json_answer,compiler,eval` | `fail` | 9 | 13.25 | {"answer":"eval_qk_uop"} |
| `json_v4_eval_compiler_009` | `json_answer,compiler,eval` | `pass` | 12 | 15.34 | {"answer":"eval_qk_beam_009"} |
| `json_v4_eval_compiler_010` | `json_answer,compiler,eval` | `pass` | 12 | 15.62 | {"answer":"eval_qk_policy_010"} |
| `json_v4_eval_compiler_011` | `json_answer,compiler,eval` | `pass` | 13 | 16.64 | {"answer":"eval_qk_suffix_cache_011"} |
| `json_v4_eval_compiler_012` | `json_answer,compiler,eval` | `pass` | 13 | 15.47 | {"answer":"eval_qk_json_axis_012"} |
| `json_v4_eval_compiler_013` | `json_answer,compiler,eval` | `fail` | 10 | 14.00 | {"answer":"eval_qk_wide_load"} |
| `json_v4_eval_compiler_014` | `json_answer,compiler,eval` | `fail` | 9 | 12.80 | {"answer":"qk_coalesced"} |
| `json_v4_eval_compiler_015` | `json_answer,compiler,eval` | `fail` | 9 | 12.79 | {"answer":"eval_qk_wavefront"} |
| `json_v4_eval_compiler_016` | `json_answer,compiler,eval` | `pass` | 13 | 16.54 | {"answer":"eval_qk_dequant_016"} |
| `json_v4_eval_compiler_017` | `json_answer,compiler,eval` | `fail` | 6 | 9.52 | {"answer":"gemv"} |
| `json_v4_eval_compiler_018` | `json_answer,compiler,eval` | `fail` | 10 | 12.67 | {"answer":"eval_qk_q4_block"} |
| `json_v4_eval_compiler_019` | `json_answer,compiler,eval` | `fail` | 10 | 13.61 | {"answer":"eval_qk_q6_block"} |
| `json_v4_eval_compiler_020` | `json_answer,compiler,eval` | `fail` | 9 | 13.30 | {"answer":"eval_qk_uop"} |
| `json_v4_eval_compiler_021` | `json_answer,compiler,eval` | `pass` | 12 | 16.90 | {"answer":"eval_qk_beam_021"} |
| `json_v4_eval_compiler_022` | `json_answer,compiler,eval` | `pass` | 12 | 15.61 | {"answer":"eval_qk_policy_022"} |
| `json_v4_eval_compiler_023` | `json_answer,compiler,eval` | `pass` | 13 | 16.49 | {"answer":"eval_qk_suffix_cache_023"} |
| `json_v4_eval_compiler_024` | `json_answer,compiler,eval` | `pass` | 13 | 15.40 | {"answer":"eval_qk_json_axis_024"} |
| `json_v4_eval_compiler_025` | `json_answer,compiler,eval` | `pass` | 14 | 17.02 | {"answer":"eval_qk_wide_load_025"} |
| `json_v4_eval_compiler_026` | `json_answer,compiler,eval` | `fail` | 9 | 12.77 | {"answer":"qk_coalesced"} |
| `json_v4_eval_compiler_027` | `json_answer,compiler,eval` | `fail` | 9 | 12.76 | {"answer":"eval_qk_wavefront"} |
| `json_v4_eval_compiler_028` | `json_answer,compiler,eval` | `pass` | 13 | 16.46 | {"answer":"eval_qk_dequant_028"} |
| `json_v4_eval_compiler_029` | `json_answer,compiler,eval` | `fail` | 6 | 9.45 | {"answer":"gemv"} |
| `json_v4_eval_compiler_030` | `json_answer,compiler,eval` | `fail` | 10 | 12.56 | {"answer":"eval_qk_q4_block"} |
| `json_v4_eval_compiler_031` | `json_answer,compiler,eval` | `fail` | 10 | 13.49 | {"answer":"eval_qk_q6_block"} |
| `json_v4_eval_compiler_032` | `json_answer,compiler,eval` | `fail` | 9 | 13.19 | {"answer":"eval_qk_uop"} |
| `json_v4_eval_compiler_033` | `json_answer,compiler,eval` | `pass` | 12 | 17.00 | {"answer":"eval_qk_beam_033"} |
| `json_v4_eval_compiler_034` | `json_answer,compiler,eval` | `pass` | 12 | 15.57 | {"answer":"eval_qk_policy_034"} |
| `json_v4_eval_string_001` | `json_answer,string,eval` | `pass` | 12 | 22.53 | {"answer":"EVALQK001AB"} |
| `json_v4_eval_string_002` | `json_answer,string,eval` | `pass` | 9 | 12.48 | {"answer":"eab002"} |
| `json_v4_eval_string_003` | `json_answer,string,eval` | `fail` | 12 | 24.01 | {"answer":"k003abqleve"} |
| `json_v4_eval_string_004` | `json_answer,string,eval` | `pass` | 12 | 22.54 | {"answer":"EVALQK004AB"} |
| `json_v4_eval_string_005` | `json_answer,string,eval` | `pass` | 9 | 15.67 | {"answer":"eab005"} |
| `json_v4_eval_string_006` | `json_answer,string,eval` | `fail` | 12 | 24.01 | {"answer":"k006abqleve"} |
| `json_v4_eval_string_007` | `json_answer,string,eval` | `pass` | 12 | 22.52 | {"answer":"EVALQK007AB"} |
| `json_v4_eval_string_008` | `json_answer,string,eval` | `pass` | 9 | 15.95 | {"answer":"eab008"} |
| `json_v4_eval_string_009` | `json_answer,string,eval` | `fail` | 12 | 23.98 | {"answer":"k009abqleve"} |
| `json_v4_eval_string_010` | `json_answer,string,eval` | `pass` | 12 | 20.45 | {"answer":"EVALQK010AB"} |
| `json_v4_eval_string_011` | `json_answer,string,eval` | `pass` | 9 | 15.76 | {"answer":"eab011"} |
| `json_v4_eval_string_012` | `json_answer,string,eval` | `fail` | 12 | 23.85 | {"answer":"kcb210kqve"} |
| `json_v4_eval_string_013` | `json_answer,string,eval` | `pass` | 12 | 22.42 | {"answer":"EVALQK013AB"} |
| `json_v4_eval_string_014` | `json_answer,string,eval` | `pass` | 9 | 15.89 | {"answer":"eab014"} |
| `json_v4_eval_string_015` | `json_answer,string,eval` | `fail` | 12 | 23.88 | {"answer":"kcb510kqale"} |
| `json_v4_eval_string_016` | `json_answer,string,eval` | `pass` | 12 | 22.00 | {"answer":"EVALQK016AB"} |
| `json_v4_eval_string_017` | `json_answer,string,eval` | `pass` | 9 | 15.43 | {"answer":"eab017"} |
| `json_v4_eval_string_018` | `json_answer,string,eval` | `fail` | 12 | 23.93 | {"answer":"k810qklaev"} |
| `json_v4_eval_string_019` | `json_answer,string,eval` | `pass` | 12 | 17.54 | {"answer":"EVALQK019AB"} |
| `json_v4_eval_string_020` | `json_answer,string,eval` | `pass` | 9 | 11.47 | {"answer":"eab020"} |
| `json_v4_eval_string_021` | `json_answer,string,eval` | `fail` | 12 | 21.12 | {"answer":"kcb210kqve"} |
| `json_v4_eval_string_022` | `json_answer,string,eval` | `pass` | 12 | 22.54 | {"answer":"EVALQK022AB"} |
| `json_v4_eval_string_023` | `json_answer,string,eval` | `fail` | 6 | 11.63 | {"answer":"EAB"} |
| `json_v4_eval_string_024` | `json_answer,string,eval` | `fail` | 12 | 23.89 | {"answer":"k420kqalev"} |
| `json_v4_eval_string_025` | `json_answer,string,eval` | `pass` | 12 | 22.55 | {"answer":"EVALQK025AB"} |
| `json_v4_eval_string_026` | `json_answer,string,eval` | `fail` | 6 | 11.57 | {"answer":"EAB"} |
| `json_v4_eval_string_027` | `json_answer,string,eval` | `fail` | 12 | 23.99 | {"answer":"b2702kqale"} |
| `json_v4_eval_string_028` | `json_answer,string,eval` | `pass` | 12 | 22.42 | {"answer":"EVALQK028AB"} |
| `json_v4_eval_string_029` | `json_answer,string,eval` | `pass` | 9 | 15.87 | {"answer":"eab029"} |
| `json_v4_eval_string_030` | `json_answer,string,eval` | `fail` | 12 | 23.91 | {"answer":"k030abqleve"} |
| `json_v4_eval_string_031` | `json_answer,string,eval` | `pass` | 12 | 22.53 | {"answer":"EVALQK031AB"} |
| `json_v4_eval_string_032` | `json_answer,string,eval` | `pass` | 9 | 15.89 | {"answer":"eab032"} |
| `json_v4_eval_string_033` | `json_answer,string,eval` | `fail` | 12 | 23.84 | {"answer":"k330qklaev"} |
| `json_v4_eval_string_034` | `json_answer,string,eval` | `pass` | 12 | 14.23 | {"answer":"EVALQK034AB"} |
| `json_v4_eval_categorization_001` | `json_answer,categorization,eval` | `pass` | 11 | 12.11 | {"answer":"eval_odd_bucket_001"} |
| `json_v4_eval_categorization_002` | `json_answer,categorization,eval` | `pass` | 11 | 13.15 | {"answer":"eval_even_bucket_002"} |
| `json_v4_eval_categorization_003` | `json_answer,categorization,eval` | `pass` | 11 | 13.11 | {"answer":"eval_odd_bucket_003"} |
| `json_v4_eval_categorization_004` | `json_answer,categorization,eval` | `pass` | 11 | 13.13 | {"answer":"eval_even_bucket_004"} |
| `json_v4_eval_categorization_005` | `json_answer,categorization,eval` | `pass` | 11 | 13.11 | {"answer":"eval_odd_bucket_005"} |
| `json_v4_eval_categorization_006` | `json_answer,categorization,eval` | `pass` | 11 | 13.09 | {"answer":"eval_even_bucket_006"} |
| `json_v4_eval_categorization_007` | `json_answer,categorization,eval` | `pass` | 11 | 13.16 | {"answer":"eval_odd_bucket_007"} |
| `json_v4_eval_categorization_008` | `json_answer,categorization,eval` | `pass` | 11 | 13.11 | {"answer":"eval_even_bucket_008"} |
| `json_v4_eval_categorization_009` | `json_answer,categorization,eval` | `pass` | 11 | 13.18 | {"answer":"eval_odd_bucket_009"} |
| `json_v4_eval_categorization_010` | `json_answer,categorization,eval` | `pass` | 11 | 12.91 | {"answer":"eval_even_bucket_010"} |
| `json_v4_eval_categorization_011` | `json_answer,categorization,eval` | `pass` | 11 | 13.08 | {"answer":"eval_odd_bucket_011"} |
| `json_v4_eval_categorization_012` | `json_answer,categorization,eval` | `pass` | 11 | 13.16 | {"answer":"eval_even_bucket_012"} |
| `json_v4_eval_categorization_013` | `json_answer,categorization,eval` | `pass` | 11 | 13.13 | {"answer":"eval_odd_bucket_013"} |
| `json_v4_eval_categorization_014` | `json_answer,categorization,eval` | `fail` | 11 | 13.12 | {"answer":"eval_odd_bucket_014"} |
| `json_v4_eval_categorization_015` | `json_answer,categorization,eval` | `pass` | 11 | 13.12 | {"answer":"eval_odd_bucket_015"} |
| `json_v4_eval_categorization_016` | `json_answer,categorization,eval` | `pass` | 11 | 13.16 | {"answer":"eval_even_bucket_016"} |
| `json_v4_eval_categorization_017` | `json_answer,categorization,eval` | `pass` | 11 | 13.14 | {"answer":"eval_odd_bucket_017"} |
| `json_v4_eval_categorization_018` | `json_answer,categorization,eval` | `pass` | 11 | 13.12 | {"answer":"eval_even_bucket_018"} |
| `json_v4_eval_categorization_019` | `json_answer,categorization,eval` | `pass` | 11 | 13.10 | {"answer":"eval_odd_bucket_019"} |
| `json_v4_eval_categorization_020` | `json_answer,categorization,eval` | `pass` | 11 | 12.91 | {"answer":"eval_even_bucket_020"} |
| `json_v4_eval_categorization_021` | `json_answer,categorization,eval` | `pass` | 11 | 13.16 | {"answer":"eval_odd_bucket_021"} |
| `json_v4_eval_categorization_022` | `json_answer,categorization,eval` | `pass` | 11 | 13.12 | {"answer":"eval_even_bucket_022"} |
| `json_v4_eval_categorization_023` | `json_answer,categorization,eval` | `pass` | 11 | 13.12 | {"answer":"eval_odd_bucket_023"} |
| `json_v4_eval_categorization_024` | `json_answer,categorization,eval` | `pass` | 11 | 13.08 | {"answer":"eval_even_bucket_024"} |
| `json_v4_eval_categorization_025` | `json_answer,categorization,eval` | `pass` | 11 | 13.07 | {"answer":"eval_odd_bucket_025"} |
| `json_v4_eval_categorization_026` | `json_answer,categorization,eval` | `pass` | 11 | 13.07 | {"answer":"eval_even_bucket_026"} |
| `json_v4_eval_categorization_027` | `json_answer,categorization,eval` | `pass` | 11 | 13.08 | {"answer":"eval_odd_bucket_027"} |
| `json_v4_eval_categorization_028` | `json_answer,categorization,eval` | `pass` | 11 | 13.05 | {"answer":"eval_even_bucket_028"} |
| `json_v4_eval_categorization_029` | `json_answer,categorization,eval` | `pass` | 11 | 13.13 | {"answer":"eval_odd_bucket_029"} |
| `json_v4_eval_categorization_030` | `json_answer,categorization,eval` | `pass` | 11 | 12.89 | {"answer":"eval_even_bucket_030"} |
| `json_v4_eval_categorization_031` | `json_answer,categorization,eval` | `pass` | 11 | 13.13 | {"answer":"eval_odd_bucket_031"} |
| `json_v4_eval_categorization_032` | `json_answer,categorization,eval` | `pass` | 11 | 13.11 | {"answer":"eval_even_bucket_032"} |
| `json_v4_eval_categorization_033` | `json_answer,categorization,eval` | `pass` | 11 | 13.17 | {"answer":"eval_odd_bucket_033"} |
| `json_v4_eval_categorization_034` | `json_answer,categorization,eval` | `fail` | 11 | 13.12 | {"answer":"eval_odd_bucket_034"} |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `arithmetic` | 27 | 34 | 0.79 |
| `categorization` | 32 | 34 | 0.94 |
| `code` | 34 | 34 | 1.00 |
| `compiler` | 14 | 34 | 0.41 |
| `eval` | 162 | 204 | 0.79 |
| `fact` | 34 | 34 | 1.00 |
| `json_answer` | 162 | 204 | 0.79 |
| `string` | 21 | 34 | 0.62 |

## JSON Quality Axes

| axis | passed | scored | pass rate [95% CI] |
|---|---:|---:|---:|
| `parse_valid` | 199 | 204 | 0.98 [0.94, 0.99] |
| `no_extra_text` | 199 | 204 | 0.98 [0.94, 0.99] |
| `schema_ok` | 199 | 204 | 0.98 [0.94, 0.99] |
| `type_ok` | 199 | 204 | 0.98 [0.94, 0.99] |
| `value_correct` | 162 | 204 | 0.79 [0.73, 0.84] |
| `strict_pass` | 162 | 204 | 0.79 [0.73, 0.84] |
