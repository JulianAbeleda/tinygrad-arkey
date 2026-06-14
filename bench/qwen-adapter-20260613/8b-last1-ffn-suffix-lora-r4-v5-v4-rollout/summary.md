# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter: `bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5`
- dataset: `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `204`
- generated tokens: `1952`
- quality: `fail` (105/204)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `json_v4_eval_arithmetic_001` | `json_answer,arithmetic,eval` | `pass` | 10 | 0.37 | {"answer": 10521} |
| `json_v4_eval_arithmetic_002` | `json_answer,arithmetic,eval` | `pass` | 9 | 17.39 | {"answer": 5110} |
| `json_v4_eval_arithmetic_003` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.50 | {"answer": 3509} |
| `json_v4_eval_arithmetic_004` | `json_answer,arithmetic,eval` | `pass` | 10 | 8.05 | {"answer": 10575} |
| `json_v4_eval_arithmetic_005` | `json_answer,arithmetic,eval` | `fail` | 20 | 15.55 | {"answer": 5200}\n</think>\n\n{"answer": 5200} |
| `json_v4_eval_arithmetic_006` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.46 | {"answer": 3521} |
| `json_v4_eval_arithmetic_007` | `json_answer,arithmetic,eval` | `pass` | 10 | 21.52 | {"answer": 10629} |
| `json_v4_eval_arithmetic_008` | `json_answer,arithmetic,eval` | `fail` | 9 | 16.83 | {"answer": 5288} |
| `json_v4_eval_arithmetic_009` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.45 | {"answer": 3533} |
| `json_v4_eval_arithmetic_010` | `json_answer,arithmetic,eval` | `pass` | 10 | 20.94 | {"answer": 10683} |
| `json_v4_eval_arithmetic_011` | `json_answer,arithmetic,eval` | `fail` | 20 | 27.88 | {"answer": 5155}\n</think>\n\n{"answer": 5155} |
| `json_v4_eval_arithmetic_012` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.49 | {"answer": 3545} |
| `json_v4_eval_arithmetic_013` | `json_answer,arithmetic,eval` | `pass` | 10 | 21.61 | {"answer": 10737} |
| `json_v4_eval_arithmetic_014` | `json_answer,arithmetic,eval` | `pass` | 9 | 17.40 | {"answer": 5272} |
| `json_v4_eval_arithmetic_015` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.55 | {"answer": 3557} |
| `json_v4_eval_arithmetic_016` | `json_answer,arithmetic,eval` | `pass` | 10 | 21.75 | {"answer": 10791} |
| `json_v4_eval_arithmetic_017` | `json_answer,arithmetic,eval` | `fail` | 9 | 16.85 | {"answer": 5377} |
| `json_v4_eval_arithmetic_018` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.52 | {"answer": 3569} |
| `json_v4_eval_arithmetic_019` | `json_answer,arithmetic,eval` | `pass` | 10 | 20.01 | {"answer": 10845} |
| `json_v4_eval_arithmetic_020` | `json_answer,arithmetic,eval` | `fail` | 20 | 27.87 | {"answer": 5100}\n</think>\n\n{"answer": 5100} |
| `json_v4_eval_arithmetic_021` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.49 | {"answer": 3581} |
| `json_v4_eval_arithmetic_022` | `json_answer,arithmetic,eval` | `pass` | 10 | 21.41 | {"answer": 10899} |
| `json_v4_eval_arithmetic_023` | `json_answer,arithmetic,eval` | `fail` | 9 | 17.29 | {"answer": 5264} |
| `json_v4_eval_arithmetic_024` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.00 | {"answer": 3593} |
| `json_v4_eval_arithmetic_025` | `json_answer,arithmetic,eval` | `pass` | 10 | 21.73 | {"answer": 10953} |
| `json_v4_eval_arithmetic_026` | `json_answer,arithmetic,eval` | `fail` | 9 | 16.90 | {"answer": 5466} |
| `json_v4_eval_arithmetic_027` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.56 | {"answer": 3605} |
| `json_v4_eval_arithmetic_028` | `json_answer,arithmetic,eval` | `pass` | 10 | 20.95 | {"answer": 11007} |
| `json_v4_eval_arithmetic_029` | `json_answer,arithmetic,eval` | `pass` | 9 | 17.45 | {"answer": 5245} |
| `json_v4_eval_arithmetic_030` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.55 | {"answer": 3617} |
| `json_v4_eval_arithmetic_031` | `json_answer,arithmetic,eval` | `pass` | 10 | 21.74 | {"answer": 11061} |
| `json_v4_eval_arithmetic_032` | `json_answer,arithmetic,eval` | `fail` | 9 | 17.43 | {"answer": 5136} |
| `json_v4_eval_arithmetic_033` | `json_answer,arithmetic,eval` | `pass` | 9 | 18.47 | {"answer": 3629} |
| `json_v4_eval_arithmetic_034` | `json_answer,arithmetic,eval` | `pass` | 10 | 21.74 | {"answer": 11115} |
| `json_v4_eval_fact_001` | `json_answer,fact,eval` | `pass` | 5 | 14.78 | {"answer":"At"} |
| `json_v4_eval_fact_002` | `json_answer,fact,eval` | `pass` | 6 | 20.31 | {"answer":"Rn"} |
| `json_v4_eval_fact_003` | `json_answer,fact,eval` | `pass` | 5 | 18.06 | {"answer":"Fr"} |
| `json_v4_eval_fact_004` | `json_answer,fact,eval` | `pass` | 5 | 17.57 | {"answer":"Ra"} |
| `json_v4_eval_fact_005` | `json_answer,fact,eval` | `pass` | 5 | 18.17 | {"answer":"Ac"} |
| `json_v4_eval_fact_006` | `json_answer,fact,eval` | `pass` | 5 | 18.02 | {"answer":"Th"} |
| `json_v4_eval_fact_007` | `json_answer,fact,eval` | `pass` | 5 | 9.56 | {"answer":"Pa"} |
| `json_v4_eval_fact_008` | `json_answer,fact,eval` | `pass` | 5 | 10.13 | {"answer":"U"} |
| `json_v4_eval_fact_009` | `json_answer,fact,eval` | `pass` | 6 | 17.39 | {"answer":"Np"} |
| `json_v4_eval_fact_010` | `json_answer,fact,eval` | `pass` | 5 | 17.65 | {"answer":"Pu"} |
| `json_v4_eval_fact_011` | `json_answer,fact,eval` | `pass` | 5 | 18.11 | {"answer":"Am"} |
| `json_v4_eval_fact_012` | `json_answer,fact,eval` | `pass` | 6 | 20.37 | {"answer":"Cm"} |
| `json_v4_eval_fact_013` | `json_answer,fact,eval` | `pass` | 6 | 19.89 | {"answer":"Bk"} |
| `json_v4_eval_fact_014` | `json_answer,fact,eval` | `pass` | 6 | 20.34 | {"answer":"Cf"} |
| `json_v4_eval_fact_015` | `json_answer,fact,eval` | `pass` | 5 | 18.11 | {"answer":"Es"} |
| `json_v4_eval_fact_016` | `json_answer,fact,eval` | `pass` | 6 | 20.30 | {"answer":"Fm"} |
| `json_v4_eval_fact_017` | `json_answer,fact,eval` | `pass` | 5 | 17.12 | {"answer":"Md"} |
| `json_v4_eval_fact_018` | `json_answer,fact,eval` | `pass` | 5 | 17.73 | {"answer":"No"} |
| `json_v4_eval_fact_019` | `json_answer,fact,eval` | `pass` | 6 | 19.83 | {"answer":"Lr"} |
| `json_v4_eval_fact_020` | `json_answer,fact,eval` | `pass` | 6 | 19.82 | {"answer":"Rf"} |
| `json_v4_eval_fact_021` | `json_answer,fact,eval` | `pass` | 5 | 17.15 | {"answer":"Db"} |
| `json_v4_eval_fact_022` | `json_answer,fact,eval` | `pass` | 6 | 19.33 | {"answer":"Sg"} |
| `json_v4_eval_fact_023` | `json_answer,fact,eval` | `pass` | 6 | 20.39 | {"answer":"Bh"} |
| `json_v4_eval_fact_024` | `json_answer,fact,eval` | `pass` | 6 | 20.00 | {"answer":"Hs"} |
| `json_v4_eval_fact_025` | `json_answer,fact,eval` | `pass` | 5 | 17.13 | {"answer":"Mt"} |
| `json_v4_eval_fact_026` | `json_answer,fact,eval` | `pass` | 5 | 9.66 | {"answer":"Ds"} |
| `json_v4_eval_fact_027` | `json_answer,fact,eval` | `pass` | 6 | 19.27 | {"answer":"Rg"} |
| `json_v4_eval_fact_028` | `json_answer,fact,eval` | `pass` | 6 | 19.36 | {"answer":"Cn"} |
| `json_v4_eval_fact_029` | `json_answer,fact,eval` | `pass` | 5 | 17.65 | {"answer":"Nh"} |
| `json_v4_eval_fact_030` | `json_answer,fact,eval` | `pass` | 5 | 17.20 | {"answer":"Fl"} |
| `json_v4_eval_fact_031` | `json_answer,fact,eval` | `pass` | 5 | 17.63 | {"answer":"Mc"} |
| `json_v4_eval_fact_032` | `json_answer,fact,eval` | `pass` | 5 | 17.60 | {"answer":"Lv"} |
| `json_v4_eval_fact_033` | `json_answer,fact,eval` | `pass` | 5 | 17.63 | {"answer":"Ts"} |
| `json_v4_eval_fact_034` | `json_answer,fact,eval` | `fail` | 5 | 17.59 | {"answer":"O"} |
| `json_v4_eval_code_001` | `json_answer,code,eval` | `fail` | 7 | 12.54 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_002` | `json_answer,code,eval` | `fail` | 7 | 22.29 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_003` | `json_answer,code,eval` | `fail` | 7 | 22.31 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_004` | `json_answer,code,eval` | `fail` | 7 | 22.34 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_005` | `json_answer,code,eval` | `fail` | 7 | 22.35 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_006` | `json_answer,code,eval` | `fail` | 7 | 22.33 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_007` | `json_answer,code,eval` | `pass` | 11 | 14.45 | {"answer":"eval_json_handler_007"} |
| `json_v4_eval_code_008` | `json_answer,code,eval` | `fail` | 7 | 13.76 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_009` | `json_answer,code,eval` | `pass` | 11 | 14.58 | {"answer":"eval_json_handler_009"} |
| `json_v4_eval_code_010` | `json_answer,code,eval` | `fail` | 7 | 12.28 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_011` | `json_answer,code,eval` | `fail` | 7 | 11.23 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_012` | `json_answer,code,eval` | `fail` | 7 | 11.75 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_013` | `json_answer,code,eval` | `fail` | 7 | 22.22 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_014` | `json_answer,code,eval` | `fail` | 7 | 22.12 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_015` | `json_answer,code,eval` | `pass` | 11 | 28.19 | {"answer":"eval_json_handler_015"} |
| `json_v4_eval_code_016` | `json_answer,code,eval` | `fail` | 7 | 22.11 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_017` | `json_answer,code,eval` | `fail` | 7 | 22.05 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_018` | `json_answer,code,eval` | `fail` | 7 | 22.18 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_019` | `json_answer,code,eval` | `fail` | 7 | 22.25 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_020` | `json_answer,code,eval` | `fail` | 7 | 21.63 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_021` | `json_answer,code,eval` | `fail` | 7 | 22.19 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_022` | `json_answer,code,eval` | `fail` | 7 | 22.22 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_023` | `json_answer,code,eval` | `fail` | 7 | 22.22 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_024` | `json_answer,code,eval` | `fail` | 7 | 22.28 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_025` | `json_answer,code,eval` | `pass` | 11 | 28.21 | {"answer":"eval_json_handler_025"} |
| `json_v4_eval_code_026` | `json_answer,code,eval` | `fail` | 7 | 22.21 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_027` | `json_answer,code,eval` | `fail` | 7 | 22.18 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_028` | `json_answer,code,eval` | `fail` | 7 | 16.63 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_029` | `json_answer,code,eval` | `fail` | 7 | 17.41 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_030` | `json_answer,code,eval` | `pass` | 11 | 16.66 | {"answer":"eval_json_handler_030"} |
| `json_v4_eval_code_031` | `json_answer,code,eval` | `pass` | 11 | 27.85 | {"answer":"eval_json_handler_031"} |
| `json_v4_eval_code_032` | `json_answer,code,eval` | `fail` | 7 | 21.88 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_033` | `json_answer,code,eval` | `fail` | 7 | 22.24 | {"answer":"eval_json_handler"} |
| `json_v4_eval_code_034` | `json_answer,code,eval` | `pass` | 11 | 28.34 | {"answer":"eval_json_handler_034"} |
| `json_v4_eval_compiler_001` | `json_answer,compiler,eval` | `fail` | 24 | 19.90 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_002` | `json_answer,compiler,eval` | `fail` | 24 | 23.97 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_003` | `json_answer,compiler,eval` | `fail` | 5 | 7.59 | {"answer":"eval"} |
| `json_v4_eval_compiler_004` | `json_answer,compiler,eval` | `fail` | 24 | 23.71 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_005` | `json_answer,compiler,eval` | `fail` | 24 | 24.56 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_006` | `json_answer,compiler,eval` | `fail` | 24 | 22.10 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_007` | `json_answer,compiler,eval` | `fail` | 5 | 7.39 | {"answer":"eval"} |
| `json_v4_eval_compiler_008` | `json_answer,compiler,eval` | `fail` | 5 | 7.83 | {"answer":"eval"} |
| `json_v4_eval_compiler_009` | `json_answer,compiler,eval` | `fail` | 5 | 5.08 | {"answer":"eval"} |
| `json_v4_eval_compiler_010` | `json_answer,compiler,eval` | `fail` | 7 | 9.82 | {"answer":"eval_qk"} |
| `json_v4_eval_compiler_011` | `json_answer,compiler,eval` | `fail` | 5 | 7.41 | {"answer":"eval"} |
| `json_v4_eval_compiler_012` | `json_answer,compiler,eval` | `fail` | 5 | 6.69 | {"answer":"eval"} |
| `json_v4_eval_compiler_013` | `json_answer,compiler,eval` | `fail` | 24 | 24.03 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_014` | `json_answer,compiler,eval` | `fail` | 24 | 24.06 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_015` | `json_answer,compiler,eval` | `fail` | 5 | 7.56 | {"answer":"eval"} |
| `json_v4_eval_compiler_016` | `json_answer,compiler,eval` | `fail` | 5 | 7.43 | {"answer":"eval"} |
| `json_v4_eval_compiler_017` | `json_answer,compiler,eval` | `fail` | 24 | 24.60 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_018` | `json_answer,compiler,eval` | `fail` | 24 | 22.12 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_019` | `json_answer,compiler,eval` | `fail` | 5 | 7.42 | {"answer":"eval"} |
| `json_v4_eval_compiler_020` | `json_answer,compiler,eval` | `fail` | 5 | 7.79 | {"answer":"eval"} |
| `json_v4_eval_compiler_021` | `json_answer,compiler,eval` | `fail` | 5 | 8.06 | {"answer":"eval"} |
| `json_v4_eval_compiler_022` | `json_answer,compiler,eval` | `fail` | 7 | 8.50 | {"answer":"eval_qk"} |
| `json_v4_eval_compiler_023` | `json_answer,compiler,eval` | `fail` | 5 | 6.58 | {"answer":"eval"} |
| `json_v4_eval_compiler_024` | `json_answer,compiler,eval` | `fail` | 5 | 5.88 | {"answer":"eval"} |
| `json_v4_eval_compiler_025` | `json_answer,compiler,eval` | `fail` | 24 | 16.86 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_026` | `json_answer,compiler,eval` | `fail` | 24 | 19.58 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_027` | `json_answer,compiler,eval` | `fail` | 5 | 7.11 | {"answer":"eval"} |
| `json_v4_eval_compiler_028` | `json_answer,compiler,eval` | `fail` | 24 | 16.30 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_029` | `json_answer,compiler,eval` | `fail` | 24 | 22.81 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_compiler_030` | `json_answer,compiler,eval` | `fail` | 5 | 6.70 | {"answer":"eval"} |
| `json_v4_eval_compiler_031` | `json_answer,compiler,eval` | `fail` | 5 | 7.41 | {"answer":"eval"} |
| `json_v4_eval_compiler_032` | `json_answer,compiler,eval` | `fail` | 5 | 7.83 | {"answer":"eval"} |
| `json_v4_eval_compiler_033` | `json_answer,compiler,eval` | `fail` | 5 | 7.21 | {"answer":"eval"} |
| `json_v4_eval_compiler_034` | `json_answer,compiler,eval` | `fail` | 7 | 9.34 | {"answer":"eval_qk"} |
| `json_v4_eval_string_001` | `json_answer,string,eval` | `pass` | 12 | 20.64 | {"answer":"EVALQK001AB"} |
| `json_v4_eval_string_002` | `json_answer,string,eval` | `fail` | 6 | 10.43 | {"answer":"eab"} |
| `json_v4_eval_string_003` | `json_answer,string,eval` | `fail` | 12 | 22.20 | {"answer":"kcb3003qale"} |
| `json_v4_eval_string_004` | `json_answer,string,eval` | `pass` | 12 | 16.62 | {"answer":"EVALQK004AB"} |
| `json_v4_eval_string_005` | `json_answer,string,eval` | `fail` | 6 | 11.03 | {"answer":"eab"} |
| `json_v4_eval_string_006` | `json_answer,string,eval` | `fail` | 12 | 22.16 | {"answer":"kcb600qkale"} |
| `json_v4_eval_string_007` | `json_answer,string,eval` | `pass` | 12 | 20.97 | {"answer":"EVALQK007AB"} |
| `json_v4_eval_string_008` | `json_answer,string,eval` | `fail` | 6 | 11.04 | {"answer":"eab"} |
| `json_v4_eval_string_009` | `json_answer,string,eval` | `fail` | 11 | 20.95 | {"answer":"k009abqle"} |
| `json_v4_eval_string_010` | `json_answer,string,eval` | `pass` | 12 | 20.95 | {"answer":"EVALQK010AB"} |
| `json_v4_eval_string_011` | `json_answer,string,eval` | `fail` | 6 | 11.15 | {"answer":"eab"} |
| `json_v4_eval_string_012` | `json_answer,string,eval` | `fail` | 12 | 22.18 | {"answer":"kcb210qkva"} |
| `json_v4_eval_string_013` | `json_answer,string,eval` | `pass` | 12 | 21.01 | {"answer":"EVALQK013AB"} |
| `json_v4_eval_string_014` | `json_answer,string,eval` | `fail` | 6 | 11.09 | {"answer":"eab"} |
| `json_v4_eval_string_015` | `json_answer,string,eval` | `fail` | 12 | 22.25 | {"answer":"kcb510kqale"} |
| `json_v4_eval_string_016` | `json_answer,string,eval` | `pass` | 12 | 20.99 | {"answer":"EVALQK016AB"} |
| `json_v4_eval_string_017` | `json_answer,string,eval` | `fail` | 6 | 11.10 | {"answer":"eab"} |
| `json_v4_eval_string_018` | `json_answer,string,eval` | `fail` | 11 | 21.05 | {"answer":"k810qkale"} |
| `json_v4_eval_string_019` | `json_answer,string,eval` | `pass` | 12 | 21.01 | {"answer":"EVALQK019AB"} |
| `json_v4_eval_string_020` | `json_answer,string,eval` | `fail` | 6 | 10.11 | {"answer":"eab"} |
| `json_v4_eval_string_021` | `json_answer,string,eval` | `fail` | 12 | 21.58 | {"answer":"kcb210kqale"} |
| `json_v4_eval_string_022` | `json_answer,string,eval` | `pass` | 12 | 20.98 | {"answer":"EVALQK022AB"} |
| `json_v4_eval_string_023` | `json_answer,string,eval` | `fail` | 6 | 11.10 | {"answer":"eab"} |
| `json_v4_eval_string_024` | `json_answer,string,eval` | `fail` | 12 | 22.17 | {"answer":"kcb420kqale"} |
| `json_v4_eval_string_025` | `json_answer,string,eval` | `pass` | 12 | 21.01 | {"answer":"EVALQK025AB"} |
| `json_v4_eval_string_026` | `json_answer,string,eval` | `fail` | 24 | 28.08 | {"answer":"E","":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_v4_eval_string_027` | `json_answer,string,eval` | `fail` | 11 | 14.31 | {"answer":"bk720kqale"} |
| `json_v4_eval_string_028` | `json_answer,string,eval` | `pass` | 12 | 20.96 | {"answer":"EVALQK028AB"} |
| `json_v4_eval_string_029` | `json_answer,string,eval` | `fail` | 6 | 10.98 | {"answer":"eab"} |
| `json_v4_eval_string_030` | `json_answer,string,eval` | `fail` | 11 | 21.12 | {"answer":"k030abqle"} |
| `json_v4_eval_string_031` | `json_answer,string,eval` | `pass` | 12 | 20.99 | {"answer":"EVALQK031AB"} |
| `json_v4_eval_string_032` | `json_answer,string,eval` | `fail` | 6 | 11.04 | {"answer":"eab"} |
| `json_v4_eval_string_033` | `json_answer,string,eval` | `fail` | 11 | 21.05 | {"answer":"k330qkale"} |
| `json_v4_eval_string_034` | `json_answer,string,eval` | `pass` | 12 | 20.98 | {"answer":"EVALQK034AB"} |
| `json_v4_eval_categorization_001` | `json_answer,categorization,eval` | `fail` | 11 | 11.32 | {"answer":"eval_even_bucket_001"} |
| `json_v4_eval_categorization_002` | `json_answer,categorization,eval` | `pass` | 11 | 12.38 | {"answer":"eval_even_bucket_002"} |
| `json_v4_eval_categorization_003` | `json_answer,categorization,eval` | `fail` | 11 | 11.88 | {"answer":"eval_even_bucket_003"} |
| `json_v4_eval_categorization_004` | `json_answer,categorization,eval` | `pass` | 11 | 11.99 | {"answer":"eval_even_bucket_004"} |
| `json_v4_eval_categorization_005` | `json_answer,categorization,eval` | `fail` | 11 | 12.03 | {"answer":"eval_even_bucket_005"} |
| `json_v4_eval_categorization_006` | `json_answer,categorization,eval` | `pass` | 11 | 12.06 | {"answer":"eval_even_bucket_006"} |
| `json_v4_eval_categorization_007` | `json_answer,categorization,eval` | `fail` | 11 | 12.34 | {"answer":"eval_even_bucket_007"} |
| `json_v4_eval_categorization_008` | `json_answer,categorization,eval` | `pass` | 11 | 11.86 | {"answer":"eval_even_bucket_008"} |
| `json_v4_eval_categorization_009` | `json_answer,categorization,eval` | `pass` | 11 | 12.28 | {"answer":"eval_odd_bucket_009"} |
| `json_v4_eval_categorization_010` | `json_answer,categorization,eval` | `pass` | 11 | 11.34 | {"answer":"eval_even_bucket_010"} |
| `json_v4_eval_categorization_011` | `json_answer,categorization,eval` | `pass` | 11 | 12.00 | {"answer":"eval_odd_bucket_011"} |
| `json_v4_eval_categorization_012` | `json_answer,categorization,eval` | `pass` | 11 | 11.83 | {"answer":"eval_even_bucket_012"} |
| `json_v4_eval_categorization_013` | `json_answer,categorization,eval` | `pass` | 11 | 12.11 | {"answer":"eval_odd_bucket_013"} |
| `json_v4_eval_categorization_014` | `json_answer,categorization,eval` | `pass` | 11 | 12.43 | {"answer":"eval_even_bucket_014"} |
| `json_v4_eval_categorization_015` | `json_answer,categorization,eval` | `pass` | 11 | 12.44 | {"answer":"eval_odd_bucket_015"} |
| `json_v4_eval_categorization_016` | `json_answer,categorization,eval` | `pass` | 11 | 12.44 | {"answer":"eval_even_bucket_016"} |
| `json_v4_eval_categorization_017` | `json_answer,categorization,eval` | `pass` | 11 | 12.36 | {"answer":"eval_odd_bucket_017"} |
| `json_v4_eval_categorization_018` | `json_answer,categorization,eval` | `pass` | 11 | 12.42 | {"answer":"eval_even_bucket_018"} |
| `json_v4_eval_categorization_019` | `json_answer,categorization,eval` | `pass` | 11 | 12.42 | {"answer":"eval_odd_bucket_019"} |
| `json_v4_eval_categorization_020` | `json_answer,categorization,eval` | `pass` | 11 | 12.22 | {"answer":"eval_even_bucket_020"} |
| `json_v4_eval_categorization_021` | `json_answer,categorization,eval` | `fail` | 11 | 12.43 | {"answer":"eval_even_bucket_021"} |
| `json_v4_eval_categorization_022` | `json_answer,categorization,eval` | `pass` | 11 | 12.43 | {"answer":"eval_even_bucket_022"} |
| `json_v4_eval_categorization_023` | `json_answer,categorization,eval` | `pass` | 11 | 12.44 | {"answer":"eval_odd_bucket_023"} |
| `json_v4_eval_categorization_024` | `json_answer,categorization,eval` | `pass` | 11 | 12.47 | {"answer":"eval_even_bucket_024"} |
| `json_v4_eval_categorization_025` | `json_answer,categorization,eval` | `pass` | 11 | 11.62 | {"answer":"eval_odd_bucket_025"} |
| `json_v4_eval_categorization_026` | `json_answer,categorization,eval` | `pass` | 11 | 6.89 | {"answer":"eval_even_bucket_026"} |
| `json_v4_eval_categorization_027` | `json_answer,categorization,eval` | `fail` | 11 | 7.68 | {"answer":"eval_even_bucket_027"} |
| `json_v4_eval_categorization_028` | `json_answer,categorization,eval` | `pass` | 11 | 7.11 | {"answer":"eval_even_bucket_028"} |
| `json_v4_eval_categorization_029` | `json_answer,categorization,eval` | `pass` | 11 | 7.89 | {"answer":"eval_odd_bucket_029"} |
| `json_v4_eval_categorization_030` | `json_answer,categorization,eval` | `pass` | 11 | 8.67 | {"answer":"eval_even_bucket_030"} |
| `json_v4_eval_categorization_031` | `json_answer,categorization,eval` | `pass` | 11 | 6.87 | {"answer":"eval_odd_bucket_031"} |
| `json_v4_eval_categorization_032` | `json_answer,categorization,eval` | `pass` | 11 | 6.21 | {"answer":"eval_even_bucket_032"} |
| `json_v4_eval_categorization_033` | `json_answer,categorization,eval` | `fail` | 11 | 7.99 | {"answer":"eval_even_bucket_033"} |
| `json_v4_eval_categorization_034` | `json_answer,categorization,eval` | `pass` | 11 | 12.47 | {"answer":"eval_even_bucket_034"} |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `arithmetic` | 26 | 34 | 0.76 |
| `categorization` | 27 | 34 | 0.79 |
| `code` | 7 | 34 | 0.21 |
| `compiler` | 0 | 34 | 0.00 |
| `eval` | 105 | 204 | 0.51 |
| `fact` | 33 | 34 | 0.97 |
| `json_answer` | 105 | 204 | 0.51 |
| `string` | 12 | 34 | 0.35 |

## JSON Quality Axes

| axis | passed | scored | pass rate [95% CI] |
|---|---:|---:|---:|
| `parse_valid` | 190 | 204 | 0.93 [0.89, 0.96] |
| `no_extra_text` | 187 | 204 | 0.92 [0.87, 0.95] |
| `schema_ok` | 190 | 204 | 0.93 [0.89, 0.96] |
| `type_ok` | 190 | 204 | 0.93 [0.89, 0.96] |
| `value_correct` | 107 | 204 | 0.52 [0.46, 0.59] |
| `strict_pass` | 105 | 204 | 0.51 [0.45, 0.58] |
