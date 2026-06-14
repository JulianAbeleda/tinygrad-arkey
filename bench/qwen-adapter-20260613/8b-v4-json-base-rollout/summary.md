# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- dataset: `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `204`
- generated tokens: `4896`
- quality: `fail` (0/204)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `json_v4_eval_arithmetic_001` | `json_answer,arithmetic,eval` | `fail` | 24 | 0.90 | <think>\nOkay, the user is asking for the sum of 7011 and 3510. |
| `json_v4_eval_arithmetic_002` | `json_answer,arithmetic,eval` | `fail` | 24 | 32.45 | <think>\nOkay, let's see. The user is asking for the result of (22 multiplied by 5) |
| `json_v4_eval_arithmetic_003` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.69 | <think>\nOkay, let's see. The user is asking for the result of 7033 minus 3 |
| `json_v4_eval_arithmetic_004` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.87 | <think>\nOkay, the user is asking for the sum of 7044 and 3531. |
| `json_v4_eval_arithmetic_005` | `json_answer,arithmetic,eval` | `fail` | 24 | 31.89 | <think>\nOkay, let's see. The user is asking for the result of (25 multiplied by 8) |
| `json_v4_eval_arithmetic_006` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.48 | <think>\nOkay, the user is asking for the result of 7066 minus 3545. |
| `json_v4_eval_arithmetic_007` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.87 | <think>\nOkay, the user is asking for the sum of 7077 and 3552. |
| `json_v4_eval_arithmetic_008` | `json_answer,arithmetic,eval` | `fail` | 24 | 31.76 | <think>\nOkay, let's see. The user is asking for the result of (28 multiplied by 11 |
| `json_v4_eval_arithmetic_009` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.51 | <think>\nOkay, let's see. The user is asking for the result of 7099 minus 3 |
| `json_v4_eval_arithmetic_010` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.53 | <think>\nOkay, the user is asking for the sum of 7110 and 3573. |
| `json_v4_eval_arithmetic_011` | `json_answer,arithmetic,eval` | `fail` | 24 | 31.98 | <think>\nOkay, let's see. The user is asking for the result of (31 multiplied by 5) |
| `json_v4_eval_arithmetic_012` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.26 | <think>\nOkay, let's see. The user is asking for the result of 7132 minus 3 |
| `json_v4_eval_arithmetic_013` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.43 | <think>\nOkay, the user is asking for the sum of 7143 and 3594. |
| `json_v4_eval_arithmetic_014` | `json_answer,arithmetic,eval` | `fail` | 24 | 32.37 | <think>\nOkay, let's see. The user is asking for the result of (34 multiplied by 8) |
| `json_v4_eval_arithmetic_015` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.50 | <think>\nOkay, let's see. The user is asking for the result of 7165 minus 3 |
| `json_v4_eval_arithmetic_016` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.83 | <think>\nOkay, the user is asking for the sum of 7176 and 3615. |
| `json_v4_eval_arithmetic_017` | `json_answer,arithmetic,eval` | `fail` | 24 | 31.83 | <think>\nOkay, let's see. The user is asking for the result of (37 multiplied by 11 |
| `json_v4_eval_arithmetic_018` | `json_answer,arithmetic,eval` | `fail` | 24 | 32.82 | <think>\nOkay, the user is asking for the result of 7198 minus 3629. |
| `json_v4_eval_arithmetic_019` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.43 | <think>\nOkay, the user is asking for the sum of 7209 and 3636. |
| `json_v4_eval_arithmetic_020` | `json_answer,arithmetic,eval` | `fail` | 24 | 32.54 | <think>\nOkay, let's see. The user is asking for the result of (40 multiplied by 5) |
| `json_v4_eval_arithmetic_021` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.43 | <think>\nOkay, the user is asking for the result of 7231 minus 3650. |
| `json_v4_eval_arithmetic_022` | `json_answer,arithmetic,eval` | `fail` | 24 | 35.12 | <think>\nOkay, the user is asking for the sum of 7242 and 3657. |
| `json_v4_eval_arithmetic_023` | `json_answer,arithmetic,eval` | `fail` | 24 | 32.67 | <think>\nOkay, let's see. The user is asking for the result of (43 multiplied by 8) |
| `json_v4_eval_arithmetic_024` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.73 | <think>\nOkay, let's see. The user is asking for the result of 7264 minus 3 |
| `json_v4_eval_arithmetic_025` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.53 | <think>\nOkay, the user is asking for the sum of 7275 and 3678. |
| `json_v4_eval_arithmetic_026` | `json_answer,arithmetic,eval` | `fail` | 24 | 31.61 | <think>\nOkay, let's see. The user is asking for the result of (46 multiplied by 11 |
| `json_v4_eval_arithmetic_027` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.76 | <think>\nOkay, let's see. The user is asking for the result of 7297 minus 3 |
| `json_v4_eval_arithmetic_028` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.47 | <think>\nOkay, the user is asking for the sum of 7308 and 3699. |
| `json_v4_eval_arithmetic_029` | `json_answer,arithmetic,eval` | `fail` | 24 | 32.41 | <think>\nOkay, let's see. The user is asking for the result of (49 multiplied by 5) |
| `json_v4_eval_arithmetic_030` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.44 | <think>\nOkay, let's see. The user is asking for the result of 7330 minus 3 |
| `json_v4_eval_arithmetic_031` | `json_answer,arithmetic,eval` | `fail` | 24 | 35.01 | <think>\nOkay, the user is asking for the sum of 7341 and 3720. |
| `json_v4_eval_arithmetic_032` | `json_answer,arithmetic,eval` | `fail` | 24 | 30.02 | <think>\nOkay, let's see. The user is asking for the result of (52 multiplied by 8) |
| `json_v4_eval_arithmetic_033` | `json_answer,arithmetic,eval` | `fail` | 24 | 33.83 | <think>\nOkay, the user is asking for the result of 7363 minus 3734. |
| `json_v4_eval_arithmetic_034` | `json_answer,arithmetic,eval` | `fail` | 24 | 34.76 | <think>\nOkay, the user is asking for the sum of 7374 and 3741. |
| `json_v4_eval_fact_001` | `json_answer,fact,eval` | `fail` | 24 | 36.11 | <think>\nOkay, the user is asking for the chemical symbol of Astatine. Let me recall the periodic table. |
| `json_v4_eval_fact_002` | `json_answer,fact,eval` | `fail` | 24 | 39.36 | <think>\nOkay, the user is asking for the chemical symbol for Radon. Let me think. Radon is a |
| `json_v4_eval_fact_003` | `json_answer,fact,eval` | `fail` | 24 | 39.65 | <think>\nOkay, the user is asking for the chemical symbol of Francium. Let me recall the periodic table. Franc |
| `json_v4_eval_fact_004` | `json_answer,fact,eval` | `fail` | 24 | 39.60 | <think>\nOkay, the user is asking for the chemical symbol for Radium. Let me think. Radium is a |
| `json_v4_eval_fact_005` | `json_answer,fact,eval` | `fail` | 24 | 39.27 | <think>\nOkay, the user is asking for the chemical symbol of Actinium. Let me recall the periodic table. Act |
| `json_v4_eval_fact_006` | `json_answer,fact,eval` | `fail` | 24 | 39.21 | <think>\nOkay, the user is asking for the chemical symbol of Thorium. Let me recall the periodic table. Thor |
| `json_v4_eval_fact_007` | `json_answer,fact,eval` | `fail` | 24 | 36.63 | <think>\nOkay, the user is asking for the chemical symbol of Protactinium. Let me start by recalling what I |
| `json_v4_eval_fact_008` | `json_answer,fact,eval` | `fail` | 24 | 39.30 | <think>\nOkay, the user is asking for the chemical symbol of Uranium. Let me recall the periodic table. Uran |
| `json_v4_eval_fact_009` | `json_answer,fact,eval` | `fail` | 24 | 38.53 | <think>\nOkay, the user is asking for the chemical symbol of Neptunium. Let me recall the periodic table |
| `json_v4_eval_fact_010` | `json_answer,fact,eval` | `fail` | 24 | 39.02 | <think>\nOkay, the user is asking for the chemical symbol for Plutonium. Let me think. Plutonium |
| `json_v4_eval_fact_011` | `json_answer,fact,eval` | `fail` | 24 | 39.45 | <think>\nOkay, the user is asking for the chemical symbol for Americium. Let me think. Americium is a |
| `json_v4_eval_fact_012` | `json_answer,fact,eval` | `fail` | 24 | 39.30 | <think>\nOkay, the user is asking for the chemical symbol for Curium. Let me think. Curium is a |
| `json_v4_eval_fact_013` | `json_answer,fact,eval` | `fail` | 24 | 38.26 | <think>\nOkay, the user is asking for the chemical symbol for Berkelium. Let me think. First, I |
| `json_v4_eval_fact_014` | `json_answer,fact,eval` | `fail` | 24 | 39.46 | <think>\nOkay, the user is asking for the chemical symbol of Californium. Let me think. Californium is a |
| `json_v4_eval_fact_015` | `json_answer,fact,eval` | `fail` | 24 | 38.67 | <think>\nOkay, the user is asking for the chemical symbol of Einsteinium. Let me think. First, I need |
| `json_v4_eval_fact_016` | `json_answer,fact,eval` | `fail` | 24 | 39.57 | <think>\nOkay, the user is asking for the chemical symbol of Fermium. Let me think. Fermium is a |
| `json_v4_eval_fact_017` | `json_answer,fact,eval` | `fail` | 24 | 38.26 | <think>\nOkay, the user is asking for the chemical symbol of Mendelevium. Let me think. First, |
| `json_v4_eval_fact_018` | `json_answer,fact,eval` | `fail` | 24 | 38.40 | <think>\nOkay, the user is asking for the chemical symbol of Nobelium. Let me think. First, I need |
| `json_v4_eval_fact_019` | `json_answer,fact,eval` | `fail` | 24 | 39.42 | <think>\nOkay, the user is asking for the chemical symbol of Lawrencium. Let me think. First, I |
| `json_v4_eval_fact_020` | `json_answer,fact,eval` | `fail` | 24 | 36.27 | <think>\nOkay, the user is asking for the chemical symbol of Rutherfordium. Let me recall what I know about |
| `json_v4_eval_fact_021` | `json_answer,fact,eval` | `fail` | 24 | 39.19 | <think>\nOkay, the user is asking for the chemical symbol of Dubnium. Let me think. First, I |
| `json_v4_eval_fact_022` | `json_answer,fact,eval` | `fail` | 24 | 38.60 | <think>\nOkay, the user is asking for the chemical symbol of Seaborgium. Let me think. First, |
| `json_v4_eval_fact_023` | `json_answer,fact,eval` | `fail` | 24 | 39.56 | <think>\nOkay, the user is asking for the chemical symbol of Bohrium. Let me recall the periodic table. Boh |
| `json_v4_eval_fact_024` | `json_answer,fact,eval` | `fail` | 24 | 39.10 | <think>\nOkay, the user is asking for the chemical symbol of Hassium. Let me think. First, I need |
| `json_v4_eval_fact_025` | `json_answer,fact,eval` | `fail` | 24 | 38.42 | <think>\nOkay, the user is asking for the chemical symbol of Meitnerium. Let me start by recalling what |
| `json_v4_eval_fact_026` | `json_answer,fact,eval` | `fail` | 24 | 38.21 | <think>\nOkay, the user is asking for the chemical symbol of Darmstadtium. Let me think. First, |
| `json_v4_eval_fact_027` | `json_answer,fact,eval` | `fail` | 24 | 38.83 | <think>\nOkay, the user is asking for the chemical symbol of Roentgenium. Let me start by recalling what |
| `json_v4_eval_fact_028` | `json_answer,fact,eval` | `fail` | 24 | 37.48 | <think>\nOkay, the user is asking for the chemical symbol of Copernicium. Let me think. First, |
| `json_v4_eval_fact_029` | `json_answer,fact,eval` | `fail` | 24 | 39.34 | <think>\nOkay, the user is asking for the chemical symbol of Nihonium. Let me start by recalling what I |
| `json_v4_eval_fact_030` | `json_answer,fact,eval` | `fail` | 24 | 37.65 | <think>\nOkay, the user is asking for the chemical symbol of Flerovium. Let me think. First, |
| `json_v4_eval_fact_031` | `json_answer,fact,eval` | `fail` | 24 | 38.72 | <think>\nOkay, the user is asking for the chemical symbol of Moscovium. Let me think. First, I |
| `json_v4_eval_fact_032` | `json_answer,fact,eval` | `fail` | 24 | 39.19 | <think>\nOkay, the user is asking for the chemical symbol of Livermorium. Let me think. First, I |
| `json_v4_eval_fact_033` | `json_answer,fact,eval` | `fail` | 24 | 38.65 | <think>\nOkay, the user is asking for the chemical symbol of Tennessine. Let me start by recalling what I |
| `json_v4_eval_fact_034` | `json_answer,fact,eval` | `fail` | 24 | 38.29 | <think>\nOkay, the user is asking for the chemical symbol of Oganesson. Let me think. First, I |
| `json_v4_eval_code_001` | `json_answer,code,eval` | `fail` | 24 | 30.03 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_002` | `json_answer,code,eval` | `fail` | 24 | 39.51 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_003` | `json_answer,code,eval` | `fail` | 24 | 39.15 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_004` | `json_answer,code,eval` | `fail` | 24 | 37.13 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_005` | `json_answer,code,eval` | `fail` | 24 | 39.37 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_006` | `json_answer,code,eval` | `fail` | 24 | 39.58 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_007` | `json_answer,code,eval` | `fail` | 24 | 39.31 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_008` | `json_answer,code,eval` | `fail` | 24 | 39.06 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_009` | `json_answer,code,eval` | `fail` | 24 | 37.01 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_010` | `json_answer,code,eval` | `fail` | 24 | 38.68 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_011` | `json_answer,code,eval` | `fail` | 24 | 39.54 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_012` | `json_answer,code,eval` | `fail` | 24 | 39.63 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_013` | `json_answer,code,eval` | `fail` | 24 | 39.18 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_014` | `json_answer,code,eval` | `fail` | 24 | 39.51 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_015` | `json_answer,code,eval` | `fail` | 24 | 37.56 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_016` | `json_answer,code,eval` | `fail` | 24 | 39.25 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_017` | `json_answer,code,eval` | `fail` | 24 | 39.22 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_018` | `json_answer,code,eval` | `fail` | 24 | 36.87 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_019` | `json_answer,code,eval` | `fail` | 24 | 38.71 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_020` | `json_answer,code,eval` | `fail` | 24 | 39.15 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_021` | `json_answer,code,eval` | `fail` | 24 | 38.68 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_022` | `json_answer,code,eval` | `fail` | 24 | 39.23 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_023` | `json_answer,code,eval` | `fail` | 24 | 39.07 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_024` | `json_answer,code,eval` | `fail` | 24 | 38.51 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_025` | `json_answer,code,eval` | `fail` | 24 | 38.55 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_026` | `json_answer,code,eval` | `fail` | 24 | 39.58 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_027` | `json_answer,code,eval` | `fail` | 24 | 39.68 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_028` | `json_answer,code,eval` | `fail` | 24 | 39.27 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_029` | `json_answer,code,eval` | `fail` | 24 | 36.12 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_030` | `json_answer,code,eval` | `fail` | 24 | 39.08 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_031` | `json_answer,code,eval` | `fail` | 24 | 38.91 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_032` | `json_answer,code,eval` | `fail` | 24 | 39.52 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_033` | `json_answer,code,eval` | `fail` | 24 | 38.72 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_code_034` | `json_answer,code,eval` | `fail` | 24 | 39.72 | <think>\nOkay, the user is asking about a Python-style identifier that's shown between backticks: `eval_json_handler |
| `json_v4_eval_compiler_001` | `json_answer,compiler,eval` | `fail` | 24 | 21.30 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_002` | `json_answer,compiler,eval` | `fail` | 24 | 26.24 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_003` | `json_answer,compiler,eval` | `fail` | 24 | 24.53 | <think>\nOkay, let's tackle this question. The user is asking about the glossary key that names the concept of |
| `json_v4_eval_compiler_004` | `json_answer,compiler,eval` | `fail` | 24 | 24.69 | <think>\nOkay, let's see. The user is asking about the glossary key in tinygrad GPU that refers to |
| `json_v4_eval_compiler_005` | `json_answer,compiler,eval` | `fail` | 24 | 23.26 | <think>\nOkay, let's see. The user is asking about the tinygrad GPU glossary. They mentioned that ` |
| `json_v4_eval_compiler_006` | `json_answer,compiler,eval` | `fail` | 24 | 22.82 | <think>\nOkay, let's see. The user is asking about a glossary key in the tinygrad GPU documentation. |
| `json_v4_eval_compiler_007` | `json_answer,compiler,eval` | `fail` | 24 | 25.63 | <think>\nOkay, let's see. The user is asking about the glossary key in tinygrad GPU that refers to |
| `json_v4_eval_compiler_008` | `json_answer,compiler,eval` | `fail` | 24 | 26.37 | <think>\nOkay, let's tackle this question. The user is asking about the glossary key that names the concept of |
| `json_v4_eval_compiler_009` | `json_answer,compiler,eval` | `fail` | 24 | 27.28 | <think>\nOkay, the user is asking about the tinygrad GPU glossary. They mentioned that `eval_qk_beam |
| `json_v4_eval_compiler_010` | `json_answer,compiler,eval` | `fail` | 24 | 24.85 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_011` | `json_answer,compiler,eval` | `fail` | 24 | 25.61 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_012` | `json_answer,compiler,eval` | `fail` | 24 | 22.47 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_013` | `json_answer,compiler,eval` | `fail` | 24 | 25.74 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_014` | `json_answer,compiler,eval` | `fail` | 24 | 25.82 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_015` | `json_answer,compiler,eval` | `fail` | 24 | 26.07 | <think>\nOkay, let's tackle this question. The user is asking about the glossary key that names the concept of |
| `json_v4_eval_compiler_016` | `json_answer,compiler,eval` | `fail` | 24 | 24.81 | <think>\nOkay, let's see. The user is asking about the glossary key in tinygrad GPU that refers to |
| `json_v4_eval_compiler_017` | `json_answer,compiler,eval` | `fail` | 24 | 25.78 | <think>\nOkay, let's see. The user is asking about the tinygrad GPU glossary. They mentioned that ` |
| `json_v4_eval_compiler_018` | `json_answer,compiler,eval` | `fail` | 24 | 23.66 | <think>\nOkay, let's see. The user is asking about a glossary key in the tinygrad GPU documentation. |
| `json_v4_eval_compiler_019` | `json_answer,compiler,eval` | `fail` | 24 | 25.77 | <think>\nOkay, let's see. The user is asking about the glossary key in tinygrad GPU that refers to |
| `json_v4_eval_compiler_020` | `json_answer,compiler,eval` | `fail` | 24 | 24.55 | <think>\nOkay, let's tackle this question. The user is asking about the glossary key that names the concept of |
| `json_v4_eval_compiler_021` | `json_answer,compiler,eval` | `fail` | 24 | 27.00 | <think>\nOkay, the user is asking about the tinygrad GPU glossary. They mentioned that `eval_qk_beam |
| `json_v4_eval_compiler_022` | `json_answer,compiler,eval` | `fail` | 24 | 25.93 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_023` | `json_answer,compiler,eval` | `fail` | 24 | 24.89 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_024` | `json_answer,compiler,eval` | `fail` | 24 | 23.30 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_025` | `json_answer,compiler,eval` | `fail` | 24 | 26.25 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_026` | `json_answer,compiler,eval` | `fail` | 24 | 26.16 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_compiler_027` | `json_answer,compiler,eval` | `fail` | 24 | 26.19 | <think>\nOkay, let's tackle this question. The user is asking about the glossary key that names the concept of |
| `json_v4_eval_compiler_028` | `json_answer,compiler,eval` | `fail` | 24 | 25.27 | <think>\nOkay, let's see. The user is asking about the glossary key in tinygrad GPU that refers to |
| `json_v4_eval_compiler_029` | `json_answer,compiler,eval` | `fail` | 24 | 26.59 | <think>\nOkay, let's see. The user is asking about the tinygrad GPU glossary. They mentioned that ` |
| `json_v4_eval_compiler_030` | `json_answer,compiler,eval` | `fail` | 24 | 23.89 | <think>\nOkay, let's see. The user is asking about a glossary key in the tinygrad GPU documentation. |
| `json_v4_eval_compiler_031` | `json_answer,compiler,eval` | `fail` | 24 | 25.62 | <think>\nOkay, let's see. The user is asking about the glossary key in tinygrad GPU that refers to |
| `json_v4_eval_compiler_032` | `json_answer,compiler,eval` | `fail` | 24 | 26.31 | <think>\nOkay, let's tackle this question. The user is asking about the glossary key that names the concept of |
| `json_v4_eval_compiler_033` | `json_answer,compiler,eval` | `fail` | 24 | 27.23 | <think>\nOkay, the user is asking about the tinygrad GPU glossary. They mentioned that `eval_qk_beam |
| `json_v4_eval_compiler_034` | `json_answer,compiler,eval` | `fail` | 24 | 25.98 | <think>\nOkay, let's tackle this question. The user is asking about a specific term in the tinygrad GPU gloss |
| `json_v4_eval_string_001` | `json_answer,string,eval` | `fail` | 24 | 32.25 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk001ab'. Let me |
| `json_v4_eval_string_002` | `json_answer,string,eval` | `fail` | 24 | 29.48 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_003` | `json_answer,string,eval` | `fail` | 24 | 33.53 | <think>\nOkay, the user wants me to reverse the string "evalqk003ab". Let me think |
| `json_v4_eval_string_004` | `json_answer,string,eval` | `fail` | 24 | 31.91 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk004ab'. Let me |
| `json_v4_eval_string_005` | `json_answer,string,eval` | `fail` | 24 | 28.40 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_006` | `json_answer,string,eval` | `fail` | 24 | 32.59 | <think>\nOkay, the user wants me to reverse the string "evalqk006ab". Let me think |
| `json_v4_eval_string_007` | `json_answer,string,eval` | `fail` | 24 | 31.86 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk007ab'. Let me |
| `json_v4_eval_string_008` | `json_answer,string,eval` | `fail` | 24 | 29.54 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_009` | `json_answer,string,eval` | `fail` | 24 | 31.92 | <think>\nOkay, the user wants me to reverse the string "evalqk009ab". Let me think |
| `json_v4_eval_string_010` | `json_answer,string,eval` | `fail` | 24 | 31.80 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk010ab'. Let me |
| `json_v4_eval_string_011` | `json_answer,string,eval` | `fail` | 24 | 29.66 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_012` | `json_answer,string,eval` | `fail` | 24 | 33.12 | <think>\nOkay, the user wants me to reverse the string "evalqk012ab". Let me think |
| `json_v4_eval_string_013` | `json_answer,string,eval` | `fail` | 24 | 31.84 | <think>\nOkay, the user wants me to return the uppercase form of the string 'evalqk013ab |
| `json_v4_eval_string_014` | `json_answer,string,eval` | `fail` | 24 | 29.66 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_015` | `json_answer,string,eval` | `fail` | 24 | 33.88 | <think>\nOkay, the user wants me to reverse the string "evalqk015ab". Let me think |
| `json_v4_eval_string_016` | `json_answer,string,eval` | `fail` | 24 | 32.42 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk016ab'. Let me |
| `json_v4_eval_string_017` | `json_answer,string,eval` | `fail` | 24 | 29.64 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_018` | `json_answer,string,eval` | `fail` | 24 | 31.84 | <think>\nOkay, the user wants me to reverse the string "evalqk018ab". Let me think |
| `json_v4_eval_string_019` | `json_answer,string,eval` | `fail` | 24 | 31.95 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk019ab'. Let me |
| `json_v4_eval_string_020` | `json_answer,string,eval` | `fail` | 24 | 28.54 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_021` | `json_answer,string,eval` | `fail` | 24 | 33.91 | <think>\nOkay, the user wants me to reverse the string "evalqk021ab". Let me think |
| `json_v4_eval_string_022` | `json_answer,string,eval` | `fail` | 24 | 32.22 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk022ab'. Let me |
| `json_v4_eval_string_023` | `json_answer,string,eval` | `fail` | 24 | 30.01 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_024` | `json_answer,string,eval` | `fail` | 24 | 31.68 | <think>\nOkay, the user wants me to reverse the string "evalqk024ab". Let me think |
| `json_v4_eval_string_025` | `json_answer,string,eval` | `fail` | 24 | 32.13 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk025ab'. Let me |
| `json_v4_eval_string_026` | `json_answer,string,eval` | `fail` | 24 | 28.99 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_027` | `json_answer,string,eval` | `fail` | 24 | 33.74 | <think>\nOkay, the user wants me to reverse the string "evalqk027ab". Let me think |
| `json_v4_eval_string_028` | `json_answer,string,eval` | `fail` | 24 | 32.06 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk028ab'. Let me |
| `json_v4_eval_string_029` | `json_answer,string,eval` | `fail` | 24 | 29.91 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_030` | `json_answer,string,eval` | `fail` | 24 | 31.59 | <think>\nOkay, the user wants me to reverse the string "evalqk030ab". Let me think |
| `json_v4_eval_string_031` | `json_answer,string,eval` | `fail` | 24 | 30.87 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk031ab'. Let me |
| `json_v4_eval_string_032` | `json_answer,string,eval` | `fail` | 24 | 28.68 | <think>\nOkay, let's see. The user wants me to take the first letters of each word in the phrase " |
| `json_v4_eval_string_033` | `json_answer,string,eval` | `fail` | 24 | 32.38 | <think>\nOkay, the user wants me to reverse the string "evalqk033ab". Let me think |
| `json_v4_eval_string_034` | `json_answer,string,eval` | `fail` | 24 | 29.80 | <think>\nOkay, the user wants the uppercase version of the string 'evalqk034ab'. Let me |
| `json_v4_eval_categorization_001` | `json_answer,categorization,eval` | `fail` | 24 | 17.83 | <think>\nOkay, the user wants me to classify 9001 as even or odd. Let me think. |
| `json_v4_eval_categorization_002` | `json_answer,categorization,eval` | `fail` | 24 | 22.60 | <think>\nOkay, the user wants me to classify 9002 as even or odd. Let me think. |
| `json_v4_eval_categorization_003` | `json_answer,categorization,eval` | `fail` | 24 | 22.29 | <think>\nOkay, the user wants me to classify the number 9003 as even or odd. Let me |
| `json_v4_eval_categorization_004` | `json_answer,categorization,eval` | `fail` | 24 | 22.78 | <think>\nOkay, the user wants me to classify 9004 as even or odd. Let me think. |
| `json_v4_eval_categorization_005` | `json_answer,categorization,eval` | `fail` | 24 | 23.26 | <think>\nOkay, the user wants me to classify 9005 as even or odd. Let me think. |
| `json_v4_eval_categorization_006` | `json_answer,categorization,eval` | `fail` | 24 | 23.43 | <think>\nOkay, the user wants me to classify 9006 as even or odd. Let me think. |
| `json_v4_eval_categorization_007` | `json_answer,categorization,eval` | `fail` | 24 | 23.22 | <think>\nOkay, the user wants me to classify 9007 as even or odd. Let me think. |
| `json_v4_eval_categorization_008` | `json_answer,categorization,eval` | `fail` | 24 | 22.19 | <think>\nOkay, the user wants me to classify 9008 as even or odd. Let me think. |
| `json_v4_eval_categorization_009` | `json_answer,categorization,eval` | `fail` | 24 | 23.58 | <think>\nOkay, the user wants me to classify 9009 as even or odd. Let me think. |
| `json_v4_eval_categorization_010` | `json_answer,categorization,eval` | `fail` | 24 | 19.65 | <think>\nOkay, the user wants me to classify 9010 as even or odd. Let me think. |
| `json_v4_eval_categorization_011` | `json_answer,categorization,eval` | `fail` | 24 | 23.10 | <think>\nOkay, the user wants me to classify 9011 as even or odd. Let me think. |
| `json_v4_eval_categorization_012` | `json_answer,categorization,eval` | `fail` | 24 | 23.58 | <think>\nOkay, the user wants me to classify the number 9012 as even or odd. Let me |
| `json_v4_eval_categorization_013` | `json_answer,categorization,eval` | `fail` | 24 | 23.29 | <think>\nOkay, the user wants me to classify the number 9013 as even or odd. Let me |
| `json_v4_eval_categorization_014` | `json_answer,categorization,eval` | `fail` | 24 | 23.14 | <think>\nOkay, the user wants me to classify the number 9014 as even or odd. Let me |
| `json_v4_eval_categorization_015` | `json_answer,categorization,eval` | `fail` | 24 | 15.41 | <think>\nOkay, the user wants me to classify the number 9015 as even or odd. Let me |
| `json_v4_eval_categorization_016` | `json_answer,categorization,eval` | `fail` | 24 | 22.92 | <think>\nOkay, the user wants me to classify the number 9016 as even or odd. Let me |
| `json_v4_eval_categorization_017` | `json_answer,categorization,eval` | `fail` | 24 | 22.81 | <think>\nOkay, the user wants me to classify the number 9017 as even or odd. Let me |
| `json_v4_eval_categorization_018` | `json_answer,categorization,eval` | `fail` | 24 | 22.60 | <think>\nOkay, the user wants me to classify the number 9018 as even or odd. Let me |
| `json_v4_eval_categorization_019` | `json_answer,categorization,eval` | `fail` | 24 | 23.29 | <think>\nOkay, the user wants me to classify the number 9019 as even or odd. Let me |
| `json_v4_eval_categorization_020` | `json_answer,categorization,eval` | `fail` | 24 | 19.87 | <think>\nOkay, the user wants me to classify 9020 as even or odd. Let me think. |
| `json_v4_eval_categorization_021` | `json_answer,categorization,eval` | `fail` | 24 | 23.02 | <think>\nOkay, the user wants me to classify 9021 as even or odd. Let me think. |
| `json_v4_eval_categorization_022` | `json_answer,categorization,eval` | `fail` | 24 | 23.56 | <think>\nOkay, the user wants me to classify the number 9022 as even or odd. Let me |
| `json_v4_eval_categorization_023` | `json_answer,categorization,eval` | `fail` | 24 | 21.99 | <think>\nOkay, the user wants me to classify the number 9023 as even or odd. Let me |
| `json_v4_eval_categorization_024` | `json_answer,categorization,eval` | `fail` | 24 | 22.28 | <think>\nOkay, the user wants me to classify the number 9024 as even or odd. Let me |
| `json_v4_eval_categorization_025` | `json_answer,categorization,eval` | `fail` | 24 | 23.58 | <think>\nOkay, the user wants me to classify 9025 as even or odd. Let me think. |
| `json_v4_eval_categorization_026` | `json_answer,categorization,eval` | `fail` | 24 | 23.02 | <think>\nOkay, the user wants me to classify the number 9026 as even or odd. Let me |
| `json_v4_eval_categorization_027` | `json_answer,categorization,eval` | `fail` | 24 | 23.55 | <think>\nOkay, the user wants me to classify the number 9027 as even or odd. Let me |
| `json_v4_eval_categorization_028` | `json_answer,categorization,eval` | `fail` | 24 | 23.35 | <think>\nOkay, the user wants me to classify the number 9028 as even or odd. Let me |
| `json_v4_eval_categorization_029` | `json_answer,categorization,eval` | `fail` | 24 | 23.36 | <think>\nOkay, the user wants me to classify the number 9029 as even or odd. Let me |
| `json_v4_eval_categorization_030` | `json_answer,categorization,eval` | `fail` | 24 | 23.22 | <think>\nOkay, the user wants me to classify 9030 as even or odd. Let me think. |
| `json_v4_eval_categorization_031` | `json_answer,categorization,eval` | `fail` | 24 | 23.13 | <think>\nOkay, the user wants me to classify the number 9031 as even or odd. Let me |
| `json_v4_eval_categorization_032` | `json_answer,categorization,eval` | `fail` | 24 | 19.01 | <think>\nOkay, the user wants me to classify the number 9032 as even or odd. Let me |
| `json_v4_eval_categorization_033` | `json_answer,categorization,eval` | `fail` | 24 | 23.42 | <think>\nOkay, the user wants me to classify the number 9033 as even or odd. Let me |
| `json_v4_eval_categorization_034` | `json_answer,categorization,eval` | `fail` | 24 | 18.11 | <think>\nOkay, the user wants me to classify the number 9034 as even or odd. Let me |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `arithmetic` | 0 | 34 | 0.00 |
| `categorization` | 0 | 34 | 0.00 |
| `code` | 0 | 34 | 0.00 |
| `compiler` | 0 | 34 | 0.00 |
| `eval` | 0 | 204 | 0.00 |
| `fact` | 0 | 34 | 0.00 |
| `json_answer` | 0 | 204 | 0.00 |
| `string` | 0 | 34 | 0.00 |

## JSON Quality Axes

| axis | passed | scored | pass rate [95% CI] |
|---|---:|---:|---:|
| `parse_valid` | 0 | 204 | 0.00 [0.00, 0.02] |
| `no_extra_text` | 0 | 204 | 0.00 [0.00, 0.02] |
| `schema_ok` | 0 | 204 | 0.00 [0.00, 0.02] |
| `type_ok` | 0 | 204 | 0.00 [0.00, 0.02] |
| `value_correct` | 0 | 204 | 0.00 [0.00, 0.02] |
| `strict_pass` | 0 | 204 | 0.00 [0.00, 0.02] |
