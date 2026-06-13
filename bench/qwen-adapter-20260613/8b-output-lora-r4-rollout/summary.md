# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter: `bench/qwen-adapter-20260613/8b-output-lora-r4`
- dataset: `bench/qwen-rollout-20260612/dataset-small.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `75`
- generated tokens: `608`
- quality: `pass` (75/75)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `math_add_001` | `math_short,math` | `pass` | 6 | 0.32 | <think>\n\n</think>\n\n42 |
| `math_sub_002` | `math_short,math` | `pass` | 6 | 0.81 | <think>\n\n</think>\n\n63 |
| `math_mul_003` | `math_short,math` | `pass` | 6 | 8.35 | <think>\n\n</think>\n\n96 |
| `math_div_004` | `math_short,math` | `pass` | 6 | 8.26 | <think>\n\n</think>\n\n12 |
| `math_square_005` | `math_short,math` | `pass` | 7 | 9.01 | <think>\n\n</think>\n\n121 |
| `math_half_006` | `math_short,math` | `pass` | 6 | 8.32 | <think>\n\n</think>\n\n25 |
| `math_percent_007` | `math_short,math` | `pass` | 5 | 7.20 | <think>\n\n</think>\n\n8 |
| `math_double_008` | `math_short,pattern` | `pass` | 6 | 6.81 | <think>\n\n</think>\n\n48 |
| `math_plus3_009` | `math_short,pattern` | `pass` | 6 | 7.35 | <think>\n\n</think>\n\n17 |
| `math_min_010` | `math_short,comparison` | `pass` | 6 | 7.24 | <think>\n\n</think>\n\n19 |
| `math_max_011` | `math_short,comparison` | `pass` | 6 | 7.83 | <think>\n\n</think>\n\n15 |
| `math_mod_012` | `math_short,math` | `pass` | 5 | 7.61 | <think>\n\n</think>\n\n2 |
| `math_avg_013` | `math_short,math` | `pass` | 5 | 6.90 | <think>\n\n</think>\n\n6 |
| `math_order_014` | `math_short,math` | `pass` | 6 | 8.20 | <think>\n\n</think>\n\n14 |
| `math_negative_015` | `math_short,math` | `pass` | 5 | 7.71 | <think>\n\n</think>\n\n7 |
| `code_len_001` | `code_short,code` | `pass` | 6 | 6.56 | <think>\n\n</think>\n\nlen() |
| `code_append_002` | `code_short,code` | `pass` | 6 | 6.63 | <think>\n\n</think>\n\nappend() |
| `code_loop_003` | `code_short,code` | `pass` | 5 | 6.27 | <think>\n\n</think>\n\nfor |
| `code_dict_004` | `code_short,code` | `pass` | 5 | 6.58 | <think>\n\n</think>\n\nDictionary |
| `code_import_005` | `code_short,code` | `pass` | 5 | 6.79 | <think>\n\n</think>\n\nimport |
| `code_comment_006` | `code_short,code` | `pass` | 5 | 6.19 | <think>\n\n</think>\n\n# |
| `code_equal_007` | `code_short,code` | `pass` | 5 | 6.61 | <think>\n\n</think>\n\n== |
| `code_true_008` | `code_short,code` | `pass` | 5 | 6.24 | <think>\n\n</think>\n\nTrue |
| `code_print_009` | `code_short,code` | `pass` | 5 | 6.07 | <think>\n\n</think>\n\nprint |
| `code_index_010` | `code_short,code` | `pass` | 5 | 6.47 | <think>\n\n</think>\n\n0 |
| `format_json_001` | `format_following,format,json` | `pass` | 17 | 9.71 | <think>\n\n</think>\n\n```json\n{\n  "result": "ok"\n}\n``` |
| `format_oneword_002` | `format_following,format` | `pass` | 5 | 6.83 | <think>\n\n</think>\n\nbanana |
| `format_csv_003` | `format_following,format` | `pass` | 7 | 7.58 | <think>\n\n</think>\n\nname,age |
| `format_heading_004` | `format_following,format` | `pass` | 6 | 7.68 | <think>\n\n</think>\n\n# Done |
| `format_yaml_005` | `format_following,format` | `pass` | 12 | 9.75 | <think>\n\n</think>\n\n```yaml\nstatus: pass\n``` |
| `format_letters_006` | `format_following,format` | `pass` | 5 | 6.94 | <think>\n\n</think>\n\nAMD |
| `format_lower_007` | `format_following,format` | `pass` | 5 | 7.23 | <think>\n\n</think>\n\ngreen |
| `format_bullets_008` | `format_following,format` | `pass` | 15 | 9.91 | <think>\n\n</think>\n\n- alpha: First item  \n- beta: Second item |
| `format_quote_009` | `format_following,format` | `pass` | 6 | 7.66 | <think>\n\n</think>\n\nsearch works |
| `format_colon_010` | `format_following,format` | `pass` | 8 | 8.41 | <think>\n\n</think>\n\nanswer: ready. |
| `fact_capital_france_001` | `facts_short,facts` | `pass` | 6 | 6.93 | <think>\n\n</think>\n\nParis. |
| `fact_capital_japan_002` | `facts_short,facts` | `pass` | 6 | 8.88 | <think>\n\n</think>\n\nTokyo |
| `fact_water_003` | `facts_short,facts` | `pass` | 7 | 9.03 | <think>\n\n</think>\n\nH₂O |
| `fact_red_planet_004` | `facts_short,facts` | `pass` | 6 | 7.92 | <think>\n\n</think>\n\nMars |
| `fact_ocean_005` | `facts_short,facts` | `pass` | 14 | 10.16 | <think>\n\n</think>\n\nThe largest ocean on Earth is the Pacific Ocean. |
| `fact_hamlet_006` | `facts_short,facts` | `pass` | 7 | 8.79 | <think>\n\n</think>\n\nWilliam Shakespeare. |
| `fact_freezing_007` | `facts_short,facts` | `pass` | 6 | 8.14 | <think>\n\n</think>\n\n0°C |
| `fact_egg_mammal_008` | `facts_short,facts` | `pass` | 8 | 8.99 | <think>\n\n</think>\n\nPlatypus |
| `fact_us_currency_009` | `facts_short,facts` | `pass` | 15 | 10.30 | <think>\n\n</think>\n\nThe currency of the United States is the US Dollar. |
| `fact_egypt_010` | `facts_short,facts` | `pass` | 5 | 7.83 | <think>\n\n</think>\n\nAfrica |
| `reason_alpha_001` | `reasoning_short,reasoning` | `pass` | 5 | 6.89 | <think>\n\n</think>\n\napple |
| `reason_days_002` | `reasoning_short,reasoning` | `pass` | 5 | 7.57 | <think>\n\n</think>\n\nWednesday |
| `reason_opposite_003` | `reasoning_short,reasoning` | `pass` | 5 | 7.56 | <think>\n\n</think>\n\nSouth |
| `reason_mammal_004` | `reasoning_short,reasoning` | `pass` | 6 | 6.09 | <think>\n\n</think>\n\nYes. |
| `reason_heavier_005` | `reasoning_short,reasoning` | `pass` | 14 | 8.58 | <think>\n\n</think>\n\n1 kilogram is heavier than 1 gram. |
| `reason_warmest_006` | `reasoning_short,reasoning` | `pass` | 10 | 9.12 | <think>\n\n</think>\n\nSteam is the warmest. |
| `reason_month_007` | `reasoning_short,reasoning` | `pass` | 5 | 7.26 | <think>\n\n</think>\n\nJanuary |
| `reason_last_letter_008` | `reasoning_short,reasoning` | `pass` | 5 | 6.30 | <think>\n\n</think>\n\nD |
| `reason_sort_009` | `reasoning_short,reasoning` | `pass` | 11 | 7.69 | <think>\n\n</think>\n\n1, 2, 3 |
| `reason_mix_010` | `reasoning_short,reasoning` | `pass` | 5 | 6.15 | <think>\n\n</think>\n\nPurple |
| `compiler_jit_001` | `compiler_tinygrad,compiler` | `pass` | 9 | 7.65 | <think>\n\n</think>\n\nJust-In-Time compilation. |
| `compiler_matvec_002` | `compiler_tinygrad,compiler` | `pass` | 8 | 8.26 | <think>\n\n</think>\n\nMatrix-vector multiplication. |
| `compiler_gemv_003` | `compiler_tinygrad,compiler` | `pass` | 15 | 9.98 | <think>\n\n</think>\n\nGEMV multiplies a matrix by a vector. |
| `compiler_tensor_004` | `compiler_tinygrad,tinygrad` | `pass` | 11 | 9.46 | <think>\n\n</think>\n\nTinygrad mainly operates on tensors. |
| `compiler_pass_005` | `compiler_tinygrad,compiler` | `pass` | 20 | 10.75 | <think>\n\n</think>\n\nA compiler optimization pass tries to improve the performance or efficiency of the generated code. |
| `compiler_quant_006` | `compiler_tinygrad,compiler` | `pass` | 25 | 10.71 | <think>\n\n</think>\n\nQuantization reduces the precision of model weights, typically converting them from floating-point to lower-bit integer representations. |
| `compiler_dequant_007` | `compiler_tinygrad,compiler` | `pass` | 18 | 9.99 | <think>\n\n</think>\n\nDequantization converts quantized values back toward their original continuous range. |
| `compiler_gpu_008` | `compiler_tinygrad,compiler` | `pass` | 8 | 8.41 | <think>\n\n</think>\n\nGraphics Processing Unit. |
| `compiler_kernel_009` | `compiler_tinygrad,compiler` | `pass` | 25 | 10.02 | <think>\n\n</think>\n\nA kernel is a function executed on a GPU by multiple threads in parallel to perform computations on large datasets. |
| `compiler_bandwidth_010` | `compiler_tinygrad,compiler` | `pass` | 21 | 11.27 | <think>\n\n</think>\n\nMemory bandwidth measures the rate at which data can be read from or written to memory. |
| `instr_even_001` | `instruction_following,instruction` | `pass` | 6 | 6.90 | <think>\n\n</think>\n\nYes. |
| `instr_upper_002` | `instruction_following,instruction` | `pass` | 15 | 9.82 | <think>\n\n</think>\n\nThe uppercase of "cat" is **CAT**. |
| `instr_lower_003` | `instruction_following,instruction` | `pass` | 14 | 9.76 | <think>\n\n</think>\n\nThe lowercase of "DOG" is "dog". |
| `instr_repeat_004` | `instruction_following,instruction` | `pass` | 6 | 4.80 | <think>\n\n</think>\n\nhi hi |
| `instr_option_005` | `instruction_following,instruction` | `pass` | 5 | 4.75 | <think>\n\n</think>\n\nblue |
| `instr_vowels_006` | `instruction_following,instruction` | `pass` | 5 | 6.09 | <think>\n\n</think>\n\n1 |
| `instr_initials_007` | `instruction_following,instruction` | `pass` | 5 | 5.77 | <think>\n\n</think>\n\nAB |
| `instr_final_word_008` | `instruction_following,instruction` | `pass` | 5 | 6.76 | <think>\n\n</think>\n\ncompute |
| `instr_underscore_009` | `instruction_following,instruction` | `pass` | 6 | 6.73 | <think>\n\n</think>\n\namd_gpu |
| `instr_ok_010` | `instruction_following,instruction` | `pass` | 5 | 7.23 | <think>\n\n</think>\n\nOK |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `code` | 10 | 10 | 1.00 |
| `code_short` | 10 | 10 | 1.00 |
| `comparison` | 2 | 2 | 1.00 |
| `compiler` | 9 | 9 | 1.00 |
| `compiler_tinygrad` | 10 | 10 | 1.00 |
| `facts` | 10 | 10 | 1.00 |
| `facts_short` | 10 | 10 | 1.00 |
| `format` | 10 | 10 | 1.00 |
| `format_following` | 10 | 10 | 1.00 |
| `instruction` | 10 | 10 | 1.00 |
| `instruction_following` | 10 | 10 | 1.00 |
| `json` | 1 | 1 | 1.00 |
| `math` | 11 | 11 | 1.00 |
| `math_short` | 15 | 15 | 1.00 |
| `pattern` | 2 | 2 | 1.00 |
| `reasoning` | 10 | 10 | 1.00 |
| `reasoning_short` | 10 | 10 | 1.00 |
| `tinygrad` | 1 | 1 | 1.00 |
