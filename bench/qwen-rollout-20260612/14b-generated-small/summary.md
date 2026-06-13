# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/14b/policy.json`
- dataset: `bench/qwen-rollout-20260612/dataset-small.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `75`
- generated tokens: `644`
- quality: `pass` (75/75)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `math_add_001` | `math_short,math` | `pass` | 6 | 0.27 | <think>\n\n</think>\n\n42 |
| `math_sub_002` | `math_short,math` | `pass` | 6 | 0.68 | <think>\n\n</think>\n\n63 |
| `math_mul_003` | `math_short,math` | `pass` | 6 | 12.88 | <think>\n\n</think>\n\n96 |
| `math_div_004` | `math_short,math` | `pass` | 6 | 11.60 | <think>\n\n</think>\n\n12 |
| `math_square_005` | `math_short,math` | `pass` | 7 | 15.76 | <think>\n\n</think>\n\n121 |
| `math_half_006` | `math_short,math` | `pass` | 6 | 12.35 | <think>\n\n</think>\n\n25 |
| `math_percent_007` | `math_short,math` | `pass` | 5 | 9.64 | <think>\n\n</think>\n\n8 |
| `math_double_008` | `math_short,pattern` | `pass` | 6 | 7.58 | <think>\n\n</think>\n\n48 |
| `math_plus3_009` | `math_short,pattern` | `pass` | 6 | 9.30 | <think>\n\n</think>\n\n17 |
| `math_min_010` | `math_short,comparison` | `pass` | 6 | 9.29 | <think>\n\n</think>\n\n19 |
| `math_max_011` | `math_short,comparison` | `pass` | 6 | 11.09 | <think>\n\n</think>\n\n15 |
| `math_mod_012` | `math_short,math` | `pass` | 5 | 11.43 | <think>\n\n</think>\n\n2 |
| `math_avg_013` | `math_short,math` | `pass` | 5 | 8.03 | <think>\n\n</think>\n\n6 |
| `math_order_014` | `math_short,math` | `pass` | 6 | 11.51 | <think>\n\n</think>\n\n14 |
| `math_negative_015` | `math_short,math` | `pass` | 5 | 10.17 | <think>\n\n</think>\n\n7 |
| `code_len_001` | `code_short,code` | `pass` | 5 | 6.26 | <think>\n\n</think>\n\nlen |
| `code_append_002` | `code_short,code` | `pass` | 5 | 6.66 | <think>\n\n</think>\n\nappend |
| `code_loop_003` | `code_short,code` | `pass` | 5 | 6.92 | <think>\n\n</think>\n\nfor |
| `code_dict_004` | `code_short,code` | `pass` | 5 | 7.48 | <think>\n\n</think>\n\nDictionary |
| `code_import_005` | `code_short,code` | `pass` | 5 | 8.02 | <think>\n\n</think>\n\nimport |
| `code_comment_006` | `code_short,code` | `pass` | 5 | 6.41 | <think>\n\n</think>\n\n# |
| `code_equal_007` | `code_short,code` | `pass` | 5 | 7.85 | <think>\n\n</think>\n\n== |
| `code_true_008` | `code_short,code` | `pass` | 5 | 6.78 | <think>\n\n</think>\n\nTrue |
| `code_print_009` | `code_short,code` | `pass` | 5 | 7.01 | <think>\n\n</think>\n\nprint |
| `code_index_010` | `code_short,code` | `pass` | 5 | 7.33 | <think>\n\n</think>\n\n0 |
| `format_json_001` | `format_following,format,json` | `pass` | 10 | 12.68 | <think>\n\n</think>\n\n{"result": "ok"} |
| `format_oneword_002` | `format_following,format` | `pass` | 5 | 8.20 | <think>\n\n</think>\n\nbanana |
| `format_csv_003` | `format_following,format` | `pass` | 7 | 10.24 | <think>\n\n</think>\n\nname,age |
| `format_heading_004` | `format_following,format` | `pass` | 6 | 10.30 | <think>\n\n</think>\n\n# Done |
| `format_yaml_005` | `format_following,format` | `pass` | 12 | 16.63 | <think>\n\n</think>\n\n```yaml\nstatus: pass\n``` |
| `format_letters_006` | `format_following,format` | `pass` | 5 | 8.04 | <think>\n\n</think>\n\nAMD |
| `format_lower_007` | `format_following,format` | `pass` | 5 | 9.68 | <think>\n\n</think>\n\ngreen |
| `format_bullets_008` | `format_following,format` | `pass` | 25 | 22.47 | <think>\n\n</think>\n\n- **Alpha**: A bullet item named Alpha.  \n- **Beta**: A bullet item named Beta. |
| `format_quote_009` | `format_following,format` | `pass` | 7 | 11.56 | <think>\n\n</think>\n\nSearch works. |
| `format_colon_010` | `format_following,format` | `pass` | 8 | 12.69 | <think>\n\n</think>\n\nanswer: ready. |
| `fact_capital_france_001` | `facts_short,facts` | `pass` | 6 | 8.17 | <think>\n\n</think>\n\nParis. |
| `fact_capital_japan_002` | `facts_short,facts` | `pass` | 11 | 20.33 | <think>\n\n</think>\n\nThe capital of Japan is Tokyo. |
| `fact_water_003` | `facts_short,facts` | `pass` | 14 | 21.50 | <think>\n\n</think>\n\nThe chemical formula for water is H₂O. |
| `fact_red_planet_004` | `facts_short,facts` | `pass` | 7 | 12.77 | <think>\n\n</think>\n\nMars. |
| `fact_ocean_005` | `facts_short,facts` | `pass` | 14 | 19.41 | <think>\n\n</think>\n\nThe largest ocean on Earth is the Pacific Ocean. |
| `fact_hamlet_006` | `facts_short,facts` | `pass` | 7 | 14.41 | <think>\n\n</think>\n\nWilliam Shakespeare. |
| `fact_freezing_007` | `facts_short,facts` | `pass` | 6 | 11.27 | <think>\n\n</think>\n\n0°C |
| `fact_egg_mammal_008` | `facts_short,facts` | `pass` | 9 | 15.01 | <think>\n\n</think>\n\nPlatypus. |
| `fact_us_currency_009` | `facts_short,facts` | `pass` | 17 | 21.01 | <think>\n\n</think>\n\nThe currency of the United States is the **Dollar**. |
| `fact_egypt_010` | `facts_short,facts` | `pass` | 5 | 10.77 | <think>\n\n</think>\n\nAfrica |
| `reason_alpha_001` | `reasoning_short,reasoning` | `pass` | 5 | 8.05 | <think>\n\n</think>\n\napple |
| `reason_days_002` | `reasoning_short,reasoning` | `pass` | 13 | 18.78 | <think>\n\n</think>\n\nTwo days after Monday is **Wednesday**. |
| `reason_opposite_003` | `reasoning_short,reasoning` | `pass` | 6 | 10.10 | <think>\n\n</think>\n\nSouth. |
| `reason_mammal_004` | `reasoning_short,reasoning` | `pass` | 6 | 6.38 | <think>\n\n</think>\n\nYes. |
| `reason_heavier_005` | `reasoning_short,reasoning` | `pass` | 10 | 10.43 | <think>\n\n</think>\n\n1 kilogram is heavier. |
| `reason_warmest_006` | `reasoning_short,reasoning` | `pass` | 9 | 13.66 | <think>\n\n</think>\n\nSteam is warmest. |
| `reason_month_007` | `reasoning_short,reasoning` | `pass` | 6 | 11.00 | <think>\n\n</think>\n\nJanuary. |
| `reason_last_letter_008` | `reasoning_short,reasoning` | `pass` | 5 | 6.78 | <think>\n\n</think>\n\nD |
| `reason_sort_009` | `reasoning_short,reasoning` | `pass` | 11 | 10.61 | <think>\n\n</think>\n\n1, 2, 3 |
| `reason_mix_010` | `reasoning_short,reasoning` | `pass` | 14 | 14.42 | <think>\n\n</think>\n\nMixing red and blue paint usually makes purple. |
| `compiler_jit_001` | `compiler_tinygrad,compiler` | `pass` | 9 | 10.17 | <think>\n\n</think>\n\nJust-In-Time compilation. |
| `compiler_matvec_002` | `compiler_tinygrad,compiler` | `pass` | 8 | 11.33 | <think>\n\n</think>\n\nMatrix-vector multiplication. |
| `compiler_gemv_003` | `compiler_tinygrad,compiler` | `pass` | 16 | 17.88 | <think>\n\n</think>\n\nGEMV multiplies a general matrix by a vector. |
| `compiler_tensor_004` | `compiler_tinygrad,tinygrad` | `pass` | 11 | 14.42 | <think>\n\n</think>\n\nTinygrad mainly operates on tensors. |
| `compiler_pass_005` | `compiler_tinygrad,compiler` | `pass` | 20 | 22.75 | <think>\n\n</think>\n\nA compiler optimization pass tries to improve the efficiency and performance of the generated code. |
| `compiler_quant_006` | `compiler_tinygrad,compiler` | `pass` | 25 | 23.91 | <think>\n\n</think>\n\nQuantization usually reduces the precision of model weights, typically converting them from floating-point to lower-bit integers. |
| `compiler_dequant_007` | `compiler_tinygrad,compiler` | `pass` | 18 | 21.15 | <think>\n\n</think>\n\nDequantization converts quantized values back toward their original continuous values. |
| `compiler_gpu_008` | `compiler_tinygrad,compiler` | `pass` | 8 | 11.18 | <think>\n\n</think>\n\nGraphics Processing Unit. |
| `compiler_kernel_009` | `compiler_tinygrad,compiler` | `pass` | 24 | 21.08 | <think>\n\n</think>\n\nA kernel is a function that is executed in parallel on multiple threads of a GPU to perform computations. |
| `compiler_bandwidth_010` | `compiler_tinygrad,compiler` | `pass` | 23 | 25.50 | <think>\n\n</think>\n\nMemory bandwidth measures the rate at which data can be read from or written to a memory unit. |
| `instr_even_001` | `instruction_following,instruction` | `pass` | 6 | 8.15 | <think>\n\n</think>\n\nYes. |
| `instr_upper_002` | `instruction_following,instruction` | `pass` | 15 | 19.33 | <think>\n\n</think>\n\nThe uppercase of "cat" is **CAT**. |
| `instr_lower_003` | `instruction_following,instruction` | `pass` | 14 | 19.13 | <think>\n\n</think>\n\nThe lowercase of "DOG" is "dog". |
| `instr_repeat_004` | `instruction_following,instruction` | `pass` | 6 | 9.09 | <think>\n\n</think>\n\nhi hi |
| `instr_option_005` | `instruction_following,instruction` | `pass` | 5 | 6.38 | <think>\n\n</think>\n\nblue |
| `instr_vowels_006` | `instruction_following,instruction` | `pass` | 5 | 6.78 | <think>\n\n</think>\n\n1 |
| `instr_initials_007` | `instruction_following,instruction` | `pass` | 5 | 6.40 | <think>\n\n</think>\n\nAB |
| `instr_final_word_008` | `instruction_following,instruction` | `pass` | 6 | 9.35 | <think>\n\n</think>\n\nCompute. |
| `instr_underscore_009` | `instruction_following,instruction` | `pass` | 6 | 7.74 | <think>\n\n</think>\n\namd_gpu |
| `instr_ok_010` | `instruction_following,instruction` | `pass` | 6 | 11.16 | <think>\n\n</think>\n\nOK. |

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
