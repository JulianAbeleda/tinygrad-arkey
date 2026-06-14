# LLM Suffix Adapter Train

This gate caches hidden states at the selected suffix boundary, then trains
LoRA tensors only inside the suffix. It is a diagnostic path for internal
adapters when full-model adapter backprop is too slow or too memory-heavy.

## Summary

- status: `pass`
- adapter: `suffix_lora` rank `4` alpha `8.0`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- targets: `last1_ffn`
- suffix start block: `35` (`1` blocks)
- parity: `pass` max_abs `0.0`
- cache bytes: train `3829235712`, eval `1965211648`
- rows: `612` (`408` train, `204` eval)
- examples: `4361` train, `2235` eval
- adapter L2 delta: `3.538910`

## Metrics

| split | initial loss | final loss | loss delta | initial acc | final acc | acc delta |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 5.4982 | 0.3050 | 5.1932 | 0.6406 | 0.9219 | 0.2812 |
| `eval` | 5.2144 | 0.4231 | 4.7913 | 0.6562 | 0.9219 | 0.2656 |

## History

| step | example | loss |
|---:|---|---:|
| 1 | `json_v4_train_code_005:tok0` | 27.7750 |
| 30 | `json_v4_train_compiler_052:tok10` | -0.0000 |
| 60 | `json_v4_train_fact_009:tok3` | 0.0000 |
| 90 | `json_v4_train_compiler_051:tok10` | 0.0000 |
| 120 | `json_v4_train_string_040:tok9` | -0.0000 |

## Eval Source IDs

json_v4_eval_arithmetic_001, json_v4_eval_arithmetic_002, json_v4_eval_arithmetic_003, json_v4_eval_arithmetic_004, json_v4_eval_arithmetic_005, json_v4_eval_arithmetic_006, json_v4_eval_arithmetic_007, json_v4_eval_arithmetic_008, json_v4_eval_arithmetic_009, json_v4_eval_arithmetic_010, json_v4_eval_arithmetic_011, json_v4_eval_arithmetic_012, json_v4_eval_arithmetic_013, json_v4_eval_arithmetic_014, json_v4_eval_arithmetic_015, json_v4_eval_arithmetic_016, json_v4_eval_arithmetic_017, json_v4_eval_arithmetic_018, json_v4_eval_arithmetic_019, json_v4_eval_arithmetic_020, json_v4_eval_arithmetic_021, json_v4_eval_arithmetic_022, json_v4_eval_arithmetic_023, json_v4_eval_arithmetic_024, json_v4_eval_arithmetic_025, json_v4_eval_arithmetic_026, json_v4_eval_arithmetic_027, json_v4_eval_arithmetic_028, json_v4_eval_arithmetic_029, json_v4_eval_arithmetic_030, json_v4_eval_arithmetic_031, json_v4_eval_arithmetic_032, json_v4_eval_arithmetic_033, json_v4_eval_arithmetic_034, json_v4_eval_categorization_001, json_v4_eval_categorization_002, json_v4_eval_categorization_003, json_v4_eval_categorization_004, json_v4_eval_categorization_005, json_v4_eval_categorization_006, json_v4_eval_categorization_007, json_v4_eval_categorization_008, json_v4_eval_categorization_009, json_v4_eval_categorization_010, json_v4_eval_categorization_011, json_v4_eval_categorization_012, json_v4_eval_categorization_013, json_v4_eval_categorization_014, json_v4_eval_categorization_015, json_v4_eval_categorization_016, json_v4_eval_categorization_017, json_v4_eval_categorization_018, json_v4_eval_categorization_019, json_v4_eval_categorization_020, json_v4_eval_categorization_021, json_v4_eval_categorization_022, json_v4_eval_categorization_023, json_v4_eval_categorization_024, json_v4_eval_categorization_025, json_v4_eval_categorization_026, json_v4_eval_categorization_027, json_v4_eval_categorization_028, json_v4_eval_categorization_029, json_v4_eval_categorization_030, json_v4_eval_categorization_031, json_v4_eval_categorization_032, json_v4_eval_categorization_033, json_v4_eval_categorization_034, json_v4_eval_code_001, json_v4_eval_code_002, json_v4_eval_code_003, json_v4_eval_code_004, json_v4_eval_code_005, json_v4_eval_code_006, json_v4_eval_code_007, json_v4_eval_code_008, json_v4_eval_code_009, json_v4_eval_code_010, json_v4_eval_code_011, json_v4_eval_code_012, json_v4_eval_code_013, json_v4_eval_code_014, json_v4_eval_code_015, json_v4_eval_code_016, json_v4_eval_code_017, json_v4_eval_code_018, json_v4_eval_code_019, json_v4_eval_code_020, json_v4_eval_code_021, json_v4_eval_code_022, json_v4_eval_code_023, json_v4_eval_code_024, json_v4_eval_code_025, json_v4_eval_code_026, json_v4_eval_code_027, json_v4_eval_code_028, json_v4_eval_code_029, json_v4_eval_code_030, json_v4_eval_code_031, json_v4_eval_code_032, json_v4_eval_code_033, json_v4_eval_code_034, json_v4_eval_compiler_001, json_v4_eval_compiler_002, json_v4_eval_compiler_003, json_v4_eval_compiler_004, json_v4_eval_compiler_005, json_v4_eval_compiler_006, json_v4_eval_compiler_007, json_v4_eval_compiler_008, json_v4_eval_compiler_009, json_v4_eval_compiler_010, json_v4_eval_compiler_011, json_v4_eval_compiler_012, json_v4_eval_compiler_013, json_v4_eval_compiler_014, json_v4_eval_compiler_015, json_v4_eval_compiler_016, json_v4_eval_compiler_017, json_v4_eval_compiler_018, json_v4_eval_compiler_019, json_v4_eval_compiler_020, json_v4_eval_compiler_021, json_v4_eval_compiler_022, json_v4_eval_compiler_023, json_v4_eval_compiler_024, json_v4_eval_compiler_025, json_v4_eval_compiler_026, json_v4_eval_compiler_027, json_v4_eval_compiler_028, json_v4_eval_compiler_029, json_v4_eval_compiler_030, json_v4_eval_compiler_031, json_v4_eval_compiler_032, json_v4_eval_compiler_033, json_v4_eval_compiler_034, json_v4_eval_fact_001, json_v4_eval_fact_002, json_v4_eval_fact_003, json_v4_eval_fact_004, json_v4_eval_fact_005, json_v4_eval_fact_006, json_v4_eval_fact_007, json_v4_eval_fact_008, json_v4_eval_fact_009, json_v4_eval_fact_010, json_v4_eval_fact_011, json_v4_eval_fact_012, json_v4_eval_fact_013, json_v4_eval_fact_014, json_v4_eval_fact_015, json_v4_eval_fact_016, json_v4_eval_fact_017, json_v4_eval_fact_018, json_v4_eval_fact_019, json_v4_eval_fact_020, json_v4_eval_fact_021, json_v4_eval_fact_022, json_v4_eval_fact_023, json_v4_eval_fact_024, json_v4_eval_fact_025, json_v4_eval_fact_026, json_v4_eval_fact_027, json_v4_eval_fact_028, json_v4_eval_fact_029, json_v4_eval_fact_030, json_v4_eval_fact_031, json_v4_eval_fact_032, json_v4_eval_fact_033, json_v4_eval_fact_034, json_v4_eval_string_001, json_v4_eval_string_002, json_v4_eval_string_003, json_v4_eval_string_004, json_v4_eval_string_005, json_v4_eval_string_006, json_v4_eval_string_007, json_v4_eval_string_008, json_v4_eval_string_009, json_v4_eval_string_010, json_v4_eval_string_011, json_v4_eval_string_012, json_v4_eval_string_013, json_v4_eval_string_014, json_v4_eval_string_015, json_v4_eval_string_016, json_v4_eval_string_017, json_v4_eval_string_018, json_v4_eval_string_019, json_v4_eval_string_020, json_v4_eval_string_021, json_v4_eval_string_022, json_v4_eval_string_023, json_v4_eval_string_024, json_v4_eval_string_025, json_v4_eval_string_026, json_v4_eval_string_027, json_v4_eval_string_028, json_v4_eval_string_029, json_v4_eval_string_030, json_v4_eval_string_031, json_v4_eval_string_032, json_v4_eval_string_033, json_v4_eval_string_034
