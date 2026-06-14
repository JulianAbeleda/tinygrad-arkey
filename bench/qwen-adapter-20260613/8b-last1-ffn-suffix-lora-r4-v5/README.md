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
- cache bytes: train `187072512`, eval `48971776`
- rows: `60` (`48` train, `12` eval)
- examples: `319` train, `80` eval
- adapter L2 delta: `4.624648`

## Metrics

| split | initial loss | final loss | loss delta | initial acc | final acc | acc delta |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 7.1041 | 0.2817 | 6.8224 | 0.5417 | 0.9583 | 0.4167 |
| `eval` | 7.4458 | 0.2680 | 7.1777 | 0.5000 | 0.9167 | 0.4167 |

## History

| step | example | loss |
|---:|---|---:|
| 1 | `json_math_037:tok6` | 0.1491 |
| 20 | `json_fact_054:tok1` | -0.0000 |
| 40 | `json_code_003:tok5` | 0.0026 |
| 60 | `json_sort_059:tok5` | 0.0000 |
| 80 | `json_math_041:tok2` | 13.0336 |

## Eval Source IDs

json_code_055, json_compiler_020, json_compiler_030, json_compiler_040, json_compiler_060, json_fact_010, json_math_015, json_math_025, json_math_035, json_math_045, json_math_050, json_pattern_005
