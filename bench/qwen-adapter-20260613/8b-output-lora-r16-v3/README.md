# LLM Adapter Train

This is the first real Qwen adapter-training gate. It trains output-head
LoRA tensors only, with the base GGUF model frozen. It is not QLoRA and
does not claim model-quality improvement.

## Summary

- status: `pass`
- adapter: `output_lora` rank `16` alpha `16.0`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- rows: `60` (`48` train, `12` eval)
- examples: `319` train, `80` eval
- adapter L2 delta: `7.954963`

## Metrics

| split | initial loss | final loss | loss delta | initial acc | final acc | acc delta |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 6.5342 | 0.4130 | 6.1212 | 0.5417 | 0.9167 | 0.3750 |
| `eval` | 7.8166 | 0.7877 | 7.0289 | 0.5000 | 0.8542 | 0.3542 |

## History

| step | example | loss |
|---:|---|---:|
| 1 | `json_math_037:tok6` | 0.1446 |
| 60 | `json_sort_059:tok5` | -0.0000 |
| 120 | `json_math_041:tok4` | 0.0001 |
| 180 | `json_reason_049:tok1` | 0.0000 |
| 240 | `json_compiler_017:tok2` | -0.0000 |

## Eval Source IDs

json_code_055, json_compiler_020, json_compiler_030, json_compiler_040, json_compiler_060, json_fact_010, json_math_015, json_math_025, json_math_035, json_math_045, json_math_050, json_pattern_005
