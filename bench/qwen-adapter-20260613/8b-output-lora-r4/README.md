# LLM Adapter Train

This is the first real Qwen adapter-training gate. It trains output-head
LoRA tensors only, with the base GGUF model frozen. It is not QLoRA and
does not claim model-quality improvement.

## Summary

- status: `pass`
- adapter: `output_lora` rank `4` alpha `8.0`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- rows: `10` (`8` train, `2` eval)
- examples: `48` train, `12` eval
- adapter L2 delta: `0.003541`

## Metrics

| split | initial loss | final loss | loss delta | initial acc | final acc | acc delta |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| `eval` | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |

## History

| step | example | loss |
|---:|---|---:|
| 1 | `8b-generated-small:math_double_008:tok1` | 0.0000 |
| 2 | `8b-generated-small:math_div_004:tok4` | -0.0000 |
| 3 | `8b-generated-small:math_div_004:tok1` | 0.0000 |
| 4 | `8b-generated-small:math_min_010:tok3` | 0.0000 |

## Eval Source IDs

math_add_001, math_mul_003
