# LLM Adapter Train

This is the first real Qwen adapter-training gate. It trains output-head
LoRA tensors only, with the base GGUF model frozen. It is not QLoRA and
does not claim model-quality improvement.

## Summary

- status: `pass`
- adapter: `output_lora` rank `8` alpha `16.0`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- rows: `60` (`48` train, `12` eval)
- examples: `48` train, `12` eval
- adapter L2 delta: `20.133744`

## Metrics

| split | initial loss | final loss | loss delta | initial acc | final acc | acc delta |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 39.3462 | 0.0000 | 39.3462 | 0.0000 | 1.0000 | 1.0000 |
| `eval` | 39.2907 | 0.0000 | 39.2907 | 0.0000 | 1.0000 | 1.0000 |

## History

| step | example | loss |
|---:|---|---:|
| 1 | `sentinel_039:tok0` | 39.7874 |
| 20 | `sentinel_054:tok0` | -0.0000 |
| 40 | `sentinel_003:tok0` | -0.0000 |
| 60 | `sentinel_059:tok0` | -0.0000 |
| 80 | `sentinel_042:tok0` | -0.0000 |

## Eval Source IDs

sentinel_005, sentinel_010, sentinel_015, sentinel_020, sentinel_025, sentinel_030, sentinel_035, sentinel_040, sentinel_045, sentinel_050, sentinel_055, sentinel_060
