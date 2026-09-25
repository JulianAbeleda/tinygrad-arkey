# Tinygrad LoRA self-training MVP

This is a real optimizer/update proof over rollout-shaped SFT rows. It uses a deterministic frozen byte-context base
to qualify completion-only masking, tinygrad backward/Adam, LoRA-only mutation, adapter persistence, and held-out evaluation.
It is substrate evidence, not a claim that Qwen3-8B has been fine-tuned.

## Summary

- status: `pass`
- rows: `24` (`19` train, `5` eval)
- completion examples: `767` train, `199` eval
- device: `NV`
- steps/rank: `60` / `8`
- frozen base unchanged: `True`
- adapter reload max abs: `0.0`

## Metrics

| split | initial loss | final loss | delta | initial accuracy | final accuracy |
|---|---:|---:|---:|---:|---:|
| `train` | 4.8299 | 0.0756 | -4.7543 | 0.0248 | 0.9752 |
| `eval` | 4.8277 | 0.1001 | -4.7275 | 0.0251 | 0.9749 |

## Held-out source IDs

turn-00, turn-05, turn-10, turn-15, turn-20
