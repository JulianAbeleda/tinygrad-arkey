# LLM SFT Smoke Train

This is a tinygrad optimization smoke test over rollout-derived SFT rows.
It validates training/eval plumbing only; it is not a Qwen fine-tune or
adapter.

## Summary

- status: `pass`
- rows: `150` (`120` train, `30` eval)
- byte examples: `14817` train, `3848` eval
- model: `byte_context_softmax`
- device: `AMD`
- optimizer: `Adam`, steps `80`, batch `512`, lr `0.04`

## Metrics

| split | initial loss | final loss | loss delta | initial acc | final acc | acc delta |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 4.8475 | 0.8681 | 3.9794 | 0.0063 | 0.7629 | 0.7566 |
| `eval` | 4.8483 | 1.5290 | 3.3193 | 0.0065 | 0.6320 | 0.6255 |

## History

| step | batch loss |
|---:|---:|
| 1 | 4.4424 |
| 20 | 1.5729 |
| 40 | 1.2462 |
| 60 | 1.1342 |
| 80 | 0.9525 |

## Eval Source IDs

code_append_002, code_index_010, compiler_bandwidth_010, compiler_kernel_009, fact_capital_france_001, fact_hamlet_006, format_bullets_008, format_letters_006, instr_even_001, instr_option_005, math_add_001, math_max_011, math_order_014, reason_alpha_001, reason_mix_010
