# LLM Training Data Probe

This is a dry-run exporter from rollout artifacts to SFT-style rows. It
validates dataset shape and filtering only; it is not a training loop.

## Summary

- input rows: `150`
- exported rows: `150`
- filtered rows: `0`
- average prompt tokens: `36.00`
- average completion tokens: `8.35`

## Tag Distribution

| tag | rows |
|---|---:|
| `code` | 20 |
| `code_short` | 20 |
| `comparison` | 4 |
| `compiler` | 18 |
| `compiler_tinygrad` | 20 |
| `facts` | 20 |
| `facts_short` | 20 |
| `format` | 20 |
| `format_following` | 20 |
| `instruction` | 20 |
| `instruction_following` | 20 |
| `json` | 2 |
| `math` | 22 |
| `math_short` | 30 |
| `pattern` | 4 |
| `reasoning` | 20 |
| `reasoning_short` | 20 |
| `tinygrad` | 2 |

## Sample Rows

| id | tags | prompt | completion |
|---|---|---|---|
| `8b-generated-small:math_add_001` | `math_short,math` | Answer with digits only. /no_think\nWhat is 14 + 28? | <think>\n\n</think>\n\n42 |
| `8b-generated-small:math_sub_002` | `math_short,math` | Answer with digits only. /no_think\nWhat is 100 - 37? | <think>\n\n</think>\n\n63 |
| `8b-generated-small:math_mul_003` | `math_short,math` | Answer with digits only. /no_think\nWhat is 12 * 8? | <think>\n\n</think>\n\n96 |
| `8b-generated-small:math_div_004` | `math_short,math` | Answer with digits only. /no_think\nWhat is 144 / 12? | <think>\n\n</think>\n\n12 |
| `8b-generated-small:math_square_005` | `math_short,math` | Answer with digits only. /no_think\nWhat is 11 squared? | <think>\n\n</think>\n\n121 |
| `8b-generated-small:math_half_006` | `math_short,math` | Answer with digits only. /no_think\nWhat is half of 50? | <think>\n\n</think>\n\n25 |
| `8b-generated-small:math_percent_007` | `math_short,math` | Answer with digits only. /no_think\nWhat is 10 percent of 80? | <think>\n\n</think>\n\n8 |
| `8b-generated-small:math_double_008` | `math_short,pattern` | Answer with digits only. /no_think\nGive the next number: 3, 6, 12, 24, | <think>\n\n</think>\n\n48 |
| `8b-generated-small:math_plus3_009` | `math_short,pattern` | Answer with digits only. /no_think\nGive the next number: 5, 8, 11, 14, | <think>\n\n</think>\n\n17 |
| `8b-generated-small:math_min_010` | `math_short,comparison` | Answer with digits only. /no_think\nWhich number is smaller: 19 or 27? | <think>\n\n</think>\n\n19 |
