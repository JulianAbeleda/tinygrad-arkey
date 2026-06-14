# LLM Rollout Compare Report

Deterministic offline comparison for committed rollout artifacts. This checks
quality scores, prompt coverage, output equality, and eval-loop timing sanity.
It is not an LLM-as-judge or broad capability benchmark.

## Artifacts

| label | mode | prompts | quality | generated | tok/s | dataset |
|---|---|---:|---:|---:|---:|---|
| `8b-v4-json-base-rollout` | `generated` | 204 | `fail` 0/204 | 4896 | 26.00 | `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl` |
| `8b-output-lora-r16-v3-v4-rollout` | `generated` | 204 | `fail` 69/204 | 2298 | 7.94 | `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl` |

## Comparisons

### `8b-v4-json-base-rollout` vs `8b-output-lora-r16-v3-v4-rollout`

- quality delta: `69` (0 -> 69)
- regressions: `0`
- improvements: `69`
- text changed: `204/204`
- tokens changed: `204/204`
- generated token delta: `-2598`
- eval-loop tok/s ratio: `0.3055`

| tag | baseline passed | candidate passed | delta | scored |
|---|---:|---:|---:|---:|
| `arithmetic` | 0 | 0 | 0 | 34 |
| `categorization` | 0 | 24 | 24 | 34 |
| `code` | 0 | 0 | 0 | 34 |
| `compiler` | 0 | 0 | 0 | 34 |
| `eval` | 0 | 69 | 69 | 204 |
| `fact` | 0 | 33 | 33 | 34 |
| `json_answer` | 0 | 69 | 69 | 204 |
| `string` | 0 | 12 | 12 | 34 |

| JSON axis | baseline rate | candidate rate | delta | scored |
|---|---:|---:|---:|---:|
| `no_extra_text` | 0.00 | 0.80 | 164 | 204 |
| `parse_valid` | 0.00 | 0.80 | 164 | 204 |
| `schema_ok` | 0.00 | 0.80 | 164 | 204 |
| `strict_pass` | 0.00 | 0.34 | 69 | 204 |
| `type_ok` | 0.00 | 0.80 | 163 | 204 |
| `value_correct` | 0.00 | 0.34 | 69 | 204 |

| id | quality | tokens equal | text |
|---|---|---:|---|
| `json_v4_eval_arithmetic_001` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_002` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_003` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_004` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_005` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_006` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_007` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_008` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_009` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_010` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_011` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_012` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_013` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_014` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_015` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_016` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_017` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_018` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_019` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_020` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_021` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_022` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_023` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_024` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_v4_eval_arithmetic_025` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| ... | ... | ... | 179 more changed rows omitted from markdown |
