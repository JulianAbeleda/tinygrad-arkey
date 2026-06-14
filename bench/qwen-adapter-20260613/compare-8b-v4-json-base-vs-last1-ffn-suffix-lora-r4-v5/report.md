# LLM Rollout Compare Report

Deterministic offline comparison for committed rollout artifacts. This checks
quality scores, prompt coverage, output equality, and eval-loop timing sanity.
It is not an LLM-as-judge or broad capability benchmark.

## Artifacts

| label | mode | prompts | quality | generated | tok/s | dataset |
|---|---|---:|---:|---:|---:|---|
| `8b-v4-json-base-rollout` | `generated` | 204 | `fail` 0/204 | 4896 | 26.00 | `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl` |
| `8b-last1-ffn-suffix-lora-r4-v5-v4-rollout` | `generated` | 204 | `fail` 105/204 | 1952 | 12.68 | `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl` |

## Comparisons

### `8b-v4-json-base-rollout` vs `8b-last1-ffn-suffix-lora-r4-v5-v4-rollout`

- quality delta: `105` (0 -> 105)
- regressions: `0`
- improvements: `105`
- text changed: `204/204`
- tokens changed: `204/204`
- generated token delta: `-2944`
- eval-loop tok/s ratio: `0.4876`

| tag | baseline passed | candidate passed | delta | scored |
|---|---:|---:|---:|---:|
| `arithmetic` | 0 | 26 | 26 | 34 |
| `categorization` | 0 | 27 | 27 | 34 |
| `code` | 0 | 7 | 7 | 34 |
| `compiler` | 0 | 0 | 0 | 34 |
| `eval` | 0 | 105 | 105 | 204 |
| `fact` | 0 | 33 | 33 | 34 |
| `json_answer` | 0 | 105 | 105 | 204 |
| `string` | 0 | 12 | 12 | 34 |

| JSON axis | baseline rate | candidate rate | delta | scored |
|---|---:|---:|---:|---:|
| `no_extra_text` | 0.00 | 0.92 | 187 | 204 |
| `parse_valid` | 0.00 | 0.93 | 190 | 204 |
| `schema_ok` | 0.00 | 0.93 | 190 | 204 |
| `strict_pass` | 0.00 | 0.51 | 105 | 204 |
| `type_ok` | 0.00 | 0.93 | 190 | 204 |
| `value_correct` | 0.00 | 0.52 | 107 | 204 |

| id | quality | tokens equal | text |
|---|---|---:|---|
| `json_v4_eval_arithmetic_001` | `fail -> pass` | `False` | {"answer": 10521} |
| `json_v4_eval_arithmetic_002` | `fail -> pass` | `False` | {"answer": 5110} |
| `json_v4_eval_arithmetic_003` | `fail -> pass` | `False` | {"answer": 3509} |
| `json_v4_eval_arithmetic_004` | `fail -> pass` | `False` | {"answer": 10575} |
| `json_v4_eval_arithmetic_005` | `fail -> fail` | `False` | {"answer": 5200}\n</think>\n\n{"answer": 5200} |
| `json_v4_eval_arithmetic_006` | `fail -> pass` | `False` | {"answer": 3521} |
| `json_v4_eval_arithmetic_007` | `fail -> pass` | `False` | {"answer": 10629} |
| `json_v4_eval_arithmetic_008` | `fail -> fail` | `False` | {"answer": 5288} |
| `json_v4_eval_arithmetic_009` | `fail -> pass` | `False` | {"answer": 3533} |
| `json_v4_eval_arithmetic_010` | `fail -> pass` | `False` | {"answer": 10683} |
| `json_v4_eval_arithmetic_011` | `fail -> fail` | `False` | {"answer": 5155}\n</think>\n\n{"answer": 5155} |
| `json_v4_eval_arithmetic_012` | `fail -> pass` | `False` | {"answer": 3545} |
| `json_v4_eval_arithmetic_013` | `fail -> pass` | `False` | {"answer": 10737} |
| `json_v4_eval_arithmetic_014` | `fail -> pass` | `False` | {"answer": 5272} |
| `json_v4_eval_arithmetic_015` | `fail -> pass` | `False` | {"answer": 3557} |
| `json_v4_eval_arithmetic_016` | `fail -> pass` | `False` | {"answer": 10791} |
| `json_v4_eval_arithmetic_017` | `fail -> fail` | `False` | {"answer": 5377} |
| `json_v4_eval_arithmetic_018` | `fail -> pass` | `False` | {"answer": 3569} |
| `json_v4_eval_arithmetic_019` | `fail -> pass` | `False` | {"answer": 10845} |
| `json_v4_eval_arithmetic_020` | `fail -> fail` | `False` | {"answer": 5100}\n</think>\n\n{"answer": 5100} |
| `json_v4_eval_arithmetic_021` | `fail -> pass` | `False` | {"answer": 3581} |
| `json_v4_eval_arithmetic_022` | `fail -> pass` | `False` | {"answer": 10899} |
| `json_v4_eval_arithmetic_023` | `fail -> fail` | `False` | {"answer": 5264} |
| `json_v4_eval_arithmetic_024` | `fail -> pass` | `False` | {"answer": 3593} |
| `json_v4_eval_arithmetic_025` | `fail -> pass` | `False` | {"answer": 10953} |
| ... | ... | ... | 179 more changed rows omitted from markdown |
