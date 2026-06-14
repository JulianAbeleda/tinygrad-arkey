# LLM Rollout Compare Report

Deterministic offline comparison for committed rollout artifacts. This checks
quality scores, prompt coverage, output equality, and eval-loop timing sanity.
It is not an LLM-as-judge or broad capability benchmark.

## Artifacts

| label | mode | prompts | quality | generated | tok/s | dataset |
|---|---|---:|---:|---:|---:|---|
| `8b-last1-ffn-suffix-lora-r4-v5-v4-rollout` | `generated` | 204 | `fail` 105/204 | 1952 | 12.68 | `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl` |
| `8b-last1-ffn-suffix-lora-r4-v6-gold-v4-rollout` | `generated` | 204 | `fail` 162/204 | 2017 | 14.76 | `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl` |

## Comparisons

### `8b-last1-ffn-suffix-lora-r4-v5-v4-rollout` vs `8b-last1-ffn-suffix-lora-r4-v6-gold-v4-rollout`

- quality delta: `57` (105 -> 162)
- regressions: `2`
- improvements: `59`
- text changed: `121/204`
- tokens changed: `121/204`
- generated token delta: `65`
- eval-loop tok/s ratio: `1.1646`

| tag | baseline passed | candidate passed | delta | scored |
|---|---:|---:|---:|---:|
| `arithmetic` | 26 | 27 | 1 | 34 |
| `categorization` | 27 | 32 | 5 | 34 |
| `code` | 7 | 34 | 27 | 34 |
| `compiler` | 0 | 14 | 14 | 34 |
| `eval` | 105 | 162 | 57 | 204 |
| `fact` | 33 | 34 | 1 | 34 |
| `json_answer` | 105 | 162 | 57 | 204 |
| `string` | 12 | 21 | 9 | 34 |

| JSON axis | baseline rate | candidate rate | delta | scored |
|---|---:|---:|---:|---:|
| `no_extra_text` | 0.92 | 0.98 | 12 | 204 |
| `parse_valid` | 0.93 | 0.98 | 9 | 204 |
| `schema_ok` | 0.93 | 0.98 | 9 | 204 |
| `strict_pass` | 0.51 | 0.79 | 57 | 204 |
| `type_ok` | 0.93 | 0.98 | 9 | 204 |
| `value_correct` | 0.52 | 0.79 | 55 | 204 |

| id | quality | tokens equal | text |
|---|---|---:|---|
| `json_v4_eval_arithmetic_001` | `pass -> pass` | `False` | {"answer":10521} |
| `json_v4_eval_arithmetic_003` | `pass -> pass` | `False` | {"answer":3509} |
| `json_v4_eval_arithmetic_004` | `pass -> pass` | `False` | {"answer":10575} |
| `json_v4_eval_arithmetic_005` | `fail -> pass` | `False` | {"answer": 5200} |
| `json_v4_eval_arithmetic_006` | `pass -> pass` | `False` | {"answer":3521} |
| `json_v4_eval_arithmetic_007` | `pass -> pass` | `False` | {"answer":10629} |
| `json_v4_eval_arithmetic_008` | `fail -> fail` | `False` | {"answer": 52800000000000000000 |
| `json_v4_eval_arithmetic_009` | `pass -> pass` | `False` | {"answer":3533} |
| `json_v4_eval_arithmetic_010` | `pass -> pass` | `False` | {"answer":10683} |
| `json_v4_eval_arithmetic_011` | `fail -> fail` | `False` | {"answer":155500000000000000000 |
| `json_v4_eval_arithmetic_012` | `pass -> pass` | `False` | {"answer":3545} |
| `json_v4_eval_arithmetic_013` | `pass -> pass` | `False` | {"answer":10737} |
| `json_v4_eval_arithmetic_015` | `pass -> pass` | `False` | {"answer":3557} |
| `json_v4_eval_arithmetic_016` | `pass -> pass` | `False` | {"answer":10791} |
| `json_v4_eval_arithmetic_017` | `fail -> fail` | `False` | {"answer":37*11+5000} |
| `json_v4_eval_arithmetic_018` | `pass -> pass` | `False` | {"answer":3569} |
| `json_v4_eval_arithmetic_019` | `pass -> pass` | `False` | {"answer":10845} |
| `json_v4_eval_arithmetic_020` | `fail -> fail` | `False` | {"answer": 51000000000000000000 |
| `json_v4_eval_arithmetic_021` | `pass -> pass` | `False` | {"answer":3581} |
| `json_v4_eval_arithmetic_022` | `pass -> pass` | `False` | {"answer":10899} |
| `json_v4_eval_arithmetic_023` | `fail -> fail` | `False` | {"answer":346400000000000000000 |
| `json_v4_eval_arithmetic_024` | `pass -> pass` | `False` | {"answer":3593} |
| `json_v4_eval_arithmetic_025` | `pass -> pass` | `False` | {"answer":10953} |
| `json_v4_eval_arithmetic_027` | `pass -> pass` | `False` | {"answer":3605} |
| `json_v4_eval_arithmetic_028` | `pass -> pass` | `False` | {"answer":11007} |
| ... | ... | ... | 96 more changed rows omitted from markdown |
