# LLM Rollout Compare Report

Deterministic offline comparison for committed rollout artifacts. This checks
quality scores, prompt coverage, output equality, and eval-loop timing sanity.
It is not an LLM-as-judge or broad capability benchmark.

## Artifacts

| label | mode | prompts | quality | generated | tok/s | dataset |
|---|---|---:|---:|---:|---:|---|
| `8b-json-base-rollout` | `generated` | 12 | `fail` 0/12 | 288 | 8.61 | `bench/qwen-adapter-20260613/training-data-v3/eval-prompts.jsonl` |
| `8b-output-lora-r16-v3-rollout` | `generated` | 12 | `fail` 3/12 | 142 | 3.43 | `bench/qwen-adapter-20260613/training-data-v3/eval-prompts.jsonl` |

## Comparisons

### `8b-json-base-rollout` vs `8b-output-lora-r16-v3-rollout`

- quality delta: `3` (0 -> 3)
- regressions: `0`
- improvements: `3`
- text changed: `12/12`
- tokens changed: `12/12`
- generated token delta: `-146`
- eval-loop tok/s ratio: `0.3989`

| tag | baseline passed | candidate passed | delta | scored |
|---|---:|---:|---:|---:|
| `code` | 0 | 1 | 1 | 1 |
| `compiler` | 0 | 0 | 0 | 4 |
| `eval` | 0 | 3 | 3 | 12 |
| `fact` | 0 | 1 | 1 | 1 |
| `json_answer` | 0 | 3 | 3 | 12 |
| `math` | 0 | 1 | 1 | 5 |
| `pattern` | 0 | 0 | 0 | 1 |

| id | quality | tokens equal | text |
|---|---|---:|---|
| `json_code_055` | `fail -> pass` | `False` | {"answer":"#"} |
| `json_compiler_020` | `fail -> fail` | `False` | {"answer":"rounding"} |
| `json_compiler_030` | `fail -> fail` | `False` | {"answer":"continuous"} |
| `json_compiler_040` | `fail -> fail` | `False` | {"answer":"a kernel"} |
| `json_compiler_060` | `fail -> fail` | `False` | {"answer":"Central Processing Unit"} |
| `json_fact_010` | `fail -> pass` | `False` | {"answer":"purple"} |
| `json_math_015` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_math_025` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_math_035` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
| `json_math_045` | `fail -> fail` | `False` | {"answer":"11"} |
| `json_math_050` | `fail -> pass` | `False` | {"answer":"15"} |
| `json_pattern_005` | `fail -> fail` | `False` | {"answer":"111111111111111111111 |
