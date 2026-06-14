# LLM Rollout Compare Report

Deterministic offline comparison for committed rollout artifacts. This checks
quality scores, prompt coverage, output equality, and eval-loop timing sanity.
It is not an LLM-as-judge or broad capability benchmark.

## Artifacts

| label | mode | prompts | quality | generated | tok/s | dataset |
|---|---|---:|---:|---:|---:|---|
| `8b-output-lora-r16-v3-rollout` | `generated` | 12 | `fail` 3/12 | 142 | 3.43 | `bench/qwen-adapter-20260613/training-data-v3/eval-prompts.jsonl` |
| `8b-last1-ffn-suffix-lora-r4-v5-rollout` | `generated` | 12 | `fail` 4/12 | 103 | 3.12 | `bench/qwen-adapter-20260613/training-data-v3/eval-prompts.jsonl` |

## Comparisons

### `8b-output-lora-r16-v3-rollout` vs `8b-last1-ffn-suffix-lora-r4-v5-rollout`

- quality delta: `1` (3 -> 4)
- regressions: `1`
- improvements: `2`
- text changed: `10/12`
- tokens changed: `10/12`
- generated token delta: `-39`
- eval-loop tok/s ratio: `0.9089`

| tag | baseline passed | candidate passed | delta | scored |
|---|---:|---:|---:|---:|
| `code` | 1 | 1 | 0 | 1 |
| `compiler` | 0 | 1 | 1 | 4 |
| `eval` | 3 | 4 | 1 | 12 |
| `fact` | 1 | 1 | 0 | 1 |
| `json_answer` | 3 | 4 | 1 | 12 |
| `math` | 1 | 0 | -1 | 5 |
| `pattern` | 0 | 1 | 1 | 1 |

| id | quality | tokens equal | text |
|---|---|---:|---|
| `json_compiler_020` | `fail -> fail` | `False` | {"answer":"quantization reduces the precision of model weights"} |
| `json_compiler_030` | `fail -> fail` | `False` | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_compiler_040` | `fail -> fail` | `False` | {"answer":""} |
| `json_compiler_060` | `fail -> pass` | `False` | {"answer":"central processing unit"} |
| `json_math_015` | `fail -> fail` | `False` | {"answer": 25} |
| `json_math_025` | `fail -> fail` | `False` | {"answer": 8}\n</think>\n\n{"answer": 8} |
| `json_math_035` | `fail -> fail` | `False` | {"answer":14} |
| `json_math_045` | `fail -> fail` | `False` | {"answer": 6} |
| `json_math_050` | `pass -> fail` | `False` | {"answer": 15} |
| `json_pattern_005` | `fail -> pass` | `False` | {"answer":"48"} |
