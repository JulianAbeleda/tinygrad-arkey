# LLM Rollout Compare Report

Deterministic offline comparison for committed rollout artifacts. This checks
quality scores, prompt coverage, output equality, and eval-loop timing sanity.
It is not an LLM-as-judge or broad capability benchmark.

## Artifacts

| label | mode | prompts | quality | generated | tok/s | dataset |
|---|---|---:|---:|---:|---:|---|
| `8b-generated-small` | `generated` | 75 | `pass` 75/75 | 608 | 10.39 | `bench/qwen-rollout-20260612/dataset-small.jsonl` |
| `8b-output-lora-r4-rollout` | `generated` | 75 | `pass` 75/75 | 608 | 6.16 | `bench/qwen-rollout-20260612/dataset-small.jsonl` |

## Comparisons

### `8b-generated-small` vs `8b-output-lora-r4-rollout`

- quality delta: `0` (75 -> 75)
- regressions: `0`
- improvements: `0`
- text changed: `0/75`
- tokens changed: `0/75`
- generated token delta: `0`
- eval-loop tok/s ratio: `0.5925`

| tag | baseline passed | candidate passed | delta | scored |
|---|---:|---:|---:|---:|
| `code` | 10 | 10 | 0 | 10 |
| `code_short` | 10 | 10 | 0 | 10 |
| `comparison` | 2 | 2 | 0 | 2 |
| `compiler` | 9 | 9 | 0 | 9 |
| `compiler_tinygrad` | 10 | 10 | 0 | 10 |
| `facts` | 10 | 10 | 0 | 10 |
| `facts_short` | 10 | 10 | 0 | 10 |
| `format` | 10 | 10 | 0 | 10 |
| `format_following` | 10 | 10 | 0 | 10 |
| `instruction` | 10 | 10 | 0 | 10 |
| `instruction_following` | 10 | 10 | 0 | 10 |
| `json` | 1 | 1 | 0 | 1 |
| `math` | 11 | 11 | 0 | 11 |
| `math_short` | 15 | 15 | 0 | 15 |
| `pattern` | 2 | 2 | 0 | 2 |
| `reasoning` | 10 | 10 | 0 | 10 |
| `reasoning_short` | 10 | 10 | 0 | 10 |
| `tinygrad` | 1 | 1 | 0 | 1 |
