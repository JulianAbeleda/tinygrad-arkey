# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter: `bench/qwen-adapter-20260613/8b-output-lora-r16-v3`
- dataset: `bench/qwen-adapter-20260613/training-data-v3/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `12`
- generated tokens: `142`
- quality: `fail` (3/12)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `json_pattern_005` | `json_answer,pattern,eval` | `fail` | 24 | 0.89 | {"answer":"111111111111111111111 |
| `json_fact_010` | `json_answer,fact,eval` | `pass` | 5 | 7.04 | {"answer":"purple"} |
| `json_math_015` | `json_answer,math,eval` | `fail` | 24 | 8.87 | {"answer":"111111111111111111111 |
| `json_compiler_020` | `json_answer,compiler,eval` | `fail` | 6 | 7.81 | {"answer":"rounding"} |
| `json_math_025` | `json_answer,math,eval` | `fail` | 24 | 11.56 | {"answer":"111111111111111111111 |
| `json_compiler_030` | `json_answer,compiler,eval` | `fail` | 5 | 7.20 | {"answer":"continuous"} |
| `json_math_035` | `json_answer,math,eval` | `fail` | 24 | 6.54 | {"answer":"111111111111111111111 |
| `json_compiler_040` | `json_answer,compiler,eval` | `fail` | 6 | 7.83 | {"answer":"a kernel"} |
| `json_math_045` | `json_answer,math,eval` | `fail` | 6 | 7.18 | {"answer":"11"} |
| `json_math_050` | `json_answer,math,eval` | `pass` | 6 | 8.05 | {"answer":"15"} |
| `json_code_055` | `json_answer,code,eval` | `pass` | 5 | 7.25 | {"answer":"#"} |
| `json_compiler_060` | `json_answer,compiler,eval` | `fail` | 7 | 8.90 | {"answer":"Central Processing Unit"} |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `code` | 1 | 1 | 1.00 |
| `compiler` | 0 | 4 | 0.00 |
| `eval` | 3 | 12 | 0.25 |
| `fact` | 1 | 1 | 1.00 |
| `json_answer` | 3 | 12 | 0.25 |
| `math` | 1 | 5 | 0.20 |
| `pattern` | 0 | 1 | 0.00 |
