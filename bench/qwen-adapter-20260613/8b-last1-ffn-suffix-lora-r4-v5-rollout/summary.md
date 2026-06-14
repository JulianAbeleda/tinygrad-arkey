# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter: `bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5`
- dataset: `bench/qwen-adapter-20260613/training-data-v3/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `12`
- generated tokens: `103`
- quality: `fail` (4/12)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `json_pattern_005` | `json_answer,pattern,eval` | `pass` | 6 | 0.21 | {"answer":"48"} |
| `json_fact_010` | `json_answer,fact,eval` | `pass` | 5 | 12.82 | {"answer":"purple"} |
| `json_math_015` | `json_answer,math,eval` | `fail` | 7 | 19.62 | {"answer": 25} |
| `json_compiler_020` | `json_answer,compiler,eval` | `fail` | 12 | 23.88 | {"answer":"quantization reduces the precision of model weights"} |
| `json_math_025` | `json_answer,math,eval` | `fail` | 14 | 25.26 | {"answer": 8}\n</think>\n\n{"answer": 8} |
| `json_compiler_030` | `json_answer,compiler,eval` | `fail` | 24 | 33.46 | {"answer":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":"":" |
| `json_math_035` | `json_answer,math,eval` | `fail` | 6 | 16.58 | {"answer":14} |
| `json_compiler_040` | `json_answer,compiler,eval` | `fail` | 4 | 11.31 | {"answer":""} |
| `json_math_045` | `json_answer,math,eval` | `fail` | 6 | 10.98 | {"answer": 6} |
| `json_math_050` | `json_answer,math,eval` | `fail` | 7 | 19.49 | {"answer": 15} |
| `json_code_055` | `json_answer,code,eval` | `pass` | 5 | 13.49 | {"answer":"#"} |
| `json_compiler_060` | `json_answer,compiler,eval` | `pass` | 7 | 21.04 | {"answer":"central processing unit"} |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `code` | 1 | 1 | 1.00 |
| `compiler` | 1 | 4 | 0.25 |
| `eval` | 4 | 12 | 0.33 |
| `fact` | 1 | 1 | 1.00 |
| `json_answer` | 4 | 12 | 0.33 |
| `math` | 0 | 5 | 0.00 |
| `pattern` | 1 | 1 | 1.00 |
