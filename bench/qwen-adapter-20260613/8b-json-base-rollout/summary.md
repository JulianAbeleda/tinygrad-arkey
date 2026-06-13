# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- dataset: `bench/qwen-adapter-20260613/training-data-v3/eval-prompts.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `12`
- generated tokens: `288`
- quality: `fail` (0/12)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `json_pattern_005` | `json_answer,pattern,eval` | `fail` | 24 | 0.93 | <think>\nOkay, let's see. The user is asking for the next number in the sequence: 3,  |
| `json_fact_010` | `json_answer,fact,eval` | `fail` | 24 | 34.39 | <think>\nOkay, the user is asking what color red and blue paint usually make. Let me think. When you mix |
| `json_math_015` | `json_answer,math,eval` | `fail` | 24 | 36.63 | <think>\nOkay, the user is asking for half of 50. Let me calculate that. Half of 5 |
| `json_compiler_020` | `json_answer,compiler,eval` | `fail` | 24 | 34.70 | <think>\nOkay, the user is asking what quantization does to model weights. Let me recall. Quantization is a |
| `json_math_025` | `json_answer,math,eval` | `fail` | 24 | 34.54 | <think>\nOkay, the user is asking for 10 percent of 80. Let me calculate that.  |
| `json_compiler_030` | `json_answer,compiler,eval` | `fail` | 24 | 34.63 | <think>\nOkay, the user is asking what dequantization converts quantized values toward. Let me start by recalling what |
| `json_math_035` | `json_answer,math,eval` | `fail` | 24 | 36.17 | <think>\nOkay, the user is asking for the sum of 2 and 12. Let me calculate that. |
| `json_compiler_040` | `json_answer,compiler,eval` | `fail` | 24 | 34.90 | <think>\nOkay, the user is asking about what a kernel is in GPU programming. Let me start by recalling my knowledge |
| `json_math_045` | `json_answer,math,eval` | `fail` | 24 | 31.93 | <think>\nOkay, the user is asking for the average of 4, 6, and 8. Let me |
| `json_math_050` | `json_answer,math,eval` | `fail` | 24 | 37.21 | <think>\nOkay, the user is asking for the sum of 15 and 0. Let me think. Adding |
| `json_code_055` | `json_answer,code,eval` | `fail` | 24 | 35.24 | <think>\nOkay, the user is asking which character starts a single-line comment in Python. Let me think. I remember |
| `json_compiler_060` | `json_answer,compiler,eval` | `fail` | 24 | 34.58 | <think>\nOkay, the user is asking what CPU stands for. Let me recall. CPU is a common acronym, so |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `code` | 0 | 1 | 0.00 |
| `compiler` | 0 | 4 | 0.00 |
| `eval` | 0 | 12 | 0.00 |
| `fact` | 0 | 1 | 0.00 |
| `json_answer` | 0 | 12 | 0.00 |
| `math` | 0 | 5 | 0.00 |
| `pattern` | 0 | 1 | 0.00 |
