# LLM Rollout Summary

This is a dataset-style rollout artifact. It is useful for practical eval,
future SFT/RLVR data generation, and compiler/search behavior gates. Timing
is eval-loop sanity data, not the canonical decode benchmark.

## Summary

- mode: `generated`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- dataset: `bench/qwen-rollout-20260612/dataset-smoke.jsonl`
- storage: `shared`
- prompt format: `chat`
- prompts: `10`
- generated tokens: `109`
- quality: `pass` (10/10)

| id | tags | quality | generated | tok/s | text |
|---|---|---:|---:|---:|---|
| `math_sum` | `math,short` | `pass` | 6 | 0.31 | <think>\n\n</think>\n\n42 |
| `math_product` | `math,short` | `pass` | 6 | 0.87 | <think>\n\n</think>\n\n63 |
| `sequence_next` | `math,pattern` | `pass` | 6 | 12.34 | <think>\n\n</think>\n\n32 |
| `capital_france` | `facts,short` | `pass` | 6 | 17.78 | <think>\n\n</think>\n\nParis. |
| `python_len` | `code,short` | `pass` | 6 | 14.19 | <think>\n\n</think>\n\nlen() |
| `json_ok` | `format,json` | `pass` | 13 | 23.76 | <think>\n\n</think>\n\n{\n  "result": "ok"\n} |
| `compiler_short` | `compiler,explain` | `pass` | 29 | 30.47 | <think>\n\n</think>\n\nA compiler optimization pass improves the efficiency of the generated code by applying transformations that preserve correctness while enhancing performance or reducing resource usage. |
| `tinygrad_short` | `tinygrad,explain` | `pass` | 27 | 38.64 | <think>\n\n</think>\n\nTinygrad is a minimalistic deep learning framework written in Python that allows for low-level control over neural network computations. |
| `alphabetical_first` | `instruction,short` | `pass` | 5 | 9.50 | <think>\n\n</think>\n\napple |
| `exact_letters` | `instruction,format` | `pass` | 5 | 15.15 | <think>\n\n</think>\n\nAMD |

## Quality By Tag

| tag | passed | scored | pass rate |
|---|---:|---:|---:|
| `code` | 1 | 1 | 1.00 |
| `compiler` | 1 | 1 | 1.00 |
| `explain` | 2 | 2 | 1.00 |
| `facts` | 1 | 1 | 1.00 |
| `format` | 2 | 2 | 1.00 |
| `instruction` | 2 | 2 | 1.00 |
| `json` | 1 | 1 | 1.00 |
| `math` | 3 | 3 | 1.00 |
| `pattern` | 1 | 1 | 1.00 |
| `short` | 5 | 5 | 1.00 |
| `tinygrad` | 1 | 1 | 1.00 |
