# Qwen Eval Harness

This is a smallest-real rollout/eval check for the accepted QK generated-policy path.
It compares explicit Q4/Q6 primitive flags against the generated shared-storage policy
on fixed prompts with greedy decoding. The correctness gate is exact generated-token
parity between the two modes.

## Summary

- status: `pass`
- model: `/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/14b/policy.json`
- storage: `shared`
- prompts: `10`
- token parity: `True`
- quality status: `pass`
- quality score: `10/10`
- timing note: this harness is a rollout correctness gate; first-prompt
  timings include session/JIT effects and are not the canonical decode benchmark.

| mode | generated tokens | elapsed s | tok/s |
|---|---:|---:|---:|
| `explicit` | 103 | 39.910 | 2.58 |
| `generated` | 103 | 37.756 | 2.73 |

## Prompt Outputs

| id | tags | token match | quality | explicit tok/s | generated tok/s | generated text |
|---|---|---:|---:|---:|---:|---|
| `alphabetical_first` | `instruction,short` | `True` | `pass` | 5.42 | 5.88 | <think>\n\n</think>\n\napple |
| `capital_france` | `facts,short` | `True` | `pass` | 9.61 | 11.62 | <think>\n\n</think>\n\nParis. |
| `compiler_short` | `compiler,explain` | `True` | `pass` | 12.53 | 16.22 | <think>\n\n</think>\n\nA compiler optimization pass improves the efficiency and performance of the generated code. |
| `exact_letters` | `instruction,format` | `True` | `pass` | 8.46 | 9.78 | <think>\n\n</think>\n\nAMD |
| `json_ok` | `format,json` | `True` | `pass` | 10.86 | 13.49 | <think>\n\n</think>\n\n{"result": "ok"} |
| `math_product` | `math,short` | `True` | `pass` | 0.67 | 0.68 | <think>\n\n</think>\n\n63 |
| `math_sum` | `math,short` | `True` | `pass` | 0.30 | 0.31 | <think>\n\n</think>\n\n42. |
| `python_len` | `code,short` | `True` | `pass` | 14.82 | 19.25 | <think>\n\n</think>\n\nThe built-in function `len()` returns the length of a list in Python. |
| `sequence_next` | `math,pattern` | `True` | `pass` | 6.77 | 7.62 | <think>\n\n</think>\n\n32 |
| `tinygrad_short` | `tinygrad,explain` | `True` | `pass` | 17.33 | 22.88 | <think>\n\n</think>\n\nTinygrad is a small, flexible machine learning framework designed for research and experimentation. |

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

## Training Readiness Verdict

The faster inference path is ready to be used as a deterministic rollout/eval backend
when the model and policy artifact are pinned. This does not establish a tinygrad
LLM training path; it only validates the decode side needed by future SFT/RLVR
experiments.
