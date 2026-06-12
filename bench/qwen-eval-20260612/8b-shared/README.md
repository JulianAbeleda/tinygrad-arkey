# Qwen Eval Harness

This is a smallest-real rollout/eval check for the accepted QK generated-policy path.
It compares explicit Q4/Q6 primitive flags against the generated shared-storage policy
on fixed prompts with greedy decoding. The correctness gate is exact generated-token
parity between the two modes.

## Summary

- status: `pass`
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`
- policy: `bench/qk-shared-storage-20260612/8b/policy.json`
- storage: `shared`
- prompts: `10`
- token parity: `True`
- quality status: `pass`
- quality score: `10/10`
- timing note: this harness is a rollout correctness gate; first-prompt
  timings include session/JIT effects and are not the canonical decode benchmark.

| mode | generated tokens | elapsed s | tok/s |
|---|---:|---:|---:|
| `explicit` | 109 | 30.692 | 3.55 |
| `generated` | 109 | 30.537 | 3.57 |

## Prompt Outputs

| id | tags | token match | quality | explicit tok/s | generated tok/s | generated text |
|---|---|---:|---:|---:|---:|---|
| `alphabetical_first` | `instruction,short` | `True` | `pass` | 9.23 | 9.59 | <think>\n\n</think>\n\napple |
| `capital_france` | `facts,short` | `True` | `pass` | 17.52 | 18.16 | <think>\n\n</think>\n\nParis. |
| `compiler_short` | `compiler,explain` | `True` | `pass` | 27.60 | 30.68 | <think>\n\n</think>\n\nA compiler optimization pass improves the efficiency of the generated code by applying transformations that preserve correctness while enhancing performance or reducing resource usage. |
| `exact_letters` | `instruction,format` | `True` | `pass` | 15.09 | 15.38 | <think>\n\n</think>\n\nAMD |
| `json_ok` | `format,json` | `True` | `pass` | 23.70 | 24.09 | <think>\n\n</think>\n\n{\n  "result": "ok"\n} |
| `math_product` | `math,short` | `True` | `pass` | 0.87 | 0.87 | <think>\n\n</think>\n\n63 |
| `math_sum` | `math,short` | `True` | `pass` | 0.31 | 0.31 | <think>\n\n</think>\n\n42 |
| `python_len` | `code,short` | `True` | `pass` | 13.80 | 13.69 | <think>\n\n</think>\n\nlen() |
| `sequence_next` | `math,pattern` | `True` | `pass` | 12.23 | 12.20 | <think>\n\n</think>\n\n32 |
| `tinygrad_short` | `tinygrad,explain` | `True` | `pass` | 37.85 | 38.88 | <think>\n\n</think>\n\nTinygrad is a minimalistic deep learning framework written in Python that allows for low-level control over neural network computations. |

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
