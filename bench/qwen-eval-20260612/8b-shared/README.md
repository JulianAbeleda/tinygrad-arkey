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
- prompts: `3`
- token parity: `True`
- timing note: this harness is a rollout correctness gate; first-prompt
  timings include session/JIT effects and are not the canonical decode benchmark.

| mode | generated tokens | elapsed s | tok/s |
|---|---:|---:|---:|
| `explicit` | 48 | 26.656 | 1.80 |
| `generated` | 48 | 27.651 | 1.74 |

## Prompt Outputs

| id | token match | explicit tok/s | generated tok/s | explicit text | generated text |
|---|---:|---:|---:|---|---|
| `compiler_short` | `True` | 2.30 | 2.19 | <think>\nOkay, the user is asking for a one-sentence explanation of what | <think>\nOkay, the user is asking for a one-sentence explanation of what |
| `math_sum` | `True` | 0.84 | 0.81 | <think>\nOkay, let's see. The user is asking for the final number | <think>\nOkay, let's see. The user is asking for the final number |
| `sequence_next` | `True` | 21.94 | 22.59 | <think>\nOkay, let's see. The sequence given is 2,  | <think>\nOkay, let's see. The sequence given is 2,  |

## Training Readiness Verdict

The faster inference path is ready to be used as a deterministic rollout/eval backend
when the model and policy artifact are pinned. This does not establish a tinygrad
LLM training path; it only validates the decode side needed by future SFT/RLVR
experiments.
