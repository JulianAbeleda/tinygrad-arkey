# Qwen Adapter Artifacts

This directory contains the first Qwen adapter-training gates for the Track 1
practical loop.

## V0 Self-Distilled Plumbing Gate

The first adapter is intentionally narrow:

- model: Qwen3-8B Q4_K_M
- base inference path: generated QK policy with `QK_PRIMITIVE_STORAGE=shared`
- adapter: output-head LoRA only, rank `4`, alpha `8`
- training data: `bench/qwen-rollout-20260612/training-data-v1/sft.jsonl`

Artifacts:

- `8b-output-lora-r4/README.md`: adapter training summary and saved adapter.
- `8b-output-lora-r4/adapter.json`: adapter config.
- `8b-output-lora-r4/adapter.npz`: adapter tensors.
- `8b-output-lora-r4-rollout/summary.md`: 75-prompt rollout with adapter loaded.
- `compare-8b-base-vs-output-lora/report.md`: deterministic base-vs-adapter
  comparison.

Current result:

- adapter weights changed: L2 delta `0.003541`
- train/eval loss was already saturated on self-generated rows and stayed
  saturated
- adapter rollout quality: `75/75`
- base vs adapter: `0` regressions, `0/75` text changes, `0/75` token changes
- eval-loop speed ratio: `0.5925` vs base generated rollout

This proves adapter install, save/load, rollout, comparison, and contract
plumbing. It does not prove Qwen quality improvement. The training data is
self-distilled from the base model, so it has almost no supervised loss signal
for this output-head adapter.

## V2 Sentinel Behavior-Change Gate

The second adapter uses a non-self-generated supervised target:

- dataset: `training-data-v2/sft.jsonl`
- held-out eval prompts: `training-data-v2/eval-prompts.jsonl`
- target behavior: answer ordinary questions with exactly the one-token
  sentinel `OK`
- adapter: output-head LoRA only, rank `8`, alpha `16`

Artifacts:

- `training-data-v2/README.md`: generated SFT/eval dataset and exact-match
  held-out prompt set.
- `8b-sentinel-base-rollout/summary.md`: base model probe on held-out prompts.
- `8b-output-lora-r8-v2/README.md`: adapter training summary and saved adapter.
- `8b-output-lora-r8-v2-rollout/summary.md`: held-out rollout with adapter
  loaded.
- `compare-8b-base-vs-output-lora-r8-v2/report.md`: deterministic base-vs-v2
  comparison.

Current result:

- base held-out rollout: `0/12` exact-match (`<think>` first token)
- adapter held-out rollout: `12/12` exact-match (`OK`)
- compare improvement: `+12` passes, `0` regressions
- teacher-forced held-out accuracy: `0.0 -> 1.0`
- eval-loop speed ratio: `0.9685` vs the base sentinel rollout

This proves the adapter loop can produce a contract-gated behavior change on
held-out prompts. It is still a deliberately synthetic sentinel override, not a
general capability or preference-training result.

## V3 Strict JSON Answer Gate

The third gate replaces the sentinel with a human-authored strict JSON-answer
dataset:

- dataset: `training-data-v3/sft.jsonl`
- held-out eval prompts: `training-data-v3/eval-prompts.jsonl`
- target behavior: return only compact JSON with exactly one key, `answer`
- adapter: output-head LoRA only, rank `16`, alpha `16`, trained with EOS
  targets

Artifacts:

- `training-data-v3/README.md`: SFT/eval dataset and strict JSON schema.
- `8b-json-base-rollout/summary.md`: base model probe on held-out prompts.
- `8b-output-lora-r16-v3/README.md`: adapter training summary and saved
  adapter.
- `8b-output-lora-r16-v3-rollout/summary.md`: held-out rollout with adapter
  loaded.
- `compare-8b-json-base-vs-output-lora-r16-v3/report.md`: deterministic
  base-vs-v3 comparison.

Current result:

- base held-out rollout: `0/12`
- adapter held-out rollout: `3/12`
- compare improvement: `+3` passes, `0` regressions
- teacher-forced held-out token accuracy: `0.5000 -> 0.8542`
- eval-loop speed ratio: `0.3989` vs the base JSON rollout

Verdict: failed promotion. Output-only LoRA learned to suppress `<think>` and
emit JSON-shaped text, but it did not reliably produce the correct held-out
answer values and sometimes collapsed numeric answers into repeated `1`s. Do
not lower the gate or keep LR-tuning this path as a win. The next adapter
experiment should target more conditional capacity than the output head alone,
for example a small allowlisted set of FFN/attention projection adapters, using
the same strict JSON rollout/compare gate.
