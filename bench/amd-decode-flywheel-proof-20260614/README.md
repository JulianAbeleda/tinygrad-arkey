# AMD Decode Flywheel Proof Artifacts

Date: 2026-06-14

This directory records the first historical test of the model-to-kernel side of
the AMD decode optimization flywheel.

The question is narrow: can a model triage or rank kernel experiments better
than simple baselines before outcomes are known?

## Artifacts

- `kernel-triage-v0/`: Phase 1 dataset with `83` historical kernel candidate
  rows, `45` train rows, and `38` family-split holdout rows.
- `triage-baselines-v0/`: Phase 2 deterministic baseline and model scoring.
- `triage-qwen3-8b-base-v0/`: no-adapter Qwen3-8B generated-policy rollout on
  the holdout prompts.
- `triage-protocol-diagnostic-v0/`: Phase 3.0 diagnostic that extracts the
  JSON-shaped part of the base rollout without changing the official strict
  score.
- `triage-sft-v0/`: Phase 3.1 strict JSON SFT export with `45` train rows and
  `38` eval/holdout rows.
- `triage-adapter-v0-attempt/`: Phase 3.2 first-candidate training attempt;
  blocked by current training-loop latency before a rollout artifact existed.

## Current Result

Best deterministic baseline:

- `mechanism_prior` / `simple_family_heuristic`
- accuracy `0.289`
- macro-F1 `0.185`
- false-positive accept rate `0.000`
- precision@3 `0.083`
- NDCG `0.218`

No-adapter model result:

- `qwen3_8b_base`
- accuracy `0.000`
- macro-F1 `0.000`
- `38/38` predictions scored as `invalid_output`

The Qwen prompts include `/no_think`, but this model still emits empty
`<think>` tags before JSON-shaped text and often uses reason values outside the
allowed taxonomy. Under the strict compact-JSON contract, the result is a real
schema failure and does not beat any baseline.

## Conclusion

Phase 2 conclusion is `no_signal` for the current strict no-adapter 8B model.
The full flywheel is not proven. The next flywheel-relevant step must show that
a schema-capable model or adapter beats `mechanism_prior` on this holdout before
it is allowed to influence kernel experiment ordering.

Phase 3.0/3.1 update: extracting the JSON object from the base rollout fixes
parse/schema but not triage. Extracted macro-F1 is only `0.036`, with
false-positive accept rate `0.763`, below the `mechanism_prior` macro-F1
baseline of `0.185`. The current base model is wrong on triage, not merely
badly formatted. The strict SFT export is ready, but the first suffix-cache
adapter candidate did not produce a practical training artifact in this run.
