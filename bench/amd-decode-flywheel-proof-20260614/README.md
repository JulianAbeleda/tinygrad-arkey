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
