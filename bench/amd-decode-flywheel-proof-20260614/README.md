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
- `triage-adapter-smoke-v0/`: Phase 3.2A tiny suffix-cache adapter smoke with
  progress logging and split row caps.
- `triage-adapter-smoke-v0-rollout/`: held-out rollout for the tiny adapter
  smoke.
- `triage-adapter-smoke-v0-eval/`: strict Phase 2-style score for the tiny
  adapter smoke.
- `triage-adapter-smoke-v0-protocol-diagnostic/`: extraction diagnostic for the
  tiny adapter smoke rollout.
- `triage-cost-model-v0/`: Phase 3B learned cost-model triage with leak-free
  pre-result features, optional XGBoost, and a centroid fallback.
- `triage-feature-audit-v0/`: Phase 3C feature/data coverage audit that turns
  the Phase 3B negative into concrete data and feature targets.
- `kernel-triage-v1/`: Phase 3D dataset preserving the v0 split while adding
  normalized mechanisms and the frozen `candidate_outcome_v1` schema.
- `triage-feature-audit-v1/`: Phase 3D audit over the v1 schema.
- `kernel-triage-v1-featured/`: Phase 3E dataset preserving v1 rows while
  adding real source/compile features where committed artifacts expose them.
- `triage-feature-audit-v1-featured/`: Phase 3E audit over the featured
  schema.
- `triage-coverage-plan-v1/`: Phase 3E targeted outcome plan required before
  rerunning the cost model as a decision point.

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

Phase 3.2A update: instrumentation now shows the latency source. In the tiny
smoke, caching `4` train prefixes took `32.8s`, and caching `2` eval prefixes
took `21.0s`. The adapter changed weights and reduced teacher-forced loss on
the tiny slice, but held-out generation did not move: strict score stayed
`0/38`, extracted macro-F1 stayed `0.036`, and false-positive accept rate stayed
`0.763`. This confirms the negative rather than rescuing it.

Phase 3B update: the learned-cost-model version of triage is also recorded.
`extra/qk_flywheel_cost_model.py` extracts only pre-result candidate/context
features and audits out target/result leakage. Local XGBoost `3.2.0` ran with a
native `rank:ndcg` ranker, but the result still loses to `mechanism_prior`:
macro-F1 `0.137` versus `0.185`, precision@3 `0.000` versus `0.083`, and NDCG
`0.189` versus `0.218`, with false-positive accept rate `0.000`. XGBoost is
the right tool class for cost-model triage, but the current `45` train rows and
feature policy do not yet prove the hard half of the flywheel.

Phase 3C update: `extra/qk_flywheel_feature_audit.py` scopes why Phase 3B
failed and what data to collect next. Current audit:
`needs_data_and_feature_expansion`, `24` unseen holdout categorical values,
`56` weak rows, `9` post-full-decode train rows, and no target/result leakage.
The highest-priority gaps are label coverage for `construction_blocked`,
`raw_accept_unconfirmed`, and `diagnostic_only`; normalization of `18`
`unknown` mechanism holdout rows; mechanism coverage for
`packed_word_lane_unroll`, `qk_block_dot`, `vector_load`, and
`wide_load_only`; and richer tinygrad/UOp/profile features for rows without
structural kernel detail.

Phase 3D update: `extra/qk_flywheel_dataset_v1.py` adds the frozen
`candidate_outcome_v1` schema and normalizes semantic mechanisms while
preserving the same `45` train / `38` holdout family split. Unknown mechanisms
drop to `0`, with `26` rows changed from v0 names. The v1 audit improves
coverage but still concludes `needs_data_and_feature_expansion`: unseen
holdout categorical values fall from `24` to `15`, weak rows fall from `56` to
`43`, and no target/result leakage is detected. The remaining blocker is real
data and features: `33` holdout rows still have mechanisms unseen in train,
label coverage is thin, and current UOp features are proxy estimates rather
than first-class tinygrad/UOp/profile extraction.

Phase 3E update: `extra/qk_flywheel_feature_enrich.py` adds real source/compile
features where committed artifacts expose load-width or compile-gate evidence.
The featured dataset still has `83` rows, `45` train, and `38` holdout rows; it
does not synthesize outcomes or move holdout rows into train. Real UOp/source
features are now available on `13` rows (`7` train, `6` holdout): `7`
`tile_custom`, `2` `packed_word_lane_unroll`, `2` `qk_block_dot`, and `2`
`vector_load`. The featured audit remains clean on target/result leakage, but
the decision is still blocked: unseen holdout categorical values remain `15`,
weak rows remain `43`, and `33` holdout rows still have mechanisms unseen in
train. `triage-coverage-plan-v1/` therefore keeps `rerun_phase3b_allowed=false`
and calls for a real targeted outcome batch before another XGBoost decision
run.
