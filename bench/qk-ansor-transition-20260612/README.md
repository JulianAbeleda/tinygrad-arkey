# QK Ansor Transition

This directory is the first explicit bridge from the current generated-policy
Q4_K/Q6_K primitive work toward an Ansor-style search loop.

It does not add runtime behavior, kernels, or BEAM. It defines the current
objective, bottleneck attribution, and semantic descriptor layer from committed
artifacts.

## Regenerate

```sh
PYTHONPATH=. .venv/bin/python extra/qk_llama_scorecard.py \
  bench/qk-shared-storage-20260612/8b \
  bench/qk-shared-storage-20260612/14b \
  bench/qk-shared-storage-20260612/32b \
  --rollout-compare bench/qwen-rollout-20260612/compare-8b-small/report.json \
  --json bench/qk-ansor-transition-20260612/scorecard.json \
  --md bench/qk-ansor-transition-20260612/scorecard.md

PYTHONPATH=. .venv/bin/python extra/qk_gap_profile.py \
  bench/qk-shared-storage-20260612/8b \
  bench/qk-shared-storage-20260612/14b \
  bench/qk-shared-storage-20260612/32b \
  --json bench/qk-ansor-transition-20260612/gap-profile.json \
  --md bench/qk-ansor-transition-20260612/gap-profile.md

for model in 8b 14b 32b; do
  upper=$(printf '%s' "$model" | tr '[:lower:]' '[:upper:]')
  PYTHONPATH=. .venv/bin/python extra/qk_semantic_descriptor.py \
    --policy bench/qk-shared-storage-20260612/$model/policy.json \
    --model-label "$upper" \
    --json bench/qk-ansor-transition-20260612/descriptors/$model.json \
    --md bench/qk-ansor-transition-20260612/descriptors/$model.md
done
```

## Current Scorecard

- 8B: `52.07 tok/s`, `51.46%` of llama.cpp
- 14B: `40.55 tok/s`, `61.63%` of llama.cpp
- 32B: `17.23 tok/s`, `55.94%` of llama.cpp
- all three pass greedy A/B in the QK harness matrix
- 8B rollout comparator: generated vs explicit is token-identical on `75/75`
  prompts

The first comparable-speed target is `>=70%` llama.cpp on all three rows.
Current minimum is `51.46%`, so the next performance work still needs a real
QK schedule/codegen improvement.

## Current Gap Profile

Committed shared-storage profiles exist for 14B and 32B. The shared 8B profile
is missing and is explicitly marked as a fresh-profile requirement before using
8B for bottleneck attribution.

For the profiled rows, named attribution says QK GEMV still dominates:

- 14B named AMD: Q4+Q6 primitive GEMV is `30.08 ms/tok`
- 32B named AMD: Q4+Q6 primitive GEMV is `82.44 ms/tok`

This points the next research step at QK semantic schedule/codegen, not more
rollout/eval infrastructure.

## Descriptor Layer

`descriptors/{8b,14b,32b}.json` convert the accepted generated policies into a
machine-readable semantic object:

- format: Q4_K/Q6_K
- tensor role
- shape
- packed-layout metadata
- current lowering family
- parts/local/reduction choices
- storage and benefit metadata

This is not pure Ansor yet. It is the representation layer that makes a later
candidate generator/search loop possible.
