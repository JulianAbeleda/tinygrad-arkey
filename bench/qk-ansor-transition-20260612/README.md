# QK Ansor Transition

This directory is the first explicit bridge from the current generated-policy
Q4_K/Q6_K primitive work toward an Ansor-style search loop.

It does not add runtime behavior, kernels, or BEAM. It defines the current
objective, bottleneck attribution, semantic descriptor layer, and the first
static candidate/search-loop surface from committed artifacts.

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

  PYTHONPATH=. .venv/bin/python extra/qk_descriptor_policy.py \
    --descriptor bench/qk-ansor-transition-20260612/descriptors/$model.json \
    --accepted bench/qk-shared-storage-20260612/$model/policy.json \
    --policy-json bench/qk-ansor-transition-20260612/reproduced/$model-policy.json \
    --diff-json bench/qk-ansor-transition-20260612/reproduced/$model-diff.json \
    --diff-md bench/qk-ansor-transition-20260612/reproduced/$model-diff.md

  PYTHONPATH=. .venv/bin/python extra/qk_candidate_generator.py \
    --descriptor bench/qk-ansor-transition-20260612/descriptors/$model.json \
    --json bench/qk-ansor-transition-20260612/candidates/$model-candidates.json \
    --md bench/qk-ansor-transition-20260612/candidates/$model-candidates.md

  PYTHONPATH=. .venv/bin/python extra/qk_candidate_static_gate.py \
    --candidates bench/qk-ansor-transition-20260612/candidates/$model-candidates.json \
    --json bench/qk-ansor-transition-20260612/static-gates/$model-static-gate.json \
    --md bench/qk-ansor-transition-20260612/static-gates/$model-static-gate.md \
    --fail-on-reject

  mkdir -p bench/qk-ansor-transition-20260612/search/$model/policies
  PYTHONPATH=. .venv/bin/python extra/qk_ansor_transition_loop.py \
    --candidates bench/qk-ansor-transition-20260612/candidates/$model-candidates.json \
    --static-gate bench/qk-ansor-transition-20260612/static-gates/$model-static-gate.json \
    --scorecard bench/qk-ansor-transition-20260612/scorecard.json \
    --gap-profile bench/qk-ansor-transition-20260612/gap-profile.json \
    --json bench/qk-ansor-transition-20260612/search/$model/run.json \
    --md bench/qk-ansor-transition-20260612/search/$model/run.md \
    --policies-dir bench/qk-ansor-transition-20260612/search/$model/policies \
    --max-to-benchmark 6
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

Committed shared-storage profiles now exist for 8B, 14B, and 32B. Named
attribution says QK GEMV still dominates:

- 8B named AMD: Q4+Q6 primitive GEMV is `14.91 ms/tok`
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

## Candidate Loop v0

The descriptor layer now round-trips back into a runtime policy with no semantic
diff against the accepted generated policies. From that descriptor, the v0
candidate generator creates bounded policy variants:

- 8B: `21` candidates
- 14B: `29` candidates
- 32B: `33` candidates

The static gate currently passes every generated candidate because v0 only
varies supported Q4_K/Q6_K primitive `parts` and `LOCAL` settings. The search
loop is intentionally static planning: it writes `current` plus six ranked
`benchmark_next` policy files per model. Promotion still requires running those
policies through the QK harness with correctness and stability gates.
