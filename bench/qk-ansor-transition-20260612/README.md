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

base=bench/qk-ansor-transition-20260612/semantic-schedules
for model in 8b 14b; do
  PYTHONPATH=. .venv/bin/python extra/qk_semantic_schedule.py \
    --descriptor bench/qk-ansor-transition-20260612/descriptors/$model.json \
    --json $base/$model/candidates.json \
    --md $base/$model/candidates.md \
    --gate-json $base/$model/static-gate.json \
    --gate-md $base/$model/static-gate.md
done

PYTHONPATH=. .venv/bin/python extra/qk_semantic_schedule_verdict.py \
  --base bench/qk-ansor-transition-20260612/semantic-schedules \
  --json bench/qk-ansor-transition-20260612/semantic-schedules/verdict.json \
  --md bench/qk-ansor-transition-20260612/semantic-schedules/verdict.md

base=bench/qk-ansor-transition-20260612/semantic-codegen-v1
for model in 8b 14b; do
  PYTHONPATH=. .venv/bin/python extra/qk_semantic_codegen.py \
    --descriptor bench/qk-ansor-transition-20260612/descriptors/$model.json \
    --json $base/$model/candidates.json \
    --md $base/$model/candidates.md \
    --gate-json $base/$model/static-gate.json \
    --gate-md $base/$model/static-gate.md
done

PYTHONPATH=. .venv/bin/python extra/qk_semantic_codegen_verdict.py \
  --base bench/qk-ansor-transition-20260612/semantic-codegen-v1 \
  --json bench/qk-ansor-transition-20260612/semantic-codegen-v1/verdict.json \
  --md bench/qk-ansor-transition-20260612/semantic-codegen-v1/verdict.md
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

- 8B: `19` candidates
- 14B: `27` candidates
- 32B: `32` candidates

The static gate currently passes every generated candidate because v0 only
varies supported Q4_K/Q6_K primitive `parts` and `LOCAL` settings. The search
loop is intentionally static planning: it writes `current` plus six ranked
`benchmark_next` policy files per model. Promotion still requires running those
policies through the QK harness with correctness and stability gates.

## Loop Benchmark Verdict

`extra/qk_loop_benchmark.py` benchmarked the six `benchmark_next` policies per
model against each model's current accepted generated policy with
`QK_PRIMITIVE_STORAGE=shared`. This is policy-vs-policy, not explicit-flags vs
candidate.

Result: `benchmarks/verdict.md` marks the loop v0 frontier exhausted.

- 8B: `0` accepts, `2` ties, `3` rejects, `1` needs-rerun.
- 14B: `0` accepts, `2` ties, `4` rejects.
- 32B: one raw accept, `001-ffn_gate LOCAL:64 -> LOCAL:32`, at `+3.24%`, but a
  fresh confirmation rerun was a tie at `-2.29%`, so it is not promoted.

Conclusion: simple descriptor-level `parts`/`LOCAL` knob search is not enough to
move toward the `>=70%` llama.cpp target. The next research step needs a real
semantic schedule/codegen change, not another hand sweep over the same primitive
knobs.

## Semantic Schedule v0

`extra/qk_semantic_schedule.py` generated the first second-stage semantic
schedule/codegen surface from the accepted descriptors. This pass is gated only
on 8B and 14B by default; 32B is intentionally skipped unless both target models
show promise.

The generated surface adds schedule specs over the existing runtime families:

- `direct_out` for Q4_K `parts=1` microbench only;
- `row_upcast2` over the current primitive family;
- `reduce_unroll4` over the current primitive family;
- `two_dim_local4` over the current primitive family.

Static gate result:

- 8B: `15` total candidates, `14` microbenchable, `13` full-decode supported.
- 14B: `15` total candidates, `14` microbenchable, `13` full-decode supported.

Microbench result:

- 8B: `2` accepts, only `009-attn-q-blk-0-attn-q-weight-row-upcast2` is
  full-decode supported.
- 14B: `1` accept, `009-attn-q-blk-0-attn-q-weight-row-upcast2`.

Full-decode gate result:

| model | candidate | explicit tok/s | generated tok/s | gain | A/B | verdict |
|---|---|---:|---:|---:|---|---|
| 8B | `009-attn-q...row-upcast2` | `53.27` | `47.79` | `-10.28%` | pass | reject |
| 14B | `009-attn-q...row-upcast2` | `38.13` | `36.14` | `-5.21%` | pass | reject |

Verdict: `semantic_schedule_v0_rejected`. The isolated attention microbench
win did not survive full decode, so 32B was skipped. The next research step
needs a richer semantic layout/codegen capability, not another sweep over these
same schedule sketches.

Artifacts: `semantic-schedules/verdict.md`.

Measurement commands used on native AMD:

```sh
base=bench/qk-ansor-transition-20260612/semantic-schedules
for model in 8b 14b; do
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_semantic_schedule_bench.py \
    --model $model \
    --candidates $base/$model/candidates.json \
    --static-gate $base/$model/static-gate.json \
    --out $base/$model/microbench-runs \
    --json $base/$model/microbench.json \
    --md $base/$model/microbench.md \
    --iters 3 \
    --min-gain 0.03 \
    --tie-band 0.03
done

for model in 8b 14b; do
  model_file=Qwen3-${model^^}-Q4_K_M.gguf
  cand=009-attn-q-blk-0-attn-q-weight-row-upcast2
  DEV=AMD PYTHONPATH=. QK_PRIMITIVE_STORAGE=shared .venv/bin/python \
    extra/qk_policy_pipeline.py \
    --model ~/models/$model_file \
    --out $base/$model/full-benchmark/$cand \
    --device AMD \
    --level 2 \
    --iters 2 \
    --benchmark 128 \
    --reference-mode policy \
    --reference-policy bench/qk-ansor-transition-20260612/search/$model/policies/current.policy.json \
    --input-policy $base/$model/microbench-runs/$cand/policy.json \
    --repeats 3 \
    --max-extra-repeats 4 \
    --ab-tokens 32 \
    --profile never \
    --accept-gain 0.03 \
    --tie-band 0.03
done
```

## Semantic Codegen v1

`extra/qk_semantic_codegen.py` promoted the concrete Q4_K direct-output kernel
into a runtime-supported generated-policy family:
`q4_k_packed_u32_direct`. Unlike semantic schedule v0, these candidates are
exact-tensor overrides, so a full-decode candidate would only change the
specific tensor that won its microbench.

Static gate result:

- 8B: `4` total candidates, `3` microbenchable and full-decode supported.
- 14B: `5` total candidates, `4` microbenchable and full-decode supported.

Microbench result:

| model | accepts | ties | rejects | best tie | verdict |
|---|---:|---:|---:|---|---|
| 8B | `0` | `2` | `1` | `ffn_gate +2.57%` | no full-decode candidate |
| 14B | `0` | `2` | `2` | `ffn_gate +2.41%` | no full-decode candidate |

Verdict: `semantic_codegen_v1_rejected`. Direct-output Q4 removes the partial
reduction kernel, but the measured per-tensor gains stayed inside the fixed
`3%` tie band or regressed. No candidate reached the gate for full decode, so
32B was skipped by policy.

Artifacts: `semantic-codegen-v1/verdict.md`.

Measurement commands used on native AMD:

```sh
base=bench/qk-ansor-transition-20260612/semantic-codegen-v1
for model in 8b 14b; do
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_semantic_schedule_bench.py \
    --model $model \
    --candidates $base/$model/candidates.json \
    --static-gate $base/$model/static-gate.json \
    --out $base/$model/microbench-runs \
    --json $base/$model/microbench.json \
    --md $base/$model/microbench.md \
    --iters 3 \
    --min-gain 0.03 \
    --tie-band 0.03
done

PYTHONPATH=. .venv/bin/python extra/qk_semantic_codegen_verdict.py \
  --base $base \
  --json $base/verdict.json \
  --md $base/verdict.md
```
