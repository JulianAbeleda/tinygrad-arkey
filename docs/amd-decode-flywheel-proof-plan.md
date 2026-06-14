# AMD Decode Flywheel Proof Plan

Date: 2026-06-14

Status: plan of record for proving or falsifying the full kernel-optimization
flywheel.

Parent architecture note:
`docs/amd-decode-kernel-optimization-flywheel.md`.

## Goal

Prove whether the structured model/eval loop can make the AMD kernel
optimization loop better, not just cheaper.

The full flywheel claim is:

```text
faster kernels
-> cheaper model/eval experiments
-> more structured kernel artifacts
-> better model triage/proposals
-> fewer wasted kernel experiments
-> accepted kernel optimizations
```

The first half is already useful. This plan exists to prove or falsify the
second half.

## Success Standard

The full flywheel is not proven by better compiler vocabulary alone. It is
proven only if model-assisted triage or proposals improve the kernel workflow
against baselines and deterministic gates.

Minimum proof:

- The model beats reject-all, random ranking, and a simple hand heuristic on
  held-out historical kernel candidates.
- In live shadow mode, the model continues to rank or triage candidates better
  than those baselines before outcomes are known.
- In controlled assist mode, model rankings reduce wasted kernel experiments or
  surface a candidate that passes the normal static, correctness, microbench,
  and full-decode gates.

Kernel speed is still decided only by deterministic gates. The model never
promotes a kernel by itself.

## Phase 0: Scope And Commit The Hypothesis

Purpose:

- Freeze the claim as a hypothesis.
- Prevent "flywheel" language from justifying indefinite work without proof.

Inputs:

- `docs/amd-decode-kernel-optimization-flywheel.md`
- current Phase 4.2 compiler-data artifacts
- current QK/Ansor-transition verdict docs

Outputs:

- documented hypothesis, baselines, and stop rules
- links from kernel docs, eval docs, handoff, and checklist

Gate:

- The docs must say the model-to-kernel link is unproven.
- The next proof step must be a triage/ranking benchmark, not another training
  run by default.

Current status:

- Mostly complete in the working tree, not committed.

## Phase 1: Build Kernel History Dataset

Purpose:

- Turn existing kernel experiments into a structured dataset for triage and
  ranking.

Inputs:

- `bench/qk-ansor-transition-20260612/`
- `bench/qk-packed-*`
- `bench/qk-block-dot-*`
- `bench/qk-threeway-load-microbench-20260613/`
- relevant docs under `docs/amd-decode-*`

Example row shape:

```json
{
  "id": "semantic_codegen_v3_8b_ffn_gate",
  "family": "semantic_codegen_v3",
  "model": "Qwen3-8B-Q4_K_M",
  "role": "ffn_gate",
  "format": "Q4_K",
  "mechanism": "packed_word_lane_unroll",
  "pre_result_context": {
    "profile_bottleneck": "QK_GEMV",
    "hypothesis": "packed-load memory access",
    "static_gate": "pass"
  },
  "label": "reject",
  "reason": "microbench_tie",
  "evidence": {
    "gain_pct": -0.65,
    "source_evidence": "scalar_u32_loads"
  }
}
```

Important split:

- `pre_result_context` is what the model may see.
- `label`, `reason`, and final `evidence` are hidden during prediction.

Outputs:

- `extra/qk_flywheel_dataset.py`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/kernel-triage-v0/examples.jsonl`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/kernel-triage-v0/summary.json`
- dataset README with label counts and split policy

Gate:

- At least enough examples to make a benchmark meaningful. If the dataset is too
  small or too imbalanced, stop and record that the current repo history cannot
  prove the flywheel yet.
- Use time-split or family-split holdout. Do not use random split as the main
  claim because it leaks repeated-family structure.

## Phase 2: Historical Triage Benchmark

Purpose:

- Test whether existing models can predict or rank kernel outcomes better than
  baselines before any new training.

Tasks:

- Verdict prediction: accept, reject, tie, needs-rerun, or construction-blocked.
- Ranking: order candidate experiments by expected value.
- Dead-branch detection: identify mechanisms that should not be retried.

Baselines:

- majority-class / reject-all
- random ranking
- simple heuristic based on mechanism family and prior family verdict
- human-selected next step where the historical next step is available

Models:

- base Qwen3-8B generated path
- current best structured adapter if available
- optional larger local Qwen model as an analysis-only upper bound

Outputs:

- `extra/qk_flywheel_triage_eval.py`
- rollout artifacts under
  `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-baseline-v0/`
- aggregate metrics JSON/Markdown

Metrics:

- macro-F1 and accuracy for verdict prediction
- precision@k or NDCG for ranking
- false-positive rate on known dead branches
- calibration of high-confidence accepts

Gate:

- If no model beats simple baselines, the full flywheel is not proven. Continue
  treating this as two loops with a one-way benefit.
- If a model beats baselines but remains weak, continue to Phase 3 as a training
  experiment.

## Phase 3: Train A Kernel-Triage Adapter

Purpose:

- Test whether repo-specific structured kernel history can improve model
  triage beyond prompting alone.

Inputs:

- Phase 1 dataset
- Phase 2 baseline metrics
- Phase 4.2 stable compiler vocabulary data

Training target:

- Strict JSON triage outputs, not free-form essays.
- Example:

```json
{"label":"reject","reason":"microbench_regression","retry":false}
```

Outputs:

- combined SFT dataset for compiler vocabulary plus kernel triage
- suffix-cache adapter artifact
- held-out triage rollout and compare artifacts

Gate:

- Must beat Phase 2 baselines on family-split or time-split holdout.
- Must keep false-positive accepts low. A model that confidently recommends
  known dead branches is not useful even if average accuracy improves.

Stop rule:

- If training only memorizes family names or fails family-split holdout, do not
  call it a flywheel. The model loop remains a structured-output capability.

## Phase 4: Live Shadow Mode

Purpose:

- Test the model on new kernel decisions without letting it steer the work yet.

Method:

- Before running new candidate gates, ask the model to predict/rank outcomes.
- Freeze predictions in an artifact before seeing results.
- Run the normal human/deterministic kernel loop unchanged.
- Score the model after outcomes are known.

Outputs:

- `bench/amd-decode-flywheel-proof-YYYYMMDD/shadow-v0/`
- prediction JSONL
- outcome JSONL
- shadow score report

Gate:

- Model ranking must beat baselines on fresh candidates.
- The model must reduce dead-branch recommendations versus simple heuristics.

Stop rule:

- If shadow mode fails, do not allow model-ranked execution order. Keep using
  the model only for documentation or artifact extraction.

## Phase 5: Controlled Assist Mode

Purpose:

- Let the model influence low-risk ordering, not correctness or promotion.

Allowed influence:

- Rank candidates inside an already-approved candidate family.
- Suggest which artifact evidence to inspect first.
- Flag likely duplicate dead branches.

Not allowed:

- No direct runtime integration because the model suggests it.
- No bypassing static, correctness, microbench, or full-decode gates.
- No expanding to 32B or risky search solely because the model recommends it.

Gate:

- Measure GPU time or candidate count per decisive outcome.
- The assisted ordering must reduce wasted experiments or surface a real
  accepted candidate earlier than baseline ordering.

## Phase 6: Full Flywheel Proof

Purpose:

- Prove the model loop improved kernel optimization, not just documentation.

Acceptable proof:

- A model-ranked or model-proposed candidate passes the normal deterministic
  gates through full decode and improves speed, with an audit trail showing the
  model materially changed ranking or selection.

Alternative proof:

- Across multiple fresh candidate families, model-assisted ordering reduces
  wasted GPU experiments by a meaningful margin versus baseline ordering, even
  if no speedup is found in that window.

Required artifacts:

- frozen model predictions before outcomes
- baseline ordering comparison
- deterministic gate outputs
- final promote/reject verdict
- postmortem on whether the model added value

Gate:

- If a full-decode speedup lands, it is still a kernel win first. The flywheel
  claim requires showing model assistance mattered.
- If no model assistance advantage is measurable, record the result as two
  parallel loops with one-way benefit.

## Phase 7: Maintenance Loop

Purpose:

- Keep the flywheel honest if it is proven.

Rules:

- Every new candidate outcome appends a structured example.
- Every adapter refresh reruns the held-out triage benchmark.
- Every live model-assisted decision records prediction-before-outcome.
- Promotion gates remain deterministic and model-independent.

Regression rule:

- If the model stops beating baselines, remove it from execution ordering and
  return to shadow mode.

## Strategic Stop Rules

Stop calling this a flywheel if:

- the historical triage benchmark cannot beat simple baselines;
- family-split holdout collapses after training;
- live shadow predictions fail on fresh candidates;
- model-assisted ordering does not reduce wasted experiments;
- no accepted kernel optimization or measurable workflow improvement can be
  attributed to model assistance.

In that case the strategy is still useful, but narrower:

```text
kernel loop = performance work
model/eval loop = structured-output capability
connection = faster kernels make model experiments cheaper
```

That is a valid outcome. It just is not a full compounding flywheel.
