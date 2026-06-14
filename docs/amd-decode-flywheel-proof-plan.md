# AMD Decode Flywheel Proof Plan

Date: 2026-06-14

Status: plan of record for proving or falsifying the full kernel-optimization
flywheel. Phase 1 is built, and the first strict Phase 2 no-adapter baseline is
measured.

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

- Complete and linked from the handoff/checklist. This phase froze the
  hypothesis and made the model-to-kernel link explicitly unproven.

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

Primary source classes:

- Policy/descriptor candidates:
  `bench/qk-ansor-transition-20260612/candidates/`,
  `search/`, `static-gates/`, and `benchmarks/`.
- Semantic candidate families:
  `semantic-schedules/`, `semantic-codegen-v1/`,
  `semantic-codegen-v2/`, `semantic-codegen-v3/`, and
  `semantic-codegen-v4/`.
- Packed-load / semantic-op diagnostics:
  `bench/qk-packed-tile-*`, `bench/qk-packed-semantic-op-20260613/`,
  `bench/qk-block-dot-*`, and
  `bench/qk-threeway-load-microbench-20260613/`.
- Accepted runtime baselines:
  `bench/qk-shared-storage-20260612/`,
  `bench/qk-policy-pipeline-20260612/`, and
  `bench/qk-harness-20260612/`.

Dataset unit:

- Prefer one row per candidate outcome, not one row per top-level artifact.
- A "candidate outcome" can be a static-gate rejection, construction-blocked
  candidate, microbench result, full-decode result, or accepted baseline.
- Keep top-level family verdict rows too, but mark them as `row_kind=family`
  so they are not mixed with individual candidates in ranking metrics.

Example row shape:

```json
{
  "id": "semantic_codegen_v3_8b_ffn_gate",
  "row_kind": "candidate",
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

Required fields:

- `id`: stable repo-relative row id.
- `row_kind`: `candidate`, `family`, `baseline`, or `diagnostic`.
- `family`: candidate family or artifact family.
- `model`: model id when applicable.
- `tensor`: tensor name when applicable.
- `role`: `ffn_gate`, `ffn_up`, `ffn_down`, `attn_q`, `attn_k`,
  `attn_output`, or `unknown`.
- `format`: `Q4_K`, `Q6_K`, `q8_1`, or `unknown`.
- `mechanism`: compact stable mechanism key.
- `pre_result_context`: model-visible context.
- `label`: hidden target label.
- `reason`: hidden target reason.
- `evidence`: hidden target evidence.
- `source_files`: repo-relative artifact paths used to build the row.

Initial labels:

- `accept`: passed the relevant gate and was promoted for its scope.
- `reject`: failed a static, construction, microbench, or full-decode gate.
- `tie`: did not clear the promotion bar but was not materially worse.
- `raw_accept_unconfirmed`: microbench or isolated win without full-decode
  confirmation.
- `needs_rerun`: artifact says more measurement is needed.
- `construction_blocked`: candidate could not be built or compiled in the
  intended form.
- `diagnostic_only`: useful evidence but not a promotion candidate.

Initial reason taxonomy:

- `static_gate_fail`
- `construction_blocked`
- `correctness_fail`
- `microbench_regression`
- `microbench_tie`
- `full_decode_regression`
- `confirmation_failed`
- `insufficient_gain`
- `memory_pressure`
- `unsupported_runtime_scope`
- `diagnostic_only`
- `accepted_runtime_path`
- `needs_rerun`

Mechanism taxonomy v0:

- `parts_local_policy`
- `direct_output`
- `row_grouping`
- `packed_word_lane_unroll`
- `vector_load`
- `tile_custom`
- `qk_block_dot`
- `wide_load_only`
- `shared_storage`
- `storage_cap`
- `semantic_descriptor_replay`
- `unknown`

Context redaction rules:

- The model-visible prompt may include hypothesis, candidate mechanism, shape,
  source/gate status before final measurement, and prior family context.
- The model-visible prompt must not include final speed delta, final verdict,
  final reason, or post-result prose from the verdict doc.
- If a row is testing "predict after static gate", static-gate status may be
  visible. If testing "predict before static gate", static-gate status must be
  hidden. Record this as `prediction_stage`.

Splits:

- `time_split`: train on earlier families, hold out later families.
- `family_split`: hold out entire semantic families such as
  `semantic_codegen_v3`, `qk_block_dot`, or `threeway_load`.
- `random_split`: allowed only as a diagnostic sanity check, never as the main
  flywheel claim.

Summary requirements:

- label counts by split
- mechanism counts by split
- family counts by split
- number of rows with complete numeric evidence
- number of rows with source/disassembly evidence
- warnings for missing fields or ambiguous labels

Outputs:

- `extra/qk_flywheel_dataset.py`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/kernel-triage-v0/examples.jsonl`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/kernel-triage-v0/prompts.jsonl`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/kernel-triage-v0/summary.json`
- dataset README with label counts and split policy
- focused tests for schema validation and at least one representative source
  artifact per source class

Gate:

- At least enough examples to make a benchmark meaningful. If the dataset is too
  small or too imbalanced, stop and record that the current repo history cannot
  prove the flywheel yet.
- Use time-split or family-split holdout. Do not use random split as the main
  claim because it leaks repeated-family structure.
- Every emitted prompt row must have a hidden label row with the same id.
- No prompt row may contain final verdict keywords from its own target evidence.
- The extractor must fail loudly on malformed JSON artifacts instead of silently
  dropping rows.

Phase 1 completion checklist:

- [x] dataset builder checked in:
  `extra/qk_flywheel_dataset.py`
- [x] dataset artifact generated:
  `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v0/`
- [x] schema tests pass
- [x] summary proves the repo has enough examples for Phase 2
- [x] examples were not too few, so Phase 2 model eval was allowed

Current status:

- The v0 dataset has `83` examples, `45` train rows, and `38` family-split
  holdout rows.
- Holdout families are `semantic_schedule_v0`, `semantic_codegen_v3`,
  `semantic_codegen_v4`, `qk_block_dot`, and `threeway_load`.
- Labels cover accepted runtime paths, rejected candidates, ties,
  construction-blocked candidates, raw unconfirmed accepts, needs-rerun rows,
  and diagnostic-only rows.
- Prompt rows include Qwen's `/no_think` control and a strict compact-JSON
  instruction, but strict scoring still rejects assistant outputs that include
  empty `<think>` tags or out-of-taxonomy values.

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

Prompt contract:

- Input is the Phase 1 `prompts.jsonl` row.
- Output must be compact JSON only.
- Qwen prompt rows include `/no_think`, but the scorer still treats generated
  `<think>` tags, markdown, prose, or any other wrapper as invalid output.
- Required output schema:

```json
{"label":"reject","reason":"microbench_regression","retry":false}
```

Allowed `label` and `reason` values must match the Phase 1 taxonomy.
`retry` means "worth running this same mechanism again with small parameter
changes"; it is not permission to bypass gates.

Evaluation modes:

- `verdict_only`: predict only `label`.
- `verdict_reason`: predict `label` and `reason`.
- `dead_branch`: predict `retry`.
- `ranking`: rank candidates within the same family/model group by expected
  value before final outcomes.

Baselines in detail:

- `majority_label`: always predicts the most common training label.
- `reject_all`: always predicts `reject`, useful because most kernel candidates
  should fail.
- `random_label`: seeded random label using train-set label frequencies.
- `simple_family_heuristic`: if a prior family is rejected, reject later
  candidates in the same family unless the mechanism changes.
- `mechanism_prior`: predicts from historical mechanism-level outcome rates in
  the training split.
- `oracle_human_path`: optional non-competitive reference when the historical
  human next step is explicitly documented.

Metrics in detail:

- label accuracy
- macro-F1, so the reject-heavy class distribution cannot hide failure
- reason accuracy on rows where label matches
- retry precision and recall
- false-positive accept rate
- false-positive retry rate on known dead branches
- optional Brier score or expected calibration error if a future schema adds
  confidence
- ranking precision@1, precision@3, and NDCG by candidate group

Minimum pass condition:

- Beat `reject_all` and `mechanism_prior` on macro-F1.
- Keep false-positive accepts below a predeclared threshold, initially `5%`.
- Improve ranking precision@k over random and mechanism-prior baselines on the
  held-out split.

Expected first interpretation:

- If base/current models lose to `mechanism_prior`, do not train V7 for flywheel
  reasons yet. The historical signal is either too small, too obvious, or not
  learnable by the current prompt/model.
- If a larger model beats baselines but 8B does not, the task may be real but
  the local adapter may need training or a stronger base.
- If 8B beats baselines before training, move to live shadow mode before
  spending on adapter training.

Outputs:

- `extra/qk_flywheel_triage_eval.py`
- rollout artifacts under
  `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-baselines-v0/`
- aggregate metrics JSON/Markdown
- per-row predictions JSONL
- baseline predictions JSONL
- confusion matrix and ranking table

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
- If a model beats baselines clearly, run at least one live shadow batch before
  letting it influence ordering.

Phase 2 completion checklist:

- [x] evaluator checked in:
  `extra/qk_flywheel_triage_eval.py`
- [x] baselines checked in and deterministic
- [x] at least one no-adapter model evaluated
- [x] metrics artifact generated:
  `bench/amd-decode-flywheel-proof-20260614/triage-baselines-v0/`
- [x] conclusion states one of: `no_signal`, `trainable_signal`,
  `shadow_ready`, or `dataset_insufficient`

Current status:

- Deterministic baseline artifact:
  `bench/amd-decode-flywheel-proof-20260614/triage-baselines-v0/`.
- Best deterministic baseline is `mechanism_prior` / `simple_family_heuristic`
  at accuracy `0.289`, macro-F1 `0.185`, false-positive accept rate `0.000`,
  precision@3 `0.083`, and NDCG `0.218`.
- `reject_all` and `majority_label` reach accuracy `0.237` and macro-F1
  `0.077`.
- Qwen3-8B base generated-policy rollout:
  `bench/amd-decode-flywheel-proof-20260614/triage-qwen3-8b-base-v0/`.
- Qwen3-8B base strict result is accuracy `0.000`, macro-F1 `0.000`, and
  `38/38` `invalid_output` predictions. With `/no_think`, the model emits
  empty `<think>` tags plus JSON-shaped content, but strict parse/schema still
  fails; several generated reasons are also outside the allowed taxonomy.
- Conclusion: `no_signal` for the current strict no-adapter 8B model. This does
  not prove the full flywheel; it says the current base model cannot close the
  model-to-kernel link under the declared contract.

Immediate implication:

- Do not start Phase 3 because "the flywheel should work." Start Phase 3 only
  as a targeted experiment to make a schema-capable triage model beat
  `mechanism_prior` on this holdout.

## Phase 3: Train A Kernel-Triage Adapter

Purpose:

- Test whether repo-specific structured kernel history can improve model
  triage beyond prompting alone.
- Separate strict-output capability from actual kernel-triage skill.
- Decide whether there is enough signal to justify live shadow mode.

Inputs:

- Phase 1 dataset:
  `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v0/`
- Phase 2 baseline metrics:
  `bench/amd-decode-flywheel-proof-20260614/triage-baselines-v0/`
- Phase 2 no-adapter rollout:
  `bench/amd-decode-flywheel-proof-20260614/triage-qwen3-8b-base-v0/`
- Phase 4.2 stable compiler vocabulary data, only as format/schema support.
  It must not leak kernel-triage holdout answers.
- Existing strict-JSON adapter/training infrastructure under
  `extra/llm_adapter*.py`, reused only if it fits the data contract.

Starting facts:

- Train split: `45` examples.
- Holdout split: `38` examples.
- Holdout is family-split, not random-split.
- Baseline to beat: `mechanism_prior` / `simple_family_heuristic`, macro-F1
  `0.185`, accuracy `0.289`, false-positive accept rate `0.000`,
  precision@3 `0.083`, NDCG `0.218`.
- Current strict no-adapter 8B result: macro-F1 `0.000`, accuracy `0.000`,
  `38/38` invalid outputs.

Non-goals:

- No new kernel candidates.
- No kernel code changes.
- No generated-policy promotion.
- No live ordering decisions.
- No 32B or risky schedule/search work.
- No training on holdout prompts, holdout labels, or model outputs derived from
  holdout labels.
- No counting deterministic post-processing alone as flywheel proof. A JSON
  extractor or taxonomy repair can be measured as a diagnostic baseline, but it
  cannot prove model-to-kernel reasoning by itself.

Phase 3.0: Protocol Diagnostic

Purpose:

- Determine how much of the Phase 2 failure is strict-output protocol versus
  wrong kernel triage.

Tasks:

- Score the existing no-adapter rollout under the strict scorer. This is already
  the official Phase 2 result.
- Add an optional diagnostic parser that extracts the first JSON object after
  empty `<think>` tags and scores it separately.
- Add an optional deterministic taxonomy-repair diagnostic that maps only
  predeclared aliases to the Phase 1 reason taxonomy.
- Record these diagnostics as non-competitive unless they are predeclared as a
  separate baseline in the evaluator artifact.

Outputs:

- optional diagnostic artifact under
  `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-protocol-diagnostic-v0/`
- parser/repair code, if added, must be tested and must never modify the
  official strict score in place.

Gate:

- If the extracted/repair diagnostic is still below `mechanism_prior`, proceed
  only if the goal is strict-output training, not because the model has shown
  kernel skill.
- If the diagnostic beats `mechanism_prior`, rerun with a predeclared scorer and
  treat it as a protocol-fix baseline, then continue to live shadow only after
  strict-output behavior is also solved.

Phase 3.1: Export Kernel-Triage SFT Data

Purpose:

- Convert the Phase 1 train split into adapter-ready strict JSON examples.

Dataset rules:

- Inputs come from `prompts-train.jsonl`.
- Targets come from the matching hidden labels in `examples.jsonl`.
- The model input must include only the prompt text, not `expected_json`.
- The target must be exactly compact JSON with keys `label`, `reason`, and
  `retry`.
- Holdout rows may be copied into an eval manifest, but never into the training
  JSONL.
- Because the train split has only `45` rows, oversampling is allowed only on
  train rows and must be recorded in the artifact.
- Format/schema support rows from V4/V4.1 strict-JSON work may be mixed in only
  if they are tagged separately, carry no kernel outcome labels, and cannot
  teach the holdout answers.

Proposed output files:

- `extra/qk_flywheel_triage_sft.py`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-sft-v0/train.jsonl`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-sft-v0/holdout-prompts.jsonl`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-sft-v0/summary.json`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-sft-v0/README.md`

Required summary:

- source row counts
- oversampled row counts
- label counts before and after sampling
- reason counts before and after sampling
- whether any schema-support rows were mixed in
- proof that holdout ids are absent from training rows

Phase 3.2: Train Adapter Candidates

Purpose:

- Produce a schema-capable Qwen3-8B triage adapter without changing the kernel
  runtime.

First candidate:

- Base: Qwen3-8B Q4_K_M.
- Runtime: existing generated-policy/shared-storage path.
- Adapter path: reuse suffix-cache internal-adapter training if compatible,
  starting with the smallest previously viable policy, such as `last1_ffn`
  rank `4`.
- Temperature: `0.0` for evaluation rollouts.
- Token cap: `64`, matching Phase 1 prompts.

Escalation policy:

- Expand rank or suffix depth only after the small candidate produces valid
  strict JSON on the holdout.
- Do not run broad capacity sweeps before a clean artifact shows where the
  failure is: parse/schema, taxonomy, label reasoning, retry reasoning, or
  ranking.
- Teacher-forced loss and token accuracy are diagnostic only. They do not
  promote Phase 3.

Proposed artifacts:

- `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-adapter-v0/`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-adapter-v0-rollout/`
- `bench/amd-decode-flywheel-proof-YYYYMMDD/triage-adapter-v0-compare/`

Phase 3.3: Score Adapter Against Phase 2

Purpose:

- Use the exact Phase 2 evaluator so the adapter is compared against the same
  deterministic baselines.

Required command shape:

```sh
PYTHONPATH=. .venv/bin/python extra/qk_flywheel_triage_eval.py \
  --examples bench/amd-decode-flywheel-proof-20260614/kernel-triage-v0/examples.jsonl \
  --out bench/amd-decode-flywheel-proof-YYYYMMDD/triage-adapter-eval-v0 \
  --rollout adapter_v0=bench/amd-decode-flywheel-proof-YYYYMMDD/triage-adapter-v0-rollout
```

Required metrics:

- strict JSON pass count
- label accuracy
- macro-F1
- reason accuracy on label matches
- false-positive accept rate
- retry precision and recall
- precision@1, precision@3, and NDCG
- confusion matrix
- per-family and per-mechanism breakdown

Minimum Phase 3 pass:

- Strict JSON parse/schema/type pass on at least `37/38` holdout rows.
- Macro-F1 must beat `mechanism_prior` (`0.185`) on the family-split holdout.
- False-positive accept rate must stay at or below `5%` (`<=1` false-positive
  accept on the current holdout).
- Ranking precision@3 or NDCG must improve over `mechanism_prior`, not just
  label macro-F1.

Shadow-ready bar:

- Macro-F1 improves over `mechanism_prior` by a meaningful margin, initially
  `+0.05` absolute or better.
- Strict JSON output is effectively solved (`37/38` minimum, `38/38`
  preferred).
- No evidence that the model is only memorizing family names or always choosing
  a low-risk label.
- The model reduces wasted-experiment recommendations without increasing
  false-positive accepts.

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
- Phase 3 score report that directly imports the Phase 2 baseline numbers

Gate:

- Must beat Phase 2 baselines on family-split or time-split holdout.
- Must keep false-positive accepts low. A model that confidently recommends
  known dead branches is not useful even if average accuracy improves.
- Must pass strict JSON output. A model that requires hand repair at inference
  time is not ready to steer kernel ordering.

Stop rule:

- If training only memorizes family names or fails family-split holdout, do not
  call it a flywheel. The model loop remains a structured-output capability.
- If strict JSON improves but macro-F1 does not beat `mechanism_prior`, Phase 3
  is a useful adapter-capability result but not flywheel evidence.
- If macro-F1 improves but false-positive accepts rise above the threshold, do
  not enter shadow mode; a kernel assistant that recommends dead branches is
  operationally expensive even when aggregate metrics look better.

Phase 3 completion checklist:

- [ ] protocol diagnostic scoped or explicitly skipped
- [ ] SFT exporter checked in and tested
- [ ] training artifact generated with holdout-contamination audit
- [ ] adapter rollout generated on the Phase 1 holdout prompts
- [ ] adapter scored by `extra/qk_flywheel_triage_eval.py`
- [ ] result classified as one of: `schema_fail`, `baseline_fail`,
  `unsafe_accepts`, `shadow_ready`

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
