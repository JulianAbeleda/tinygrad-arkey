# AMD Decode Flywheel Proof Plan

Date: 2026-06-14

Status: plan of record for proving or falsifying the full kernel-optimization
flywheel. Phase 1/2 and the Phase 3.0 through 3F diagnostic/data subphases are
built where marked. Phase 3F+ currently has a strong `xgboost` result on the
same 38-holdout split (`macro-F1 0.891`, `accuracy 0.895`, `false_accept 0.000`),
but the rerun is still gated by residual mechanism-coverage requirements.

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

## Phase 2–4 Scope Snapshot

This is the current actionable map from proof work to "ready for live shadow":

- Phase 2 (`historical triage benchmark`): complete.
  - Baseline split is in place; `mechanism_prior` is the current strong
    deterministic baseline.
  - Baseline gap is explicit and reproducible.
- Phase 3 (`adapter / protocol`): complete as diagnostic.
  - 8B adapter path and protocol instrumentation are implemented, but strict
    and noisy results did not beat the baseline.
  - The compounding claim is not yet supported by adapter behavior alone.
- Phase 3B/3C/3D/3E (`cost-model experiments`): complete as data-framing.
  - A real XGBoost rerun with v1+ featured data is strong, but still blocked by
    unresolved mechanism/label coverage requirements.
  - Coverage work is now explicit in the plus-plan: still need
    `3 packed_word_lane_unroll`, `2 qk_block_dot`, and `1 wide_load_only` train rows
    (plus one unseen holdout prediction-stage value).
- Phase 3F (`targeted real outcomes`): complete as partial pass.
  - Added rows improved real-feature density and held out/known-family integrity.
  - No model-quality claim yet; rerun of cost-model remains blocked by coverage.
- Phase 4 (`live shadow mode`): not started.
  - Entry condition is still "cost-model rerun unblocked and above baseline" on
    the current split protocol.
  - Until then, shadow mode is not a valid claim of flywheel compounding.

Phase sequence from here:
1. Collect the remaining 6 missing mechanism rows (3 `packed_word_lane_unroll`,
   2 `qk_block_dot`, 1 `wide_load_only`) and add a holdout-stage categorical value
   fix so coverage no longer reports `after_microbench_before_full_decode` as unseen.
2. Rerun cost-model candidate ordering/ranking from the same protocol and
   prove gain against `mechanism_prior`, p@k and NDCG.
3. Only if that passes, open Phase 4.2 shadow trials.

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

- [x] protocol diagnostic scoped and generated:
  `bench/amd-decode-flywheel-proof-20260614/triage-protocol-diagnostic-v0/`
- [x] SFT exporter checked in and tested:
  `extra/qk_flywheel_triage_sft.py`
- [x] SFT artifact generated with holdout-contamination audit:
  `bench/amd-decode-flywheel-proof-20260614/triage-sft-v0/`
- [ ] training artifact generated with holdout-contamination audit
- [ ] adapter rollout generated on the Phase 1 holdout prompts
- [ ] adapter scored by `extra/qk_flywheel_triage_eval.py`
- [x] current result classified as `training_path_latency_blocked` before
  adapter rollout

Current status:

- Phase 3.0 diagnostic result:
  `bench/amd-decode-flywheel-proof-20260614/triage-protocol-diagnostic-v0/`.
  Strict text remains `0/38` parseable. JSON extraction makes `38/38`
  parse/schema-valid and taxonomy repair can make `38/38` taxonomy-valid, but
  extracted labels still reach only accuracy `0.053`, macro-F1 `0.036`, and
  false-positive accept rate `0.763`. This is below `mechanism_prior`
  macro-F1 `0.185`, so protocol repair is not enough.
- Phase 3.1 SFT export result:
  `bench/amd-decode-flywheel-proof-20260614/triage-sft-v0/`. It has `45`
  train rows, `38` eval/holdout rows, `0` oversampled rows, `0`
  schema-support rows, and `0` holdout ids in train.
- Phase 3.2 first-candidate attempt:
  `bench/amd-decode-flywheel-proof-20260614/triage-adapter-v0-attempt/`.
  Two `last1_ffn` rank-4 suffix-cache attempts were terminated after repeated
  30 second polls with no stdout and no adapter artifact: one generated-mode
  attempt and one baseline-mode attempt. No adapter rollout or Phase 3.3 score
  exists yet.
- Phase 3.2A instrumentation/smoke:
  `bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0/`.
  The suffix trainer now emits progress to stdout and `progress.jsonl`, and has
  split-specific `--max-train-rows` / `--max-eval-rows` controls. The smoke used
  `4` train rows, `2` eval rows, and `8` optimizer steps. It changed adapter
  weights (`adapter_l2=1.582253`) and reduced tiny-slice teacher-forced loss
  (`train -0.8872`, `eval -0.9574`), but it did not improve held-out
  generation: strict score remained `0/38`, extracted macro-F1 remained
  `0.036`, and false-positive accept rate remained `0.763`.
- The instrumentation explains the prior silent runs: on the smoke, caching `4`
  train prefixes took `32.8s`, and caching `2` eval prefixes took `21.0s`, with
  prompts around `269-425` tokens. A full adapter run on the current prompt
  shape is expected to spend many minutes in cache/eval before useful feedback.

Immediate implication:

- Do not continue by sweeping adapter ranks. The next Phase 3.2 step should
  make the training loop observable/practical first: add progress reporting,
  shorten/compress the long kernel-context prompts, or create a smaller
  predeclared smoke candidate that can complete before retrying a full held-out
  adapter rollout.
- Phase 3.2A confirms the low-expectation smoke result: `45` examples did not
  rescue the local 8B. The stronger strategic next test is a stronger proposer
  on the same benchmark, or a prompt-compressed local-adapter experiment if the
  goal is specifically to keep testing local 8B.

## Phase 3B: Learned Cost Model Triage

Purpose:

- Test the compiler-autotuning version of triage: a small learned cost model
  over structured features, not an LLM over prose prompts.
- Keep triage/ranking separate from novel mechanism proposal. Triage is a cost
  model job; proposal remains a stronger-LLM or human-reasoning job.
- Decide whether the current `45` train rows and pre-result features already
  contain enough signal to beat `mechanism_prior`.

Implementation:

- `extra/qk_flywheel_cost_model.py`
- Artifact:
  `bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v0/`
- Feature policy: `pre_result_analytical_context_v0`.
- Uses only pre-result fields from the Phase 1 examples:
  row kind, family, model size, role, format, mechanism, prediction stage,
  schedule/context booleans and numeric knobs, parsed opt flags, and analytical
  proxy features for reuse, ILP, warp concurrency, load width, imbalance, and
  schedule complexity.
- Explicitly excludes raw ids as categorical features and excludes target or
  result fields: `label`, `reason`, `retry`, `evidence`, `status`, `gain`,
  `gain_pct`, candidate/current GB/s, correctness decisions, source files, and
  split markers.
- Backend behavior:
  - `xgboost` if installed, using the native `DMatrix`/`train` API so
    `scikit-learn` is not required.
  - `centroid_cost_model` fallback for tests and no-dependency environments.

Current Phase 3B result:

- XGBoost was available locally as `3.2.0` and the native `rank:ndcg` ranker ran
  with integer relevance labels.
- Feature count: `127` (`76` numeric, `51` train-seen categorical one-hot
  columns).
- The family-split holdout still contains many unseen categorical values:
  `24` ignored holdout categories, including all holdout families and several
  holdout mechanisms/schedule names.
- Leakage audit: no raw-id categorical features and no target/result fields.

Metrics on the same `38` holdout rows:

| scorer | accuracy | macro-F1 | false accept | p@3 | NDCG |
|---|---:|---:|---:|---:|---:|
| `mechanism_prior` | `0.289` | `0.185` | `0.000` | `0.083` | `0.218` |
| `centroid_cost_model` | `0.105` | `0.039` | `0.263` | `0.000` | `0.153` |
| `xgboost_cost_model` | `0.237` | `0.137` | `0.000` | `0.000` | `0.189` |

Conclusion:

- Phase 3B is `no_signal` on the current historical benchmark. XGBoost is the
  right tool class for learned triage/ranking, but the current dataset/features
  do not beat the deterministic prior.
- This does not disprove cost models in general. It says the present `45`-row
  train split, family-split holdout, and feature extraction are too thin for a
  useful learned triage model.
- Do not build a cost model from scratch before it has more data and richer
  first-class features. The ML piece should stay off-the-shelf; the novel work
  is extracting better tinygrad/UOp/profile features and collecting more
  labeled candidate outcomes.
- Do not promote Phase 4 shadow mode from this result. A future Phase 3B retry
  must beat `mechanism_prior` on macro-F1, keep false-positive accepts low, and
  improve ranking metrics by a meaningful margin.

## Phase 3C: Cost-Model Data And Feature Upgrade

Purpose:

- Turn the Phase 3B negative into a concrete data/feature collection plan.
- Separate missing-data failures from missing-feature failures.
- Define the next batch before running more kernels, so the cost-model dataset
  grows toward coverage instead of random experiment accumulation.

Implementation:

- `extra/qk_flywheel_feature_audit.py`
- Artifact:
  `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v0/`
- Inputs: the same Phase 1 `examples.jsonl` and the Phase 3B leak-free feature
  extractor.
- Outputs:
  - `summary.json`: train/holdout coverage, target rows, weak-row reasons,
    stage viability, and leakage audit.
  - `row-audit.jsonl`: per-row feature-quality flags.
  - `README.md`: short human-readable target list.

Current Phase 3C result:

- Conclusion: `needs_data_and_feature_expansion`.
- Train rows: `45`; holdout rows: `38`.
- Unseen holdout categorical values: `24`.
- Weak rows: `56`.
- Post-full-decode train rows: `9`; these are useful historical outcomes but
  weak signal for pre-outcome triage.
- Leakage audit: no target/result feature names detected.

Highest priority targets:

1. Add label coverage for labels present in holdout but absent or undercovered
   in train:
   - `construction_blocked`: `1` train / `19` holdout, needs `4` more train
     rows to hit the initial `5`-row floor.
   - `raw_accept_unconfirmed`: `0` train / `3` holdout, needs `5`.
   - `diagnostic_only`: `0` train / `1` holdout, needs `5`.
2. Normalize `unknown` mechanisms before treating them as learnable classes:
   `18` holdout rows are currently `unknown`.
3. Add targeted mechanism coverage for holdout mechanisms with fewer than five
   train rows:
   - `packed_word_lane_unroll`: `0` train / `2` holdout, needs `5`.
   - `qk_block_dot`: `0` train / `2` holdout, needs `5`.
   - `vector_load`: `0` train / `2` holdout, needs `5`.
   - `wide_load_only`: `0` train / `1` holdout, needs `5`.
4. Reduce categorical train/holdout mismatch:
   - unseen families: `qk_block_dot`, `semantic_codegen_v3`,
     `semantic_codegen_v4`, `semantic_schedule_v0`, `threeway_load`.
   - unseen schedule names: `direct_out`, `reduce_unroll4`, `row_upcast2`,
     `two_dim_local4`.
   - unseen schedule families: `q4_k_packed_u32`, `q6_k_packed_u16`.
5. Add richer first-class tinygrad/UOp/profile features for rows with
   `no_structural_kernel_detail` (`26` rows), instead of relying on top-level
   labels and candidate names.

Next implementation scope:

- Phase 3D has added the canonical candidate-outcome log schema and normalized
  mechanism layer. The remaining work is feature/data collection, not another
  cost-model score run.
- Add real UOp/profile feature extraction for candidate rows:
  UOp op counts, global load/store counts, scalar versus vector load evidence,
  estimated bytes, arithmetic-intensity proxy, local/shared memory use, loop
  and opt counts, generated source/body size, and static-gate failure reason.
- Generate a small targeted data batch that fills the label/mechanism holes
  above. The goal is cost-model coverage first, not immediate kernel speedup.
- Rerun Phase 3B only after this data/feature upgrade. Do not promote Phase 4
  until the rerun beats `mechanism_prior` under the same gates.

## Phase 3D: Cost-Model Feature Schema v1

Purpose:

- Make the cost-model rows look like compiler-autotuning data instead of prose
  history.
- Normalize mechanism names so holdout mechanisms are visible as real classes,
  not `unknown`.
- Freeze a candidate-record schema that separates model-visible features from
  final outcome fields.

Implementation:

- `extra/qk_flywheel_dataset_v1.py`
- `extra/qk_flywheel_cost_model.py`
- `extra/qk_flywheel_feature_audit.py`
- `test/external/test_qk_flywheel_phase3d.py`
- Artifacts:
  `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1/` and
  `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1/`

Schema:

- Each row has `schema_version="kernel_triage_v1"`.
- Each row has a top-level `candidate_record` with
  `schema_version="candidate_outcome_v1"`.
- `candidate_record.static_features`, `candidate_record.uop_features`, and
  `candidate_record.profile_features` are the model-visible feature groups.
- `candidate_record.outcome` stores `label`, `reason`, `retry`, and
  `source_files`, but prompts and feature extraction remove that group.
- Current `uop_features` are explicit proxy estimates with
  `uop_available=false`; they are not yet first-class tinygrad UOp counts.

Current Phase 3D result:

- Dataset rows are unchanged: `83` rows, `45` train rows, and `38`
  family-split holdout rows.
- The v0 split policy is preserved as `family_split_v0_preserved`.
- Normalization removes the v0 `unknown` mechanism hole:
  `18` unknown-mechanism holdout rows in the v0 audit become `0` unknown
  mechanism rows in v1.
- `26` rows changed mechanism names from v0, mostly semantic schedule/codegen
  rows such as `row_upcast`, `reduce_unroll`, and `two_dim_local`.
- The feature audit improves but does not clear the gate:
  unseen holdout categorical values fall from `24` to `15`, and weak rows fall
  from `56` to `43`.
- Leakage audit remains clean: no target/result feature names are used.

Remaining gaps:

- `33` holdout rows still have mechanisms unseen in train, now because the
  mechanisms are named correctly rather than hidden as `unknown`.
- Label coverage is still thin for `construction_blocked`,
  `raw_accept_unconfirmed`, and `diagnostic_only`.
- The model still needs real UOp/profile features; current v1 UOp features are
  analytical proxies.
- Do not rerun or promote XGBoost as a decision point until the targeted
  mechanism/label rows and first-class UOp/profile features exist.

Next implementation scope:

- Build a targeted candidate/outcome batch for the uncovered mechanisms:
  `packed_word_lane_unroll`, `qk_block_dot`, `reduce_unroll`, `row_upcast`,
  `two_dim_local`, `vector_load`, and `wide_load_only`.
- Add first-class extracted features for source/UOp/profile evidence, replacing
  proxy-only `uop_available=false` rows where possible.
- Rerun Phase 3B only after the v1 schema has materially better coverage.

## Phase 3E: Real Feature Extraction And Coverage Plan

Purpose:

- Add real source/compile evidence to the v1 candidate schema where committed
  artifacts already expose it.
- Keep final outcomes out of model-visible features.
- Produce the targeted data-collection plan needed before another cost-model
  score run.

Implementation:

- `extra/qk_flywheel_feature_enrich.py`
- `extra/qk_flywheel_coverage_plan.py`
- `extra/qk_flywheel_feature_audit.py`
- `test/external/test_qk_flywheel_phase3e.py`
- Artifacts:
  `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured/`,
  `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured/`,
  and
  `bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1/`.

Feature policy:

- Before microbench, use only static/source/compile evidence:
  load-width reports, compile instruction counts, memory-instruction counts,
  target global-load shape, source vector-type evidence, source line counts,
  workgroup/local/group shape, and selected instruction-class counts.
- Do not use final labels, reasons, retry flags, evidence blobs, source-file
  paths in prompts, final verdict status, speed gains, candidate/current GB/s,
  correctness decisions, or A/B outcomes as model-visible features.
- Current Phase 3E does not synthesize outcomes and does not move holdout rows
  into train.

Current Phase 3E result:

- Dataset rows are unchanged: `83` rows, `45` train rows, and `38`
  family-split holdout rows.
- Feature schema: `candidate_outcome_v1_featured`.
- Real UOp/source rows: `13` total, `7` train and `6` holdout.
- Real feature coverage by mechanism:
  - `tile_custom`: `7`
  - `packed_word_lane_unroll`: `2`
  - `qk_block_dot`: `2`
  - `vector_load`: `2`
- Source/load-width report rows: `11`.
- Compile-report rows: `2`.
- Leakage audit remains clean: no target/result feature names detected.

What did not improve:

- Unseen holdout categorical values remain `15`.
- Weak rows remain `43`.
- `33` holdout rows still have mechanisms unseen in train.
- The missing train coverage is now the dominant blocker, not feature plumbing.

Coverage plan:

- `triage-coverage-plan-v1/` keeps `rerun_phase3b_allowed=false`.
- It calls for at least `35` new mechanism-coverage outcome rows:
  `5` each for `packed_word_lane_unroll`, `qk_block_dot`, `reduce_unroll`,
  `row_upcast`, `two_dim_local`, `vector_load`, and `wide_load_only`.
- It also records `14` label-coverage row targets:
  `4` for `construction_blocked`, `5` for `diagnostic_only`, and `5` for
  `raw_accept_unconfirmed`. These can overlap with mechanism rows if the
  natural outcomes match; labels must not be forced.

Gate:

- Do not rerun Phase 3B/XGBoost as a decision point yet.
- The next useful implementation is a real post-Phase-3E candidate/outcome
  batch that fills the mechanism and label holes above.
- After that batch, regenerate `kernel-triage-v1-featured/`, rerun the feature
  audit, and only then run the cost model again against `mechanism_prior`.

## Phase 3F: Targeted Outcome Batch v1

Purpose:

- Convert unused committed real probe/source diagnostics into train rows where
  they fill the Phase 3E coverage gaps.
- Keep the family-split holdout untouched.
- Make the remaining blocker explicit instead of treating a partial batch as a
  cost-model rerun trigger.

Implementation:

- `extra/qk_flywheel_targeted_outcomes.py`
- `extra/qk_flywheel_dataset_v1.py`
- `extra/qk_flywheel_feature_enrich.py`
- `extra/qk_flywheel_coverage_plan.py`
- `test/external/test_qk_flywheel_phase3f.py`
- Artifacts:
  `bench/amd-decode-flywheel-proof-20260614/targeted-outcomes-v1/`,
  `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured-plus/`,
  `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured-plus/`,
  `bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v1-plus/`,
  and
  `bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1-plus/`.

Current Phase 3F result:

- Added `47` real train rows from committed diagnostics:
  `5` `direct_output`, `10` `row_upcast`, `8` `reduce_unroll`,
  `8` `two_dim_local`, `6` `vector_load`, `4` `wide_load_only`,
  `4` `tile_custom`, `3` `qk_block_dot`, and
  `2` `packed_word_lane_unroll`.
- Labels added naturally: `21` `construction_blocked`, `7` `raw_accept_unconfirmed`,
  `7` `diagnostic_only`, `7` `reject`, and `6` `tie`.
- The plus dataset has `130` rows: `92` train and the original `38` holdout.
- Real UOp/source rows increase from `13` to `20`.
- Design-only `QK_BLOCK_DOT` semantic-op contract rows remain excluded from
  training labels because they have no runtime lowering or outcome.
- Leakage remains clean; prompts still omit `candidate_record.outcome`,
  source-file paths, and feature-source paths.

- Cost-model pass on `kernel-triage-v1-featured-plus` now has a strong
  XGBoost signal (`macro-F1` `0.891`, `accuracy` `0.895`) versus
  `mechanism_prior` (`macro-F1` `0.552`) on the same `38` holdout rows.

Audit delta versus Phase 3E:

- Unseen holdout categorical values improve from `15` to `1`.
- Weak rows improve from `43` to `9`.
- Remaining mechanism coverage need falls from `35` rows to `6`.
- Remaining label coverage need is now `0`.

Remaining blocker:

- `triage-coverage-plan-v1-plus/` still keeps
  `rerun_phase3b_allowed=false`.
- Still needed before another XGBoost decision run:
  `3` `packed_word_lane_unroll`, `2` `qk_block_dot`, and
  `1` `wide_load_only` train rows, plus `after_microbench_before_full_decode`
  prediction-stage coverage.

Gate:

- Do not rerun Phase 3B/XGBoost as a decision point yet.
- Continue with a second targeted real-outcome batch, focused on the remaining
  semantic schedule mechanisms and natural raw-accept opportunities.

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
