# AMD Decode Flywheel Proof Plan

Date: 2026-06-14

Status: plan of record for proving or falsifying the full kernel-optimization
flywheel. Phases 1–6 (the triage line) are built and **falsified**: the holdout
`xgboost` result that once looked strong (`macro-F1 0.891`) was a metric artifact
(the kernel outcomes were scored on wall-clock throughput dominated by ~0.27 ms
launch overhead), and on the corrected device metric a cheap deterministic rule
matches the learned model on every honest re-test (Phases 4.x/M, `metric-audit-m0/`).
The live frontier is **Phase B**: re-basing the metric and reducing to primitives
showed the real lever is batching / a fused Q4_K GEMM. Read the postmortem first.

Parent architecture note:
`docs/amd-decode-kernel-optimization-flywheel.md`.

Postmortem (read this first for the honest arc and what is actually real):
`docs/amd-decode-flywheel-postmortem.md` — the triage premise was a dead end (noise
metric + a deterministic rule matches the model), but re-basing the metric and reducing
to primitives produced a real target: batching / a fused Q4_K GEMM (Phase B).

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

## Phase 3G: Coverage Closure Batch

Purpose:

- Close the residual Phase 3F+ coverage gate without changing the family-split
  holdout.
- Add only real candidate outcomes; no synthetic labels, duplicated holdout
  rows, or design-only contract rows.
- Make the next cost-model rerun decision-grade instead of a lucky fit on thin
  coverage.

Required rows:

- `packed_word_lane_unroll`: add `3` train rows.
  - Run packed-load lane-unroll candidates on additional Q4_K tensors/roles.
  - Require generated-source load-width evidence before timing.
- `qk_block_dot`: add `2` train rows.
  - Repeat the block-local semantic-op compile gate on more dominant Q4_K
    tensors.
  - Microbench only candidates that pass the compile/static shape checks.
- `wide_load_only`: add `1` train row.
  - Continue the three-way load diagnostic controls.
  - Keep this as branch-bounding evidence unless it passes the normal runtime
    integration gates.

Prediction-stage coverage:

- Add or normalize a train row at `after_microbench_before_full_decode`, because
  this is the only remaining unseen holdout categorical value.
- If the row is a real microbench-pass candidate awaiting full-decode
  confirmation, record it at that stage. If it is not genuinely at that stage,
  do not force the stage just to satisfy the audit.

Implementation scope:

- Extend `extra/qk_flywheel_targeted_outcomes.py` or add a narrowly named
  companion script only if the source artifacts no longer fit the Phase 3F
  extractor cleanly.
- Regenerate:
  `targeted-outcomes-v1/`,
  `kernel-triage-v1-featured-plus/`,
  `triage-feature-audit-v1-featured-plus/`,
  `triage-coverage-plan-v1-plus/`, and
  `triage-cost-model-v1-plus/`.
- Extend `test/external/test_qk_flywheel_phase3f.py` with the new expected row
  counts and coverage checks, or split a Phase 3G test if the batch becomes a
  separate script.

Exit gate:

- `triage-coverage-plan-v1-plus/summary.json` must report
  `rerun_phase3b_allowed=true` or have no mechanism/categorical blockers left.
- The rerun cost-model result must still beat `mechanism_prior` on macro-F1,
  precision@k, and NDCG with `false_positive_accept_rate <= 0.05`.
- Only after that should Phase 4 live shadow mode start.

Status: met (2026-06-14). The coverage-closure batch added `6` real mechanism
rows (`3` `packed_word_lane_unroll`, `2` `qk_block_dot`, `1` `wide_load_only`)
plus the `after_microbench_before_full_decode` stage row on the `blk.2` raw_accept
candidate. `triage-coverage-plan-v1-plus/` now reports
`rerun_phase3b_allowed=true` with no blockers, and the rerun keeps XGBoost ahead
of `mechanism_prior` (macro-F1 `0.873` vs `0.479`, p@1 `0.500` vs `0.000`, p@3
`0.250` vs `0.167`, NDCG `0.500` vs `0.253`, false-positive accept rate `0.0`).
Phase 4 is unblocked.

## Phase 4: Live Shadow Mode

Purpose:

- Test the model on new kernel decisions without letting it steer the work yet.

Method:

- Before running new candidate gates, ask the model to predict/rank outcomes.
- Freeze predictions in an artifact before seeing results.
- Run the normal human/deterministic kernel loop unchanged.
- Score the model after outcomes are known.

State entering Phase 4:

- The Phase 3G exit gate is met: `triage-coverage-plan-v1-plus/` reports
  `rerun_phase3b_allowed=true`, and `triage-cost-model-v1-plus/` keeps XGBoost
  ahead of `mechanism_prior` on the fixed `38`-row holdout (macro-F1 `0.873` vs
  `0.479`, `false_positive_accept_rate=0.0`). The labeled corpus is the `136`-row
  `kernel-triage-v1-featured-plus/` dataset.

Required harness:

- Split the coupled fit+predict in `extra/qk_flywheel_cost_model.py` into a
  reusable train -> freeze -> predict path, or add a thin
  `extra/qk_flywheel_shadow.py` that imports `extract_feature_map`,
  `FeatureVectorizer`, the XGBoost classifier/ranker fit, and `_label_policy`
  from the cost-model module. Do not fork the feature logic; reuse it so the
  shadow predictor and the audited cost model share one leak-free feature path.
- Train on the entire labeled `kernel-triage-v1-featured-plus/` corpus (all
  `136` rows; the family-split holdout is not special in shadow mode because the
  test set is the fresh batch, not the holdout). Persist the fitted vectorizer
  feature vocab, the classifier, the ranker, and the label/reason policy.
- Predict on fresh, unlabeled candidate rows built from static descriptor
  metadata only (shape, role, format, mechanism, opts, prediction stage
  `after_static_before_microbench`). Real source/compile/microbench features are
  absent before running and default in the vectorizer; v0 shadow is therefore a
  blind static-stage prediction. Staged re-prediction (after compile, after
  microbench) is a Phase 4.x extension, not v0.

Fresh candidate batch (instance-level generalization, new tensors / same
mechanism families):

- `3` `packed_word_lane_unroll` packed-load candidates on untouched dominant
  Q4_K `ffn_gate` tensors not in the corpus (for example
  `blk.4/5/6.ffn_gate.weight`), via the same descriptor -> v3 -> schedule_bench
  path used in Phase 3G.
- `2` `qk_block_dot` compile-gate + microbench candidates on untouched dominant
  Q4_K tensors (for example `blk.0.attn_output.weight` and
  `blk.1.ffn_up.weight`).
- `1` `wide_load_only` three-way load diagnostic on an untouched tensor (for
  example `blk.0.attn_output.weight`).
- Rationale: these generators produce label diversity (raw_accept, tie,
  reject, construction_blocked, diagnostic_only), so the shadow score is not a
  single-label artifact. None of these exact tensors appears in train.

Freeze protocol (the load-bearing honesty rule):

- Write `shadow-v0/predictions.jsonl` with the model label, reason, retry,
  confidence, and rank score for every fresh candidate, plus a freeze record
  (`shadow-v0/freeze.json`) carrying the corpus content hash, the trained-model
  parameter hash, the candidate-set hash, and the git commit, before any fresh
  GPU run exists.
- Commit the frozen predictions before producing outcomes. The outcome run must
  not be able to influence the predictions; a test asserts `predictions.jsonl`
  and `freeze.json` are unchanged after `outcomes.jsonl` is written.

Run and score:

- Run the normal deterministic generators on the fresh batch to produce
  `shadow-v0/outcomes.jsonl` with the same extractor labels used for training
  (reuse the Phase 3G extractor functions; do not hand-label).
- Score with `extra/qk_flywheel_triage_eval.py`: macro-F1, precision@k, NDCG,
  and `false_positive_accept_rate` for the model versus `mechanism_prior`,
  `simple_family_heuristic`, and `reject_all` on the fresh batch.
- Add a dead-branch metric to the shadow scorer: experiments-to-first-live
  (count of dead candidates -- reject / construction_blocked / diagnostic_only
  -- ranked above the first live candidate) under model ranking versus baseline
  ranking, plus the model's false-positive accept rate on the fresh batch.

Outputs:

- `bench/amd-decode-flywheel-proof-YYYYMMDD/shadow-v0/`
- `predictions.jsonl` (frozen before outcomes)
- `freeze.json` (corpus/model/candidate hashes + commit)
- `outcomes.jsonl`
- `summary.json` + `README.md` shadow score report

Tests:

- Extend `test/external/` with a Phase 4 test that asserts: predictions are
  frozen before outcomes (hash stability), the shadow feature path reuses the
  audited leak-free features (no `FORBIDDEN_FEATURE_SOURCES` token in shadow
  feature names), and the score report compares the model against all three
  baselines on the fresh batch.

Exit gate:

- Model ranking must beat `mechanism_prior` on the fresh batch on macro-F1 and
  at least one of precision@k or NDCG, with `false_positive_accept_rate <= 0.05`.
- The model must reduce dead-branch recommendations versus
  `simple_family_heuristic` (lower experiments-to-first-live or fewer dead
  branches in the top-k).
- Only after that should Phase 5 controlled assist mode start.

Stop rule:

- If shadow mode fails, do not allow model-ranked execution order. Keep using
  the model only for documentation or artifact extraction.

Result (v0, 2026-06-14): ran and recorded in `shadow-v0/`. The gate is **not
met** (honest negative). `extra/qk_flywheel_shadow.py` froze 6 blind static-stage
predictions (commit `d9365daed`) before the fresh GPU run; the committed
`predictions.jsonl`/`freeze.json` are unchanged after outcomes (hash-verified).
The deterministic generators produced `3` tie, `1` reject, `1` construction_blocked
(`blk.1.ffn_up` correctness-gate fail), and `1` diagnostic_only — **zero live
candidates**, so ranking is undefined. XGBoost collapsed to all-`reject` on thin
static features (macro-F1 `0.071`) and lost to `mechanism_prior` (`0.667`). The
fixed-holdout cost-model win (macro-F1 `0.873`) does **not** generalize to fresh
tensors at the blind static stage. Per the stop rule the model stays
documentation-only; this is not promoted to Phase 5. The harness, freeze
protocol, and scorer are proven and reusable for Phase 4.x.

Phase 4.x follow-ups (to give the gate a fair, live-bearing test): a larger /
repeated fresh batch that includes live candidates (raw_accept), and staged
re-prediction after compile/microbench evidence — not blind static only.

Out of scope for v0 (future Phase 4.x / 5):

- Staged re-prediction after compile/microbench evidence becomes available.
- Cross-model generalization (14B/32B) and genuinely new mechanism families.
- Any runtime integration or gate bypass driven by the model.

## Phase 4.1: Cost-Aware Staged Shadow

Purpose:

- Move directly toward the realistic flywheel proof (Phase 6 alternative:
  model-assisted ordering reduces wasted GPU experiments), in shadow, with the
  objective and prediction stage corrected by the v0 negative.

What v0 taught (grounded in the `136`-row corpus):

- Live outcomes (`accept` / `raw_accept_unconfirmed` / `needs_rerun`) are `21/136`
  (~`15%`) and cluster in semantic-schedule families (`parts_local_policy` `7`,
  `row_upcast` `6`, `direct_output` `3`, `shared_storage` `3`). The memory-access
  probes used in v0 (`qk_block_dot`, `wide_load_only`) have **zero** live rows in
  the whole corpus and `packed_word_lane_unroll` has `1`: v0 sampled the
  mechanisms least able to win, so the ranking gate was vacuous.
- The richest real source/compile features sit at
  `after_compile_before_microbench` (`14` of `22` real-feature rows), not at the
  blind static stage v0 predicted from.
- Within identical-shape, same-mechanism candidates the win/loss is set by weight
  magnitudes the model cannot observe, so no model can beat a mechanism prior
  there. The model's only honest edge is cross-shape / role / config / compile
  signal. Beating `mechanism_prior` on labels may therefore be impossible with the
  current feature set; the flywheel-relevant question is cost, not label accuracy.

Objective reframe:

- Stop optimizing macro-F1 against `mechanism_prior`. Optimize **wasted GPU
  reduction**: at the compile stage (cheap), decide which candidates are worth the
  expensive microbench / full-decode, and measure GPU-seconds saved versus
  running every static-pass candidate (the current deterministic loop), at fixed
  live-recall (never skip a true live candidate). This is exactly the Phase 6
  alternative proof, measured in shadow before any real gating.

Method:

- Predict at `after_compile_before_microbench`: run only the cheap compile gate
  per candidate (no microbench), feed the real compile-stage features the model
  was trained on, and freeze a per-candidate keep/skip decision + live-probability
  before the expensive stage runs.
- Reuse the `extra/qk_flywheel_shadow.py` freeze protocol and leak-free path; add a
  compile-stage candidate builder and a per-stage cost model.
- Run the deterministic loop fully (microbench every candidate) so the true label
  and the counterfactual cost are known, then score.

Fresh batch (diverse, live-bearing, cross-feature):

- Draw from live-capable families so the batch contains real live and dead
  instances: semantic-schedule candidates (`parts_local_policy` / `row_upcast` /
  `direct_output`) on fresh tensors with varying `parts`/`opts`, plus a block of
  `packed_word_lane_unroll` ffn_gate candidates (Phase 3G shows ~1-in-N is
  `raw_accept`), plus the cheap-to-gate memory-access probes (`qk_block_dot` /
  `wide_load_only`) as known dead branches. Target enough candidates (`~20`) that
  the live count is not zero and cost differences are meaningful.

Cost model:

- Capture measured GPU-seconds per stage from the generator artifacts
  (`elapsed_s` / device timing): compile (cheap), microbench (expensive),
  full-decode (most expensive). Wasted GPU = microbench/full-decode seconds spent
  on candidates that turn out dead.

Metrics / gate:

- Primary: GPU-seconds saved versus run-everything at `100%` live-recall, for
  (a) `mechanism_prior` gating and (b) the cost-model gating. Report both; the
  flywheel-via-prior result is real if either beats run-everything without
  dropping a live candidate.
- Secondary (the open question): does the cost-model gate save more GPU than the
  `mechanism_prior` gate at equal live-recall? Pre-register that the prior may
  win; that is still a decisive flywheel result (the deterministic prior is the
  practical tool, the learned model adds no value at the current feature set).
- Keep the freeze-before-outcomes hash discipline and the leak-free audit.

Exit gate:

- Some pre-result gate (model or prior) reduces wasted GPU versus run-everything
  by a meaningful margin at full live-recall, with the keep/skip decision frozen
  before outcomes. If the cost model beats the prior, that is the signal to enter
  Phase 5 with the model; if only the prior wins, enter Phase 5 with the prior and
  keep the model documentation-only.

Out of scope:

- Letting any gate skip a real microbench/full-decode in the live loop (that is
  Phase 5). 14B/32B. New mechanism families. Intra-identical-shape discrimination
  (unobservable; explicitly not attempted).

Result (2026-06-14): ran and recorded in `shadow-staged/`. The gate is **met, and
the cost model beats the prior** — the first evidence the learned model adds value
over the deterministic baseline. `extra/qk_flywheel_shadow.py` froze keep/skip rank
scores for `16` fresh semantic-schedule candidates (commit `f7979eb4a`) before the
microbench; the committed `predictions.jsonl`/`freeze.json` are unchanged after
outcomes (hash-verified), and the feature path is leak-free. Microbench outcomes:
`2` `raw_accept` (live), `5` tie, `1` reject, `8` construction_blocked. The model
ranked the two live candidates (`row_upcast` on `blk.1/2.attn_q.weight`, `+4.5%` /
`+6.9%`) at the very top, so its gate would run `2` microbenches instead of `16` and
catch both winners: **`14/16` experiments saved at `100%` live-recall, versus `0`
for `mechanism_prior` and `simple_family_heuristic`**.

Why it is real, not luck or leakage: a 2/2 top-rank hit on 16 candidates with 2
live is ~`0.8%` by chance; none of the `4` fresh tensors is in the corpus; and the
corpus carries a genuine observable interaction the mechanism-only prior ignores --
`row_upcast` wins `6/8` on `attn_q` but `0/4` on `ffn_gate`. The model learned the
(role x mechanism) interaction and generalized it to fresh attn_q tensors.

Honest caveats: only `2` live candidates in this batch, so the margin rests on one
specific learned pattern; a hand-coded (role x mechanism) prior would likely match
it -- the model is extracting an available feature interaction the *shipped*
baselines miss, not doing something unobtainable. Validate on larger/more diverse
batches before strong generalization claims. This is shadow only (no gate skipped a
real run). Per the exit gate the model has earned entry to Phase 5; keep the prior
as a fallback.

## Phase 4.2: Generalization Replication and Minimal-Gate Ablation

Purpose:

- Before letting any gate skip a real microbench in the live loop (Phase 5),
  confirm the 4.1 positive is not a 2-live-candidate fluke, and determine the
  SIMPLEST deterministic gate that captures the available signal. The flywheel
  should ship the cheapest gate that works, not the learned model by default.

What 4.1 left open:

- 4.1 beat `mechanism_prior`, but only `2` live candidates carried the result, both
  from one pattern (`attn_q` `row_upcast`). That is thin.
- The mechanism-only prior is a weak baseline. The grounded corpus win-structure
  shows the live signal is a (role x mechanism) interaction, not mechanism alone:
  `attn_q` x `row_upcast` is `75%` live and `attn_q` x `direct_output` is `42%`
  live, while every `ffn_gate` / `ffn_down` / `reduce_unroll` / `two_dim_local`
  cell is `0%` live. A trivial role x mechanism lookup already encodes this, so the
  4.1 model edge may evaporate against it. That is the real question.

Reframe -- minimal-gate ablation:

- Treat the gate as a ladder of increasing richness and find the simplest rung
  that meaningfully beats run-everything at `100%` live-recall:
  1. `run_all` (current loop, 0 savings).
  2. `mechanism_prior` (mechanism -> majority label).
  3. `role_mechanism_prior` (NEW: (role, mechanism) -> majority label, fall back to
     mechanism-only when the cell is empty).
  4. learned cost model.
- The flywheel adopts the lowest rung that works. Pre-register the expected and
  acceptable result: if `role_mechanism_prior` matches the model, ship the lookup
  and keep the model documentation-only -- that is a win for the flywheel (a cheap
  deterministic gate reduces wasted GPU), not a loss.

Fresh batch (bigger, multi-pattern, multi-block):

- Center on `attn_q` across many fresh blocks (e.g. `blk.3..blk.10.attn_q.weight`)
  so both winning patterns appear with `>=5` live candidates total: `row_upcast`
  (replicates 4.1) and `direct_output` (a second, distinct winning combo the model
  must also catch). Add `ffn_gate` fresh blocks as dead controls, and optionally
  `ffn_down` (Q6_K) as an all-dead new (role, format) region the gate must skip
  (note the Q6_K family/parts difference in the descriptor clone). Target `~30-40`
  candidates so the live count is not `2`.

Metric and per-pattern reporting:

- Same safe-skip metric: max microbench experiments a gate can skip while keeping
  every live candidate (skip below the lowest-scored live candidate).
- Report per (role x mechanism) cell, not just aggregate, so a single dominant
  pattern cannot mask a miss. A gate that catches `attn_q` `row_upcast` but drops
  every `attn_q` `direct_output` winner has failed generalization even if aggregate
  savings look fine.

Generalization checks:

- Does `attn_q` `row_upcast` keep winning on fresh blocks not in the corpus
  (replication)?
- Does the chosen gate also catch the `attn_q` `direct_output` winners (a second
  pattern), or did 4.1 only memorize one combo?
- Keep the freeze-before-outcomes hash discipline and the leak-free audit.

Implementation:

- Extend `STAGED_SCHEDULE_TENSORS` in `extra/qk_flywheel_shadow.py`; add a
  `role_mechanism_prior` baseline to the staged scorer; emit the per-cell breakdown
  in `shadow-staged-v2/summary.json`. Reuse the freeze protocol and safe-skip
  scorer unchanged.

Exit gate (authorizes Phase 5):

- `>= 5` live candidates across `>= 2` winning patterns (not inconclusive).
- The simplest sufficient gate saves a meaningful fraction of experiments at `100%`
  live-recall and replicates on fresh blocks per-pattern.
- Phase 5 then proceeds with whichever gate won the ablation (deterministic lookup
  preferred when it ties the model).

Failure modes (pre-registered, not bugs):

- `< 5` live -> inconclusive; enlarge the batch, do not re-roll to a lucky pass.
- Model ties `role_mechanism_prior` -> ship the lookup, model stays
  documentation-only.
- No gate beats run-all at full recall -> the signal does not generalize; stay in
  shadow and keep running everything.

Out of scope:

- Skipping any real run in the live loop (Phase 5). 14B/32B. New mechanism families
  beyond the four schedule mechanisms. Intra-identical-shape discrimination.

Result (2026-06-14): ran and recorded in `shadow-staged-v2/`. `40` fresh candidates,
keep/skip scores frozen before the microbench (commit `8844e160e`,
hash-verified unchanged, leak-free). Outcomes: `7` live (raw_accept) across `3`
patterns -- `attn_q` x `row_upcast` (`3`), `attn_q` x `direct_output` (`3`), and a
surprise `ffn_gate` x `row_upcast` (`1`); `12` tie, `1` reject, `20`
construction_blocked. Gate ladder, safe-skips at `100%` live-recall:
`run_all`=`0`, `mechanism_prior`=`0`, `role_mechanism_prior`=`0`, **cost model=`23`
(skips all `23` definitely-dead construction_blocked candidates while keeping every
winner)**. Pre-registered rule -> the model strictly beats the lookup, so it earns
Phase 5 entry.

But read the margin honestly. On the `6` expected `attn_q` winners the lookup and
model agree (both score them high). The entire `23`-vs-`0` gap comes from ONE
surprise: a fresh `ffn_gate` x `row_upcast` won despite `0/4` historical live in
that cell. Both priors scored that true winner at `0.0` -- the same score as the
dead candidates -- so to keep `100%` recall they could not skip anything (their
safe-skip floor collapsed to `0`). The model scored it `0.383`, above the dead
candidates, so it banked the obvious construction_blocked skips. The safe-skip
metric is hostage to the worst-ranked true winner (recorded per gate as
`floor_setter`), so a single surprise inflates the gap; a soft per-cell lookup
(`P(live)` = `0` for that cell) fails identically. The robust, qualitative finding:
the learned model generalizes "`row_upcast` can win" across roles and does not
catastrophically write off a surprise winner -- exactly the property a
don't-miss-a-winner gate needs -- but the `23`-vs-`0` magnitude should be replicated
before it is trusted. Phase 5 keeps the lookup as a cheap fallback and keeps
validating the robustness-to-surprise effect.

## Phase 4.3: Robustness Replication (shadow)

Purpose:

- Convert the 4.2 result from an interesting one-off into a trustworthy signal
  BEFORE any gate skips a real run. The 4.2 margin (`23` vs `0`) rested on a single
  surprise winner and the safe-skip metric is hostage to the worst-ranked true
  winner. 4.3 tests whether the model's edge -- not catastrophically writing off a
  surprise winner -- replicates across batches and survives a less brittle metric.

Hypothesis (pre-registered):

- The learned model keeps surprise winners that the role x mechanism lookup writes
  off, consistently across independent frozen batches, and its
  experiments-saved advantage persists when the all-or-nothing `100%`-recall
  constraint is relaxed.

Method:

- Run `K >= 3` more frozen batches (`shadow-staged-v3/`, `-v4/`, `-v5/`), each
  reusing the staged freeze protocol and leak-free path. Seed each with
  surprise-prone cells: (role x mechanism) combinations with thin or zero live
  history that can still win -- e.g. `ffn_gate` x `row_upcast` and `ffn_gate` x
  `direct_output` across many fresh blocks (blk.13..35), plus fresh `attn_q` blocks
  (the lookup's confident-live region, to check the model matches there too).
- Freeze keep/skip scores before each microbench; run; score the same gate ladder.

Metric upgrade (fixes the 4.2 brittleness):

- Add a recall-vs-savings curve: experiments saved at `100%`, `95%`, and `90%`
  live-recall, per gate, per batch. At `100%` one surprise winner dominates; at
  `95%` the typical value shows. Report the curve, not just the `100%` point.
- Pool across batches: in how many of `K` batches does the model save more than the
  lookup at each recall level; and the surprise-winner keep-rate (fraction of live
  candidates in low-historical-live cells that each gate's score would keep).
- Keep the `floor_setter` diagnostic so single-winner effects stay visible.

Exit gate (decides Phase 5's gate source):

- If the model saves more than the lookup in a majority of the `K` batches AND its
  advantage persists at `95%` recall (not only the brittle `100%` point), the model
  earns model-driven Phase 5.
- If the model only ties the lookup once pooled / recall-relaxed, the 4.2 win was a
  single-batch artifact: Phase 5 proceeds with the deterministic lookup and the
  model stays documentation-only.

Failure modes (pre-registered, not bugs):

- Report all `K` batches; do not drop or re-roll a batch to manufacture a majority.
- A batch with `< 5` live or `< 2` patterns is inconclusive for that batch and is
  enlarged, not discarded.

Out of scope:

- Skipping any real run (still Phase 5). 14B/32B. New mechanism families.

Result (2026-06-14): ran `3` frozen batches (`shadow-staged-v3/-v4/-v5`, `32`
candidates each, predictions committed in `8288ad28b` before the microbench,
hash-verified). Pooled: `96` candidates, `13` live. Under the pre-registered
safe-skip metric the model "won" `3/3` batches (`48` vs `0` pooled at `100%` and
`95%`) -- but that result is a metric artifact and is reported as such. Adding the
fair baseline that the pre-registration missed -- a deterministic class-skip gate that
skips the schedule classes which are `100%` construction_blocked in training
(`reduce_unroll` / `two_dim_local` / `ffn_gate` `vector_load`) -- it saves the SAME
`48` at `100%` live-recall with `0` missed winners. The model skips exactly those same
construction_blocked candidates and nothing more.

Conclusion: `deterministic_class_skip_matches_model_ship_the_lookup_model_adds_no_value`.
The `48`-vs-`0` advantage over `role_mechanism_prior` is a floor-collapse artifact of
the safe-skip metric: that metric penalizes a discrete gate for tying a surprise
winner with the dead mass, which is not a real cost. Against a fair deterministic gate
the learned model adds no value at the current feature set. This retroactively reframes
the 4.1/4.2 "model beats prior" results: those margins were largely the same artifact,
not evidence the model triages better than a cheap deterministic rule. **Phase 5's gate
source is therefore the deterministic class-skip gate, not the model; the model stays
documentation-only** unless a future feature set lets it strictly beat full-recall
determinism. This is a decisive, honest flywheel result -- the cheap deterministic gate
is the tool.

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

Gate source (decided by Phase 4.3 -> deterministic class-skip):

- 4.3 showed the learned model does not beat a fair deterministic gate, so this phase
  uses the deterministic class-skip gate: skip candidates in (role, mechanism) cells
  that are `100%` construction_blocked in training (known-broken schedule classes).
  No learned model in the live loop. Re-open model-driven gating only if a future
  feature set lets the model strictly beat full-recall determinism in shadow.

Method (constrained):

- Take the normal deterministic loop's static-pass candidate stream. Before the
  microbench, the gate marks each candidate keep or skip; the loop actually skips
  the skip-marked microbenches (the first real model influence on work).
- Start with the safest skip class only: `construction_blocked`-predicted candidates
  in cells with zero live history (in 4.2, `20/20` such candidates were genuine
  construction failures). Widen the skip class only after the audit (below) shows
  zero missed winners.

Safety rails:

- Conservative union: skip a candidate only if BOTH the model and the lookup agree
  it is dead; if either says keep, keep. This bounds recall risk to the better of
  the two gates.
- Never skip a candidate scored within a margin of the gate's own live floor.
- Freeze the keep/skip decision (and gate source) before outcomes, as in 4.1/4.2.

Audit (the load-bearing safety mechanism):

- Randomly sample a fraction of SKIPPED candidates and run them anyway (a shadow
  audit inside the live loop). This is the only way to measure the real missed-winner
  rate -- you cannot claim full recall in the live loop without spot-checking skips.
- Record experiments saved, GPU time saved per decisive outcome, and the audited
  missed-winner rate.

Exit gate (to Phase 6):

- Assisted ordering reduces wasted experiments by a meaningful margin AND the audit
  shows the missed-winner rate within a pre-declared tolerance (target: zero missed
  winners in the audit sample).

Stop rule:

- If the audit catches a missed winner above tolerance, revert to run-everything,
  keep the model documentation-only, and do not enter Phase 6.

Out of scope:

- Full runtime integration of a model-proposed kernel (Phase 6). 14B/32B. Any bypass
  of static/correctness/microbench/full-decode gates. Skipping outside the approved
  safe class.

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

## Phase M: Metric Re-base and Bottleneck Diagnosis (precursor to everything downstream)

This corrects the two errors G0 surfaced and gates all further triage/generation work.
The whole 3F-4.x line optimized a wall-clock metric dominated by ~0.27 ms launch
overhead, and G0 then searched ILP knobs (UPCAST/UNROLL) on a kernel that achieves only
**~19% of peak HBM bandwidth** (`~183` of `~960` GB/s on `attn_q`, a `~5.2x` roofline
gap). The kernel is NOT bandwidth-saturated, so there IS real headroom -- we measured the
wrong quantity and turned the wrong dials. Re-base the metric and find the real
bottleneck before chasing the headroom.

Why headroom is plausible (roofline, grounded): Q4_K decode GEMV has arithmetic
intensity `~14` ops/byte, LEFT of the RX7900XTX FP32 ridge (`~64` ops/byte), so it should
be bandwidth-bound and approach the roof -- yet it sits at `19%`. So the bandwidth itself
is not being saturated. Likely causes: poor/narrow memory access, low occupancy, or the
INT dequant path (nibble shift/mask + scale/min) being the true limiter -- the FP32 ridge
understates integer compute pressure, so the diagnosis must roofline the int path too.

### M0a: Metric re-base (deterministic)

- Confirm the roofline denominator: measure this GPU's actual achievable HBM bandwidth
  with a pure streaming-copy benchmark, rather than assuming the `960` GB/s datasheet
  value. The canonical metric becomes roofline-relative achieved bandwidth =
  `device_q4_eff` / measured_peak.
- Re-score the existing candidate space (the G0 grid and the 4.x schedule candidates) on
  this metric. Recompute every "gain" on device bandwidth.
- Re-audit the 4.x `raw_accept` labels honestly: were ANY of them genuine device-bandwidth
  improvements over `v1_partial`, or all wall-clock noise? Record the verdict.
- Fix the root cause: `qk_semantic_schedule_bench`'s `Q4_RESULT_RE` captures `q4_eff`
  (wall); switch the gain metric to `device_q4_eff`, and propagate the metric choice to
  the cost-model outcome labels if triage is revived.

### M0b: Bottleneck diagnosis (profiling)

- Profile `v1_partial` on `attn_q` to name what caps it at `19%`: dequant-compute-bound
  (INT unpack), load-width / coalescing-bound, or occupancy-bound. Use DEBUG=7 source +
  instruction mix (already used in 3G/G0), device counters / the tinygrad profiler, and a
  roofline placement that accounts for the INT dequant ops, not just FP32.
- Output: a named bottleneck and the search dimensions that address it.

### M0c: Redefine the real search space

- From the bottleneck, define the candidate axes that actually matter and DROP the ones
  G0 proved irrelevant (UPCAST/UNROLL). Examples by bottleneck: dequant-bound -> vectorized
  unpack, lookup-table dequant, fused scale/min, b128 packed loads; load-bound -> wider /
  coalesced global loads, vector dtypes; occupancy-bound -> `parts` / LDS / register
  pressure tuning.
- This redefined space + re-based metric is the input to a re-run headroom probe (G0').

Exit gate:

- A measured peak bandwidth; the re-based metric applied to the candidate space; the 4.x
  labels re-audited with an honest verdict; a named bottleneck; and a redefined search
  space. Only then do generation (G0'/G1/G2) or any revived triage resume.

Pre-registered honesty:

- If the re-based metric shows the 4.x wins were all noise, record it (the triage line
  optimized noise, confirmed).
- If the diagnosis shows most of the `5x` gap is irreducible (e.g. the kernel is near its
  own INT-dequant compute roof), record the realistic headroom -- it may be far below
  `5x`. Only a confirmed, addressable gap revives the flywheel target.

Result (2026-06-14, `extra/qk_metric_audit.py` + DEBUG=7 profiling, `metric-audit-m0/`):

- **M0a metric re-base.** Measured achievable peak = `859` GB/s (warm streaming copy, this
  GPU, 89% of the 960 datasheet). On the device metric `v1_partial` sits at `~20%` of peak
  on `attn_q` (4096x4096) and `~47%` on `ffn_gate` (12288x4096) -- real, **shape-dependent**
  headroom of `~5x` and `~2x`. Re-auditing a sample of `7` of the `22` distinct 4.x
  `raw_accept` configs on device: **`0` beat `v1_partial` by more than the `2%` noise band**
  (median device gain `-38.6%`; `row_upcast` is `-47..-51%`, `direct_output` is a tie). The
  entire 3F-4.x "win" signal was wall-clock noise -- confirmed. Root cause:
  `qk_semantic_schedule_bench` scored `q4_eff` (wall, overhead-dominated), not `device_q4_eff`.
- **M0b bottleneck.** The kernel already issues wide `b128` loads (`38` b128 vs `20` b32), so
  load width is NOT the cap. The body is dominated by the Q4_K dequant: `~3862` vector-ALU
  ops per kernel (`~55` per global load) -- integer nibble-unpack (shift/mask/cndmask/alignbit)
  + ubyte->fp32 conversion + scale/min. Both bandwidth (`20-47%`) and ALU utilization are low,
  so it is latency/occupancy-bound on the long dequant dependency chain; small matrices (less
  parallelism to hide it) sit lowest. Bottleneck = **Q4_K dequant compute + occupancy**.
- **M0c redefined search space.** Target the dequant: lookup-table 4-bit->fp dequant,
  bit-field-extract instead of shift+mask chains, vectorized multi-nibble unpack, fused
  scale/min; and raise occupancy (esp. small matrices). **Drop** the axes G0 proved
  irrelevant: UPCAST/UNROLL and wider loads (already b128). This redefined space + the device
  metric is the input to a re-run headroom probe (G0').

Net: the hypothesis is alive with a real `~2-5x` target and a named bottleneck. The 3F-4.x
triage/generation effort was aimed at the wrong metric (wall) and the wrong axes (ILP); the
real frontier is dequant-compute reduction. Next is G0' over the dequant/occupancy axes on
`device_q4_eff`.

## Phase G: Model-Proposed Candidate Generation Track

This is the primary path to the Phase 6 flywheel proof ("a model-proposed candidate
passes the normal deterministic gates through full decode and improves speed"). The
triage line (3F-4.3) pursued Phase 6's *alternative* proof (reduce wasted experiments)
and found only a modest deterministic win with no learned-model value; generation
pursues the *primary* proof and is the harder, higher-value half of the flywheel.

Why generation is safer than triage-skipping:

- Every proposed candidate runs through the SAME static + correctness + microbench +
  full-decode gates. A bad proposal just fails a gate -- bounded wasted GPU, never a
  wrong kernel and never a bypass. There is no recall risk like Phase 5's skip gate.
  The only cost is GPU on bad proposals, which is exactly what G1 measures.

Existing infrastructure to build on (do not rebuild):

- `extra/qk_candidate_generator.py`: the deterministic grid enumerator -- a SMALL fixed
  space (`parts` in `{1,2,4}`, `LOCAL` in `{32,64}`).
- `extra/qk_semantic_schedule.py`: the four schedule mechanisms with FIXED opt args
  (`UPCAST:0:2`, `UNROLL:2:4`, `LOCAL:1:4`) -- the generation frontier lives in the
  args and compositions the grid never tries.
- `extra/qk_ansor.py`: a roofline cost model (RX7900XTX mem GB/s, FP32 TFLOPS, ridge
  point) -- a candidate scorer for model-guided search.
- `extra/qk_ansor_transition_loop.py`: an Ansor-style prioritization loop.
- The committed static/correctness/microbench gates (q4_k_bench primitive path).

### G0: Search-space headroom probe (deterministic, shadow)

Purpose:

- Establish whether any frontier exists beyond the hardcoded grid BEFORE involving a
  model -- the honest precondition, mirroring how 4.x first probed whether triage had
  signal at all.

Method:

- Expand the parametric space on the live-bearing tensors (the `attn_q`
  `row_upcast` / `direct_output` region that produces accepts): `LOCAL` in
  `{16,32,64,128,256}`, a `parts` sweep, `UPCAST` / `UNROLL` arg sweeps `{2,4,8,16}`,
  and composed/multi-axis opts (e.g. `UPCAST:0:k` + `UNROLL:2:j`, `LOCAL:0`+`LOCAL:1`).
- Run every expanded candidate through the existing static + correctness + microbench
  gates. No model yet -- this is brute-force/grid search.

Metrics:

- Best device GB/s gain found versus `v1_partial` and versus the best of the four
  hardcoded mechanisms; GPU experiments spent; size of the winning region.

Exit gate (pre-registered):

- If no expanded candidate beats the hardcoded best, parametric generation has no
  headroom: stop the parametric track or jump to G2 (structural). Record it honestly --
  it means the deterministic enumeration is already near-optimal.
- If wins exist, quantify the frontier and the brute-force GPU cost. That cost is the
  baseline G1 must beat.

Result (2026-06-14, `extra/qk_generation_g0.py`, `generation-g0/`): **no parametric
headroom**, plus a more important metric finding. `168` correctness-gated GPU runs over
`28` candidates x `2` fresh `attn_q` tensors, scored on median **device** throughput
(`device_q4_eff`, from DEBUG=2 kernel timing). On both tensors the plain `v1_partial`
baseline (`LOCAL:0:64`) is best (`~183` GB/s); the closest expanded candidate is
`LOCAL:0:32` at `~173` (`-5.7%`), and `UPCAST` / `UNROLL` / multi-axis / `parts>1`
roughly halve device throughput (`row_upcast2` `~88` GB/s). No expanded candidate beats
the baseline. The deterministic `v1_partial` is already optimal in this opt space.

Critical metric finding: this contradicts the 4.x "raw_accept" wins. Those were scored
by `qk_semantic_schedule_bench` on WALL-clock `q4_eff` (`~28-35` GB/s, dominated by the
`~0.27` ms launch overhead), where a few-percent "gain" is noise. On the reliable device
metric the same schedules (e.g. `row_upcast2`) are dramatically slower, not faster. So
the live/win labels the whole 3F-4.x line optimized may have been measurement artifacts,
and the flywheel's "find a winning candidate" target may have no real targets in this
space. The next priority is therefore not G2 but a metric audit: re-score the candidate
space on `device_q4_eff` and re-check whether ANY candidate genuinely beats `v1_partial`
on device before spending more effort on generation or assist.

### G0': Device-metric search over existing kernel strategies and occupancy

Purpose:

- With the metric fixed (`device_q4_eff` / measured peak) and the bottleneck named (Q4_K
  dequant compute + occupancy), test whether any EXISTING kernel strategy or occupancy
  setting beats `v1_partial` on the device metric -- before committing to new dequant
  codegen. G0 only searched `partial`-mode ILP knobs on the wrong (wall) metric; the opt
  knobs do not touch the dequant, but the other `q4_k_bench` primitive MODES are genuinely
  different dequant/load kernels.

Method:

- Sweep the primitive modes (`serial`, `partial`, `packed_load`, `vector_load`, `grouped`,
  `tile_custom`) x occupancy knobs (`parts` in `{1,2,4}`, `LOCAL` in `{32,64,128}`,
  `row_group` for grouped) on the worst-case small matrix (`attn_q`, ~20% of peak) and a
  large one (`ffn_gate`, ~47%). Median `device_q4_eff` per candidate, correctness-gated,
  expressed as roofline fraction. Deterministic -- no model.

Exit gate (pre-registered):

- If some mode/setting beats `v1_partial` on device by more than the `2%` noise band at full
  correctness, real headroom exists via existing codegen: quantify it and hand that space to
  G1 (model-guided vs random).
- If none beats `v1_partial`, the existing kernels are all similarly dequant/occupancy-bound,
  and the bottleneck requires NEW dequant codegen (G0'': implement LUT 4-bit->fp,
  bit-field-extract, vectorized multi-nibble unpack, fused scale/min). That is a
  kernel-authoring task, not a search -- scope it separately and do not pretend a knob sweep
  can reach it.

Out of scope:

- `UPCAST`/`UNROLL` (G0 killed them on device); the wall-clock metric.

Result (2026-06-14, `extra/qk_generation_g0prime.py`, `generation-g0prime/`): a small but
**real, reproducible device win** -- and a clear pointer to the real work. Sweeping `6` modes
x `parts {1,2,4}` (`18` candidates) x `2` tensors on the device metric: `packed_load` (parts1)
is the ONLY strategy that beats `v1_partial` -- `+6.2%` on `attn_q` (`21.5% -> 22.8%` of peak,
confirmed across `5` seeds) and `+2.1%` on `ffn_gate` (`49% -> 50%`). Every other mode is
worse (serial/vector_load/grouped below baseline; `tile_custom` broken at `~4%`); `parts>1`
always hurts (split-k overhead, no occupancy gain). Notably `packed_load` is the 3G
`packed_word_lane_unroll` mechanism -- so 3G found a genuine (small) device win while the 4.x
schedule work was wall-clock noise.

But the win is marginal: even the best existing kernel reaches only `22.8%` (attn_q) / `50%`
(ffn_gate) of peak, leaving `~4.4x` / `~2x` residual. The `18`-candidate mode x parts space is
small and fully enumerated by brute force, so there is no role for model-guided search (G1):
G1's premise is a space too large to enumerate, which does not hold here. The residual
headroom is gated by the dequant bottleneck (M0b) and needs NEW dequant codegen.

Decision: adopt `packed_load` as the new device baseline (a free `+6%/+2%`), and proceed to
**G0''** -- author alternative Q4_K dequant kernels (LUT 4-bit->fp, bit-field-extract instead
of shift+mask chains, vectorized multi-nibble unpack, fused scale/min) and benchmark them on
the device metric. That is a kernel-authoring task, not a search; G1 (model-guided) only
becomes relevant once G0'' creates a large parametric dequant-variant space worth searching.

### G0'': New dequant codegen (kernel authoring)

Purpose:

- Close the residual `~2-5x` gap by attacking the named bottleneck (Q4_K dequant compute +
  occupancy) with new kernels, since the existing mode/opt space is exhausted. This is real
  kernel engineering with uncertain payoff, not a search; correctness is non-negotiable.

Grounding (where to intervene):

- Kernel builders: `extra/q4_k_gemv_primitive.py` -- `q4k_gemv_partial_kernel` (L308),
  `q4k_gemv_packed_load_partial_kernel` (L328, the current best). Dequant helpers:
  `_q4k_weight` (L42), `_q4k_quant` (L38, shift+mask nibble extract), `_q4k_group_params`
  (L20, the 6-bit scale/min decode). New variants register a `--primitive-mode` in
  `extra/q4_k_bench.py` (choices ~L61, dispatch ~L153) and add a kernel builder -- orthogonal
  to the existing ones.
- Key insight: the scale/min (`_q4k_group_params`) decode is re-computed inside the reduce.
  `packed_load` already cut it from `32x` to `8x` per group (part of its `+6%`), and M0b's
  `326` cndmask + `216` alignbit ops show it is still not fully hoisted. Hoisting it to `1x`
  per group is the clearest remaining win against exactly the ops that dominate.

Candidate variants (ranked by grounded expected value):

1. `hoist_scale_min`: packed_load but with `_q4k_group_params` computed ONCE per group and
   reused across all positions/lanes (lift it out of the lane4/pos reduce). Directly removes
   the redundant cndmask/alignbit decode. Highest expected value, most grounded.
2. `bfe_nibble`: replace `_q4k_quant`'s shift+mask with a `CUSTOMI` `v_bfe_u32` bit-field
   extract (the `vector_load` path at L70 shows the CUSTOMI pattern). Marginal -- shift+mask
   may already lower to bfe -- so verify the instruction mix actually changes before trusting
   any delta.
3. `lut_dequant`: per-sub-block 16-entry nibble->fp table (built once per group from
   `d*sc*q - dmin*mn`), then indexed per weight. Trades 16 decodes/group for a LUT build +
   LDS indexed loads; uncertain (LDS latency vs ALU).

Method:

- Implement each as a new primitive mode. Run through the EXISTING correctness gate
  (`primitive_gemv_correctness` must PASS, exact same numerics -- a faster wrong kernel is
  worthless) and the device microbench. Score median `device_q4_eff` / measured peak vs the
  `packed_load` baseline on `attn_q` (worst, ~23%) and `ffn_gate` (~50%). Capture the DEBUG=7
  instruction mix per variant to confirm the intended op-count reduction actually happened.

Exit gate (pre-registered):

- A variant that beats `packed_load` on device by more than the `2%` noise band at full
  correctness is real progress: adopt it as the new baseline and iterate (compose variants).
- If no variant beats `packed_load`, record the achievable limit honestly: the Q4_K dequant
  GEMV is near its practical ceiling for this approach on this hardware, and the remaining
  gap is either irreducible or needs a different attack (storage layout, a fused
  decode+matmul, or a different quant format) -- not more of the same.

Out of scope:

- Any numerics change that alters the dequant result (the correctness gate forbids it).
- The wall-clock metric. Model-guided search (G1) until a variant family creates a parametric
  space too large to enumerate.

Result (2026-06-14, iteration 1, `extra/q4_k_gemv_primitive.py` `q4k_gemv_hoist_partial_kernel`,
`generation-g0pp/`): the highest-value variant `hoist_scale_min` is **correct but a clear device
regression** -- `36.8` vs packed_load `195.7` GB/s on attn_q (`-81%`) and `93.5` vs `430.2` on
ffn_gate (`-78%`), exact numerics on both. The DEBUG=7 mix explains it: the kernel has MORE ALU
(`5150` vs `3862`) and MORE int-dequant ops (`1718` vs `846`), not fewer -- collapsing pos/lane4
into a full unroll to enable the algebraic factoring bloated the body and serialized the reduce.

Lesson: the bottleneck (M0b) is occupancy/latency, NOT redundant decode op-count; restructuring
the reduce to hoist the decode backfires because reduce parallelism dominates. ALU-op reduction is
the wrong lever -- which also down-weights the other scoped ALU-level variants (`bfe_nibble`,
`lut_dequant`); they were not pursued without a new hypothesis. Decision: `packed_load` remains the
best kernel and the adopted device baseline. The residual `~2-5x` is not reachable by dequant-ALU
restructuring; closing it needs a different attack -- a reduction/occupancy structure that adds
parallelism without serializing, a different storage layout, or a fused decode+matmul -- or is
substantially irreducible for this latency-bound GEMV. The `hoist_scale_min` mode is kept as a
documented, correct-but-slow result.

### G1: Model-guided search versus brute force

Purpose (only if G0 shows headroom):

- Test whether a model reaches the good candidates with FEWER GPU experiments than
  random/grid search -- the generation analog of "beat the dumb baseline".

Method:

- Fix a GPU budget. Baseline = random search over the expanded space. Compare, all
  budget-matched: (a) roofline-guided search using `qk_ansor` to propose high-roofline
  points; (b) a quick check of the learned cost model as a proposer scorer (4.x suggests
  it is weak -- verify); (c) an LLM proposing opt combinations from the descriptor +
  hardware context, with proposals frozen before running.
- Score sample-efficiency: GPU experiments to reach the best candidate, and the best
  candidate found at a fixed budget.

Exit gate (pre-registered):

- A model must beat random search on sample-efficiency-to-best. If it ties random,
  brute-force/random search is the tool and generation needs no learned model either --
  the same honest bar the triage line held. Freeze any learned/LLM proposals before
  outcomes; every candidate is gated.

### G2: Structural / novel-mechanism proposal (later)

- The model proposes schedule structures outside the parametric grid (new mechanism
  compositions, novel reduction structures). The real prize and the biggest lift;
  deferred until G0/G1 establish that parametric headroom and model sample-efficiency
  exist.

### Phase 6 connection

- A G0/G1 candidate that passes all gates through full decode and improves speed IS
  Phase 6's primary proof. Required artifacts are unchanged: frozen proposals before
  outcomes, the random/grid baseline comparison, deterministic gate outputs, the
  full-decode verdict, and a postmortem on whether the model added value over
  brute-force search.

Out of scope:

- Bypassing any gate; 14B/32B; correctness shortcuts; treating a microbench win as
  proof without full decode.

## Phase B: Batched Q4_K Matmul Modality (the weight-reuse lever)

The G0'' postmortem (primitive analysis) showed the batch-1 decode GEMV is
latency/occupancy-bound with ZERO weight reuse: each dequantized weight is used in
exactly one multiply, so the hardware idles on the load -> dequant -> accumulate chain
and reaches only `~20-47%` of bandwidth. The structural lever is REUSE via batching:
process `B > 1` tokens at once so the operation becomes `W[M,K] . X[K,B]` (a GEMM), each
dequantized weight is reused `B` times, the dequant cost amortizes `B`-fold, and the op
transitions from memory/latency-bound to compute-bound. This is the "different attack"
G0'' pointed to -- not more dequant-ALU tuning.

Documenting the modality (when it applies):

- Prefill (the whole prompt is `B = prompt_len` rows at once).
- Batched serving (several concurrent sequences decode together; `B = batch`).
- Speculative / Medusa decode (multiple candidate tokens verified per step; `B = k`).
- It does NOT apply to single-stream greedy decode, which is irreducibly `B = 1`. So this
  modality raises THROUGHPUT (tokens/sec across a batch) and prefill speed, not the
  per-token LATENCY of one isolated stream -- state that honestly; do not oversell it as
  a decode-latency win.

### B0: Batch-size efficiency curve (runnable now)

Purpose:

- Quantify the amortization: does per-token efficiency climb with `B`, and how far toward
  the compute roof, before deciding whether a fused kernel (B1) is needed.

Method:

- Sweep `--seq-len` in `{1,2,4,8,16,32,64,128}` (q4_k_bench already supports it; `--primitive`
  is batch-1 only, so use the matmul paths). Measure device time for `decode_q4_k_plus_matmul`
  (fused dequant+matmul = the real quantized GEMM) and `matmul_decoded` (weights pre-dequantized
  to fp16 then dense matmul = the compute ceiling if dequant were free), on a small tensor
  (attn_q) and a large one (ffn_gate).
- Compute per-token device latency (`time/B`), achieved FLOPS (`2*M*K*B/time`) as a fraction
  of the measured fp16 compute roof, and the per-token speedup versus `B=1`. Locate the
  crossover batch where the op stops being weight-memory-bound and becomes compute-bound.

Exit gate (pre-registered):

- If per-token efficiency climbs steeply with `B` and the fused path approaches the dense-fp16
  ceiling, batching is the confirmed lever and the gain is quantified -- document the curve and
  the crossover batch as the practical guidance.
- If the fused `decode_q4_k_plus_matmul` path stays far below the `matmul_decoded` dense ceiling
  even at large `B`, the dequant is not amortizing well (likely the fp16 materialization
  round-trip), which motivates B1.

Result (2026-06-14, `extra/qk_batched_b0.py`, `batched-b0/`): batching is a **large, confirmed
lever -- and B1 is motivated**. Sweeping `B in {1..128}` (measured fp16 compute peak `83.6` TFLOPS):
per-token device latency drops **`26x` on attn_q** (`622 -> 24` us/token) and **`13x` on ffn_gate**
(`354 -> 26`) from `B=1` to `B=128` -- the dequant amortizes exactly as the primitive analysis
predicted. BUT the fused quantized path stays far below the dense-fp16 ceiling at the largest batch:
fused is only `17%` (attn_q) / `25%` (ffn_gate) of `matmul_decoded` throughput. Even the dense matmul
reaches only `10%` / `19%` of compute peak (small, untuned tinygrad GEMM), so there is headroom on
both axes. (The `B=4` point is a noisy outlier; the verdict uses the fused-vs-dense ratio at the
largest batch.)

Correction (2026-06-14, from grounding the B1 scope): the fused path does NOT do an fp16
round-trip. `decode_q4_k_plus_matmul` is already a single fused kernel (`kernels=1.0`,
`mem=9.57MB` = the compressed Q4_K weights, vs `matmul_decoded`'s `33.69MB` fp16). Its slowness is
**poor tiling**, not materialization: tinygrad's general matmul codegen with the inline dequant
produces a non-GEMM-optimal kernel (~351 GFLOPS, ~4% of peak). So B1 is not "avoid the round-trip"
(there is none) -- it is "tile the fused kernel like a real GEMM" (register-blocked output tile,
dequant each weight tile once into registers, reuse across the B-column tile).

### B1: Well-tiled fused Q4_K GEMM (kernel authoring)

Purpose:

- The fused dequant+matmul already exists and is already memory-light (`mem=9.57MB`, one kernel);
  it is just badly tiled (~`4%` of compute peak). B1 authors a GEMM-tiled fused kernel that
  closes the gap to `matmul_decoded` (the dense baseline, itself only `~18%` of peak) and then
  pushes both toward the `83.6` TFLOPS roof.

B1a -- characterize (largely answered by the B1-scope grounding):

- `decode_q4_k_plus_matmul` is a single fused kernel reading compressed weights (no fp16
  materialization). Open question to confirm before authoring: does the fused kernel dequantize
  each weight tile ONCE and reuse it across the `B` activation columns, or re-dequantize per
  column? Read the generated kernel (DEBUG=7) for the `B>1` shape and check whether the dequant
  ALU scales with `B` (re-decode) or is hoisted per weight-tile (reuse). That determines whether
  the win is tiling alone or tiling + dequant-reuse.

B1b -- author the tiled kernel:

- A register-blocked fused Q4_K GEMM: tile the output over `M x B`, stage a weight tile + the
  `B`-column activation tile, dequant each weight element once into registers, and accumulate
  across `K`. New primitive path in `extra/q4_k_gemv_primitive.py` + a `--seq-len>1` primitive
  mode in `extra/q4_k_bench.py` (currently `--primitive` is batch-1 only, L72). Correctness-gated
  (exact numerics) and measured as achieved FLOPS / `83.6` TFLOPS across a batch sweep.
- Heed the G0'' lesson: do NOT serialize the reduction or over-unroll. The win must come from
  GEMM tiling/reuse that preserves parallelism, not from cutting ALU at the cost of occupancy.

Metric:

- Per-token device latency and achieved FLOPS / measured fp16 compute roof (measure the compute
  peak directly, as Phase M measured the bandwidth peak). The roofline denominator shifts from
  memory bandwidth (`B=1`) to compute (large `B`); B0 locates the transition.

Exit gate (pre-registered):

- A tiled kernel that beats `decode_q4_k_plus_matmul` (and ideally `matmul_decoded`) on device
  FLOPS at full correctness is real progress -- adopt it and report how close to the roof. If a
  well-tiled fused kernel cannot beat the existing fused path, record that tinygrad's matmul
  codegen is the ceiling here and the remaining headroom needs lower-level work (custom
  WMMA/MFMA, a different framework) or is not worth it.

Connection to the learned model:

- B1b's tiling space (`M x B x K` tile sizes, LDS staging, dequant placement) is the first
  parametric space in the program large enough that model-guided search (G1) might earn its keep
  on a real, correctly-measured target -- the only honest place to revive the flywheel question.

Result (2026-06-14, `extra/q4_k_gemv_primitive.py` `q4k_gemm_packed_load_kernel` +
`extra/qk_gemm_b1.py`, `gemm-b1/`): a **real win at small batch**. The fused GEMM extends
packed_load with an `UPCAST`'d `B` axis so each dequantized weight is reused across the `B`
activation columns; it is correctness-gated (exact numerics, `rel_err < 1e-6`) and reads the
compressed Q4_K weights. Device-timed vs the fp16 dense matmul (`matmul_decoded` ceiling) on
attn_q + ffn_gate:

- `B=4`: GEMM beats fp16 dense **`3.7x` (ffn_gate) / `5.1x` (attn_q)**.
- `B=8`: GEMM beats fp16 dense **`1.8x` / `1.9x`**.
- `B>=16`: dense wins (GEMM plateaus at `~4.6-6%` of the `83.6` TFLOPS peak; dense climbs to
  `~15%`). Crossover `~B=12`.

So the fused GEMM is the right kernel for the **small-batch regime (`B<=8`: speculative/Medusa
decode, small serving batches)** -- memory-light and faster -- and the first hand-authored kernel
in the program to beat a real baseline at full correctness. It is a GEMV-derived kernel (B-unroll,
`UPCAST` capped at 16) and plateaus; beating tinygrad's matmul at large `B` needs a register-blocked
GEMM (2D `M x B` output tiling, LDS staging), a bigger lift where the model-guided tiling search
above could finally earn its keep. Adopt the fused GEMM for `B<=8`; use `matmul_decoded` (or a
future tiled GEMM) for large `B`.

Out of scope:

- Any numerics change; single-stream greedy decode (document it requires a batching source);
  the wall-clock metric.

## Phase W: Search-Competitive Fused Q4_K GEMM (machine search vs llama.cpp)

Goal (the actual program goal, restated): a MACHINE SEARCH that reaches llama.cpp-class Q4_K
matmul performance WITHOUT per-kernel hand-tuning. Not "a fast kernel" and not "a model that
invents kernels" -- the Ansor/AutoTVM model: a human defines a parametric template ONCE; search
(later cost-model-guided) tunes it across shapes/hardware. Everything before showed this ordering
is mandatory: the current kernel templates top out at `~20%` of peak (the opt search over them
found only `packed_load` +6%), so no search inside them can reach llama.cpp. The
fused-dequant->WMMA structure is the prerequisite for the search space to even CONTAIN a
competitive point. The deterministic generated policy is currently `61.6%` of llama.cpp (14B,
`current-verdicts`); this phase is about closing that gap by search over a competitive template.

Why this is the honest revival of the learned model: the flywheel died predicting
weight-determined, UNOBSERVABLE outcomes (3F-4.x). A cost model ranking TILE CONFIGS is the Ansor
role on OBSERVABLE features (tile sizes, FLOP/byte, occupancy, register pressure) -- the one job it
was ever suited for.

### W0: Establish the bar (make "competitive" a number)

- Measure llama.cpp Q4_K decode + prefill throughput on this GPU (end-to-end tok/s; the repo
  already references it -- the 14B generated policy is `61.6%` of llama.cpp). Translate to a
  kernel-level target: the roofline % (device bandwidth at `B=1`, compute at large `B`) the
  dominant matmuls must reach to close the gap.
- Pre-register the success bar, e.g. close 14B from `61.6%` toward `>=90%` of llama.cpp end-to-end,
  with the dominant matmuls hitting a target roofline %.
- Honest framing: matching a heavily hand-tuned kernel by search is hard; the win is "competitive
  ACROSS shapes/hardware without hand-tuning each," not "beat one Marlin kernel."

Result (2026-06-15): llama.cpp does `103.84` tok/s on 8B Q4_K decode (llama-bench tg64, ROCm) on
this GPU. Our deterministic generated policy is `~52` tok/s = `~50%`, so the bar is **close a ~2x
gap**.

### W1: Close the primitive gap -- tile-level fused dequant -> WMMA (the gate)

- tinygrad's WMMA matcher does not fire on a dequant expression (B0: the fused kernel is a `~4%`
  scalar reduce; only full-materialized fp16 -> WMMA reaches `~18%`). The Marlin trick: materialize
  at the TILE level, not the tensor level -- dequant a weight tile to fp16 in LDS/registers, then
  WMMA against the activation tile and accumulate; the compressed weights stay in DRAM (no `2x`
  memory).
- Two routes: (a) coax tinygrad to emit WMMA by realizing the dequant into a small fp16 LDS tile
  the matmul matcher recognizes; (b) hand-author WMMA intrinsics with inline dequant if (a) fails.
- Correctness-gated (exact numerics). Pre-registered: a fused-WMMA kernel that beats `matmul_decoded`
  while reading compressed weights -> primitive closed. If WMMA cannot be coaxed in tinygrad's model
  -> record it as a tinygrad-capability blocker; the path then needs lower-level intrinsics or a
  different framework, and W3/W4 do NOT run over a template that cannot contain a competitive point.
- Heed the G0'' lesson: win via tiling/reuse that PRESERVES parallelism; do not serialize.

Result (2026-06-15, `extra/qk_wmma_w1.py`, `wmma-w1/`): the gate is **open at the capability level,
closed at the performance level.** Forcing tensor cores (`TC_OPT=2`) makes tinygrad emit WMMA on the
FUSED dequant matmul -- the matcher only requires both MUL operands be fp16, which the
dequant-cast-to-f16 satisfies (`0` WMMA by default, `145` when forced). The resulting kernel is
**correct (exact numerics), uses matrix cores, and reads the COMPRESSED weights** (`~10` MB for
attn_q / `~30` MB for ffn_gate vs `~34` / `~103` MB materialized -- no fp16 round-trip). BUT it is
**`13-28x` slower** than the materialized-fp16 WMMA (`0.25-1.4%` vs `7-19%` of peak), because
tinygrad **recomputes the dequant inside the WMMA tiling instead of staging the dequantized tile
once in LDS** (the Marlin trick) -- forced-TC does not auto-stage a computed intermediate.

Gate decision: the capability exists (not a hard instruction-level wall), but the naive template is
not competitive, and **no autotuning search (W3/W4) fixes a per-tile dequant recompute** -- so W2-W4
do NOT run over it. A competitive template needs a hand-authored Marlin-class LDS-staged fused-WMMA
kernel (the real lift), or accept materialized-fp16 (`matmul_decoded`) for the compute-bound
large-batch regime and the small-batch fused GEMM (B1b) for the memory-bound regime. This sharpens
the "primitives to scale" answer: we have WMMA and can fuse it with dequant (correct, compressed),
but lack **automatic dequant-tile-staging** -- exactly the one primitive Marlin hand-writes.

### W1b: Marlin-class LDS-staged fused-WMMA kernel (the real lift)

Purpose:

- W1 showed forced-TC emits WMMA on the fused dequant (correct, compressed) but is `13-28x` slow
  because the dequant is recomputed inside the WMMA tiling. W1b closes that by **staging the
  dequantized weight tile in LDS once and reusing it across the WMMA ops** -- the Marlin structure.

Grounding -- a working hand-WMMA+LDS skeleton already exists in-repo (this de-risks W1b from
"uncharted" to "fork and swap"):

- `extra/gemm/amd_copy_matmul.py` (`WMMA=1`) is a hand-authored `128x128` tiled fp16 GEMM that:
  stages the A and B tiles from global into LDS via `UOp.placeholder(..., addrspace=AddrSpace.LOCAL)`
  stores (L49-60), `UOp.barrier(A_store, B_store)` (L61), then runs a **hand-placed**
  `Ops.SHAPED_WMMA` (L83, `arg=((16,16,16),'AMD',32)`) over the LDS-staged tiles, reused across the
  block's WMMA ops. `extra/gemm/amd_uop_matmul.py::eval_custom_matmul` is the run+correctness
  harness (MSE-gated vs `a.float()@b.float()`, prints REAL TFLOPS).
- Because WMMA is placed EXPLICITLY (`Ops.SHAPED_WMMA`), W1b does NOT depend on the forced-TC
  matcher firing on a dequant expression (the fragile W1 path). The dequant is staged into LDS once
  per block and the WMMA reads the LDS fp16 tile -- exactly the Marlin structure, with the
  per-tile-recompute (the W1 28x) structurally impossible.
- The crux line is L59: `A_store = ... .store(a[k_tile]...)` stores a global fp16 weight tile into
  LDS. W1b replaces `a[k_tile]` (plain fp16 load) with the **Q4_K dequant** of the compressed weight
  tile, reusing the `_q4k_*` helpers in `extra/q4_k_gemv_primitive.py`. A (the weight) is compressed;
  B (the activation) stays fp16. Everything downstream of the LDS store is unchanged.

W1b.0 -- validate the skeleton on THIS GPU (make-or-break, cheap, no new code):

- Run `WMMA=1 N=4096 PYTHONPATH=. .venv/bin/python extra/gemm/amd_copy_matmul.py` on `DEV=AMD`
  (RX7900XTX / gfx1100, RDNA3 so `is_rdna4=False`, `UNROLL_M=1`). Gate: MSE passes (correct) AND the
  generated source contains a WMMA intrinsic (`__builtin_amdgcn_wmma_*`) AND REAL TFLOPS is a sane
  fraction of the `83.6` peak. If the skeleton does NOT run/verify on gfx1100 (e.g. the WMMA path is
  MI3xx/RDNA4-shaped), that is the framework signal: either fix the skeleton for gfx1100 or fall to
  HIP/rocWMMA. Do not fork until the skeleton is green here.

W1b.1 -- fork + swap the A-tile load for a Q4_K dequant (`extra/qk_marlin_w1b.py`):

- Copy `block_128x128_gemm` / `amd_copy_matmul`. Keep B (activation) fp16. Pass A as the COMPRESSED
  Q4_K words buffer; in the A_store, compute the dequanted fp16 tile element(s) each thread owns by
  mapping its `(k_tile, BLOCK_K, BLOCK_M)` LDS coordinate back to Q4_K `(row, block, group, pos)` and
  calling the existing dequant (`_q4k_weight` / `_q4k_group_params` + `_q4k_quant`). `BLOCK_K=16`
  spans half a `Q4_K_BLOCK_ELEMS=256` super-block sub-structure -- get the index math exact (this is
  the real work). Correctness-gate against `q4_k_reference` (exact numerics), NOT a random fp16 A.

W1b.2 -- correctness + measure:

- Exact numerics (correctness-gated, `rel_err < 1e-2`). Device REAL TFLOPS / `83.6` peak vs the
  materialized-fp16 WMMA ceiling (the same skeleton run with a pre-dequanted fp16 A == `matmul_decoded`)
  and the W0 llama.cpp bar (`103.84` tok/s). Pre-registered: the staged kernel reads compressed
  weights AND lands within ~`10-20%` of the materialized-fp16 ceiling -> the competitive fused-WMMA
  primitive is built; W2-W4 (parametrize, autotune, cost-model search) now have a template that can
  contain a competitive point. If the dequant-in-store tanks throughput (e.g. the dequant ALU per
  store-thread re-bottlenecks) even with single-stage reuse -> record the ceiling honestly and decide
  regime split vs lower-level.

Risk: the LDS+WMMA plumbing is proven (the skeleton exists and -- pending W1b.0 -- runs on this GPU),
so the residual risk is concentrated in ONE place: the Q4_K-coordinate-to-LDS-tile index math in the
A_store, and whether per-store-thread dequant ALU is cheap enough not to re-bottleneck. That is real
but bounded engineering, not a framework unknown. Heed G0''/W1: stage once, do not serialize.

**W1b.0 -- Result (2026-06-15): the hand-SHAPED_WMMA skeleton is STALE against this fork's tinygrad;
fork-the-skeleton path is BLOCKED.** Findings, in order:

- The LDS+barrier plumbing itself works here: `amd_copy_matmul.py` *non-WMMA* path runs and verifies
  (MSE `0.0`, ~2 TFLOPS untuned at `N=512`). So `DEFINE_LOCAL` + `barrier` + `custom_kernel` are in
  sync -- the Marlin LDS-staging mechanism is expressible.
- But ALL FOUR in-repo hand-placed `Ops.SHAPED_WMMA` UOp kernels fail on this fork's tinygrad, each
  with a different drift error: `amd_copy_matmul` (`AFTER` wrapping `INDEX`, then `SHAPED_WMMA`
  un-lowered with ptr srcs), `amd_flash_attention` (`MAX` None-shape), `amd_uop_matmul`
  (`sint_to_uop`), `mi350x_uop_matmul` (reshape `(16,4)->(4,4)`). These are vendored from upstream
  `48a7627b0` (2026-04-09); this fork's tinygrad has since diverged.
- Two repair layers in: (1) fixed -- current spec (`spec.py:86`) forbids `AFTER` wrapping `INDEX`, so
  `.after(k)` must wrap the bare `DEFINE_REG` acc, not the indexed frag; (2) blocked -- `SHAPED_WMMA`
  then reaches `type_verify` UN-lowered with pointer sources (`half.ptr(2048,2)`). There is NO
  `SHAPED_WMMA` spec rule (only the lowered `Ops.WMMA`, `spec.py:113-114`), and this fork's
  `lower_shaped_wmma` (`rangeify.py:25-34`) did not fire on the upstream frag construction -- the
  upstream fragment-indexing convention is incompatible with this fork's lowering contract. Repair is
  unbounded reverse-engineering, not a one-liner; stopped per the pre-registered cascade rule.
- Crucially, the OTHER WMMA path -- **forced-TC** -- DOES verify and run end-to-end on this tinygrad:
  W1 emitted 145 correct WMMA ops via `TC_OPT=2`. So WMMA codegen works; the open question is staging
  the dequant in LDS WITHOUT hand-placed SHAPED_WMMA.

Route reassessment (the scoped "fork the skeleton" plan W1b.1 is dead as written). Open routes:
- **(a) Repair hand-SHAPED_WMMA** to this fork's `lower_shaped_wmma` contract, then fork for dequant.
  Surest expression of LDS-staged dequant->WMMA, but unbounded framework reverse-engineering with no
  in-repo green reference to copy.
- **(b) Forced-TC + opts staging** (recommended): build on the W1 path that already works; investigate
  whether a `GROUP`/`LOCAL` opt forces tinygrad to stage the dequanted tile in LDS once (killing the
  W1 per-MAC recompute) instead of hand-placing WMMA. Lower framework risk; uncertain whether the opt
  machinery will stage a COMPUTED value (it normally stages global loads).
- **(c) Drop to AMD assembly** (`test/amd/test_custom_kernel.py` shows raw `v_wmma_f32_16x16x16_f16`
  works): max control, max effort, least leverage on the search goal.

### W1b' -- merged routes (a)+(b): TC-opt over a hand-LDS-staged dequant (decision 2026-06-15)

Pursue (a) and (b) together. Two more framework facts (found grounding the route choice) collapse
them into one well-grounded plan and kill the naive tactics:

- **How this fork actually builds WMMA:** NOT by hand-placing `SHAPED_WMMA` (the stale upstream
  skeleton). The TC opt `_apply_tc_opt` (`postrange.py:219+`) tags a normal `REDUCE`-of-`MUL` and
  shifts axes into `LOCAL/UPCAST/UNROLL`; it requires both MUL operands' scalar dtype `== tc.dtype_in`
  (fp16, `L232`). This IS the path W1 used (145 correct WMMA ops). So the only green WMMA construction
  on this fork is **TC-opt-over-a-reduce**, and the right move is to feed it the right reduce, not to
  resurrect `SHAPED_WMMA`.
- **TC is requestable from a custom kernel:** `Opt(OptOps.TC, axis, (tc_select, tc_opt, use_tc))`
  (e.g. `(-1, 2, 1)`, `search.py:22`) can go in a custom kernel's `KernelInfo(opts_to_apply=...)`.
- **GROUP/GROUPTOP are FORBIDDEN with TC** (`postrange.py:173`: "no grouping with tensor cores"). This
  kills the naive (b) tactic (a GROUP opt to stage the dequanted tile alongside TC) AND explains the
  W1 28x: TC manages its own operand staging and will not let you inject LDS staging of a COMPUTED
  operand via GROUP -- so it recomputes the dequant per MAC.

Merged structure -- one shared diagnosis, then the fast falsifier (b) and the real build (a):

**Track 0 -- shared diagnosis (cheap, first).** Dump W1's generated source (`DEBUG>=4` / VIZ) and
*empirically confirm* the 28x mechanism: is the Q4_K dequant recomputed inside the WMMA/reduce loop
(hypothesis, currently unverified)? Read how TC-opt lays out the WMMA + its operand LDS staging, so
Track A knows the exact reduce/operand shapes that make TC fire. Output: a one-page note in the
`wmma-w1b/` artifact dir.

**Track B -- fast falsifier (hours, second).** Given GROUP+TC is forbidden, test the only remaining
(b) levers on the W1 fused reduce, via a custom-kernel matmul with explicit `opts_to_apply`: (i) a
`LOCAL` opt on the spatial axis, (ii) a `.contiguous()`/realize boundary on the dequant operand
inside the kernel. Gate: dequant staged ONCE (visible in source) AND device time approaches
`matmul_decoded`. Pre-registered: if no non-GROUP opt stages the computed dequant (the expected
outcome, since TC owns its staging) -> (b) falsified, recorded, all-in on (a). This is a deliberate
cheap attempt to kill the easy path before investing in the hard one.

**Track A -- the real Marlin kernel (the main build), `extra/qk_marlin_w1b.py`.** Hand-author a
custom kernel that stages the dequant in LDS once and lets TC build the WMMA over the LDS load:
- a.0 -- make-or-break sub-gate (NO dequant yet): `DEFINE_LOCAL` an fp16 weight tile, COPY it from a
  global fp16 weight (plain load), `barrier()`, then a normal matmul reduce `acc += Wlds[m,k]*x[b,k]`
  over the tile with `opts_to_apply=(Opt(OptOps.TC, axis, (-1,2,1)), ...)`. Gate: TC FIRES (WMMA in
  source) and result is correct. This tests the one load-bearing unknown -- *does the TC matcher
  accept a MUL operand that is a load from a DEFINE_LOCAL buffer written earlier in the same kernel?*
  If TC refuses an LDS-staged operand (e.g. it requires the operand trace back to a global PARAM),
  that is the deepest framework signal -> escalate to (c) assembly or (d) regime split. Do NOT add
  the dequant until a.0 is green.
- a.1 -- swap the LDS store's source from the global-fp16 load to the Q4_K **dequant** of the
  compressed tile (reuse `_q4k_weight`/`_q4k_group_params`/`_q4k_quant` from
  `extra/q4_k_gemv_primitive.py`). The dequant runs once per tile (the store, behind the barrier);
  the TC reduce reads plain fp16 LDS loads, so the per-MAC recompute is structurally impossible. Crux:
  the Q4_K-coordinate-to-LDS-tile index math (map each store-thread's `(m,k)` LDS slot to Q4_K
  `(row, block, group, pos)`). Correctness-gate vs `q4_k_reference` (exact numerics, `rel_err<1e-2`).
- a.2 -- correctness + measure: device REAL TFLOPS / `83.6` peak vs the materialized-fp16 ceiling
  (same kernel, pre-dequanted fp16 weight == `matmul_decoded`) and the W0 bar (`103.84` tok/s).
  Pre-registered: reads compressed AND lands within ~`10-20%` of the fp16 ceiling -> competitive
  fused-WMMA primitive built, W2-W4 proceed. Tanks even with single-stage reuse (per-store-thread
  dequant ALU re-bottlenecks) -> record the ceiling, decide regime split vs lower-level.

Sequencing: Track 0 -> Track B (fast, may falsify the easy path) -> Track A (a.0 gate -> a.1 -> a.2).
Artifacts under `bench/amd-decode-flywheel-proof-20260614/wmma-w1b/`; test
`test/external/test_qk_marlin_w1b.py`. Heed G0''/W1: stage once, do not serialize.

**W1b' -- RESULT (2026-06-15): the Marlin fused-dequant->WMMA primitive WORKS. Gate OPEN.**
Built bottom-up in `extra/qk_marlin_w1b.py` (`RESULT.md` + `summary.json` in `wmma-w1b/`):
- Track 0 confirmed the W1 28x mechanism from the rendered ISA: the slow kernel reads compressed and
  computes the Q4_K dequant INLINE feeding each `__WMMA_16_16_16_half_float` (recompute confirmed).
- a0a: `Opt(OptOps.TC, 0, (-1,2,1))` fires WMMA on a hand-built `Ops.REDUCE` matmul (correct). KEY:
  the q4k `.set/.after/.end` manual accumulator is NOT an `Ops.REDUCE`; `mul.reduce(k, arg=Ops.ADD,
  dtype=float32)` is (per `cdna_asm_gemm.py::custom_uop_gemm`). TC needs a real REDUCE.
- a0b (make-or-break): TC fires WMMA on a MUL operand loaded from a `DEFINE_LOCAL` written earlier in
  the SAME kernel. The Marlin structure IS expressible on this fork -- the load-bearing unknown is YES.
- a1: full Marlin -- dequant the compressed tile ONCE into LDS -> barrier -> WMMA. Correct on real
  GGUF weights (rel_err 1e-4). Rendered source verified: ALL dequant shifts pre-barrier, ALL WMMA
  post-barrier -- the per-MAC recompute is structurally gone.
- a2: fusing the dequant is ~FREE -- the fused kernel (reads compressed) is 1.07-1.08x FASTER than the
  materialized-fp16 WMMA ceiling on 4/5 shapes (0.89x on one large-N), mean 1.04x, all correct.

Caveat (honest): absolute throughput is tiny (0.04-0.23 TFLOPS) -- these are single-workgroup,
un-tiled, whole-tile-in-LDS shapes (M<=32, K<=1024 to fit the ~64KB LDS). W1b proves the PRIMITIVE
(correct, reads compressed, tensor cores, dequant-once, competitive with the fp16 ceiling); reaching
the 83.6 peak / 103.84 tok/s bar is the W2->W3 job (grid parallelism + K-tiling + occupancy over a
parametrized template). The template that can contain a competitive point now EXISTS.

### W2: Parametrize the template

Goal: turn the single-workgroup W1b' Marlin kernel into a `config -> kernel` template that reaches
real throughput at production shapes (e.g. `4096x4096`, batch 32-512), measured vs the
`matmul_decoded` fp16 ceiling and the `83.6` peak / `103.84` tok/s bar. The fusion is already proven
~free (W1b' a2), so W2 tunes tiling + parallelism + occupancy, NOT the dequant.

Two structural levers, sequenced by risk (cheap+safe first):

W2.0 -- **grid parallelism** (the big easy win; the W1b' kernel was tiny because it was ONE workgroup).
- Block the output `[M,N]` into `BLOCK_M x BLOCK_N` tiles, one workgroup per tile, via `AxisType.GLOBAL`
  ranges (`block_m`, `block_n`) like `amd_copy_matmul`. Each workgroup runs the proven W1b' body on
  its tile. Keep whole-K-in-LDS for now (so `BLOCK_M * K * 2 <= ~64KB`, e.g. `BLOCK_M=16, K<=2048`).
- Gate: correct, WMMA still fires, and absolute TFLOPS scales ~linearly with workgroup count toward a
  meaningful fraction of peak. This alone should move TFLOPS by 1-2 orders of magnitude.

W2.1 -- **K-tiling** (mandatory for real `K=4096`: a `16x4096` fp16 tile is 128KB > 64KB LDS).
- THE RISK: K-tiling means a per-K-tile loop (dequant tile -> LDS -> barrier -> accumulate -> next),
  but TC needs ONE `Ops.REDUCE` over the staged operand, and a manual outer K-loop with a hand
  accumulator is NOT a single REDUCE (the W1b' a0a lesson), while `GROUP`/`GROUPTOP` (the natural
  LDS-partial-reduce opt) is FORBIDDEN with TC (`postrange.py:173`). So the one-workgroup-K-loop +
  TC composition is an OPEN question -- a make-or-break sub-gate (W2.1a), tested empirically.
- If W2.1a composes (TC fires on the inner per-tile reduce inside the K-loop): build the full
  K-tiled template. If it does NOT: fall back to **split-K** (W2.1b) -- grid over `(block_m, block_n,
  k_block)` where each workgroup is EXACTLY the proven W1b' whole-`BLOCK_K`-in-LDS kernel producing a
  partial, then a cheap second kernel sums the partials over `k_block`. Split-K needs no TC/K-loop
  composition (each workgroup is the green W1b' primitive) at the cost of a partial-reduction pass.
  Pre-registered: pick whichever of W2.1a/W2.1b reaches higher device throughput; record both.

W2.2 -- **measure + parametrize**: expose `(BLOCK_M, BLOCK_N, BLOCK_K, split_k, waves, wmma_tile)` as
the config. Measure REAL TFLOPS / `83.6` peak and end-to-end-ish tok/s vs `matmul_decoded` and the
`103.84` bar across representative shapes. This config space is the W3 autotune search space and the
W4 cost-model feature space -- defined ONCE here by hand (the Ansor template), not searched into
existence. Artifacts under `wmma-w2/`; test `test/external/test_qk_marlin_w2.py`.

Pre-registered honesty: if grid+K-tiling closes most of the gap to `matmul_decoded` -> the template
contains a competitive point, W3 proceeds. If it plateaus well below the fp16 ceiling (tinygrad's
TC-opt tiling can't be driven hard enough from a custom kernel) -> record the ceiling; the competitive
template may need route (c) assembly, and W3/W4 search a template that can't reach the bar is moot.

### W3: Brute-force autotune vs the bar

- Autotune the template per (tensor shape, batch) by grid/random search on the device metric,
  correctness-gated. Does brute-force search reach the W0 bar across N representative shapes?
- Pre-registered: reaches within the bar across shapes -> machine search (no learned model) is
  competitive with llama.cpp; record the search cost (trials, GPU-hours). Cannot reach the bar even
  with the WMMA template -> the template or tinygrad codegen is the ceiling; record honestly.

### W4: Cost-model-guided search (the learned model, revived on a real target)

- ONLY if W3 reaches the bar by brute force: test whether a cost model reaches the competitive
  config with FEWER trials than grid/random (sample efficiency) -- the Ansor role on observable tile
  features. Use `qk_ansor`'s roofline model as the deterministic baseline cost model; test a learned
  cost model against it. Freeze model predictions before measuring (the shadow discipline).
- Baseline = random/grid, trial-budget-matched. Pre-registered: the model reaches competitive in
  fewer trials than brute-force across shapes -> the learned-model question is finally answered YES
  on a real, observable, correctly-measured target. Ties -> brute-force/roofline search is the tool
  (still a machine-search win, no learned model needed). Either way, record honestly.

Metric: device throughput / measured roofline (bandwidth at `B=1`, compute at large `B`) and
end-to-end tok/s vs llama.cpp. Correctness exact, always.

Sequencing / honesty: W1 is the gate. If the fused-dequant->WMMA primitive cannot be built in
tinygrad, the whole "machine search competitive with llama.cpp" goal is blocked at the framework
level -- report that and either go lower-level or accept the deterministic `61.6%` as the achievable
bar. The per-kernel hand-tuning that would trivially close the gap is explicitly OUT of scope: it
defeats the goal. The template is authored once; everything else is search.

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
