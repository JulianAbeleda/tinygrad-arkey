# Session Handoff

Date: 2026-06-14

Repo: `/home/ubuntu/tinygrad-arkey`

Branch: `master`

Implementation baseline before shared-promotion rerun:
`aa2827350b36ea477a20ad7fbff426e6db970345`
(`[docs] add AMD decode session handoff`)

Remote cache state before the shared-promotion rerun:

- `origin/master`: `aa2827350b36ea477a20ad7fbff426e6db970345`
- `upstream/master`: `51100d2c5c283fd4522eb603b2c291f34d373b1d`

## Environment

Native Ubuntu path, local PCIe AMD GPU. Use `DEV=AMD`. Do not run BEAM or risky
schedule search on Mac/TinyGPU/remote paths.

Python:

```sh
/home/ubuntu/tinygrad-arkey/.venv/bin/python --version
# Python 3.12.3
```

## Ubuntu Venv Sync Snapshot

Validation run: `git pull && uv sync --extra testing_minimal --extra costmodel` (already up to date).

```sh
/home/ubuntu/tinygrad-arkey/.venv/bin/python --version
# Python 3.12.3

/home/ubuntu/tinygrad-arkey/.venv/bin/python -c "import tinygrad, sys; print(tinygrad.__file__); print(sys.prefix)"
# /home/ubuntu/tinygrad-arkey/tinygrad/__init__.py
# /home/ubuntu/tinygrad-arkey/.venv

DEV=AMD /home/ubuntu/tinygrad-arkey/.venv/bin/python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).to('AMD').realize().numpy())"
# [1 2 3]

DEV=AMD /home/ubuntu/tinygrad-arkey/.venv/bin/python -c "from tinygrad import Tensor; Tensor.rand(8,8).to('AMD').realize(); print('AMD OK')"
# AMD OK
```

```text
.venv/bin/python -m pip freeze
execnet==2.1.2
filelock==3.29.4
fsspec==2026.4.0
hypothesis==6.155.2
iniconfig==2.3.0
Jinja2==3.1.6
MarkupSafe==3.0.3
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.6
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.3.20
nvidia-nvtx-cu12==12.8.90
packaging==26.2
pluggy==1.6.0
Pygments==2.20.0
pytest-split==0.11.0
pytest-timeout==2.4.0
pytest-xdist==3.8.0
pytest==9.1.0
scipy==1.17.1
setuptools==82.0.1
sortedcontainers==2.4.0
sympy==1.14.0
-e git+https://github.com/JulianAbeleda/tinygrad-arkey.git@58b3969e9e7c2541aa92f41f8b232bc89fa4c1a1#egg=tinygrad_arkey
torch==2.9.1
triton==3.5.1
typing_extensions==4.15.0
xgboost==3.2.0
z3-solver==4.15.3.0
```

Key dependency checks in this venv:

- `xgboost`: present (`3.2.0`)
- `scipy`: present (`1.17.1`)
- `pillow`: not installed in this extra set (not part of `testing_minimal` + `costmodel`)

ROCm system layer (host-level):

`apt list --installed | grep -i rocm | head` shows ROCm packages present (`librocm-smi64 5.7.0-1`, `rocm-core 7.2.4...`, `rocminfo 5.7.1`, etc.).
`hipconfig --version` reports `5.7.31921-0`.

Models present under `/home/ubuntu/models`:

- `Qwen3-1.7B-Q8_0.gguf`
- `Qwen3-4B-Q4_K_M.gguf`
- `Qwen3-8B-Q4_K_M.gguf`
- `Qwen3-14B-Q4_K_M.gguf`
- `Qwen3-32B-Q4_K_M.gguf`

## What Just Landed

The storage track reached a clean stopping point, then the shared-storage
promotion checks completed for 8B and 14B.

`QK_PRIMITIVE_STORAGE=shared` now lets Q4_K/Q6_K primitive wrappers view the raw
GGUF storage through typed buffer views instead of allocating duplicate sidecar
storage. Sidecar remains the default; shared storage is an opt-in mode.

Latest implementation commits before the shared-promotion rerun:

- `0da79f8ac [docs] make QK harness matrix canonical`
- `5661baecf [runtime] add shared QK primitive storage`
- `aa2827350 [docs] add AMD decode session handoff`

Latest Ansor-transition commits before the artifact-portability fix:

- `437f77772 [test] add QK Ansor transition loop v0`
- `c5f3abc7e [test] benchmark QK loop candidates`

The portability fix after those commits makes loop benchmark matrix `path` and
`policy` fields repo-relative, adds a test assertion against absolute artifact
paths, and resolves the `structure/` ignore policy by explicitly allowing only
the tracked session handoff and AMD checklist files. The next semantic-schedule
pass adds a second-stage generated surface and rejects it by the full 8B/14B
gate.

Latest semantic-gate hardening work centralizes shared schedule/codegen
candidate helpers in `extra/qk_semantic_candidate.py`, regenerates deterministic
semantic artifacts with storage deltas and correctness provenance, and changes
future semantic microbench wins to `raw_accept`. A semantic candidate is not a
promoted accept unless a matching full-decode confirmation rerun also accepts.
This did not rerun GPU benchmarks and did not change the current rejected
semantic-schedule v0 or semantic-codegen v1 verdicts.

Key implementation files:

- `tinygrad/llm/model.py`
- `tinygrad/llm/gguf.py`
- `extra/qk_decode_summary.py`
- `extra/qk_policy_pipeline.py`
- `test/external/test_qk_generated_policy_runtime.py`
- `test/external/test_qk_decode_summary.py`
- `test/external/test_qk_experiment_matrix.py`
- `test/external/test_qk_policy_pipeline.py`

## Current Verdict

Use `bench/qk-shared-storage-20260612/matrix-summary.md` as the current
8B/14B/32B source of truth.

Current fully shared-storage matrix:

| model | reference | generated | gain | note |
|---|---:|---:|---:|---|
| 8B | `50.41 tok/s` | `52.07 tok/s` | `3.31%` | shared storage, A/B pass; sidecar peak was `53.49 tok/s` |
| 14B | `21.77 tok/s` | `40.55 tok/s` | `86.29%` | shared storage, A/B pass, profile complete |
| 32B | `11.15 tok/s` | `17.23 tok/s` | `54.56%` | shared storage, explicit reference, A/B pass |

Decision: shared storage is validated across 8B, 14B, and 32B and should be the
recommended explicit generated-policy storage mode. Do not flip the runtime
default yet; sidecar remains useful as a control and is still slightly faster on
the old 8B peak artifact.

32B shared generated runtime storage:

- installed wrappers: `448`
- `storage_bytes=0`
- `shared_bytes=18,677,760,000`
- generated percent of llama.cpp reference: `55.9%`

8B shared validation:

- installed wrappers: `180`
- `storage_bytes=0`
- `shared_bytes=3,970,695,168`
- warm decode about `57 tok/s`
- full harness generated `52.07 tok/s` vs explicit `50.41 tok/s`
- greedy A/B `match=True`

14B shared validation:

- installed generated wrappers: `280`
- `storage_bytes=0`
- `shared_bytes=7,918,387,200`
- full harness generated `40.55 tok/s` vs explicit `21.77 tok/s`
- greedy A/B `match=True`
- profile complete

## Source Of Truth

Core verdicts and architecture:

- `docs/amd-decode-current-verdicts.md`
- `docs/amd-decode-harness-architecture.md`
- `docs/amd-decode-qk-storage-architecture.md`
- `docs/amd-decode-kernel-optimization-flywheel.md`
- `docs/amd-decode-flywheel-proof-plan.md`
- `docs/amd-decode-ansor-direction.md`
- `docs/amd-decode-optimization-plan.md`

Current artifacts:

- `bench/qk-shared-storage-20260612/README.md`
- `bench/qk-shared-storage-20260612/matrix-summary.md`
- `bench/qk-shared-storage-20260612/8b/README.md`
- `bench/qk-shared-storage-20260612/14b/README.md`
- `bench/qk-shared-storage-20260612/32b/README.md`
- `bench/qk-shared-storage-20260612/32b/decision.json`
- `bench/qk-shared-storage-20260612/32b/profile-report.md`
- `bench/qk-harness-20260612/README.md`
- `bench/qk-policy-cap-20260612/README.md`
- `bench/qk-storage-20260612/README.md`

## Flywheel Proof Status

Latest pushed proof-plan commit before the Phase 1/2 artifact build:
`67dfda9f3 [test] finish Phase 4.2 and flywheel phase 0`.

The model-to-kernel flywheel now has a concrete historical triage benchmark,
not just a narrative. Current artifacts:

- `docs/amd-decode-flywheel-proof-plan.md`
- `bench/amd-decode-flywheel-proof-20260614/README.md`
- `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v0/`
- `bench/amd-decode-flywheel-proof-20260614/triage-baselines-v0/`
- `bench/amd-decode-flywheel-proof-20260614/triage-qwen3-8b-base-v0/`
- `bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v0/`
- `bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v1-plus/`
- `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v0/`
- `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1/`
- `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1/`
- `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured/`
- `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured/`
- `bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1/`
- `bench/amd-decode-flywheel-proof-20260614/targeted-outcomes-v1/`
- `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured-plus/`
- `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured-plus/`
- `bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1-plus/`
- `extra/qk_flywheel_dataset.py`
- `extra/qk_flywheel_dataset_v1.py`
- `extra/qk_flywheel_feature_enrich.py`
- `extra/qk_flywheel_targeted_outcomes.py`
- `extra/qk_flywheel_triage_eval.py`
- `extra/qk_flywheel_cost_model.py`
- `extra/qk_flywheel_feature_audit.py`
- `extra/qk_flywheel_coverage_plan.py`

Phase 1 built `83` structured kernel-history examples from existing QK
artifacts: `45` train rows and `38` family-split holdout rows. Phase 2
established deterministic baselines. The best baseline is `mechanism_prior` /
`simple_family_heuristic` at accuracy `0.289`, macro-F1 `0.185`,
false-positive accept rate `0.000`, precision@3 `0.083`, and NDCG `0.218`.

The no-adapter Qwen3-8B generated-policy rollout on the holdout does not beat
the baselines. Strict result: accuracy `0.000`, macro-F1 `0.000`, and `38/38`
`invalid_output` predictions. The prompt includes `/no_think`, but the model
still emits empty `<think>` tags before JSON-shaped text and often uses reasons
outside the allowed taxonomy.

Conclusion: Phase 2 is `no_signal` for the current strict no-adapter 8B model.
The full flywheel is still unproven. Do not let the model influence kernel
experiment ordering until a schema-capable model or adapter beats
`mechanism_prior` on this holdout and then survives live shadow mode.

Phase 3 is now scoped in `docs/amd-decode-flywheel-proof-plan.md`. The next
flywheel-specific implementation should start with the Phase 3.0/3.1 boundary:
diagnose protocol-vs-reasoning failure if useful, then export strict
kernel-triage SFT rows from the `45` train examples. The promotion gate is not
teacher-forced loss; it is a held-out adapter rollout with at least `37/38`
strict JSON outputs, macro-F1 above `0.185`, low false-positive accepts, and
ranking metrics above the `mechanism_prior` baseline.

Phase 3.0/3.1 have now run. New artifacts:

- `bench/amd-decode-flywheel-proof-20260614/triage-protocol-diagnostic-v0/`
- `bench/amd-decode-flywheel-proof-20260614/triage-sft-v0/`
- `bench/amd-decode-flywheel-proof-20260614/triage-adapter-v0-attempt/`
- `extra/qk_flywheel_protocol_diagnostic.py`
- `extra/qk_flywheel_triage_sft.py`

Protocol diagnostic: extracting the JSON-shaped object from the base rollout
fixes parse/schema (`38/38`) but not triage. Extracted macro-F1 is `0.036`,
accuracy `0.053`, and false-positive accept rate `0.763`, well below
`mechanism_prior` macro-F1 `0.185`. SFT export: `45` train rows, `38`
eval/holdout rows, `0` holdout ids in train. Phase 3.2 is blocked on practical
adapter training latency for these long kernel-context prompts; two
`last1_ffn` rank-4 suffix-cache attempts produced no adapter artifact before
termination. Next step is a smaller/progress-reporting or prompt-compressed
adapter candidate, not a rank sweep.

Phase 3.2A added that progress-reporting path and ran the tiny smoke:
`bench/amd-decode-flywheel-proof-20260614/triage-adapter-smoke-v0/`. The
trainer now writes `progress.jsonl` and supports `--max-train-rows` /
`--max-eval-rows`. Smoke settings: `4` train rows, `2` eval rows, `8` steps.
It moved teacher-forced loss but not held-out generation. Strict score stayed
`0/38`; extraction diagnostic stayed at macro-F1 `0.036` and false-positive
accept rate `0.763`. The measured latency is the useful part: caching `4`
train prefixes took `32.8s`, and `2` eval prefixes took `21.0s`. This confirms
the negative for local 8B under the current setup; next high-value test is a
stronger proposer on the same benchmark or prompt compression before any full
local-adapter retry.

Phase 3B has now tested the learned-cost-model version of triage:
`extra/qk_flywheel_cost_model.py` and
`bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v0/`. The script
extracts only pre-result features, has a leakage audit, uses optional XGBoost
via the native API, and keeps a no-dependency centroid fallback for tests.
Local XGBoost `3.2.0` ran with a native `rank:ndcg` ranker. Result on the same
`38` holdout rows: XGBoost accuracy `0.237`, macro-F1 `0.137`,
false-positive accept rate `0.000`, precision@3 `0.000`, NDCG `0.189`.
This loses to `mechanism_prior` macro-F1 `0.185`, precision@3 `0.083`, and
NDCG `0.218`, so Phase 3B is also `no_signal`. The right next cost-model work
is more labeled outcomes plus richer first-class tinygrad/UOp/profile features,
not a from-scratch ML model.

Phase 3C now scopes that data/feature work:
`extra/qk_flywheel_feature_audit.py` and
`bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v0/`. The audit
uses the same leak-free feature extractor and reports `needs_data_and_feature_expansion`:
`24` unseen holdout categorical values, `56` weak rows, `9` post-full-decode
train rows, and no target/result feature leakage. Highest-priority targets:
add label coverage for `construction_blocked`, `raw_accept_unconfirmed`, and
`diagnostic_only`; normalize `18` `unknown` mechanism holdout rows; add
mechanism coverage for `packed_word_lane_unroll`, `qk_block_dot`,
`vector_load`, and `wide_load_only`; and add first-class tinygrad/UOp/profile
features for rows with `no_structural_kernel_detail`. Do not rerun XGBoost as a
decision point until those data/feature gaps are addressed.

Phase 3D added the first schema/data cleanup pass:
`extra/qk_flywheel_dataset_v1.py`,
`bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1/`, and
`bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1/`. It
preserves the same `45` train / `38` family-split holdout rows, adds
`candidate_outcome_v1`, removes the v0 unknown-mechanism hole (`0` unknown
mechanism rows after `26` mechanism normalizations), and keeps outcome fields
out of prompts/features. The v1 audit improves coverage but remains
`needs_data_and_feature_expansion`: unseen holdout categorical values are `15`
instead of `24`, weak rows are `43` instead of `56`, and no target/result
leakage is detected. Remaining blockers are real train coverage for the newly
named holdout mechanisms, label coverage, and first-class tinygrad/UOp/profile
features; current v1 UOp fields are proxy estimates with `uop_available=false`.

Phase 3E added real source/compile feature extraction where committed evidence
exists:
`extra/qk_flywheel_feature_enrich.py`,
`bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured/`,
`bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured/`,
and `bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1/`. The
featured dataset keeps the same `83` rows and the same `45` train / `38`
holdout split. It does not synthesize outcomes or move holdout rows into train.
Real UOp/source features are now available for `13` rows: `7` train and `6`
holdout, covering `tile_custom` (`7`), `packed_word_lane_unroll` (`2`),
`qk_block_dot` (`2`), and `vector_load` (`2`). Leakage audit is still clean.
The cost-model rerun is still blocked: unseen holdout categorical values remain
`15`, weak rows remain `43`, and `33` holdout rows still have mechanisms unseen
in train. The coverage plan requires real candidate outcomes before another
Phase 3B decision run: at least `35` mechanism-coverage rows and `14`
label-coverage targets, with labels recorded only if they occur naturally.

Phase 3F converted unused committed diagnostic artifacts into `47` real train
rows and generated this plus dataset state:

- `extra/qk_flywheel_targeted_outcomes.py`
- `bench/amd-decode-flywheel-proof-20260614/targeted-outcomes-v1/`
- `bench/amd-decode-flywheel-proof-20260614/kernel-triage-v1-featured-plus/`
- `bench/amd-decode-flywheel-proof-20260614/triage-feature-audit-v1-featured-plus/`
- `bench/amd-decode-flywheel-proof-20260614/triage-coverage-plan-v1-plus/`
- `bench/amd-decode-flywheel-proof-20260614/triage-cost-model-v1-plus/`

Result on the same `38` holdout rows is now strong: `xgboost` gives
`macro-F1 0.891`, `accuracy 0.895`, and `false-positive accept rate 0.0` versus
`mechanism_prior` at `macro-F1 0.552`. Plus summary is
`130` rows total (`92` train, `38` holdout).

Phase 3G coverage closure is now complete and the rerun gate has cleared
(`rerun_phase3b_allowed=true`). `extra/qk_flywheel_targeted_outcomes.py` ingests
a dated coverage-closure batch of `6` real mechanism rows on additional dominant
Q4_K tensors, keeping the `38`-row family-split holdout untouched:

- `3` `packed_word_lane_unroll` packed-load candidates on
  `blk.1/2/3.ffn_gate.weight`
  (`bench/amd-decode-flywheel-proof-20260614/phase3g-packed-load/`), each with a
  generated-source `global_load_b128` load-width report captured before timing.
- `2` `qk_block_dot` compile-gate + microbench candidates on
  `blk.0.ffn_up.weight` and `blk.0.attn_q.weight` (compile-shape-pass,
  microbench-reject at `-30.5%` / `-37.4%`).
- `1` `wide_load_only` three-way load diagnostic on `blk.0.ffn_up.weight`.
- The `blk.2` packed-load candidate (`raw_accept`, `+3.59%`) is recorded at the
  previously-unseen `after_microbench_before_full_decode` prediction stage,
  closing the last unseen holdout categorical value.

`extra/qk_flywheel_coverage_plan.py` now derives `rerun_phase3b_allowed` from the
actual audit state (no mechanism/label batches, no unseen holdout category)
instead of a hardcoded `false`. Regenerated artifacts: `targeted-outcomes-v1/`
(`53` rows), `kernel-triage-v1-featured-plus/` (`136` rows: `98` train, `38`
holdout), `triage-feature-audit-v1-featured-plus/` (`0` unseen holdout
categorical values), `triage-coverage-plan-v1-plus/` (gate cleared), and
`triage-cost-model-v1-plus/`. On the rerun, XGBoost still beats
`mechanism_prior`: macro-F1 `0.873` vs `0.479`, precision@1 `0.500` vs `0.000`,
precision@3 `0.250` vs `0.167`, NDCG `0.500` vs `0.253`, false-positive accept
rate `0.0` (`<= 0.05`). `test/external/test_qk_flywheel_phase3f.py` covers the
new row counts and the cleared gate.

Phase 4 v0 live shadow mode is implemented and run (`extra/qk_flywheel_shadow.py`,
`bench/amd-decode-flywheel-proof-20260614/shadow-v0/`). Result is an **honest
negative**: predictions were frozen before the fresh GPU run (commit `d9365daed`;
`predictions.jsonl`/`freeze.json` hash-verified unchanged after outcomes), the
fresh batch produced `3` tie / `1` reject / `1` construction_blocked / `1`
diagnostic_only (zero live candidates), and XGBoost collapsed to all-`reject`
(macro-F1 `0.071`) losing to `mechanism_prior` (`0.667`). The fixed-holdout
cost-model win does not generalize to fresh tensors at the blind static stage. Per
the stop rule the model stays documentation-only; Phase 5 is not entered.
`test/external/test_qk_flywheel_phase4.py` covers the freeze/leak-free/id-join/
scoring invariants.

Phase 4.1 Cost-Aware Staged Shadow is implemented and run
(`extra/qk_flywheel_shadow.py` staged steps, `shadow-staged/`). Result is a
**positive — the cost model beats the prior**, the first evidence the learned model
adds value over the deterministic baseline. Keep/skip rank scores for `16` fresh
semantic-schedule candidates were frozen before the microbench (commit `f7979eb4a`;
`predictions.jsonl`/`freeze.json` hash-verified unchanged after outcomes, leak-free
path). Outcomes: `2` raw_accept (live), `5` tie, `1` reject, `8`
construction_blocked. The model ranked both live candidates (`row_upcast` on
`blk.1/2.attn_q.weight`) at the top, so its gate would run `2` microbenches instead
of `16` and catch both winners: `14/16` experiments saved at `100%` live-recall vs
`0` for `mechanism_prior`. It works by learning the (role x mechanism) interaction
the mechanism-only prior ignores (`row_upcast` wins `6/8` on attn_q, `0/4` on
ffn_gate). Caveats: only `2` live in this batch; a hand-coded role x mechanism prior
would likely match it; validate on larger batches. `test_qk_flywheel_phase4.py`
covers the staged freeze/leak-free/safe-skip/integrity invariants.

Phase 4.2 Generalization Replication and Minimal-Gate Ablation is implemented and run
(`shadow-staged-v2/`, `*-staged-v2` CLI steps). Result: the cost model strictly beats
the role x mechanism lookup (`23` vs `0` experiments saved at `100%` live-recall on
`40` candidates, `7` live across `3` patterns), so per the pre-registered rule it
earns Phase 5 entry. Honest caveat (recorded in the proof plan and the `floor_setter`
field): the `6` expected attn_q winners are caught by the lookup too; the whole
margin comes from ONE surprise winner (a fresh `ffn_gate` x `row_upcast` that won
despite `0/4` historical live), which both priors scored `0.0` (collapsing their
safe-skip floor) while the model scored `0.383`. The robust finding is that the model
does not catastrophically write off a surprise winner; the `23`-vs-`0` magnitude is
inflated by the metric's hostage-to-worst-winner property and must be replicated.
`test_qk_flywheel_phase4.py` covers the role_mechanism_prior, ablation ladder, and
freeze integrity. Predictions frozen at commit `8844e160e` before the microbench.

Phase 4.3 Robustness Replication is implemented and run (`shadow-staged-v3/-v4/-v5`,
`shadow-staged-pool/`, `*-batch` + `pool-batches` CLI). Predictions for all 3 batches
were frozen in `8288ad28b` before the microbench (hash-verified). Result is a decisive
**negative for the model, honestly reached**: under the pre-registered safe-skip metric
the model "won" `3/3` batches (`48` vs `0`), but that is a floor-collapse artifact --
the metric penalizes the discrete lookup for tying a surprise winner with the dead
mass. The fair baseline the pre-registration missed, a deterministic class-skip gate
(skip the schedule classes that are `100%` construction_blocked in training:
reduce_unroll / two_dim_local / ffn_gate vector_load), saves the SAME `48` at `100%`
recall with `0` misses. The model skips exactly those same candidates and adds nothing.
Conclusion: ship the deterministic class-skip gate; the model adds no value at the
current feature set. This retroactively reframes the 4.1/4.2 "wins" as the same
artifact. `pool_batches` reports `deterministic_class_skip` and a `caveat`;
`test_qk_flywheel_phase4.py` checks the decision is judged against the fair baseline.

The triage line (3F-4.3) is complete: it pursued Phase 6's *alternative* proof
(reduce wasted experiments) and concluded a cheap deterministic class-skip gate is the
tool and the learned model adds no value at the current feature set. Phase 5
(constrained, deterministic class-skip) remains fully scoped in the proof plan as a
low-ceiling option to cash in that modest win.

Next scope is the **Phase G Generation Track**, fully written in
`docs/amd-decode-flywheel-proof-plan.md` -- the pivot to the harder, higher-value half
of the flywheel and the PRIMARY path to Phase 6's proof (a model-proposed candidate
passes the gates through full decode and improves speed). Generation is safer than
triage-skipping: every proposal runs the same static/correctness/microbench gates, so a
bad proposal wastes bounded GPU but can never produce a wrong kernel or bypass a gate
(no recall risk). Build on existing infra: `qk_candidate_generator` (the fixed grid:
parts {1,2,4}, LOCAL {32,64}), `qk_semantic_schedule` (4 mechanisms with FIXED opt args
-- the frontier lives in the args/compositions the grid never tries), `qk_ansor` (a
roofline cost model), and the committed gates.

Phase G0 is implemented and run (`extra/qk_generation_g0.py`, `generation-g0/`).
Result: **no parametric headroom** -- on the reliable device metric (`device_q4_eff`,
DEBUG=2 kernel timing), the plain `v1_partial` baseline (`LOCAL:0:64`, `~183` GB/s)
beats every expanded schedule on `2` fresh attn_q tensors (`168` correctness-gated
runs); UPCAST/UNROLL/parts roughly halve device throughput. `v1_partial` is already
optimal in the opt space.

Critical reframe (the hypothesis is NOT dead -- we measured wrong AND searched wrong):
G0 contradicts the 4.x "raw_accept" wins, which were scored on WALL-clock `q4_eff`
(~28-35 GB/s, dominated by ~0.27 ms launch overhead -- noise). But the decisive number
is the roofline one: the best kernel (`v1_partial`) achieves only **~19% of peak HBM
bandwidth** (`~183` of `~960` GB/s), a `~5.2x` gap -- it is NOT bandwidth-saturated, so
real headroom exists. G0 just searched ILP knobs (UPCAST/UNROLL) that do not address the
actual bottleneck. Arithmetic intensity `~14` ops/byte is LEFT of the FP32 ridge (`~64`),
so it should approach the roof; sitting at `19%` points to access-pattern / occupancy /
INT-dequant throughput, not FP ILP.

Phase M (Metric Re-base and Bottleneck Diagnosis) is implemented and run
(`extra/qk_metric_audit.py`, DEBUG=7 profiling, `metric-audit-m0/`). Findings:

- **M0a**: measured peak = `859` GB/s (warm streaming copy; 89% of 960 datasheet). On the
  device metric `v1_partial` is at `~20%` of peak on attn_q (~5x headroom) and `~47%` on
  ffn_gate (~2x) -- real, shape-dependent headroom. Re-audit of `7` of the `22` distinct 4.x
  raw_accept configs on device: **0 beat v1_partial beyond a 2% band** (median `-38.6%`;
  row_upcast `-47..-51%`). The 3F-4.x "win" signal was wall-clock noise, confirmed. Root
  cause: `qk_semantic_schedule_bench` scored `q4_eff` (wall), not `device_q4_eff`.
- **M0b**: loads are already wide b128, so width is not the cap. The body is dominated by
  Q4_K dequant ALU (~3862 vector ops/kernel, ~55 per load: nibble shift/mask/cndmask + ->fp32
  + scale/min). Low bandwidth AND low ALU utilization -> latency/occupancy-bound on the
  dequant dependency chain; small matrices worst. **Bottleneck = Q4_K dequant compute +
  occupancy.**
- **M0c**: real search axes = reduce dequant op count (LUT 4-bit->fp, bit-field-extract,
  vectorized multi-nibble unpack, fused scale/min) + raise occupancy. DROP UPCAST/UNROLL and
  wider loads (already b128).

G0' is implemented and run (`extra/qk_generation_g0prime.py`, `generation-g0prime/`):
swept 6 primitive modes x parts {1,2,4} on the device metric. Result: `packed_load` (parts1)
is the ONLY existing kernel that beats `v1_partial` -- `+6.2%` on attn_q (21.5%->22.8% of
peak, confirmed across 5 seeds) and `+2.1%` on ffn_gate (49%->50%); all other modes are worse
and tile_custom is broken (~4%); parts>1 always hurts. packed_load is the 3G
packed_word_lane_unroll mechanism, so 3G found a small real win while the 4.x schedule work
was noise. But it is marginal -- the best kernel still sits at 22.8%/50% of peak, leaving
~4.4x/~2x residual -- and the 18-candidate mode space is fully enumerated, so there is no
role for model-guided search (G1).

G0'' iteration 1 is done (`extra/q4_k_gemv_primitive.py` `q4k_gemv_hoist_partial_kernel`,
`--primitive-mode hoist_scale_min`, `generation-g0pp/`). Result: the highest-value variant
`hoist_scale_min` is correct (exact numerics) but a clear device regression -- 36.8 vs
packed_load 195.7 GB/s on attn_q (-81%), 93.5 vs 430.2 on ffn_gate (-78%). DEBUG=7 shows MORE
ALU (5150 vs 3862), not fewer: collapsing pos/lane4 into a full unroll to enable the algebraic
factoring bloated and serialized the kernel. Lesson: the bottleneck (M0b) is occupancy/latency,
NOT decode op-count, so ALU-reduction backfires; this down-weights the other ALU-level variants
(bfe_nibble, lut_dequant), not pursued without a new hypothesis. packed_load stays the adopted
device baseline (+6%/+2% over v1_partial, the one real win in the whole program).

The generation/optimization track concluded the residual batch-1 gap is NOT reachable by
dequant-ALU restructuring. The primitive analysis (G0'' postmortem) identified the structural
lever: WEIGHT REUSE via batching. That is now scoped as **Phase B (Batched Q4_K Matmul
Modality)** in `docs/amd-decode-flywheel-proof-plan.md` -- the next direction.

Batch-1 decode GEMV has zero weight reuse (each dequantized weight used once), so it is
latency-bound at ~20-47% of peak. Batching (B>1) makes it W[M,K].X[K,B] (a GEMM): each
dequantized weight is reused B times, the dequant amortizes B-fold, and the op moves from
memory/latency-bound to compute-bound. Applies to prefill, batched serving, and
speculative/Medusa decode -- NOT single-stream greedy decode (irreducibly B=1). So it raises
throughput/prefill, not one stream's per-token latency (do not oversell).

B0 is done (`extra/qk_batched_b0.py`, `batched-b0/`): batching is a confirmed large lever.
Per-token device latency drops 26x on attn_q (622->24 us/tok) and 13x on ffn_gate (354->26) from
B=1 to B=128 (measured fp16 compute peak 83.6 TFLOPS). BUT the fused decode_q4_k_plus_matmul path
stays at only 17%/25% of the dense matmul_decoded ceiling at B=128, because it materializes the
dequantized weights to fp16 in memory then reads them back; and even dense matmul reaches only
10%/19% of compute peak (untuned tinygrad GEMM). So the lever is real but realizing it needs B1.
(The B=4 point is a noisy outlier; the verdict uses the fused-vs-dense ratio at the largest batch.)

B1 is done. B1a: the existing fused kernel already reuses the dequant across the batch (dequant op
count is constant across B), so the slowness is tiling alone. B1b: authored a fused Q4_K GEMM
(`q4k_gemm_packed_load_kernel` in extra/q4_k_gemv_primitive.py + benchmark extra/qk_gemm_b1.py,
`gemm-b1/`) that extends packed_load with an UPCAST'd B axis (dequant reused across the B columns).
Result: correctness-gated, and it BEATS the fp16 dense matmul at small batch -- 3.7-5.1x at B=4,
1.8-1.9x at B=8 -- while reading compressed weights. The first hand-authored kernel to beat a real
baseline. It plateaus at ~4.6-6% of peak and loses at B>=16 (crossover ~B=12); tinygrad's matmul
tiles fp16 better at large batch. Adopt the fused GEMM for B<=8 (speculative/Medusa decode); use
matmul_decoded for large batch.

W0 + W1 are done (`extra/qk_wmma_w1.py`, `wmma-w1/`). W0 bar: llama.cpp = 103.84 tok/s on 8B Q4_K
decode (ROCm) on this GPU; our deterministic ~52 tok/s = ~50%, so the gap is ~2x. W1 gate verdict:
**open at capability, closed at performance.** Forcing tensor cores (TC_OPT=2) makes tinygrad emit
WMMA on the FUSED dequant matmul (the matcher only needs both MUL operands fp16; 0 WMMA by default,
145 forced). It is correct, uses matrix cores, and reads COMPRESSED weights (~10/30 MB vs ~34/103
MB materialized) -- but 13-28x SLOWER than materialized-fp16 WMMA, because tinygrad recomputes the
dequant inside the WMMA tiling instead of staging the dequantized tile once in LDS (the Marlin
trick). So W2-W4 do NOT run over this naive template (no tiling search fixes a per-tile dequant
recompute).

Decision point RESOLVED into a concrete scope: chose route (a), now fully scoped as **W1b
(Marlin-class LDS-staged fused-WMMA kernel)** in `docs/amd-decode-flywheel-proof-plan.md` (under
Phase W, between W1 and W2). The grounding flipped the approach: tinygrad DOES expose the LDS
primitives from a custom kernel -- `Ops.DEFINE_LOCAL` / `UOp.placeholder(..., addrspace=AddrSpace.LOCAL)`
(`tinygrad/uop/ops.py:1056`), `.barrier()` (`Ops.BARRIER`, `ops.py:532`), and the existing Q4_K
kernels already use `AddrSpace.REG` placeholders. So the move is NOT hand-writing `Ops.WMMA`:
**stage the dequant into an LDS tile once, then let forced-TC (TC_OPT=2) apply WMMA to the matmul
that now reads LOADS from LDS** -- dequant runs once (the store), WMMA operands are loads, the
per-tile recompute (the W1 28x) disappears. Primitives exist => not a framework wall; the lift is
kernel engineering (coordinating hand-LDS-staging with the forced-TC WMMA tiling). W1b.0 is a cheap
make-or-break sub-gate (can one custom kernel: DEFINE_LOCAL fp16 tile -> store dequant tile ->
barrier -> matmul from LDS with TC firing, correctly?) before authoring the full kernel. If TC
won't fire on an LDS-staged operand, THAT is the real framework limit -> go lower-level
(HIP/rocWMMA) or accept the regime split (route b: matmul_decoded large-batch ~19% peak + B1b fused
small-batch + deterministic ~50% bar).

UPDATE 2026-06-15 -- W1b.0 RAN and the fork-the-skeleton plan is DEAD; merged scope W1b' written.
W1b.0 ran `extra/gemm/amd_copy_matmul.py WMMA=1` on this GPU: the non-WMMA LDS path works (MSE 0.0)
but ALL FOUR in-repo hand-`SHAPED_WMMA` kernels are stale against this fork's tinygrad (4 different
drift errors; the closest, amd_copy_matmul, needed an AFTER-not-wrapping-INDEX fix then blocked on
`SHAPED_WMMA` reaching type_verify un-lowered with ptr srcs -- `lower_shaped_wmma` doesn't fire on
the upstream frag convention). No green in-repo `SHAPED_WMMA` reference exists. KEY: this fork builds
WMMA via the **TC opt over a normal reduce** (`_apply_tc_opt`, `postrange.py:219+`), NOT hand-placed
SHAPED_WMMA -- that's the W1 path (145 correct ops). Also: `Opt(OptOps.TC, axis, (-1,2,1))` is
requestable from a custom kernel's `opts_to_apply` (`search.py:22`), and GROUP/GROUPTOP are FORBIDDEN
with TC (`postrange.py:173` -- this kills the naive "GROUP-stage the dequant" tactic and explains the
W1 28x: TC owns its operand staging and recomputes the dequant per MAC).
User chose to pursue BOTH (a) and (b); they MERGE into one plan (W1b' in the proof plan, under Phase
W): Track 0 (diagnose W1 source + TC layout) -> Track B (fast falsifier: do LOCAL/contiguous opts
stage the dequant? expected NO) -> Track A (the real build, `extra/qk_marlin_w1b.py`: DEFINE_LOCAL a
weight tile, store dequant ONCE, barrier, normal matmul reduce + `Opt(OptOps.TC)` so TC builds the
WMMA over the LDS load). Track A.0 is the make-or-break sub-gate: does the TC matcher accept a MUL
operand that is a load from a DEFINE_LOCAL written earlier in the same kernel (test with a plain
fp16 copy, no dequant)? If yes -> add dequant (a.1) + measure (a.2); if TC refuses an LDS-staged
operand -> escalate to (c) assembly or (d) regime split. Next action: Track 0, then Track B, then A.0.

UPDATE 2026-06-15 (later) -- W1b' DONE; the Marlin primitive WORKS, gate OPEN. Built bottom-up in
`extra/qk_marlin_w1b.py` (artifacts `wmma-w1b/RESULT.md` + `summary.json`, test
`test/external/test_qk_marlin_w1b.py`). All gates green: a0a (TC fires on a hand `Ops.REDUCE` matmul
-- KEY: q4k `.set/.after/.end` is NOT a REDUCE; use `mul.reduce(k, arg=Ops.ADD, dtype=float32)` per
`cdna_asm_gemm.py::custom_uop_gemm`), a0b make-or-break (TC fires WMMA on a MUL operand loaded from a
`DEFINE_LOCAL` written earlier in the same kernel -> Marlin IS expressible here), a1 (full Marlin:
dequant compressed tile ONCE into LDS -> barrier -> WMMA; correct on real GGUF rel_err 1e-4; rendered
source verified ALL dequant shifts pre-barrier, ALL WMMA post-barrier -> per-MAC recompute gone), a2
(fusing the dequant is ~FREE: fused-reads-compressed is 1.07-1.08x FASTER than the materialized-fp16
WMMA ceiling on 4/5 shapes, mean 1.04x, all correct). CAVEAT: absolute TFLOPS tiny (0.04-0.23) --
single-workgroup, un-tiled, whole-tile-in-LDS (M<=32, K<=1024). W1b proved the PRIMITIVE; reaching
83.6 peak / 103.84 tok/s is W2 (parametrize: K-tiling [mandatory, 16x4096 fp16 = 128KB > 64KB LDS] +
grid parallelism + occupancy) -> W3 (autotune) -> W4 (cost model). The template that can contain a
competitive point now EXISTS. Next action: W2 -- parametrize qk_marlin_w1b with K-tiling + grid.

UPDATE 2026-06-15 (W2.0 done) -- grid parallelism works, ~70x. `extra/qk_marlin_w2.py`,
`wmma-w2/w20_summary.json` + `RESULT.md`, test `test_qk_marlin_w2.py`. Grid over M-rows (one workgroup
per BLOCK_M=16 tile, whole N+K) lifted throughput 0.046 -> 3.3-3.6 TFLOPS at 256 wg, all correct (~4%
of 83.6 peak). KEY fix: the LDS dequant-staging depends on the block_m GLOBAL range, polluting the
weight operand's ranges, so TC axis=0 picks the size-n_blocks grid range -> "no tensor core
available"; use `Opt(OptOps.TC, axis=1, ...)` -> selects (n,m,k). Marlin == fp16 ceiling at moderate N
(1.01-1.02x) but trails at large N/K (0.52-0.55x) because the dequant-to-LDS PROLOGUE is a serial
fixed cost not overlapped with WMMA. Next: W2.1 -- K-tiling (mandatory for K=4096: 16x4096 fp16 =
128KB > 64KB LDS) + double-buffering (overlap dequant(tile k+1) with WMMA(tile k)). Open risk W2.1a:
one-workgroup K-loop + TC composition (manual K-loop accumulator is not a single Ops.REDUCE; GROUP
forbidden with TC). Fallback W2.1b: split-K grid + partial-sum pass (each wg = the proven W1b' kernel).

UPDATE 2026-06-15 (W2.1 DONE -- VERDICT: fused custom kernel is NOT competitive; framework wall).
`marlin_splitk_kernel` in `extra/qk_marlin_w2.py`, `wmma-w2/w21_summary.json` + `RESULT.md`, test
`test_qk_marlin_w2.py`. Split-K K-tiling WORKS (grid over (block_m,k_block), each wg = single-REDUCE
W1b' body over BLOCK_K=2048, partials `.sum(0)`); handles real K=4096, correct. Chose split-K because
a manual K-loop accumulator fights TC's ownership of the WMMA accumulator. VERDICT: split-K fused =
2.2-5.0 TFLOPS (2.7-5.9% peak) vs NATIVE tinygrad fp16 matmul 28-82 TFLOPS (33-98%). Fused custom
kernel ~10x slower than native, and 5-6x slower even at small-N (16-64) memory-bound decode (where
reading compressed 3.5x less SHOULD win). ROOT CAUSE (robust): not the dequant (the manually-staged
fp16 ceiling also caps ~3-8%); it's that a custom kernel that MANUALLY stages LDS applies only the TC
opt, while native matmul applies TC+UPCAST*2+LOCAL to reach 98% -- appending those exact opts to the
Marlin kernel barely moves it (3.0->3.7%); BLOCK_M sweep flat. FUNDAMENTAL TENSION: manual LDS
dequant-staging (makes fusion free, W1b') BLOCKS the auto-tiling that reaches peak -- fusion OR peak
tiling, not both, in tinygrad custom_kernel+opts. IMPLICATION: a competitive FUSED quantized GEMM is
NOT expressible here; W3/W4 over the fused template are MOOT. Competitive paths: (c) hand-assembly
(Marlin/rocWMMA), or matmul_decoded (cheap dequant pass + NATIVE matmul at 33-98%, fp16 round-trip).
Machine-search/cost-model is meaningful on the NATIVE matmul opt schedule, not the fused kernel.
Next decision (user's): pursue (c) assembly, adopt matmul_decoded + redirect the search there, or stop.

UPDATE 2026-06-15 (PIVOT -- new scope: Phase N, loop substrate). User chose to engage the
native-matmul opt space (route ②->④) to maximize learning toward eventually building the flywheel
LOOP. New pre-registered hypothesis doc: `docs/amd-decode-loop-substrate.md`. Rationale (scorecard):
the loop needs ONE space that is rich + competitive + learnable; the two on-target quantized spaces
are dead (Q4_K GEMV decode = flat/unlearnable per M0+4.3; fused Q4_K WMMA = framework wall per W2.1);
native fp16 matmul is the ONLY rich+competitive space and its learnability is UNTESTED. Hypothesis:
N0 (do now, route ②) -- matmul_decoded (dequant pass + BEAM-tuned native matmul) is competitive for
the batched/prefill regime, giving a real instrumentable search space; N1 (later, route ④) -- a
learned cost model + cross-kernel TRANSFER beats BEAM sample-efficiency (the loop's make-or-break).
Honest decoupling: a positive proves the loop MECHANISM on opt spaces generally, NOT a llama.cpp
decode win (on-target spaces dead). Next action: N0a -- build matmul_decoded (Q4_K->fp16 dequant pass
+ native matmul), measure vs the W2 fused kernel across batch N on real 8B shapes; then N0b -- log
BEAM (config->device_time) trials as the learnability dataset for N1.

UPDATE 2026-06-15 -- Phase N COMPLETE through N1; THE LOOP HAS A HOME (first genuine positive).
- N0a (`extra/qk_matmul_decoded.py`): matmul_decoded (dequant pass + native matmul) beats the W2 fused
  kernel 4.5-9.6x per-call across N (dequant ~112us, fully amortized). Competitive batched path.
- N0b (`extra/qk_beam_log.py`, `beam_log.jsonl`): native-matmul opt space is RUGGED (111-223x spread),
  SHARP (2-10 near-optimal of ~250), NO universal winner (lookup fails), STRUCTURED (family clusters).
- N1 (`extra/qk_loop_{dataset,learnability}.py`, `beam_log_n1.jsonl` 3878 rec / 14 shapes,
  `n1_learnability.json` + `n1_RESULT.md`): leave-one-shape-out XGBoost. Model top1 = 0.89 of oracle
  vs LOOKUP 0.80, worth ~131 random trials. PRE-REGISTERED GATE = FAIL (overall 0.90 missed by 0.01,
  kept honest). Diagnostic: miss is 4 under-sampled small-N shapes (0.705); batched N>=256 (10 folds)
  = 0.964 of oracle, clears 0.90. TRANSFER rises 0.46(k=1)->0.89(k=13). The conditions absent in the
  dead spaces are present here: rich+competitive+learnable+transfers. The loop mechanism works on the
  native-matmul substrate (decoupled from llama.cpp decode per scope boundary).
Doc: `docs/amd-decode-loop-substrate.md`.

UPDATE 2026-06-15 -- N1.1 + N2 DONE; the loop WORKS and the strict gate is CLOSED.
- N1.1 (`qk_loop_dataset_smalln.py`, `beam_log_n1_smalln.jsonl`): added ~12 small-batch shapes ->
  merged ~26-shape dataset (loaders now merge all `beam_log_n1*.jsonl`). Re-run learnability: overall
  model top-1 = 0.922 (was 0.89) -> PRE-REGISTERED GATE PASSES (closed by coverage, threshold unmoved).
  small-N 0.705 -> 0.915. Naive lookup collapsed to 0.054 (model beats it 26/26 folds).
- N2 (`qk_loop_search.py`, `n2_loop_search.json`, test `test_qk_loop_search.py`): model-guided
  best-of-K/oracle 0.92(K1)->0.98(K5)->0.99(K20) vs random 0.48->0.72->0.85. TRIALS TO 95% OF ORACLE:
  guided median 1.0 vs random 86.3 (~86x fewer). Online flywheel: best-of-5 0.67(1 shape)->0.98(25).
  All gates PASS. The loop demonstrably works on the native-matmul substrate.
Scope boundary unchanged: general learned-autotuning on native matmul (serves quantized inference via
matmul_decoded for batched), decoupled from the llama.cpp decode bar (on-target spaces dead).
The whole arc is now COMPLETE: precisely-located negatives (GEMV flat, fused-WMMA walled, fused custom
kernel can't tile) + a demonstrated POSITIVE (the loop works where the space is rich+competitive+
learnable).

UPDATE 2026-06-15 -- FINAL REPORT written: `docs/amd-decode-final-report.md`. Synthesizes the whole
arc (mission -> triage premise dead -> metric re-base -> batching -> fused primitive works (W1b') ->
framework wall (W2) -> loop substrate (N) -> loop demonstrably works (N1/N2)), the findings ranked,
the meta-conclusion (learned kernel-search has a home iff rich+competitive+learnable; no home when
flat or physics/framework-bound), scope boundaries, methodology lessons, artifacts, and follow-ups.
This is the capstone. Remaining optional follow-ups (user's): make the loop live (BEAM warm-start +
real wall-clock speedup), scale the substrate study (more ops/shapes, cross-op transfer), or close the
decode gap separately (int8/DP4A mmvq GEMV, independent of the loop).

The full Phase W scope (W0-W4) remains in `docs/amd-decode-flywheel-proof-plan.md` -- the actual
program goal restated against what we learned. The current kernel templates top out at ~20% of peak, so no search inside them
reaches llama.cpp; the fused-dequant->WMMA structure is the prerequisite for the search space to
contain a competitive point. The deterministic generated policy is at 61.6% of llama.cpp (14B);
this phase closes that gap by search over a competitive template (the Ansor model: human authors
the template once, search tunes it). Sequence:

- W0: make "competitive" a number -- measure llama.cpp tok/s on this GPU, translate to a kernel
  roofline target; pre-register e.g. 14B from 61.6% toward >=90%.
- W1 (THE GATE): close the fused-dequant->WMMA primitive via the Marlin tile-level trick (dequant a
  weight tile to fp16 in LDS, then WMMA; compressed weights stay in DRAM, no 2x memory). tinygrad's
  WMMA matcher does NOT fire on a dequant expression today (B0: fused = ~4% scalar reduce; only
  materialized fp16 -> WMMA = ~18%). Two routes: coax tinygrad (realize dequant into a small fp16
  LDS tile the matcher recognizes) or hand-author WMMA intrinsics. Correctness-gated; beat
  matmul_decoded while reading compressed weights. If WMMA can't be coaxed -> tinygrad-capability
  blocker, do NOT run W3/W4 over a non-competitive template.
- W2: parametrize the kernel (WMMA tile MxNxK, B-block, LDS staging, dequant placement, occupancy).
- W3: brute-force autotune the template vs the W0 bar across shapes.
- W4 (learned model revived): only if W3 reaches the bar, test a cost model vs grid/random on
  trials-to-competitive (the Ansor role on OBSERVABLE tile features -- the one job the model was
  ever suited for, unlike the weight-determined noise that doomed 3F-4.x). Freeze predictions first.

Heed the G0'' lesson throughout: win via tiling/reuse that preserves parallelism, do not serialize.
Per-kernel hand-tuning is OUT of scope (defeats the goal). Other lower-priority options recorded in
the postmortem: wire the small-batch fused GEMM into the real decode path; stop and consolidate.

Historical note (from grounding the B1 scope): the existing fused path does NOT do an fp16 round-trip.
`decode_q4_k_plus_matmul` is already ONE fused kernel reading compressed weights
(`mem=9.57MB`, kernels=1.0, vs matmul_decoded's 33.69MB fp16). Its slowness is poor TILING
(~351 GFLOPS, ~4% of peak), not materialization. The proof-plan B0 result and postmortem are
corrected accordingly.

Next step is **B1 -- a well-tiled fused Q4_K GEMM** (kernel authoring), scoped in the proof plan:
- B1a (mostly answered): confirm via DEBUG=7 on a B>1 shape whether the fused kernel dequantizes
  each weight tile ONCE and reuses across the B activation columns, or re-decodes per column
  (does the dequant ALU scale with B?). That sets whether the win is tiling alone or tiling +
  dequant-reuse.
- B1b: author a register-blocked fused Q4_K GEMM (tile output over M x B, stage weight + B-column
  activation tiles, dequant each weight once into registers, accumulate over K). New primitive
  path in extra/q4_k_gemv_primitive.py + a seq-len>1 primitive mode in extra/q4_k_bench.py
  (--primitive is batch-1 only, L72). Correctness-gated; measured as achieved FLOPS / 83.6 TFLOPS
  across a batch sweep. Heed the G0'' lesson: win via tiling/reuse that PRESERVES parallelism, do
  not serialize the reduce. Target: beat decode_q4_k_plus_matmul, then matmul_decoded, then toward
  the roof. If a well-tiled kernel cannot beat the existing fused path, tinygrad's matmul codegen
  is the ceiling and the rest needs lower-level work (custom WMMA/MFMA) or is not worth it.

B1b's tiling space (M x B x K, LDS staging, dequant placement) is also the first parametric space
large enough that model-guided search (G1) might earn its keep on a correctly-measured target.

Also still open: wire device_q4_eff into any revived schedule_bench/cost-model labels (fix
q4_eff->device_q4_eff).

1. **G0 headroom probe (deterministic, shadow)**: expand the parametric space on the
   live-bearing attn_q tensors (LOCAL {16..256}, parts sweep, UPCAST/UNROLL args
   {2,4,8,16}, composed/multi-axis opts), run every candidate through the existing
   static+correctness+microbench gates. Measure best device GB/s gain vs v1_partial and
   vs the best hardcoded mechanism, and the GPU cost. Pre-registered: no expanded
   candidate beats the hardcoded best -> parametric generation has no headroom (stop or
   jump to G2 structural); wins exist -> quantify the frontier + brute-force cost (the
   baseline G1 must beat).
2. Only if G0 shows headroom: **G1** model-guided search (roofline-guided / LLM-proposed,
   frozen) vs random search, GPU-budget-matched, scored on sample-efficiency-to-best.
   Same honest bar: tie random -> brute force is the tool.

Do not bypass any gate, drive 14B/32B, or treat a microbench win as proof without full
decode.

Phase 4.3 deliverables:
1. Run `K>=3` more frozen staged batches (`shadow-staged-v3/-v4/-v5`) seeded with
   surprise-prone cells (ffn_gate x row_upcast / direct_output across fresh
   blk.13..35, plus fresh attn_q blocks). Reuse the staged freeze protocol.
2. Add a recall-vs-savings curve to the scorer (experiments saved at 100% / 95% /
   90% live-recall, per gate, per batch) to defang the single-surprise-winner
   brittleness, plus a pooled across-batch summary and surprise-winner keep-rate.
3. Pre-registered exit: model saves more than the lookup in a majority of batches AND
   the advantage persists at 95% recall -> model earns model-driven Phase 5; else
   Phase 5 uses the deterministic lookup, model documentation-only. Report all K
   batches; do not re-roll.

Phase 5 deliverables (gate source decided by 4.3): the loop actually skips skip-marked
microbenches, starting with only the construction_blocked-in-zero-live-history class;
conservative union (skip only if model AND lookup agree dead, keep if either says
keep); a random audit of SKIPPED candidates run anyway to measure the real
missed-winner rate; revert to run-everything if the audit catches a missed winner
above tolerance. Do not bypass static/correctness/microbench/full-decode gates or
drive 14B/32B. The staged harness, freeze protocol, safe-skip scorer, floor_setter
diagnostic, and role_mechanism_prior baseline are reusable.

The earlier Phase 4.2 scope (now executed) remains documented in the proof plan.
Original next-session deliverables, for reference -- a shadow validation before
Phase 5 lets any gate skip real runs. The 4.1 win rested on `2` live
candidates and one pattern, and the corpus shows the live signal is a (role x
mechanism) interaction (`attn_q` x `row_upcast` `75%` live, `attn_q` x
`direct_output` `42%`, all other cells `0%`), which a trivial role x mechanism
lookup already encodes. So 4.2 is an ablation to find the SIMPLEST deterministic
gate that captures the signal. Concrete next-session deliverables:

1. Extend `STAGED_SCHEDULE_TENSORS` to a bigger multi-block batch centered on fresh
   `attn_q` (e.g. `blk.3..blk.10.attn_q.weight`) for `>=5` live across two patterns
   (row_upcast + direct_output), with `ffn_gate` dead controls and optional
   `ffn_down` (Q6_K) dead region. Target `~30-40` candidates.
2. Add a `role_mechanism_prior` baseline ((role, mechanism) -> majority label,
   fall back to mechanism) to the staged scorer; emit a per-(role x mechanism)
   breakdown to `shadow-staged-v2/`.
3. Freeze keep/skip before microbench, run, score the ladder
   (run_all < mechanism_prior < role_mechanism_prior < model) on safe-skips at
   100% live-recall, and report per pattern.

Pre-registered: `<5` live -> inconclusive (enlarge, do not re-roll); model ties
role_mechanism_prior -> ship the lookup, model documentation-only; no gate beats
run-all at full recall -> stay in shadow. Phase 5 proceeds with whichever gate wins
the ablation (deterministic lookup preferred when it ties the model). Reuse the
staged freeze protocol and safe-skip scorer as-is.

The original Phase 4 scope (now executed) remains documented in
`docs/amd-decode-flywheel-proof-plan.md`. Earlier next-session deliverables, for
reference:

1. Add a thin `extra/qk_flywheel_shadow.py` that reuses
   `extract_feature_map`, `FeatureVectorizer`, the XGBoost classifier/ranker
   fit, and `_label_policy` from `extra/qk_flywheel_cost_model.py` (do not fork
   the leak-free feature path). Train on the full `136`-row
   `kernel-triage-v1-featured-plus/` corpus and persist the vectorizer vocab,
   classifier, ranker, and label policy.
2. Build a fresh, unlabeled candidate batch from static descriptor metadata on
   untouched dominant Q4_K tensors: `3` `packed_word_lane_unroll` packed-load
   (e.g. `blk.4/5/6.ffn_gate.weight`), `2` `qk_block_dot`
   (e.g. `blk.0.attn_output.weight`, `blk.1.ffn_up.weight`), `1` `wide_load_only`
   (e.g. `blk.0.attn_output.weight`). Predict at the
   `after_static_before_microbench` stage.
3. Freeze `shadow-v0/predictions.jsonl` + `shadow-v0/freeze.json` (corpus /
   model / candidate hashes + git commit) and commit before any fresh GPU run.
4. Run the same Phase 3G generators on the fresh batch to produce
   `shadow-v0/outcomes.jsonl` via the existing extractor labels (no hand
   labels), then score with `extra/qk_flywheel_triage_eval.py` plus a new
   dead-branch / experiments-to-first-live metric.
5. Add a Phase 4 test asserting frozen-before-outcomes hash stability, leak-free
   shadow features, and a three-baseline comparison on the fresh batch.

Phase 4 v0 exit gate: the model beats `mechanism_prior` on the fresh batch on
macro-F1 and at least one of precision@k / NDCG with
`false_positive_accept_rate <= 0.05`, and reduces dead-branch recommendations
versus `simple_family_heuristic`. Keep the `38`-row holdout fixed; v0 shadow is
blind static-stage, instance-level generalization (new tensors, same mechanism
families). Staged re-prediction, 14B/32B cross-model, and new mechanisms are
Phase 4.x / 5.

### Phase 2-4 Summary for Handoff

- Phase 2 is complete: baseline split and benchmark protocol are reproducible.
- Phase 3 protocol/adapter stack is complete as an integration/debug path; no
  decision-grade gain versus the deterministic baseline yet.
- Phase 3 cost-model path is complete through v1+ featured data and feature
  extraction, but the plus coverage gate is still open.
- Phase 4 (live shadow/controlled assist) has not started because rerun is
  still blocked until 6 mechanism rows plus the remaining prediction-stage
  coverage gap are collected, and a rerun exceeds phase baselines under that
  exact split.

## Verification Already Run

```sh
PYTHONPATH=. .venv/bin/python -m unittest \
  test.external.test_qk_generated_policy_runtime \
  test.external.test_qk_decode_summary \
  test.external.test_qk_experiment_matrix \
  test.external.test_qk_policy_pipeline
```

Result: `Ran 22 tests ... OK`.

```sh
PYTHONPATH=. .venv/bin/python -m py_compile \
  tinygrad/llm/model.py tinygrad/llm/gguf.py \
  extra/qk_decode_summary.py extra/qk_policy_pipeline.py \
  extra/qk_experiment_matrix.py

git diff --check
```

Both passed.

## Resume Commands

Run 32B shared generated policy:

```sh
cd /home/ubuntu/tinygrad-arkey
DEV=AMD QK_PRIMITIVE_STORAGE=shared \
QK_GENERATED_POLICY=bench/qk-shared-storage-20260612/32b/policy.json \
JIT=1 PYTHONPATH=. .venv/bin/python -m tinygrad.llm \
  --model ~/models/Qwen3-32B-Q4_K_M.gguf --warmup --benchmark 128
```

Reproduce the current matrix:

```sh
cd /home/ubuntu/tinygrad-arkey
PYTHONPATH=. .venv/bin/python extra/qk_experiment_matrix.py \
  bench/qk-shared-storage-20260612/8b \
  bench/qk-shared-storage-20260612/14b \
  bench/qk-shared-storage-20260612/32b \
  --json bench/qk-shared-storage-20260612/matrix-summary.json \
  --md bench/qk-shared-storage-20260612/matrix-summary.md
```

Run the targeted handoff test set:

```sh
cd /home/ubuntu/tinygrad-arkey
PYTHONPATH=. .venv/bin/python -m unittest \
  test.external.test_qk_generated_policy_runtime \
  test.external.test_qk_policy_parity \
  test.external.test_qk_ansor_transition \
  test.external.test_qk_decode_summary \
  test.external.test_qk_experiment_matrix \
  test.external.test_qk_policy_pipeline
```

Latest semantic-codegen v2 verification: `py_compile` passed for the semantic
schedule/codegen tools and transition test; the targeted QK tests ran `32`
tests across generated-policy runtime, policy parity, Ansor transition, and
decode summary, plus `11` matrix/pipeline tests. `git diff --check` passed, and
semantic-codegen v2 artifacts have no `/home/ubuntu/tinygrad-arkey`
checkout-path leaks.

## Stop Rules

- Do not add another q8 arithmetic candidate in `extra/`; that thread has named
  the wall and should stay stopped.
- Do not resume kernel search from the storage track.
- Do not chase 32B by hand. If 32B is discussed, use the harness matrix and
  shared-storage artifacts.
- Do not run BEAM or risky schedule search on Mac/TinyGPU/remote paths.
- Do not make `QK_GENERATED_POLICY` a global default. Do not flip
  `QK_PRIMITIVE_STORAGE=shared` to the runtime default without non-campaign
  soak; keep it explicit for now.
- Do not commit benchmark artifacts with machine-local absolute checkout paths;
  use repo-relative paths so clean checkouts can regenerate identical evidence.

## Next Decision

The clean default is to pause here. The project now has a consolidated local
inference result and a third scaling point.

Track 1 eval/parity is now model-agnostic at the tool layer:
`extra/llm_eval_harness.py` is the smallest-real LLM rollout/eval gate, and
`extra/llm_eval_matrix.py` is the matrix source of truth. The Qwen scripts are
compatibility wrappers with Qwen defaults. Current Qwen artifact:
`bench/qwen-eval-20260612/matrix-summary.md`. It compares explicit Q4/Q6
primitive flags against pinned generated policies using
`QK_PRIMITIVE_STORAGE=shared`, fixed prompts, greedy decoding, exact token
parity, and separate answer-quality scoring. The enabled 8B and 14B rows both
passed exact parity and scored `10/10`; 32B is listed in the manifest as an
optional heavy row. Treat timings in this harness as sanity data only; canonical
decode speed still comes from the QK harness matrix.

Track 1.3 is the dataset rollout layer. Generic runner:
`extra/llm_rollout.py`; shared scoring helpers: `extra/llm_eval_common.py`.
Current Qwen artifact:
`bench/qwen-rollout-20260612/8b-generated/summary.md`, generated from
`bench/qwen-rollout-20260612/manifest.json`. It runs Qwen3-8B with the
8B shared generated policy and scores `10/10` on the smoke dataset. The runner
is model-agnostic; the artifact directory is Qwen-specific because the manifest
pins Qwen models, Qwen policies, and `/no_think` prompts.

Track 1.4 scaled the same Qwen3-8B generated-policy path to a 75-prompt small
dataset at `bench/qwen-rollout-20260612/dataset-small.jsonl`. Current artifact:
`bench/qwen-rollout-20260612/8b-generated-small/summary.md`, quality `pass`,
`75/75`, `608` generated tokens. The dataset is a deterministic breadth gate
across math, code, format, facts, reasoning, compiler/tinygrad, and instruction
following; it is for rollout/eval plumbing, not a broad capability benchmark.

Track 1.5 added the deterministic offline comparator:
`extra/llm_rollout_compare.py`. Current Qwen artifact:
`bench/qwen-rollout-20260612/compare-8b-small/report.md`, comparing
`8b-generated-small` against `8b-explicit-small`. Result: explicit mode also
passes `75/75`; generated and explicit are token-identical on all `75` prompts
with quality delta `0`, regressions `0`, text changes `0/75`, and token changes
`0/75`. This is an offline score/output comparator, not an LLM-as-judge.

Track 1.6 extended the rollout/comparator contract to 14B and added a top-level
runtime contract. Current Qwen artifacts:
`bench/qwen-rollout-20260612/14b-generated-small/summary.md`,
`bench/qwen-rollout-20260612/14b-explicit-small/summary.md`, and
`bench/qwen-rollout-20260612/compare-14b-small/report.md`. Both 14B rollout
modes pass `75/75`, generate `644` tokens, and compare with quality delta `0`,
regressions `0`, text changes `0/75`, and token changes `0/75`.
`extra/llm_runtime_contract.py` validates the committed eval, rollout,
comparison, and training-data artifact set at
`bench/llm-runtime-contract-20260613/`; the current contract passes `8/8`
rows with no missing artifacts.

Track 1.7 added the first SFT-style training-data dry-run exporter:
`extra/llm_training_data_probe.py`. Current artifact:
`bench/qwen-rollout-20260612/training-data-v1/README.md`. It exports `150`
rows from the 8B and 14B generated rollout artifacts with `0` filtered rows.
This validates the data shape and filtering path only; it is not a training
loop.

Track 1.8 added the smallest real training/eval loop:
`extra/llm_sft_smoke_train.py`. Current artifact:
`bench/qwen-rollout-20260612/sft-smoke-v1/README.md`. It trains a tinygrad
byte-context softmax probe over the rollout-derived SFT rows (`120` train,
`30` eval) and writes `model.npz`. Current gate passes: eval loss
`4.8483 -> 1.5290`, eval accuracy `0.0065 -> 0.6320`. This proves the
training data, optimizer, eval metric, artifact, and contract path work; it is
not a Qwen adapter, LoRA stack, or model-quality claim. The practical path now
has eval parity, dataset rollout, regression comparison, training-data export,
and a reproducible tinygrad training-loop smoke test on top of the faster
generated-policy inference backend.

Track 1.9 added the first real Qwen adapter V0. Runtime additions:
`Transformer.logits()`, `extra/llm_adapter.py`, `extra/llm_adapter_train.py`,
optional `--adapter` loading in `extra/llm_rollout.py`, and adapter-aware
runtime-contract rows. Current artifacts:
`bench/qwen-adapter-20260613/8b-output-lora-r4/README.md`,
`bench/qwen-adapter-20260613/8b-output-lora-r4-rollout/summary.md`, and
`bench/qwen-adapter-20260613/compare-8b-base-vs-output-lora/report.md`.
The adapter is output-head LoRA only for Qwen3-8B (`rank=4`, `alpha=8`).
Adapter weights changed (`adapter_l2=0.003541`), adapter rollout passes
`75/75`, and base vs adapter has `0` regressions, `0/75` text changes, and
`0/75` token changes. Important caveat: the SFT rows are self-generated by the
base model, so teacher-forced loss is already saturated; this validates adapter
install/save/load/rollout/contract plumbing, not model improvement. The runtime
contract now passes `11/11` rows.

The Ansor-transition layer is now the current compiler-research foundation for
the llama.cpp-comparable goal. Tools: `extra/qk_llama_scorecard.py`,
`extra/qk_gap_profile.py`, `extra/qk_semantic_descriptor.py`,
`extra/qk_descriptor_policy.py`, `extra/qk_candidate_generator.py`,
`extra/qk_candidate_static_gate.py`, and `extra/qk_ansor_transition_loop.py`.
Artifacts live under `bench/qk-ansor-transition-20260612/`.

The scorecard freezes the objective: 8B `52.07 tok/s` (`51.46%` llama.cpp),
14B `40.55 tok/s` (`61.63%`), and 32B `17.23 tok/s` (`55.94%`), all
correctness-gated. The first target is `>=70%` llama.cpp on all three. Fresh
shared DEBUG=2 profiles now exist for 8B/14B/32B; named attribution says QK GEMV
still dominates (`14.91`, `30.08`, `82.44 ms/tok` respectively), so the next
research target remains QK semantic schedule/codegen.

The semantic descriptors convert accepted generated policies into
machine-readable Q4_K/Q6_K packed-GEMV objects: format, role, shape, packed
layout metadata, parts/local/reduction choices, and storage/benefit metadata.
Those descriptors now round-trip back into runtime policies with zero semantic
diff against the accepted generated policies. Candidate generation creates
bounded `parts`/`LOCAL` policy variants: 8B `19`, 14B `27`, 32B `32`; all pass
the static gate. The loop v0 writes `current` plus six ranked `benchmark_next`
policy files per model. It is static planning only; promotion still requires
the QK harness correctness/stability gates.

Loop-v0 benchmark verdict: `bench/qk-ansor-transition-20260612/benchmarks/`.
The six `benchmark_next` policies per model were benchmarked policy-vs-policy
against each model's current accepted generated policy. 8B had `0` accepts
(`2` ties, `3` rejects, `1` needs-rerun). 14B had `0` accepts (`2` ties,
`4` rejects). 32B had one raw accept (`ffn_gate LOCAL:64 -> LOCAL:32`,
`+3.24%`), but the fresh confirmation rerun was a tie at `-2.29%`, so no
candidate is promoted. Overall verdict:
`descriptor_knob_frontier_exhausted`.

Semantic-schedule v0 verdict:
`bench/qk-ansor-transition-20260612/semantic-schedules/verdict.md`.
Generated candidates covered `direct_out`, `row_upcast2`, `reduce_unroll4`, and
`two_dim_local4` over dominant 8B/14B descriptors. Static gates passed; isolated
microbench found attention `row_upcast2` wins. The only full-decode-supported
winner rejected on both target models:

| model | explicit/reference | generated candidate | gain | verdict |
|---|---:|---:|---:|---|
| 8B | `53.27 tok/s` | `47.79 tok/s` | `-10.28%` | reject |
| 14B | `38.13 tok/s` | `36.14 tok/s` | `-5.21%` | reject |

Greedy A/B passed for both, so this is a performance rejection, not a
correctness failure. 32B was skipped by rule because the 8B/14B gate did not
accept.

Semantic-codegen v1 verdict:
`bench/qk-ansor-transition-20260612/semantic-codegen-v1/verdict.md`.
This promoted Q4_K direct output into a real runtime-supported generated-policy
family (`q4_k_packed_u32_direct`) and tested exact-tensor overrides, avoiding
the v0 shape-wide blast radius. It still did not clear the locked `3%`
microbench gate: 8B had `0` accepts (`2` ties, `1` reject), and 14B had `0`
accepts (`2` ties, `2` rejects). No full-decode candidate was promoted, and
32B was skipped by rule.

Semantic verdict hardening: the verdict tools now separate confirmed accepts
from raw accepts. Future semantic microbench wins start as `raw_accept`, and
full-decode accepts are only promoted after a matching confirmation rerun. The
artifacts also record storage deltas and correctness provenance. CPU/Mac tests
prove reference unpacking; AMD microbench gates prove GEMV numerics; full-decode
A/B gates prove model assembly. `QK_PRIMITIVE_STORAGE=q4_ondemand` remains a
Q4-only negative storage prototype, not a generic candidate storage mode.

Semantic-codegen v2 / Family B verdict:
`bench/qk-ansor-transition-20260612/semantic-codegen-v2/verdict.md`.
Design note: `docs/amd-decode-semantic-family-b.md`. This pre-registered the
row-grouped Q4_K `ffn_down` mechanism as an activation-reuse / row-axis
scheduling probe. It rejected at microbench: 8B row-group 2 was `-31.03%`,
8B row-group 4 was `-71.54%`, 14B row-group 2 was `-52.59%`, and 14B row-group
4 hit an illegal opt. No raw accepts, no strong raw accepts, no runtime install,
no full-decode run, and no 32B run.

Bandwidth-roofline update:
`bench/qk-bandwidth-roofline-20260613/roofline.md`, generated by
`extra/qk_bandwidth_roofline.py`, compares tinygrad generated shared-storage
decode against llama.cpp on the same logical GGUF bytes. Result: tinygrad reaches
`27.27%`, `38.03%`, and `35.47%` of the RX 7900 XTX 960 GB/s peak on 8B/14B/32B;
llama.cpp reaches `53.00%`, `61.70%`, and `63.40%`. This makes the next research
surface packed-weight memory-access/load lowering, not another local schedule
knob. Design: `docs/amd-decode-packed-load-lowering.md`; prior-art framing:
`docs/amd-decode-prior-art.md`.

Semantic-codegen v3 / Family C v0 verdict:
`bench/qk-ansor-transition-20260612/semantic-codegen-v3/verdict.md`. The first
packed-load probe rewrote Q4_K `ffn_gate` to reduce over explicit packed
`uint32` lanes and unroll four nibbles from each word. It was correct but tied:
8B `206.42 -> 205.07 GB/s` (`-0.65%`), 14B `367.98 -> 366.84 GB/s`
(`-0.31%`). `load-width/report.md` confirms a distinct
`q4k_gemv_packed_load_partial_*` kernel but still scalar `u32` loads and no
vector-load evidence. No full decode, no 32B. Do not broaden this exact rewrite.

Semantic-codegen v4 / Family C v1 verdict:
`bench/qk-ansor-transition-20260612/semantic-codegen-v4/verdict.md`. Raw
aligned `uint32x4` load/store now lowers through AMD UOps, but the real Q4_K
GEMV candidate cannot yet consume the vector load: scalar lane extraction fails
the verifier and vector-lane arithmetic hits shape checks before AMD source is
emitted. No benchmark, full decode, or 32B run was promoted.

Packed-QK tile layer:
`docs/amd-decode-packed-qk-tile-design.md` and `extra/qk_packed_tile.py` add the
static representation needed before another Family C attempt. It defines
Q4_K/Q6_K block layout, legal load tiles, storage dtype, alignment, and search
axes. Family C v4 candidate artifacts now record `packed_qk_tile` and
`load_tile` metadata, including Q4_K `u32x4_aligned` with `32` q-values per
load. This is an IR/provenance step only; it does not solve vector-load GEMV
consumption.

Packed-QK tile consumption probe:
`bench/qk-packed-tile-consumption-20260613/README.md`, generated by
`extra/qk_packed_tile_consumption_probe.py`, answers the next construction
question. Normal UOps still cannot consume a Q4_K `uint32x4` tile:
`vec.gep(0)` fails verifier, and vector integer arithmetic fails shape
validation. A custom semantic kernel succeeds exactly, and DEBUG=4 load-width
parsing confirms `vector_u32x4` source. Verdict:
`semantic_custom_op_required`. Do not run vector-load Q4_K microbench/full
decode until a first-class packed QK load/decode/dot lowering or renderer
PatternMatcher rule exists.

Packed-QK custom lowering:
`bench/qk-packed-tile-lowering-20260613/README.md` records the first real Q4_K
GEMV consumer of the packed tile. `q4k_gemv_tile_custom_partial_kernel` uses
`tg_uint4` source loads, keeps fp16 activation semantics, supports the existing
partial-output shape, and passes AMD correctness for `parts=1` and `parts=4`.
DEBUG=4 parsing confirms `vector_u32x4`. Microbench is positive but below the
promotion bar: 8B `ffn_gate +7.20%`, `attn_output +5.83%` versus v1. Verdict:
`semantic_custom_lowering_constructed_but_not_promoted`. No runtime integration
or full-decode run was promoted.

Packed-QK lowering repeated analysis:
`bench/qk-packed-tile-lowering-analysis-20260613/README.md`, generated by
`extra/qk_packed_tile_lowering_analysis.py`, repeats the comparison across five
8B Q4_K tensors with five runs each. Source-shape evidence is real:
`source/load-width-report.md` reports v1 `u32_scalar` and `tile_custom`
`vector_u32x4`. Performance does not generalize: gain range `-2.04%` to
`+7.51%`, median `-0.36%`; only `ffn_up` is materially positive. Verdict:
`diagnose_only_not_promoted`. Do not run full decode or integrate the raw custom
path from this result.

Packed-QK research close-out:
`bench/qk-packed-tile-research-closeout-20260613/README.md`, generated by
`extra/qk_packed_tile_closeout_diagnostic.py`, parses DEBUG=7 target
disassembly for the 8B Q4_K `ffn_gate` shape. It explains why the raw custom
path stalled: `tile_custom` emits real target `global_load_b128` instructions
(`32` versus `1` in v1), but the kernel is workgroup-size `1` and opaque to
tinygrad scheduling, with a `1293`-instruction target body versus `296` for v1.
Verdict: `raw_custom_tile_path_closed_not_promoted`. Do not add more raw
`Ops.CUSTOM` `tg_uint4` variants.

Packed-QK semantic op contract:
`docs/amd-decode-packed-qk-semantic-op.md` and
`bench/qk-packed-semantic-op-20260613/README.md`, generated by
`extra/qk_semantic_op.py`, define the next research boundary. `QK_BLOCK_DOT`
is a Q4_K block-local packed load/decode/dot contract, not a full GEMV kernel.
It may hide Q4_K scale/min unpack, nibble extraction, lane mapping, and target
load spelling, but it must not hide row loops, K-block loops, split-K layout,
partial reduction, full GEMV body, or runtime policy selection. Artifact status:
design-only, `8` 8B/14B Q4_K contract rows, `6` Q6_K rows skipped, no runtime
lowering, no microbench, no full decode.

Packed-QK semantic compile gate:
`bench/qk-block-dot-compile-gate-20260613/README.md`, generated by
`extra/qk_block_dot_compile_gate.py`, records the first core
`Ops.QK_BLOCK_DOT` lowering check. The gate passes for the fixed 8B Q4_K
`blk.0.ffn_gate.weight` shape: AMD GEMV correctness passes, the v1 32-lane
scheduled shape is preserved, source `tg_uint4` is present, target disassembly
has `5` `global_load_b128` instructions versus `1` for v1, and target body size
is within the pre-registered 2x gate (`333` vs `296` parsed instructions).
This is compile-shape evidence only. It authorizes repeated dominant-shape
microbenching; it does not authorize runtime integration, full decode,
generated-policy promotion, or 32B work.

Packed-QK semantic microbench:
`bench/qk-block-dot-microbench-20260613/README.md`, generated by
`extra/qk_block_dot_microbench.py`, runs the repeated full-shape 8B
`blk.0.ffn_gate.weight` gate for the first `QK_BLOCK_DOT` lowering. Verdict:
`qk_block_dot_microbench_rejected`. v1 median is `407.99` device Q4 GB/s;
`QK_BLOCK_DOT` median is `285.01`; gain is `-30.14%` versus the `>=10%`
promotion bar. Correctness passes. Do not run full decode, integrate runtime
support, broaden to 14B/32B, or promote a policy from this result.

Three-way packed-load diagnostic:
`bench/qk-threeway-load-microbench-20260613/README.md`, generated by
`extra/qk_threeway_load_microbench.py`, compares v1 partial, schedulable
`vector_load`, and opaque `tile_custom` on the full 8B Q4_K
`blk.0.ffn_gate.weight` tensor using AMD device time. Verdict:
`wide_load_not_sufficient`. v1 median is `382.01` device Q4 GB/s. After fixing
the vector-lane reduction bug, schedulable `vector_load` passes correctness and
reaches `349.25` (`-8.58%`). Opaque no-LOCAL `tile_custom` passes correctness
but reaches only `36.99` device Q4 GB/s (`-90.32%`). This closes the cheap
"maybe wide loads alone are enough" branch. Do not harden `vector_load`, add
more raw `tg_uint4` variants, run full decode, or integrate runtime support
from this result.

QK hardening pass:
The bug-audit cleanup is committed after the three-way fix. It adds regression
coverage for shared uint32 vec4 devectorizer folding, tails, unaligned scalar
fallback, and non-empty `VCAT`; makes Q6K requested/effective storage reporting
explicit; hardens QK matrix/profile/eval parsers to fail loudly on malformed
artifacts; proves q8_1 vdot-parallel can remain a research winner but cannot be
promoted by generated runtime-policy selection; and changes the three-way
microbench default to fixed repeat seeds, with `--vary-seed` available for
stress runs.

When resuming, choose one track explicitly:

1. Use the inference win: build a real training loop, richer judge, or
   RLVR/SFT pipeline on top of the validated rollout/comparator backend.
2. Compiler research: continue from the Ansor-transition descriptor foundation:
   descriptor-level `parts`/`LOCAL` search is exhausted, and semantic schedule
   v0, semantic codegen v1 direct-output Q4, and semantic codegen v2 row
   grouping are rejected by their gates. Semantic codegen v3 packed-load v0 is
   rejected too; semantic codegen v4 is rejected at construction because the
   vector load cannot be consumed in the GEMV graph. The consumption probe shows
   the normal-UOp route is blocked and the custom semantic route is viable. The
   first raw custom lowering constructs and reaches target wide loads, but the
   close-out diagnostic rejects raw custom variants as the wrong integration
   shape. The semantic-op contract defines the allowed continuation, and the
   minimal `QK_BLOCK_DOT` compile gate passes, but the repeated full-shape
   microbench rejects the first C-style lowering. The three-way packed-load
   diagnostic also rejects the cheap wide-load-only branch after fixing the
   construction bug. Resume only by diagnosing instruction mix / occupancy /
   memory transactions, or by designing a lower-level renderer/assembly-quality
   lowering. Any future semantic raw accept needs a matching confirmation rerun
   before promotion.
3. Runtime-default soak: keep `QK_PRIMITIVE_STORAGE=shared` explicit for now,
   and only consider making it the runtime default after more non-campaign use.

Recommended next track if the goal is practical progress: training/eval stack.
Recommended next track if the goal is architecture quality: Ansor-style semantic
packed-layout/codegen research.

Practical adapter loop status:
The minimal output-LoRA path is now runtime-wired and contract-gated. V0
(`bench/qwen-adapter-20260613/8b-output-lora-r4/`) proved install/save/load and
rollout plumbing only: self-distilled SFT rows were already saturated and caused
no token/text changes. V2 adds a non-self-generated sentinel override dataset
(`bench/qwen-adapter-20260613/training-data-v2/`) and a rank-8 output LoRA
(`bench/qwen-adapter-20260613/8b-output-lora-r8-v2/`). Held-out generation now
changes behavior under contract: base sentinel rollout is `0/12`, adapter
rollout is `12/12`, compare improvement is `+12` with `0` regressions, and the
runtime contract is `14/14`. This is a real behavior-change plumbing win, not a
general quality result. Next practical step is to replace the synthetic sentinel
with a small human-authored format/preference dataset that still has an
automatic success metric.

That next step has now been run as V3 strict JSON-answering:
`bench/qwen-adapter-20260613/training-data-v3/`,
`bench/qwen-adapter-20260613/8b-output-lora-r16-v3/`, and
`bench/qwen-adapter-20260613/compare-8b-json-base-vs-output-lora-r16-v3/`.
The base generated rollout fails the held-out strict-JSON gate (`0/12`), so the
task is valid. Rank-16 output-head LoRA with EOS targets passes the training
loss gate and improves teacher-forced held-out token accuracy (`0.5000 ->
0.8542`), but held-out generation reaches only `3/12`; compare records `+3`
improvements, `0` regressions, and below-bar absolute pass rate. Verdict:
diagnostic negative. Output-only LoRA can force sentinel behavior and partially
control format, but it does not provide enough conditional capacity for this
strict JSON-answer task. Do not continue output-head-only LR sweeps as a
promoted path; the next practical adapter scope should install a small
allowlisted set of non-output adapters and reuse the same strict JSON
train/rollout/compare gate.

That non-output scope has now been attempted as V4 infrastructure. Code now
supports allowlisted `lastN_ffn` target groups, exact installed-target
recording, internal-adapter activation gradients, and a plain-block adapter
training path. The actual 8B full gate is blocked:
generated-QK internal training fails on unsupported quant bit-op gradients
(`Ops.OR`), `REALIZE=1` baseline training OOMs at `23.78 GB`, and the
plain-block no-REALIZE workaround is too slow for full `last4_ffn` or
`last1_ffn` gates. A one-step baseline/no-REALIZE `last4_ffn` smoke passes and
changes adapter tensors (`adapter_l2=0.876816`), so the graph can train in
principle. Next practical work is a dedicated internal-adapter training mode
that is differentiable through frozen layers, lower-memory than full fp16
realization, faster than the current plain-block path, and still load-compatible
with generated-QK inference. Do not rerun full 8B internal-adapter sweeps through
the current plain-block workaround.

The dedicated internal-adapter training path has now been added as V5:
`extra/llm_adapter_suffix_train.py` caches frozen prefix hidden states at a
`lastN_ffn` boundary and trains only the suffix. The 8B strict JSON
`last1_ffn` rank-4 run is
`bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5/`: suffix parity
passes exactly (`max_abs=0.0`), train/eval loss drops strongly (`7.1041 ->
0.2817`, `7.4458 -> 0.2680`), and teacher-forced eval token accuracy reaches
`0.9167`. The generated rollout is
`bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5-rollout/` and
reaches `4/12` strict JSON passes. Compared with base it is `+4` with `0`
regressions; compared with V3 output-LoRA it is `3/12 -> 4/12` with `2`
improvements and `1` regression, which is not a meaningful generation-quality
win at `N=12`.
Verdict: V5 fixes the practical internal-adapter training loop, but does not
solve strict JSON generation. Do not promote it as a behavior gate. The gap
between teacher-forced token accuracy (`0.9167`) and strict free-generation pass
rate (`0.3333`) points to objective/eval mismatch and exposure bias, not simply
adapter capacity. The recommended next adapter step is a larger held-out
generation set plus filtered own-generation / rejection-sampling SFT, with
generation pass rate as the gate. Do not start with another `lastN_ffn` capacity
sweep unless that objective/eval loop is in place. The plan of record is
`docs/qwen-json-eval-objective-scope.md`: Inspect-shaped local harness,
IFEval-style deterministic scoring, JSON parse/schema/value axes, Wilson CIs,
then rejection-sampling SFT using the same scorer as the filter.

Strict JSON V4 eval/objective status as of 2026-06-14:
The eval/objective foundation is now partially executed and committed.

- `505b914c4 [test] add strict JSON eval scorer`: deterministic JSON scorer
  with parse/schema/type/value/no-extra-text axes and Wilson intervals.
- `5829e8183 [test] add strict JSON V4 eval dataset`: V4 strict JSON dataset,
  `408` train rows and `204` held-out eval rows.
- `c71c8370e [test] add V4 strict JSON adapter rebaseline`: 204-prompt free
  generation rebaseline. Base is `0/204`; V3 output LoRA is `69/204`; V5
  suffix-cache adapter is `105/204` and is the current best behavior artifact.
- `12f3e368c [test] add Phase 4 gold control and RS sampler`: adds the
  resumable rejection-sampling data builder and trains the V6 gold-control
  suffix adapter. V6 teacher-forced eval accuracy is `0.921875`, but this is
  not a behavior result until free-generation rollout is run.
- `c5d54209d [test] record Phase 4 AMD recovery blocker`: records the current
  blocker. Rejection-sampling generation hit an AMD synchronization timeout.
  A bounded `DEV=AMD` smoke test then timed out. Sysfs GPU reset returned
  `Inappropriate ioctl for device`, and `rocm-smi -d 0 --gpureset` itself
  entered uninterruptible `D` state.
- `0871a767c [docs] update Phase 4 reboot handoff`: updates this handoff and
  checklist with the reboot-first resume order before the post-reboot run below.

Post-reboot Phase 4 status as of 2026-06-14:

The AMD blocker from `c5d54209d` is cleared for this session. The host rebooted
at `2026-06-14 03:35:51`, the old stuck PIDs are gone, and the required smoke
test printed `[2, 3, 4]`:

```bash
timeout 45s env DEV=AMD PYTHONPATH=. .venv/bin/python -c \
  "from tinygrad import Tensor; print((Tensor([1,2,3])+1).numpy().tolist())"
```

New uncommitted Phase 4 artifacts:

- `bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v6-gold-v4-rollout/`
- `bench/qwen-adapter-20260613/compare-8b-v4-last1-ffn-suffix-lora-r4-v5-vs-v6-gold-v4/`
- `bench/qwen-adapter-20260613/training-data-v4-rs-v5-k4/`
- `bench/qwen-adapter-20260613/training-data-v4-rs-v5-stratified-v1/`
- `bench/qwen-adapter-20260613/compiler-nearmiss-audit-v1/`
- `bench/qwen-adapter-20260613/training-data-v4_1-compiler/`
- `bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5-v4_1-compiler-rollout/`
- `bench/qwen-adapter-20260613/training-data-v4_1-compiler-rs-v5-k4/`

V6 gold-control free-generation rollout:

- model/policy/storage: Qwen3-8B generated shared storage with
  `bench/qk-shared-storage-20260612/8b/policy.json`
- adapter:
  `bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v6-gold-v4`
- dataset: `bench/qwen-adapter-20260613/training-data-v4/eval-prompts.jsonl`
- strict JSON pass: `162/204` (`0.794`, Wilson 95% CI `[0.733, 0.844]`)
- parse/schema/type/no-extra-text: `199/204`
- value-correct: `162/204`
- category passes: arithmetic `27/34`, fact `34/34`, code `34/34`,
  compiler `14/34`, string `21/34`, categorization `32/34`
- compare vs V5:
  `bench/qwen-adapter-20260613/compare-8b-v4-last1-ffn-suffix-lora-r4-v5-vs-v6-gold-v4/report.md`
  records `+57` strict passes (`105 -> 162`), `59` improvements,
  `2` regressions, and `121/204` changed texts.

Verdict: V6 gold-control proves the suffix adapter/objective setup can improve
free-generation behavior when trained on gold completions. The V5 objective was
not the whole bottleneck; generated training data quality now matters.

K=4 rejection sampling with V5 as generator completed in bounded `--resume`
chunks (`64`, `204`, then `408` train rows) without another AMD hang:

```bash
PYTHONPATH=. .venv/bin/python extra/llm_json_rejection_sample.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --policy bench/qk-shared-storage-20260612/8b/policy.json \
  --adapter bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5 \
  --input bench/qwen-adapter-20260613/training-data-v4/sft.jsonl \
  --out bench/qwen-adapter-20260613/training-data-v4-rs-v5-k4 \
  --device AMD --storage shared --prompt-format chat \
  --seed 20260614 --k 4 --temperatures 0.0 0.2 0.5 0.8 \
  --max-accepted-per-source 1 --resume --limit-train-rows 408
```

RS artifact summary:

- attempts: `1632`
- accepted attempts: `216`
- selected train rows: `215`
- eval rows carried through: `204`
- strict pass: `216/1632`
- selected train rows by category: arithmetic `61`, fact `67`, code `18`,
  compiler `0`, string `23`, categorization `46`
- compiler near misses: `79`

Decision after K=4: do not train V7 from
`bench/qwen-adapter-20260613/training-data-v4-rs-v5-k4/` as-is. Coverage is
too skewed and the compiler category has zero accepted rows. Training V7 on
this artifact would likely erase or under-train the exact category where V6
gold-control showed a meaningful gain.

Stratified V5 RS continuation:

`extra/llm_json_rejection_sample.py` now supports `--sample-categories`, which
limits appended samples to selected train categories while still rebuilding
coverage over all train rows. `extra/llm_json_rs_coverage_gate.py` gates
minimum selected rows per weak category.

The follow-up artifact copied the K=4 base into
`bench/qwen-adapter-20260613/training-data-v4-rs-v5-stratified-v1/` and appended
K=8 samples only for `code`, `compiler`, and `string`:

```bash
PYTHONPATH=. .venv/bin/python extra/llm_json_rejection_sample.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --policy bench/qk-shared-storage-20260612/8b/policy.json \
  --adapter bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5 \
  --input bench/qwen-adapter-20260613/training-data-v4/sft.jsonl \
  --out bench/qwen-adapter-20260613/training-data-v4-rs-v5-stratified-v1 \
  --device AMD --storage shared --prompt-format chat \
  --seed 20260614 --k 8 \
  --temperatures 0.0 0.2 0.5 0.8 0.05 0.1 0.15 0.25 \
  --max-accepted-per-source 1 --resume \
  --sample-categories code compiler string
```

Stratified v1 result:

- attempts: `2448`
- accepted attempts: `257`
- selected train rows: `217`
- weak-category selected rows: code `20`, compiler `0`, string `23`
- compiler near misses: `158`
- coverage gate:
  `bench/qwen-adapter-20260613/training-data-v4-rs-v5-stratified-v1/coverage-gate.md`
  fails with `compiler: selected_train_rows 0 < 20`.

Current Phase 4 decision: do not train V7 from either RS artifact. The issue is
not just sparse sampling; V5 produces compiler near misses with valid JSON form
but wrong values. More generic V5 temperature sampling is unlikely to solve this
without changing the generator, data/prompt shape, scorer normalization, or
objective.

Compiler near-miss audit:

`extra/llm_json_nearmiss_audit.py` generated
`bench/qwen-adapter-20260613/compiler-nearmiss-audit-v1/` from both RS artifacts
plus the V6 gold-control rollout. The audit chooses `prompt_data_fix`.

Key findings:

- K=4 compiler near misses: `79` over `68` unique sources, `0` accepts.
- Stratified v1 compiler near misses: `158` over `68` unique sources,
  `0` accepts.
- K=4 miss classification: prefix `54`, empty string `16`,
  stem-without-index `8`, substring `1`.
- Stratified v1 miss classification: prefix `111`, empty string `32`,
  stem-without-index `14`, substring `1`.
- Top actual answers are broad prefixes: `"train_qk"`, `"train"`, `""`, and
  `"train_qk_gemv"`.
- Expected answers are row-specific glossary keys such as
  `train_qk_gemv_005`.
- V6 gold-control can learn some of this under gold supervision (`14/34`
  compiler eval passes), but V5 RS generation does not produce exact
  row-specific compiler keys.

Decision after audit: do not treat this as a normalization fix. Accepting
prefixes/stems would change the task contract and weaken the strict-JSON gate.
That intervention has now been run as V4.1 compiler-only data:

- dataset: `bench/qwen-adapter-20260613/training-data-v4_1-compiler/`
- rows: `68` train, `34` eval
- target: stable concept keys such as `qk_gemv`, not row-specific keys such as
  `train_qk_gemv_005`
- integrity: train/eval prompt overlap `0`, template-instance overlap `0`,
  intentional answer overlap `12`, numeric suffix answers `0`

V5 on the V4.1 compiler eval:

- artifact:
  `bench/qwen-adapter-20260613/8b-last1-ffn-suffix-lora-r4-v5-v4_1-compiler-rollout/`
- strict JSON pass: `30/34` (`0.882`, Wilson 95% CI `[0.734, 0.953]`)
- parse/schema/type/no-extra-text: `34/34`
- value-correct: `30/34`

V5 rejection sampling on the V4.1 compiler train split:

- artifact:
  `bench/qwen-adapter-20260613/training-data-v4_1-compiler-rs-v5-k4/`
- attempts: `272`
- accepted attempts: `68`
- selected train rows: `68`
- compiler selected rows: `68/68`
- accepted rows came from `temperature=0.0`; higher low-temperature samples
  were not needed for coverage
- coverage gate: `pass` at min `20` compiler selected rows

Current next step if continuing practical work:

1. Build a combined RS-SFT artifact that keeps usable non-compiler rows from
   the original V4/stratified RS artifacts and replaces the compiler slice with
   `training-data-v4_1-compiler-rs-v5-k4` rows.
2. Train V7 only from that combined artifact, using the same architecture as
   V5/V6 (`last1_ffn`, rank `4`, alpha `8`).
3. Promote only by free-generation strict JSON rollout. At minimum compare V5,
   V6 gold-control, and V7 on the original V4 `204`-prompt gate plus the V4.1
   compiler eval gate.
4. Use strict JSON pass rate, Wilson intervals, regressions, and category
   deltas as promotion signals. Teacher-forced token accuracy remains
   diagnostic only.

Suggested commit prefix once artifacts/docs are finalized:
`[test] finish Phase 4 rejection-sampling SFT eval`.

UPDATE 2026-06-15 -- Phase D scoped: teach tinygrad the int8/DP4A vocabulary so search can reach the
decode GEMV (the tinygrad-pure way to close the ~56%-of-llama.cpp decode gap). Doc:
`docs/amd-decode-dp4a-vocabulary.md`. Confirmed in code: `V_DOT4_I32_I8` is in the ISA assembler
tables but NO codegen pattern emits it; we only reached DP4A via the `Ops.CUSTOMI` inline-asm escape
hatch (`extra/q4_k_gemv_primitive.py`). That vocabulary gap (not a search failure) is why the GEMV opt
space was flat (M0). DP4A is LIGHTER than WMMA (per-lane vector dot, no TensorCore dataclass/swizzles):
needs only Ops.DP4A + a fold pattern (int8-dot idiom -> DP4A, mirror rangeify lower_shaped_wmma) + a
1-line renderer rule (`__builtin_amdgcn_sdot4`, mirror cstyle WMMA emit at cstyle.py:62) + a search
action (mirror the TC action) + the q8_1 activation path in the graph. Phases: D0 make-or-break
ceiling probe FIRST (does a HAND-written DP4A GEMV reach ~llama.cpp tok/s? if not, codegen work is
moot -- roofline discipline) -> D1 fold pattern -> D2 renderer emit -> D3 search-reachable -> D4
end-to-end measure (search-found, not hand-asm, vs llama.cpp). Next action: run D0.

UPDATE 2026-06-15 -- Phase D0 RAN: gate NOT cleared; do NOT build the DP4A codegen vocabulary.
`bench/.../dp4a-d0/RESULT.md`. Fresh decode numbers: stock tinygrad 15.8 tok/s; our Q4K_PRIMITIVE
(fp) 58 tok/s / 278 GB/s; llama.cpp 104 / ~470-500. Microbench (q8_1_q4k_bench.py): best int8 variant
intdot 242 Q4-GB/s ffn_gate (+40% vs fp 173) but ~50% of llama.cpp; EXPLICIT DP4A (vdot, the v_dot4
asm Phase D targets) is the SLOWEST (35, asm volatile blocks scheduling). End-to-end intdot wired into
model.py (Q4K_INTDOT, reverted) = 28 tok/s, REGRESSED on unfused per-layer q8_1 quant. Optimistic
fused ceiling ~81 tok/s (~78% of llama.cpp) -- improvement, not parity. DIAGNOSIS (consistent with
M0): decode is MEMORY/occupancy-bound; DP4A accelerates COMPUTE (wrong axis); llama.cpp's win is
memory-side + int8 ACTIVATION (fewer bytes), not the dot instruction. DECISION: do not build Phase D;
it optimizes the wrong thing. The D0 gate did its job (caught the wrong lever with a cheap probe).
The decode gap, if pursued, is an int8-activation + occupancy/memory problem ceilinged ~81 tok/s, not
the DP4A-codegen-vocabulary path. model.py reverted to pristine.

UPDATE 2026-06-15 -- Phase L scoped (exhaustive): add the MEMORY-LATENCY-HIDING vocabulary to
tinygrad codegen (the real decode frontier per D0). Doc: `docs/amd-decode-latency-vocabulary.md`.
Architecture finding: AMD default = HIPRenderer -> HIP C -> comgr/LLVM; tinygrad emits NO s_waitcnt /
NO async-copy -- LLVM does the scheduling, so explicit pipelining's marginal value is UNCERTAIN ->
probe-first. tinygrad's 10 opts (TC/UPCAST/UNROLL/LOCAL/THREAD/GROUP/GROUPTOP/NOLOCALS/PADTO/SWAP) +
synchronous LDS cover tiling/parallelism/occupancy-via-locals but LACK async/prefetch/double-buffer/
wave-control (standard in CUTLASS/MLIR/Pallas). Decomposed the missing vocabulary cheapest-first:
L1 occupancy control (waves-per-eu/__launch_bounds__, smallest, maybe THE lever), L2 software prefetch
(issue i+1 loads ahead, UOp loop transform), L3 double-buffered LDS (ping-pong, CUTLASS multistage),
L4 explicit async copy (__builtin_amdgcn_global_load_lds, heaviest). Each needs OptOps + apply_opt +
search action (make it findable). GATING: L0 hand-probes FIRST (L0a occupancy, L0b prefetch, L0c
double-buffer) -- if the best moves decode toward ~90-104 build that lever; if NONE moves it materially
-> latency-hiding isn't the binding constraint (occupancy ceiling / LLVM already hides it) -> stop.
Pre-registered: even int8 ceilinged ~81 (D0), so decode PARITY may be unreachable; a located ceiling is
an acceptable result. Next action: run the L0 probes.

UPDATE 2026-06-15 -- Phase L0 RAN: STOP. Latency-hiding is NOT the decode bottleneck; do not build
Phase L. `bench/.../latency-L0/RESULT.md`. L0a occupancy via existing parts/LOCAL: flat (~82-86 GB/s,
re-confirms M0). L1 occupancy-FORCING (patched HIPRenderer amdgpu_waves_per_eu, end-to-end, reverted):
WAVES_PER_EU 2/4/6 -> ~30 tok/s, 8 -> 21 -- forcing higher occupancy REGRESSES decode (baseline 58).
The compiler default occupancy is already optimal; decode is NOT occupancy-starved (a starved kernel
speeds up with more waves; this slows down). L2 prefetch-via-ILP: UPCAST/UNROLL flat (M0), LLVM already
schedules, kernels already overlap end-to-end (278 aggregate vs ~85-173 per-kernel). Verdict per
pre-registered gate: STOP. Both scoped decode levers now probed NEGATIVE -- DP4A (D0, compute, wrong
axis) and latency-hiding (L0, not the constraint). The residual ~2x decode gap to llama.cpp is a Q4_K
dequant-ALU-cost + kernel-count/fusion problem the scoped codegen vocabularies do NOT address;
realistic ceiling ~81 tok/s (~78%), PARITY likely unreachable via codegen vocabulary. cstyle.py
reverted to pristine. The roofline/probe-first discipline caught two non-binding levers (DP4A, latency)
before building either subsystem.

UPDATE 2026-06-15 -- Phase Q scoped: reduce the dequant INSTRUCTION COUNT (the triangulated #1 decode
bottleneck -- "memory-starved but instruction-bound", our M0 + literature: OSDI'25, LUT-GEMM,
async-dequant papers). Doc: `docs/amd-decode-dequant-instruction-count.md`. KEY honest framing: the
int-dot path (D0 `intdot` = llama.cpp mmvq structure) ALREADY captures most of the dequant reduction
(242 Q4-GB/s microbench, +40% vs fp) but regressed end-to-end to 28 tok/s because the per-layer q8_1
quant was UNFUSED. So the concrete win is NOT LUT -- it is FUSING that quant. Two techniques: A (fuse
q8_1 quant into the int-dot GEMV, one kernel + reuse quant across q/k/v and gate/up -> capture the ~81
ceiling end-to-end, a real 58->81 +40% win), B (LUT-GEMM 16-entry per-group table -- the field's named
fix but with a GPU caveat: runtime-indexed per-thread LUT must go in LDS, and int-dot already avoids
per-weight fp dequant, so LUT may not beat A; scoped as a skeptical probe). Phases: Q0a fused int-dot
probe FIRST (concrete) -> Q0b LUT probe (only if it beats Q0a) -> Q1 productionize winner.
Pre-registered ceiling: best dequant-reduction ~81 (~78%); parity (104) likely needs hand-asm
efficiency + strided-activation (#2) + kernel-fusion (#3), not instruction-count alone. Touch points:
extra/q4_k_gemv_primitive.py (intdot + fused-quant variant), tinygrad/llm/model.py (decode dispatch +
quant fusion/reuse). Next action: run Q0a (build the fused int-dot decode GEMV, measure end-to-end).

UPDATE 2026-06-15 -- Phase X scoped: lossy-quantization-aware search (the first CROSS-LAYER rung).
Doc: `docs/amd-decode-lossy-quant-search.md`. Extends the validated schedule-search loop (N1/N2) to the
ALGORITHM layer's LOSSY quant choices, searched WITH accuracy as a co-objective -- the win tinygrad's
semantics-preserving search can't propose. Grounded in the frontier: PET (OSDI'21, partial-equivalence
+ correction), Mirage (OSDI'25, joint algebraic+schedule multi-level); lossy quant is the least-charted
corner (NAS/approximate-computing-adjacent). HONEST CEILING pre-registered: X does NOT exceed the int8
ceiling (~81); mixed precision is a weighted avg between fp-slow and int8-fast, so it can't beat
uniform-int8 on raw speed -- its unique value is ACCURACY-CONSTRAINED speed, AUTOMATICALLY (capture the
int8 win where layers tolerate it, minimal fp fallback where they don't, no manual tuning). 81->104 is
the OTHER rungs (Mirage-style algebraic+layout+instruction), out of scope. New machinery vs N-loop:
(1) lossy-transform vocabulary (per-linear precision choice), (2) accuracy evaluator+gate (the hard
new piece), (3) multi-objective search (cost model predicts speed+accuracy). Phases: X0 make-or-break
"room+heterogeneity+learnability" probe FIRST (same as N0b but on the precision axis: does int8 have
accuracy room? is tolerance heterogeneous across layers? is it learnable? -- with 3 honest null
outcomes: no-room/constant/unlearnable) -> X1 vocabulary+accuracy harness -> X2 extend N-loop to
multi-objective per-layer precision assignment -> X3 end-to-end measure (highest tok/s within accuracy
budget). Touch points: qk_loop_*, q4_k_gemv_primitive (Q int-dot), model.py per-layer dispatch, new
qk_accuracy_eval.py. Next action: run X0.

UPDATE 2026-06-15 -- Phase X0 RAN: WEAK home; ship uniform int8 (Q), lossy-quant SEARCH not justified
at int8. `bench/.../lossy-X0/RESULT.md`. Captured per-layer int8 (q8_1) activation tolerance during
real decode (162 linears, QK_CAPTURE hook in model.py, reverted). (a) ROOM: int8 err 0.51-1.07% across
ALL layers (broadly viable). (b) HETEROGENEITY: WEAK -- 2.1x spread only (vs schedule space 111-223x);
by type ffn_down/attn_output worst ~1%, ffn_gate/up best ~0.5%. (c) LEARNABILITY: yes, corr(outlier,
int8_err)=0.757. Verdict = pre-registered "constant" branch: no layer strongly needs fp -> uniform
int8 captures it, mixed-precision SEARCH adds little at int8. The search's real home is at MORE
AGGRESSIVE precision (int4-activation/mixed bit-widths, the AWQ regime), not int8. Another "search
doesn't help here; the simple thing (uniform int8) does."

MIRAGE NEXT (for the last percent, 81->104): written into `docs/amd-decode-dequant-instruction-count.md`
(section "NEXT: Mirage / multi-level superoptimization"). The 81->104 residual is the CROSS-LAYER rungs
(algebraic + layout + custom-kernel + instruction selection) that schedule-only search + a single
dequant kernel don't touch -- exactly Mirage (OSDI'25, arXiv:2405.05751) territory (joint
algebraic+schedule + custom-kernel discovery, uGraph hierarchy, probabilistic equivalence). It is
"machine search but cross-layer" -- the top rung of the thesis we validated on the schedule axis
(N1/N2). Cheap make-or-break before adopting: does Mirage's joint search find a decode-GEMV
reformulation/custom-kernel that beats the Q int-dot ~81 on this GPU? Yes -> closes the last percent
the search way; no -> residual is hand-asm/microarchitectural (the Writer). PET (OSDI'21) is the
partial-equivalence companion. This is the single most promising "close it the search way" direction.

UPDATE 2026-06-15 -- Q0a RAN: FAILED. The int-dot ~81 is a MICROBENCH ARTIFACT; fp (58) stays best.
`bench/.../q0a/RESULT.md`. Built the LDS-fused q4k_q8_1_fused_intdot_kernel (kept in
q4_k_gemv_primitive.py as a documented negative): correct (rel_err 0.0073) but standalone 10 Q4-GB/s
(~24x slower than separate int-dot's 242), end-to-end 6 tok/s (vs fp 58, D0 28). WHY: the phase-1
quant prologue is NOT hoisted -- tinygrad's lowering replicates it per OUTPUT ROW instead of once per
workgroup (the recurring fused-staging wall: W2, G0''). The D0 microbench 242/+40% assumed a FREE
pre-quantized activation; end-to-end the quant must be paid and BOTH strategies lose to fp (separate
launch 28, replicated prologue 6). Pre-registered ~75-81 gate FAILS. fp (58 tok/s, 56% of llama.cpp)
remains the best decode kernel. ALL codegen-reachable decode levers now NEGATIVE end-to-end:
DP4A(D0), latency(L0), lossy-quant-int8(X0 weak), int-dot(Q0a). The residual gap to llama.cpp is the
cross-layer rungs (Mirage OSDI'25, study queued) or the Writer/hand-asm -- exactly as the "why"
(single-layer-search-vs-cross-layer-codesign) analysis predicted. model.py reverted to pristine.

UPDATE 2026-06-15 -- Mirage probe (Mi0): BLOCKED on 3 independent grounds; no win on our target.
`docs/amd-decode-mirage-probe.md`. (1) HARDWARE: Mirage is CUDA/NVIDIA-only (runs its search on an
NVIDIA GPU, emits CUDA/Triton); this box has no nvidia-smi/CUDA/nvcc -> can't build or run. (2)
CODEGEN: its discovered fusions don't port -- tinygrad's lowering can't express efficient fused custom
kernels (the fused-staging wall: W2 dequant prologue, Q0a quant prologue replicated per row ~24x). (3)
PRIZE: limited on-target anyway -- per-kernel microbench 85-173 GB/s < end-to-end 278 GB/s, so the JIT
ALREADY pipelines the 252 launches; the 58->104 gap is per-kernel BANDWIDTH UTILIZATION (dequant
instruction count, Q, already negative), not launch fusion. VERDICT: Mirage's value is real but on
NVIDIA/CUDA; on AMD/tinygrad it's triple-blocked. HONEST END-STATE of the decode investigation: every
codegen/search-reachable lever exhausted & negative (M0/L0 schedule-occupancy flat; D0 DP4A; Q0a
int-dot; X0 lossy-quant; Mi0 cross-layer). fp 58 tok/s (56% of llama.cpp, 32% HBM peak) is the
tinygrad/AMD decode ceiling; parity needs the Writer (hand-written AMD kernels = what llama.cpp is).
The program's POSITIVE result stands: the loop on the SCHEDULE axis for the BATCHED regime (N1/N2,
33-98% peak). Single-stream quantized DECODE parity on AMD/tinygrad is not reachable without hand
kernels, on this evidence.

UPDATE 2026-06-15 -- Q0a COOP FIX: proved tinygrad CAN express a fast fused decode GEMV (409 GB/s
standalone, near llama.cpp 470), but it regresses e2e (24 tok/s) -- structural, not the kernel.
`bench/.../q0a/COOP_RESULT.md`. DIAGNOSIS of the original Q0a slowdown: NOT a lowering-hoist bug -- a
THREAD-ASSIGNMENT CONFLICT (the LOCAL opt's threads went to phase-1 quant, so phase-2's dot ran
redundantly across all threads -> 24x). FIX: hand-managed cooperative kernel
`q4k_q8_1_coop_fused_kernel` (amd_copy_matmul pattern -- workgroup=block_m rows; the block_m local
threads do BOTH cooperative quant-into-LDS AND their own row's dot). Correct (rel_err 0.004),
STANDALONE 409 Q4-GB/s (vs broken-fused 10, separate-intdot 242, fp 173). But END-TO-END 24-25 tok/s
(block_m 16/32/64 all same) -- REGRESSED vs fp 58. WHY: the fused kernel re-quantizes x PER WORKGROUP
-> needs LDS (~4.5KB, CAPS occupancy) + a BARRIER (breaks the inter-kernel PIPELINING decode lives on:
fp barrier-free e2e 278 > per-kernel 173; coop e2e 117 << standalone 409). LDS+barrier is the wrong
structure for occupancy/latency-bound small-GEMV decode. THE FIX IT REVEALS (= llama.cpp's structure):
the LDS+barrier is ONLY needed because of per-workgroup re-quant. llama.cpp quantizes x ONCE/token
(cheap global pass), GEMVs read global q8 BARRIER-FREE (pipeline). So next: AMORTIZED GLOBAL QUANT +
the barrier-free q4k_q8_1_intdot_partial_kernel -- quantize the shared activation once (attn-input ->
q/k/v; ffn-input -> gate/up), reuse across linears. D0 had the barrier-free int-dot at 242 standalone
/ 28 e2e WITH per-linear quant (7x/layer); amortizing to ~2x/layer is the untested path that could
beat fp -- needs model-forward surgery (quantize x once in the attn/ffn block, pass q8 to linears),
NOT a fused kernel. This is the FIRST decode lead that is not a dead end. model.py reverted to pristine;
coop kernel kept in q4_k_gemv_primitive.py.

UPDATE 2026-06-15 -- FINDING RECORDED + Phase A scoped. `docs/amd-decode-amortized-quant-plan.md`.
THE FINDING (reframe): the coop kernel proved tinygrad CAN express a 409 GB/s fused decode GEMV (near
llama.cpp 470) -- "the kernel is the wall" is FALSE. The decode gap is activation-quant PLACEMENT: the
fused kernel re-quantizes per workgroup (LDS+barrier, kills occupancy/pipelining -> e2e 24); the RIGHT
structure (llama.cpp) quantizes x ONCE/token + barrier-free int-dot GEMVs that pipeline -- expressible
now, never measured e2e (D0 only did per-linear quant -> 28). PHASE A plan: A0 make-or-break (amortize
quant via CACHING keyed by the input activation's UOp -- q/k/v share attn-input, gate/up share
ffn-input -> JIT graph has ONE quant feeding shared barrier-free int-dots; measure e2e vs fp 58, gate
>70 = win) -> A1 productionize (per-shape policy + accuracy/perplexity check; X0 says int8 broadly
viable) -> A2 vs llama.cpp 104. Pre-registered ceiling ~75-81 (int-dot 1.4x per-kernel; ~75% of
llama.cpp), NOT parity (residual = cross-layer/hand-asm rungs). Touch points: model.py (quant-cache +
dispatch), existing q4k_q8_1_intdot_partial_kernel (barrier-free), q8_1_quantize. Cheap to test (A0 =
cache + dispatch swap, no new kernel). FIRST decode lead that is not a dead end. Next action: build A0.

UPDATE 2026-06-15 -- Phase A0 RAN: premise REFUTED; the int-dot KERNEL (occupancy), not the quant, is
the wall. `bench/.../q0a/A0_RESULT.md`. Amortized quant via cache keyed by x_vec UOp -- cache HIT (36
hits, 22%, q/k/v + gate/up share), so amortization WORKED. But e2e = 28 tok/s / 136 GB/s = IDENTICAL
to D0, HALF of fp's 278. Amortizing the quant changed nothing -> D0 MIS-ATTRIBUTED its 28 to the quant;
the barrier-free int-dot KERNEL is the e2e bottleneck (136 GB/s; standalone 242 -> e2e 136, OPPOSITE of
fp 173 -> 278) due to int-accumulator register pressure (~16 REGs) -> low occupancy -> no pipelining.
BOTH int-dot structures lose e2e for the SAME reason (occupancy, not compute): fused-LDS coop (24),
barrier-free (28). fp (58) wins e2e via simple accumulator -> low regs -> high occupancy -> pipelines.
In occupancy-bound small-GEMV decode, kernel SIMPLICITY beats compute efficiency; the int-dot standalone
win is an e2e mirage in every structure. FINAL: fp 58 (56% of llama.cpp) is the tinygrad/AMD decode
ceiling; llama.cpp wins via occupancy-efficient hand-asm mmvq (DP4A-packed, minimal regs = the Writer)
that tinygrad doesn't produce. The reframe (tinygrad CAN express 409 GB/s standalone) STANDS but doesn't
translate e2e. Multiply-confirmed honest end-state. model.py pristine.

## 2026-06-15 — Instruction-count measurement (the consolidation's open question, RESOLVED)
Wrote `docs/amd-decode-consolidated-first-principles.md` (MEASURED/REFUTED/CONFIRMED ledger:
memory-bound REFUTED by READRAW 730 vs GEMV 365; occupancy REFUTED by VGPR 47/68/93; ALU-instruction
CONFIRMED). Then DID the measurement it called for — disassembled the emitted fp & int-dot kernels,
counted VALU/weight in the hot body:
- fp = **4.06 VALU/weight** (unpack 1.0 + int→fp convert 1.0 + scalar dot-fma 1.0 + affine 1.0).
- DP4A floor ≈ **1.35** (unpack 1.0 + v_dot4 0.25 + amortized affine) → **~3x headroom below fp, all in the dot.**
- fp is a **tinygrad-codegen floor, not a Q4_K instruction floor.** But tinygrad emits **zero v_dot4**
  (measured both kernels); its int path = scalar v_mad_i32_i24 + qsum + readfirstlane + more regs →
  worse e2e. Headroom locked behind a v_dot4 renderer lowering tinygrad lacks.
Result: `bench/.../q0a/INSTRUCTION_COUNT_RESULT.md`. Commit 66913464b, pushed. model.py pristine.
Decode end-state: the gap to llama.cpp is a ~3x dot-instruction gap, realizable ONLY by a DP4A codegen
feature (renderer), not single-layer search — quantifies the hand-asm/Writer boundary in instr/weight.

## 2026-06-15 — Phase L: made the loop LIVE (final-report follow-up #1)
N2 proved the guided loop but LOOKED UP measured times. Phase L (`extra/qk_loop_live.py`) times
candidates LIVE on device (the qk_beam_log `_time_program` path) on FRESH shapes absent from the
26-shape corpus, reusing the N1 XGBoost model + 277-config space.
- **L0** (4096,14336,128): guided@1=0.91, guided@8=0.979 of live oracle vs random 0.86; 95% in 5
  timings (random ~49); 0.89s vs 36.6s exhaustive = **41x wall-clock win**. PASS. `loop-live-L0/`.
- **L1** (5 fresh shapes): mean guided@8=0.977 vs random 0.821; median 3 timings to 95% (random ~82);
  mean **42x** wall speedup. 3/5 hit 1.0. Honest weak spot: small-N (4096,11008,64) k95=12/guided@8=0.92
  (187/277 valid) — the under-sampled N=64 regime. PASS. `loop-live-L1/` + test_qk_loop_live.py (5 green).
Commits b725e2e29 (L0), cc62c8412 (L1), pushed. The offline N1/N2 result HOLDS on real silicon for
unseen shapes — the loop is now a working live autotuner, not a simulation.
L2 (native tinygrad BEAM warm-start hook) is the remaining stretch — gated, OOD-risky, optional.

## 2026-06-15 — Phase L2: native-BEAM warm-start (honest NEGATIVE, bounds the live loop)
Wired the N1 model into tinygrad's native beam_search via an optional default-OFF hook
(search.py `_BEAM_CANDIDATE_FILTER`); model prunes each iteration's candidates to top-K.
RESULT (fresh 4096,14336,128, serial A/B): NO keep_k both saves wall-clock AND preserves quality —
k=12: 8.5x wall / q=0.60; k=24: 5.8x / 0.68; k=48: 1.9x / 0.91. Gate FAIL (pre-registered).
Cause: model trained on COMPLETE 277-config schedules is OOD on BEAM's PARTIAL schedules + has no
features for BEAM's larger action pool (SWAP/GROUP/THREAD — cold winner uses SWAP). Aggressive prune
kills quality, loose prune kills speedup. `loop-live-L2/`, harness qk_loop_beam_warmstart.py.
Two-sided Phase L COMPLETE: live loop is a 42x autotuner on ITS substrate (L0/L1) but does NOT transfer
to a structurally different search substrate (native BEAM) without retraining. Final report addendum
added. Commit a5f25bde0, pushed. (Transient AMD HW fault memory_lost=1 hit one sweep run; GPU
auto-recovered; clean rerun reported.) tinygrad default behavior unchanged (hook no-op when unset).

## 2026-06-15 — Scale-substrate (S) + v_dot4 (D): D0 major win; S1/S2 blocked
Pursued the two follow-ups (docs/amd-loop-scale-and-vdot4-plan.md).
- **D0 PASS (major)**: the schedulable builtin __builtin_amdgcn_udot4 (gfx1100, unsigned, target("dot-insts"))
  emits v_dot4 and at full occupancy = 169.6 Q4-GB/s ~= fp 173, 2.54x over asm-volatile v_dot4 (66.7),
  exact-correct, ~1.58 VALU/weight (vs fp 4.06). Phase D's "DP4A is dead" was an ASM-VOLATILE-BARRIER
  artifact; the builtin reopens the decode lever. qk_vdot4_builtin_d0.py, dp4a-d0/BUILTIN_VS_ASM_RESULT.md.
- **S1 GPU-BLOCKED**: default-off _BEAM_SCHEDULE_LOG hook (search.py) + qk_partial_schedule_log.py built,
  but native BEAM over its FULL action space HANGS gfx1100 (Wait timeout / memory_lost HW faults). Only
  the curated 277-config substrate is stable. Hook stays for a future stable run.
- **S2 BLOCKED**: conv ast builds, but matmul opt-candidate set fails on conv reduce (KernelOptError,
  different axes); conv reduce baseline 0.1 TF (likely flat). Needs a conv-specific opt set.
- **D1 PARTIAL/open**: builtin GEMV is kernel-competitive (=fp standalone); e2e needs target attr on
  tinygrad's generated kernel (core render_kernel change) + must beat the pipelining wall (int-dot 242->136
  e2e). The decisive decode-parity test, not yet run.
Commits f174d86e4, b4d10f6f8 pushed. GPU had repeated transient HW faults under heavy BEAM timing this session.

## 2026-06-15 — D1 COMPLETE: builtin v_dot4 decode GEMV finished out (kernel win, e2e null)
Finished the v_dot4 e2e test. Path: renderer emits `_dp4a` device helper (target("dot-insts") +
__builtin_amdgcn_udot4) when a CUSTOM body references it (cstyle.py, default-off);
q4k_q8_1_vdot_builtin_partial_kernel with 64-row/wg occupancy fix; Q4K_VDOT=1 decode dispatch (model.py).
- STANDALONE: builtin udot4 GEMV = 302 Q4-GB/s, 1.77x FASTER than fp (171), correct. Headroom realized.
- E2E (cli --benchmark, Qwen3-8B): 30.2 tok/s = fp 30.3 -- IDENTICAL, despite 1.77x kernel + half
  bytes/token (2036 vs 4762 MB). Decode is latency/launch-bound at the TOKEN level, not GEMV-throughput.
VERDICT: v_dot4 lever is REAL at kernel level (overturns Phase D's asm-volatile negative) but NULL e2e.
Decode gap is structural (per-token latency across ~252 launches), not a single-kernel codegen gap.
Closes the decode lever hunt. Commit cc9cacdf2 + docs. (Machine fp baseline ran 30 tok/s this session
vs historical 58 -- GPU degraded after HW faults; comparison is apples-to-apples same-run so null holds.)
Q4K_VDOT default-off; renderer _dp4a gated; default decode unchanged (verified).

## 2026-06-15 — B1 horizontal-fusion probe: NEGATIVE, relocates bottleneck, forks to speculation
Fused q/k/v->1 GEMV + gate/up->1 (concat Q4_K rows, Q4K_FUSE default-off). Result: 26.6 tok/s (-12% vs
30.3), correct, only -36/766 kernels. DECISIVE finding (kernels/token): TinyJit collapses the ~730-kernel
decode into ONE replayed graph (~6 host-kernels/token) -> HOST LAUNCH OVERHEAD ALREADY GONE. Horizontal
fusion trades launch-count for output-split ops (~break-even); GPU work unchanged. The 33ms/token is
GPU-side sequential execution of ~730 memory-latency-bound batch-1 kernels -> lever is PARALLELISM PER
KERNEL (a batch dim), not fewer launches. Gate <=0% -> PIVOT to speculation/batching (Strategy A): B0's
13-26x batching applies, validated loop (N1/N2/L0/L1) is the substrate, draft model = the fine-tuning lever.
Rules out megakernel ladder for single-stream on this stack. Commit 8bc5cedb0. fusion-probe-B1/RESULT.md.
Next make-or-break: batched-decode-forward latency curve (B=1,2,4,8) = the speculation ceiling.

## 2026-06-15 — Step 2: loop beats heuristic ~1.9x on decode-verification GEMMs (first machine-search-helps-decode)
Ceiling probe (qk_batch_ceiling_probe.py): batching amortizes weights ~2.4-3.5x, plateau ~14ms/tok =
COMPUTE-bound (16 GB/s, untuned batched matmul), NOT memory floor -> tunable. Step 2
(qk_decode_verify_loop.py): ran the GPU-safe curated loop on held-out 8B FFN verify shapes (M/K=12288,
N=8,16). Loop/heuristic (the forward's actual hand_coded schedule): 0.99/1.97/1.26/3.3x, mean 1.88x;
guided/oracle 0.97. GATE PASS. Honest: 1.9x not the misleading 42x-over-naive; win at N>=16 (heuristic
decent at N=8); plateau drop <=1.9x (Amdahl, matmul fraction TBD). First concrete machine-search-improves-
decode result, in the batched/speculative regime. Two stacking levers: ~2.4-3.5x batching x up to ~1.9x
loop-tuning. Commits cea32bf19, c2f8a3798. Next: apply loop schedules to the batched forward, re-measure
plateau (size the realized lever); then speculative scaffold for e2e tok/s.

## 2026-06-15 — Batched-decode TC realization (option 1): DEFINITIVE NEGATIVE
Chased "machine search closes decode via TC in the batched/speculative regime" to the end.
- Mis-diagnosis corrected: the no-TC plateau is a DTYPE problem (verification matmul runs fp32; RDNA3 WMMA
  needs fp16), NOT fusion. Verified fp16+TC=16.3TF, fp32+TC errors "no tensor core available".
- Fix = model fp16 cast (NOT tinygrad codegen). TC then fully applies (warmstart apply:4 on all FFN matmuls).
- BUT e2e NEGATIVE (T=16): fp32 18.1 -> fp16 19.8 (cast overhead) -> fp16+TC 26.2 ms/tok (SLOWER). TC at
  batch-16 net-negative: WMMA setup + PADTO blowup (12288=256x16x3, the 3 pads to 16 = ~5x waste) + Amdahl.
End-state: kernel-vocabulary levers (TC/fusion) do NOT close decode -- single-stream (latency-bound, v_dot4
proved it) OR speculative K=16 (TC realizes but hurts). Lever real isolated (2x), never translates e2e --
the recurring thesis. Genuine remaining decode lever = fewer bytes (lower-bit/MoE) or megakernel (tinygrad
can't express), not a kernel vocabulary. Commits a4a7e8660, 8838843f1. docs/amd-decode-option1-*.md.
All experiment flags (Q4K_VDOT/FUSE/UNFUSE/WARMSTART) default-off; normal decode unchanged.

## 2026-06-15 — DECODE ARC: 23 → 60.9 tok/s (2.65x), 22% → 58% of llama.cpp (capstone: docs/amd-decode-capstone.md)
Picked up from the "kernel-vocabulary levers don't close decode" negative and RE-LOCALIZED the bottleneck —
finding it was never the kernel. Two clean machine-search-beats-llama results landed.

MEASUREMENT RE-BASE (killed 3 confounds: Infinity Cache, launch overhead, memory-clock ramp). Corrected
several wrong conclusions (logged): "in-graph GEMV is 12%" (wrong kernel), "weight read is 95% of token"
(circular), "small kernels can't saturate" (clock ramp), "decode is host-bound" (single-graph artifact; it's
~99% GPU-busy). Standalone WIN: the int-dot (v_dot4) Q4_K GEMV sustains 76% of peak vs llama's 57% — the
machine kernel BEATS the reference standalone. (amd-decode-flywheel-proof-20260614/{KERNEL_BEATS_LLAMACPP,
prefetch-gemv/{PERLAYER,BREAKDOWN}_RESULT}.md)

ROOT CAUSE + WINS (all in tinygrad/llm/model.py, gated):
- Q6_K coverage (THE big one): Qwen3-8B-Q4_K_M is mixed-quant; the Q6_K matmuls (ffn_down 18/36 layers,
  attn_v, lm_head) had NO primitive and ran a slow fallback (r_32_32_4_48 = 59% of GPU work). The Q6_K
  primitive was built but gated off. **Q4K_PRIMITIVE now implies Q6K_PRIMITIVE (default-on): 23 → 53.5 tok/s
  (2.2x), byte-identical output** (Q6_K dequant exact). Q6K_FIX_RESULT.md.
- Q6K_COVER_MORE (attn_v + lm_head, default-on): +5% → 53.5, exact. (was wrongly "loses to fused graph".)
- B3 bit-width (SHIPPED, first shippable search-beats-llama-quant): built the missing Q4_K QUANTIZER
  (extra/qk_quantize.py, llama make_qkx2 port, validated BIT-EXACT vs llama's own Q4). ffn_down is
  over-provisioned (Q6 where Q4 is ~free); **Q6K_DEMOTE_FFNDOWN=1 requants the 18 Q6 ffn_down -> Q4 at load:
  53.4 → 60.9 tok/s (+14%) at dNLL -0.0028 (free).** Lossy (output diverges, coherent), gated, load-requant
  ~2-3min (cacheable). B3_DEMOTE_RESULT.md.

CLOSED DEAD-ENDS (measured, not assumed): B1 in-graph int-dot = per-kernel GEMV at its batch-1 occupancy
ceiling (~llama's 57%); int-dot/split-K/fusion all within noise (B1_INTDOT_RESULT.md). B5 speculative =
exact + verify-fast but net-negative (1.7B draft too costly vs 8B; S0+S1 results). S3 batched GEMM primitive
(q4k_gemm_kernel + q6k_gemm_kernel, tests) BUILT + verified, dormant (gated Q4K_BATCHED) — used by the
speculative verify; symbolic prefill can't use it.

NEXT TASK — P2 flash-decode (the biggest real-world lever): decode collapses 56→14 tok/s from ctx 8→3072,
attention-bound. Approach A (Tensor-level KV split) FAILED the S0 gate (slower; tinygrad won't parallelize
the split). Approach B custom Flash-Decoding kernel BUILT + verified exact (extra/qk_flash_decode.py,
test_qk_flash_decode, max_err ~1e-7). REMAINING = model integration: custom_kernel needs a UOp-builder fxn
(no raw-C bridge); UOp.range takes symbolic bounds so a symbolic split S=cdiv(Tc,L) works. De-risked path:
precompute scores via matmul (avoids nested reduce) + augment v with a 1s column (folds softmax denom) +
split-softmax custom_kernel (occupancy) + reduce. Finicky .set/.after/.end accumulator pattern (use q4k_gemv
_partial_kernel as template). Full plan: docs/amd-decode-flash-attention-plan.md (INTEGRATION STATUS).
All flags default-off except Q6_K-on-with-Q4K and COVER_MORE (exact). Memories: amd-decode-{next-step,
real-bottleneck,kernel-beats-llamacpp,measurement-confounds}.

## 2026-06-16 — P2 flash-decode SHIPPED: model integration done, exact, long-context collapse fixed
Closed out the open "NEXT TASK" above. The UOp flash-decode is built, wired into the model (gated
`FLASH_DECODE=1` in `TransformerBlock._attention`), and verified **byte-exact vs SDPA** (40 identical greedy
tokens at ctx 8 and ctx 1024). `extra/qk_flash_decode.py` (`flash_decode_attention` + 5 kernels), test
`test/external/test_qk_flash_decode.py::test_uop_flash_decode_attention`, bench `extra/_flash_bench.py`,
plan `docs/amd-decode-flash-attention-plan.md` (SHIPPED section).

DECODE tok/s vs context (Qwen3-8B Q4_K_M, `Q4K_PRIMITIVE=1`, RX 7900 XTX, median of 12):
| ctx | SDPA | FLASH | speedup | llama.cpp | flash % of llama |
|---|---:|---:|---:|---:|---:|
| 8    | 56.2 | 47.5 | 0.84x | 99.9 | 48% |
| 1024 | 27.6 | 34.3 | 1.24x | 98.2 | 35% |
| 3072 |  9.4 | 22.7 | **2.41x** | 94.0 | **24%** |
(llama.cpp = `llama-bench -ngl 99 -n 128 -d <ctx>`, runs flash-attention; near-flat 99.9->94.0.)

HEADLINE: the SDPA long-context **collapse** (6.0x drop 8->3072) is **flattened to 2.1x** — exactly the P2
target. Honest framing: this removes the tinygrad-specific attention collapse (ctx-3072 went 10%->24% of
llama) but does NOT make tinygrad competitive at long ctx — llama is still ~4x faster there (pre-existing
~58% per-token baseline gap + my attention kernels are less optimized: 5 separate f32 kernels vs llama's
fused fp16). Flash also doesn't fully flatten relative to llama (48%->24% of llama as ctx grows), so residual
attention scaling remains. SHIPS **gated (default off)** — crossover ~ctx 400; flash regresses short-ctx ~15%
from the 5 extra kernel launches/layer. Enable for long-context serving. `FLASH_L=256` default (won the L
sweep: 128/512 both worse at ctx 1024, 512 much worse at 3072).

DESIGN (5 single-accumulator UOp kernels — the q4k_gemv_partial pattern): precompute scores via
`grouped_q @ k^T` matmul into a concrete `[Hq,MAXC]` buffer (avoids nesting a q.k reduce in the custom
kernel), then `flash_max` (per-split max) -> `flash_partial` (exp-weighted partial out; 1s-augmented v folds
the softmax denom) -> `flash_gmax` (global max) -> `flash_den` (denom) -> `flash_combine` (LSE reduce).
WHY 5 NOT 2: tinygrad's linearizer range-ordering rejects coupled/multi-accumulator reduces in one kernel
(online-softmax's m/l/acc cross-reference, and two siblings ending the same range) — each kernel must be ONE
independent single-accumulator reduce. GOTCHAS SOLVED (the multi-iteration grind the plan predicted): (1)
multi-dim store index breaks tinygrad's local-dim auto-mask -> flatten to 1D indexing; (2) `opts_to_apply=()`
to skip the heuristic LOCAL opt; (3) symbolic split S=cdiv(start_pos+1,L) — a BIND in a custom-kernel range
AST fails type_verify, so use the UNbound `DEFINE_VAR` twin for kernel ranges and the bound `start_pos+1`
only in the score-matmul slice (which carries the value into the shared var_vals); buffers sized at concrete
`Smax`, ranges/strides use symbolic S<=Smax; tinygrad `cdiv` truncates (not ceiling) -> use `(a+b-1)//b`; (4)
**the model-only NaN bug** (exact standalone, garbage in-model): masked lanes computed `p(=0) *
vc[uninitialized KV cache](=Inf) = NaN` poisoning the accumulator — fix: clamp the out-of-range v index to a
written position so masked lanes read finite data.

NEXT (modest, optional): merge gmax/den/combine to cut the short-ctx launch overhead and push the crossover
lower; go fp16 in the attention math to close the residual long-ctx scaling vs llama. Memory updated
(`amd-decode-next-step`). All flags still default-off except Q6_K-on-with-Q4K and COVER_MORE.

## 2026-06-16 — Scorecard completed vs llama: PREFILL is the worst gap (corrects a wrong inference)
Filled the two unmeasured cells. PREFILL was assumed "likely close to llama" — it is NOT. Measured (warm
time-to-first-token, distinct warmup tokens so the prefix-cache doesn't skip prefill; bench `extra/_prefill
_bench.py`): ours **~65 tok/s** (512/1024/3072: 67/66/60) vs **llama-bench pp ~3000** (`-p 512,1024,3072
-n 0`) = **~2% of llama, ~45x behind**. Config-independent (same ~66 with and without `Q4K_PRIMITIVE`). Our
prefill runs at ~decode-speed*N — ZERO batching benefit (~1 TFLOP/s effective; a 32-token chunk costs ~32
decodes). The batched matmuls stay memory-bound instead of going compute-bound like llama's.
Full scorecard vs llama (this GPU, Qwen3-8B Q4_K_M):
| category | ours | llama | vs llama |
|---|---:|---:|---|
| Decode baseline (GEMV/FFN) | 56 | 100 | ~58% (2x behind) — hand-asm wall |
| Decode attention short | (in 56) | — | ~fine |
| Decode attention long (ctx 3072) | 22.7 | 94 | 24% (flash, today) |
| **Prefill** | **65** | **3000** | **~2% (45x behind)** |
| Batched/speculative | = prefill regime | — | same untuned-matmul story |
| lm_head | (in 56) | — | done |
KEY: the gap is NOT uniform. Decode ~58% (hand-asm wall); prefill ~2% (untuned matmul). The two are
different problems — decode is memory-bound at batch-1 (can't tune past), prefill is COMPUTE-bound but
UNTUNED, which is exactly the loop's proven substrate (N1/N2 hit 33-98% of peak on native matmul). So the
highest-leverage unrealized target is PREFILL via the curated-loop / matmul tuning (NOT native BEAM — it
hangs gfx1100). bench `extra/_prefill_bench.py`; llama bar `llama-bench -ngl 99 -p N -n 0`.
PREFILL SCOPED + DIAGNOSED (P0-P2, `docs/amd-decode-prefill-plan.md`): no prior prefill doc existed (always
"the other regime", deliberately excluded). P0: prefill = 1.3% of fp16 peak, NO WMMA; DEBUG=2 shows each block
runs as ONE fused `function` ~86 ms/32-tok (Q4_K dequant fused into the block mega-kernel -> untiled). P1: NO
flag fixes it (Q4K_UNFUSE/TC/Q4K_BATCHED no-op; REALIZE=1 WORSE at 22 tok/s) -- the fusion must be broken.
P2: fix direction PROVEN standalone -- `matmul_decoded` (dequant->fp16->NATIVE matmul) is 5-18x faster than
the fused path at batch-32 (ffn_gate 15.4% peak / 18x), projecting prefill ~2% -> ~15-25% of llama. Remaining
= WIRE it (a correctness-critical prefill-forward restructure, not a flag).
P2-WIRE ATTEMPT 1 (Linear-level `PREFILL_FP16` fp16-contiguous branch in `_fallback`): FAILED -> 28 tok/s
(worse, like REALIZE=1); reverted, model.py pristine. Root cause is MULTI-FACTOR (not a Linear edit): (1)
prefill's batch dim T is SYMBOLIC (v_toks) -> measured 2.2x slower than concrete (TC/tiling want concrete
dims); (2) untuned matmul (~2-15% peak); (3) per-chunk dequant. Real fix = prefill-DRIVER restructure:
concrete fixed-size chunks (pad to 32 -> concrete matmul dims) + amortized per-layer dequant + warm-started TC
schedules + token parity. Bigger than first scoped; the standalone 5-18x proves the ceiling exists.
PREFILL VERDICT (exhausted, PARKED as a located negative): EVERY accessible lever measured NEGATIVE on the
real in-model prefill -- Q4K_UNFUSE/TC/Q4K_BATCHED no-op; REALIZE=1 22 tok/s; PREFILL_FP16 28 (reverted);
concrete-T=32 ~same as symbolic (symbolic batch is NOT the in-model cap); orientation 1.0x; chunk_size 32/128/
512 = 67/69/39 (bigger batch does NOT help). Prefill is **95% GPU-busy** (GPU-bound, not launch overhead) with
in-model matmuls at ~1.3% of peak. The SAME matmul as a clean top-level kernel hits ~13 TF (matmul_decoded),
but inside the @function precompiled block graph tinygrad schedules it far below peak -- same class of wall as
decode (good standalone kernels, bad in-model scheduling). Fix needs (a) transferring loop-tuned schedules INTO
the @function forward (unsolved; L2 showed no cross-substrate transfer) or (b) hand-asm GEMM (the Writer) --
both out of the "wire an existing block" scope. model.py pristine. Full sweep: `docs/amd-decode-prefill-plan.md`.

## 2026-06-16 — DEFAULT FLIP: Q4K/Q6K primitives now default-ON (path-aware, shared storage). The arc's win, out-of-the-box
The recurring "biggest lever" lesson (a built win gated OFF) was still the live default: the master flag
`Q4K_PRIMITIVE` defaulted to 0, so a plain `from_gguf` ran the dense path at ~12 tok/s instead of ~55.
Flipped it path-aware after the S0 safety matrix. `tinygrad/llm/model.py`:
- `Q4K_PRIMITIVE` now defaults ON when `q4k_auto` (env unset AND gguf is a path AND `Device.DEFAULT=="AMD"`);
  OFF for a preloaded Tensor or non-AMD device. Q6K follows (auto-on, exact). `Q4K_PRIMITIVE=0` forces dense.
- Storage: when AUTO-enabled, defaults to `shared` (views the GGUF in place, storage_bytes=0) instead of
  `sidecar` (which DUPLICATES weights). Explicit `Q4K_PRIMITIVE=1` keeps the `sidecar` default unchanged;
  `QK_PRIMITIVE_STORAGE` always wins. This is the narrow shared-default the handoff deferred — coupled to
  auto-enable only, reversible, and now justified by S0 + the prior 8B/14B/32B greedy-A/B passes.
S0 safety matrix (VRAM 25.75 GB): 8B on=exact, peak 10.78 (sidecar) / **6.24 (shared, == dense)**; 14B
shared fits; **32B sidecar OOMs (~38 GB) -> shared makes it safe (storage_bytes=0)**; non-Q4K Q8_0 = graceful
no-op; Tensor input -> off. S2: default (no flags) = **55.0 tok/s vs dense 12.4 (4.4x), token-EXACT**, shared
storage, no extra VRAM; explicit `Q4K_PRIMITIVE=1` still sidecar. Probe `extra/_s0_safety.py`.
RESIDUAL RISKS (bounded by the `Q4K_PRIMITIVE=0` escape): MLA/MoE/SSM arches untested locally (only dense
Qwen3 here) — install is per-tensor so it should skip-to-dense, but unverified; load now always goes through
`gguf_load_with_metadata` (minor). NOT changed: Q6K_DEMOTE (lossy, opt-in), FLASH_DECODE (off), generated
policies (still never a default). Commit `[nn]`.
