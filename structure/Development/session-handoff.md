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
- `extra/qk_flywheel_dataset.py`
- `extra/qk_flywheel_triage_eval.py`

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
