# Session Handoff

Date: 2026-06-12

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
the tracked session handoff and AMD checklist files.

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
  test.external.test_qk_decode_summary \
  test.external.test_qk_experiment_matrix \
  test.external.test_qk_policy_pipeline
```

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

When resuming, choose one track explicitly:

1. Use the inference win: build a real training loop, richer judge, or
   RLVR/SFT pipeline on top of the validated rollout/comparator backend.
2. Compiler research: continue from the Ansor-transition descriptor foundation:
   descriptor-level `parts`/`LOCAL` candidate search is now exhausted; next work
   needs real semantic schedule/codegen, not another hand sweep over the same
   primitive knobs. Do not confuse this with more hand-written primitive tuning.
3. Runtime-default soak: keep `QK_PRIMITIVE_STORAGE=shared` explicit for now,
   and only consider making it the runtime default after more non-campaign use.

Recommended next track if the goal is practical progress: training/eval stack.
Recommended next track if the goal is architecture quality: Ansor-style semantic
packed-layout/codegen research.
