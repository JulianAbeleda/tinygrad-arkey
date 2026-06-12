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

## Next Decision

The clean default is to pause here. The project now has a consolidated local
inference result and a third scaling point.

When resuming, choose one track explicitly:

1. Use the inference win: validate a smallest-real training/eval stack using the
   faster decode path.
2. Compiler research: pursue the Ansor-style semantic packed-layout/codegen
   direction. Do not confuse this with more hand-written primitive tuning.
3. Runtime-default soak: keep `QK_PRIMITIVE_STORAGE=shared` explicit for now,
   and only consider making it the runtime default after more non-campaign use.

Recommended next track if the goal is practical progress: training/eval stack.
Recommended next track if the goal is architecture quality: Ansor-style semantic
packed-layout/codegen research.
