# NV decode overlap - Route B3.0 preflight record (G-B3-0 PASS)

Date: 2026-08-04

Branch: `nvidia-bringup-20260731` at `a43b27193`
(`[docs][scratchpad] Scope B3 execution and prove llama kernel bridge`).
Authority: B3 exhaustive execution scope section 7 (`G-B3-0`).

## 1. Worktree and authority inventory

- scope doc hash: `nv-decode-overlap-route-b3-exhaustive-execution-scope-
  20260804.md` (committed at `a43b27193`); amendment and external-review scopes
  present in `docs/task_workflow/input/`.
- B0/B1/B2 measurement records present on disk.
- `git status --short`: 3 modified docs (`docs/README.md`,
  `docs/beating-llama-first-principles-20260731.md`,
  `docs/what-makes-inference-fast.md`), plus untracked census/numerics
  tools and microbench binaries under `extra/llm_research/` and one scratchpad
  probe. These are preserved untouched; this record commits no user paths.
- toolchain: `nsys` at `/usr/local/bin/nsys`; CUDA toolkit 13.2 at
  `/usr/local/cuda-13.2` with `cuobjdump`; Python 3.12.3 (repo `.venv`).
- model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, 5027783488 bytes,
  identity SHA-256 `b8ef0be84bfa0588efae9fb84a3b3e5b7beb53f5620ada7d8c48bd3a26633605`.
- GPU: NVIDIA GeForce RTX 5090, driver 595.84, 32607 MiB total; ~28.3 GiB free
  before each run; no concurrent GPU process observed.

## 2. Hermetic preflight

```bash
.venv/bin/python -m pytest -q test/unit/test_full_token_dag_capture.py test/unit/test_cuda_graph_multi_stream_schedule.py
```

Result: 22 passed in 0.46s.

CPU-only llama bridge inspection
(`scratchpad/llama_cuda_binary_kernel_probe.py --inspect-only`) reproduces the
pinned record byte-for-byte: library SHA-256
`d0f6580892fc5940321a3dfd9af3b3febd13c01102861da9c155ae4cda86ac49`,
`dynamic_symbol_present=true`, `fusion_args_size=32`,
`embedded_sm120a_cubins=138`, `binary_reuse_candidate=true`.

## 3. CUDA route reproduction

Lock-held session (flock on `/tmp/gpu-bench.lock`; first arm additionally
verified lock-free via `nvidia-smi` and `flock -w 10` probe):

```bash
DEV=CUDA CUDA_GRAPH_STREAMS=1 QK_NMEAS=20 QK_REPS=3 QK_CKPTS=512 \
  .venv/bin/python extra/llm_research/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --out docs/task_workflow/output/nv-decode-overlap-b3-0-repro-20260804.json
```

Fixed-depth d512, D = same-model JIT with final sync:

| rep | tok/s | final token id |
| --- | ---: | ---: |
| 0 | 157.66 | 34208 |
| 1 | 161.37 | 34208 |
| 2 | 158.81 | 34208 |

Median 158.81 tok/s (6.30 ms/token). W median 178.24 tok/s (5.612 ms/token).
Historical anchors: CUDA B0.2 157.93 tok/s / 6.3319 ms, B2 S=1 159.72 tok/s.
Fresh session is inside the historical band; route re-anchored on this run.

## 4. Route structure census

```bash
flock -w 10 /tmp/gpu-bench.lock -c 'DEV=CUDA CUDA_GRAPH_STREAMS=1 ... route_kernel_census.py --depth 512 --out docs/task_workflow/output/nv-decode-overlap-b3-0-census-cuda-20260804.json'
```

- kernels per prime token: 1021 (pinned record: 1021)
- graph groups per replay token: 6 (pinned record: 6 groups 32/64/128/256/512/29)
- first token: 271 (3/3 reps identical)
- token SHA-256 `227ad3ce9621f2c382cc722a3c2f1677637d3e3f2bfbf37d6ca652f98880eb4e`
  (3/3 reps identical)
- median W tok/s 178.10

## 5. Verdict

**G-B3-0: PASS.** All hermetic tests green; d512 CUDA S=1 completes 3/3
deterministically; route structure exactly 1021 kernels / six groups as
expected; no unexplained model/driver/tooling mismatch; no user path changed.

Anchored artifacts: `docs/task_workflow/output/nv-decode-overlap-b3-0-repro-
20260804.json`, `docs/task_workflow/output/nv-decode-overlap-b3-0-census-cuda-
20260804.json`.
