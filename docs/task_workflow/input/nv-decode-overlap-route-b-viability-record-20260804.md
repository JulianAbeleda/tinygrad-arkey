# NV decode overlap - Route B viability measurement record (B0.1-B0.3)

Date: 2026-08-04
Status: measurement record for B0.1-B0.3 of
`nv-decode-overlap-route-b-viability-scope-20260804.md` (amended in place
with the B0.0 test-enabling shims). One flocked GPU session (RTX 5090,
driver 595.84), no concurrent GPU work. Branch: tinygrad
`nvidia-bringup-20260731` at `f87dc07dd` + the B0.0 shim edits
(uncommitted at record time). All numbers OBSERVED unless marked INFERRED.

## 1. B0.0 - test-enabling device-facts shims (OBSERVED gaps, fixed)

The first harness attempt failed before model load:
`RuntimeError: qwen3: selected-GGUF backing allocation is unknown from the
selected path and scanned allocation granularity`.

- Gap 1: `CUDAAllocator` (an `LRUAllocator`) has no
  `allocation_granularity`; `NVAllocator` defines 2 MiB (`ops_nv.py:354`).
  `scan_device_facts` reads it as `global_allocation_granularity`
  (`device_facts.py:243`) and `selected_gguf_backing_bytes` returns None
  without it (`gguf_memory_scan.py:109-117`). Fixed: `CUDAAllocator`
  gains `allocation_granularity -> 2 << 20` and `memory_stats`
  (`cuMemGetInfo_v2`) in `ops_cuda.py`.
- Gap 2: `_default_memory_probe` routes only NV through nvidia-smi; CUDA
  hit `_rocm_smi_memory_probe`, which exists on this box but returns
  None-valued rows, so the allocator fallback never ran (total/free None,
  facts state "unknown"). Fixed: CUDA now routes through nvidia-smi
  (`device_facts.py:291`).

After the shims, `scan_device_facts("CUDA")` reports state "ok": backend
CUDA, arch sm_120, total 31.8 GiB, free 27.6 GiB, granularity 2 MiB.

## 2. B0.1 - CUDA backend smoke (OBSERVED, PASS)

Elementwise + matmul on DEV=CUDA: max abs error 4.6e-5 vs CPU; TinyJit
replay loop runs deterministically. The CUDA backend compiles and runs on
this box.

## 3. B0.2 - decode harness on DEV=CUDA (OBSERVED)

Command: `DEV=CUDA QK_NMEAS=20 QK_REPS=3 QK_CKPTS=512
decode_runtime_overhead.py --model Qwen3-8B-Q4_K_M.gguf`.

| phase | tok/s (median over reps) | ms/token |
| --- | ---: | ---: |
| W (prefill measure) | 177.94 | 5.62 |
| D (decode measure) | 157.93 | 6.33 |

Route: `flash` per `_route` (flash decode candidate binds on CUDA). The
decode completes end-to-end; graph groups are 6 per token with
32/64/128/256/512/29 = 1021 kernels (the older 1021-kernel route
structure; NV's current route is 948 kernels with fusion promotions).

Correctness pins: **DO NOT match the NV-route pins.** First token 38835
(pin: 151936); generated-token sha256 `55f7a13b62...` (pin:
`9d6b3787ce...`), identical across all 3 reps (deterministic). The CUDA
sequence repeats a 6-token loop [38835, 34208, 13, 279, 3974, 13876],
which is a determinism signal but also a numerics-divergence risk versus
the NV/llama-validated stream (INFERRED: kernel chain differs on CUDA, so
logits differ; a repeating loop at this seed is not itself proof of a
bug, but it is not the reference stream).

## 4. B0.3 - nsys node trace of the CUDA decode (OBSERVED)

`nsys profile --cuda-graph-trace=node` (QK_NMEAS=8, QK_REPS=2), SQLite
export: 41,191 graph kernels, 12 distinct graphIds, 228 graph launches.
Decode graphs: 6 graphIds (node counts 32/64/128/256/512/29), 34 launches
each = 34 decode tokens; prefill graphs: 6 graphIds, 4 launches each.
CUPTI node tracing WORKS on this route (the lever-1 tooling blocker from
DEV=NV is gone).

Per-token replay structure (steady state, 33 tokens, median):

| quantity | value |
| --- | ---: |
| kernels per token | 1021 |
| node-sum (sum of kernel durations) | 5134.2 us |
| replay span (first start to last end, 6 launches) | 5363.8 us |
| overlap | -4.5% (span > node-sum; no overlap, plus 230 us of inter-launch gaps) |

Per-launch overlap (median, steady state): 32-node -8.7%, 64-node -3.9%,
128-node -4.7%, 256-node -4.6%, 512-node -4.5%, 29-node -0.9%. All
negative: every graph launch replays serialized, span >= node-sum with
small inter-kernel gaps.

## 5. Viability verdict

**VIABLE-WITH-PORT, with a decisive caveat: the overlap lever does NOT come
free with the existing CUDAGraph lowerer.**

What is viable (OBSERVED): the decode route runs end-to-end on DEV=CUDA
after two small device-facts shims; CUDA graphs capture and replay per
token; CUPTI/nsys node tracing works; execution is deterministic; prefill
is at parity with the NV route (177.94 vs 177.72 tok/s).

What is not: (1) correctness pins differ, so the CUDA route needs
re-pinning and a numerics-divergence investigation; (2) as-built
single-stream CUDAGraph replay is fully serialized (-4.5% overlap), so
Route B does not deliver llama-style overlap by itself. E1 established
that llama's overlap comes from graph nodes captured on multiple internal
streams with event edges; our lowerer adds all nodes on one stream. The
multi-stream capture mechanism is unbuilt and is the load-bearing unknown
for any Route B overlap work (mirror experiment: does a cuGraph whose
nodes come from 2-3 non-blocking streams with event edges overlap on this
driver? E3 says streams co-schedule; graphs must be measured).

Legal ceiling if the mechanism works: the E2 intra-group sum
(608.8 us / 11.35% no-contention) for the current grouping, plus whatever
the pre-split full-token DAG capture proves for cross-group edges
(Phase 4 tooling already exists, `full_token_dag_capture.py`). Wall
conversion is bounded by the ~50% DRAM-bandwidth utilization on decode
GEMVs (INFERRED accounting).

## 6. Artifacts

- `/tmp/b0_cuda_d512.json` (harness rows, session-scoped)
- `/tmp/b0_cuda_trace.nsys-rep` + `/tmp/b0_cuda_trace.sqlite`
  (session-scoped)
- shims: `tinygrad/runtime/ops_cuda.py`, `tinygrad/llm/device_facts.py`
- this record; exhaustive Route B implementation scope:
  `nv-decode-overlap-route-b-implementation-scope-20260804.md`
