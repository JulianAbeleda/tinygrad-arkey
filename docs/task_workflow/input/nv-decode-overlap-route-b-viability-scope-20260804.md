# NV decode overlap - Route B viability scope (DEV=CUDA + CUDAGraph)

Date: 2026-08-04
Status: measurement scope, probes + harness runs only. Authorizes (B0.1) a
CUDA backend smoke test, (B0.2) a d512 decode harness run on `DEV=CUDA`
with the correctness pins, and (B0.3) an nsys `--cuda-graph-trace=node`
capture of that run for replay span vs node-sum. Does NOT authorize: any
decode-route change, any CUDAGraph/HCQGraph change, any implementation, or
any promotion to `dev`/`exp`/`master`. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `9ae52c087` (Phase 0 G1 =
CONSTRUCTION_BLOCKED closed Route A Phases 1-4; Route B analysis note
`nv-decode-overlap-route-b-analysis-note-20260804.md`).

## 0. Trigger and question

Route A's native multi-compute-queue path is closed (RM rejects BIND;
without BIND extra channels never execute). Route B - run decode through
`DEV=CUDA` with the existing `CUDAGraph` lowerer - is the only remaining
overlap route candidate and is analysis-only until this test. Question:
is Route B viable on this box, meaning the decode kernel set compiles and
runs on the CUDA backend, correctness pins survive, CUDA graphs capture and
replay the decode token, and replay structure is measurable via CUPTI?

## 1. Knowns and unknowns

Known: `CUDADevice` (`ops_cuda.py:100-119`) with CUDARenderer/
PTXRenderer/NVCCRenderer and the `CUDAGraph` lowerer (`graph/cuda.py:10-60`,
cuGraphCreate/AddKernelNode/AddMemcpyNode/Instantiate/Launch, per-replay
setParams) exist and are CUPTI-visible. CUDA streams co-schedule on this
device (E3: 48-65%). llama's base CUDA graph scheduling overlaps 22.4%
(E1). The harness selects `Device[Device.DEFAULT]`, so `DEV=CUDA` routes it.

Unknown: whether the decode kernel set (flash decode, q4k/q6k GEMV
decoders, vocab scatter, norms) compiles and runs under the CUDA backend's
renderers; the decode routes are per-target promotion-gated
(`decode_routes.py`, `kernel_program.py`) with NV sm_120 records, so CUDA
may fall back to the legacy kernel chain (different kernels, possibly
different tokens); whether `CUDAGraph` can capture the 948-node decode
graph; and what overlap our own CUDA graphs realize.

## 2. Protocol (one flocked GPU session, RTX 5090 / driver 595.84)

### B0.0 Test-enabling device-facts shims (CUDA-side only, closed-default)

The first harness attempt (B0.2) failed before model load with
`RuntimeError: qwen3: selected-GGUF backing allocation is unknown from the
selected path and scanned allocation granularity`. Cause (OBSERVED): the
memory-plan admission path consumes `allocator.allocation_granularity`
(`device_facts.py:243` -> `gguf_memory_scan.py:109`); `NVAllocator` defines
it (`ops_nv.py:354`, 2 MiB) but `CUDAAllocator` (an `LRUAllocator`) does
not, so `global_allocation_granularity` is None and the GGUF backing size
is unknown. A second seam is `memory_stats` (`device_facts.py:314-320`):
only AMD/Metal allocators publish it, so CUDA's memory probe would fall
through to `_allocator_memory_probe` and error.

Authorized (B0.0): add `allocation_granularity` (2 MiB, matching the NV
large-tier value) and `memory_stats` (`cuMemGetInfo_v2`) to
`CUDAAllocator`, and route the default memory probe for CUDA through
`nvidia-smi` (an NVIDIA device; rocm-smi exists on this box but returns
None-valued rows, so the allocator fallback never runs). CUDA-side only,
properties with no behavior change unless consumed by device-facts
scanning; no decode-route, graph, or NV changes. These shims exist to make
the viability test runnable; they are not the Route B implementation.

### B0.1 CUDA backend smoke

`DEV=CUDA python3 -c "..."` with elementwise + matmul + a TinyJit replay
loop: assert numerics vs CPU and that graph capture happened (JIT graph
instances > 0). This isolates backend/compile issues from model issues.

### B0.2 Decode harness on DEV=CUDA (correctness + wall)

```
DEV=CUDA QK_NMEAS=20 QK_REPS=3 QK_CKPTS=512 python3 \
  extra/llm_research/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --out /tmp/b0_cuda_d512.json
```

Record: final token id, token evidence hashes, median tok/s, per-rep rows,
and which route ran (flash vs sdpa per `_route`). Correctness pins from the
forward scope: token sha `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`,
first token `151936`; decode sha `0721c16f...` (recompute if tokens match).

### B0.3 nsys node trace of the CUDA decode (overlap structure)

Same command family under `nsys profile --cuda-graph-trace=node` (smaller
`QK_NMEAS`/`QK_REPS` to bound capture), export the `.nsys-rep` to SQLite,
and compute per graph instance: node count, node-sum, replay span, overlap
fraction (span below node-sum), per llama's E1 method (762 nodes, 22.4%
reference). CUPTI works on this path (CUDA Driver API), unlike DEV=NV.

## 3. Viability criteria (belief-flip)

- VIABLE: harness completes on DEV=CUDA; correctness pins match 3/3; nsys
  shows captured CUDA graphs with > 0 nodes and measurable replay span and
  node-sum (overlap fraction is reported, not required for viability).
- VIABLE-WITH-PORT: harness completes but pins differ or a named kernel
  class falls back to the legacy chain; viability holds subject to a
  re-pinning scope; the record names the exact gap.
- NOT-VIABLE: DEV=CUDA cannot compile/run the decode kernel set (crash,
  unsupported op, renderer seam), or CUDAGraph cannot capture the decode
  graph. Record the exact failing boundary (kernel, op, or graph step).

## 4. Conventions and bans

GPU sessions sequential, flocked (`flock /tmp/nv_gpu.lock`); evidence
classes OBSERVED/INFERRED; commits `[docs]`/`[test]` on
`nvidia-bringup-20260731` only; never touch user files (`docs/README.md`,
`docs/beating-llama-*`, `docs/what-makes-a-token-fast-*`,
`extra/llm_research/microbench/*` binaries, `scratchpad/t6_metal_admission_probe.py`);
no route changes, no graph changes, no promotion; `git diff --check` clean.

## 5. Deliverables

This scope; a measurement record
(`nv-decode-overlap-route-b-viability-record-20260804.md`) with the B0.1-B0.3
rows, pins, and the viability verdict; anchored JSON artifacts.
