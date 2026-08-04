# Decode replay overlap measurement record

Date: 2026-08-03 (measured same day, same RTX 5090 box, sequential sessions)
Status: measurement record answering the taskgraph scope's P0 question ("does
our decode replay overlap independent nodes?") with real GPU timestamps, and
correcting the fallback inference in `five-lever-test-record-20260803.md`
section 1. No implementation changed.
Branch: tinygrad `nvidia-bringup-20260731`, HEAD `ec1c23577` (five-lever test
record). Evidence class OBSERVED unless marked INFERRED.

## 1. Why the fallback signal was wrong

The five-lever record inferred ~6% replay overlap from W wall < DEBUG=2
node-sum (5.63 ms vs 5.981 ms at d512). That comparison is invalid: the DEBUG=2
`tm` values are host-synchronized per-kernel walls from the eager path
(`tinygrad/engine/realize.py`, `Device[device].synchronize()` per kernel), so
each carries per-kernel launch/sync overhead. 948 kernels x ~0.65 us overhead
reproduces the entire 0.6 ms gap. It is not GPU overlap.

## 2. Why nsys cannot see the replay (profiler/backend mismatch)

The scope's P0 protocol (`nsys --cuda-graph-trace=node`) is inapplicable to the
executing backend, not blocked by tooling:

- Nsight Systems 2026.1.3 is installed and functional: a native CUDA smoke test
  records CUDA API calls and GPU kernel events, and existing llama reports
  contain CUDA graph/node data under the same host settings
  (`perf_event_paranoid=4`, `RmProfilingAdminOnly=1`; those restrict CPU
  sampling and GPU performance counters, not CUDA activity tracing).
- The same Nsight command around a tinygrad DEV=NV kernel collects no CUDA
  events, because DEV=NV never calls the CUDA launch API: `NVDevice`
  (`tinygrad/runtime/ops_nv.py:598`) drives the hardware through native
  NVIDIA ioctls, QMD command construction, and direct GPFIFO submission.
- The native counterpart works: `PROFILE=1 HCQ_GRAPH_PROFILE_JSON=...` exports
  per-node timestamps and dependencies (exporter at
  `tinygrad/runtime/graph/hcq.py:345`; fields: kernel name, start/end,
  duration, dependency indices, semantic metadata). Smoke artifact anchored at
  `docs/five-lever-test-20260803-hcq-graph-smoke.json`.

## 3. Method: HCQ profile timestamps

The NV runtime measures per-kernel GPU time directly: with `PROFILE=1` and
`HCQ_GRAPH_PROFILE_JSON` set, every HCQGraph replay writes per-kernel start/end
timestamps (microseconds, `HCQSignal.timestamp`) for each graph group
(`tinygrad/runtime/graph/hcq.py`, `collect_timestamps`). The probe
(`extra/llm_research/decode/replay_overlap_probe.py`) runs one measured decode
token with this enabled, collects the graph profile lines for that token (the
flush replay triggers the collect), and computes per-group node-sum (sum of
member durations) vs group span (max end - min start). Artifacts:
`docs/five-lever-test-20260803-overlap-d512.json`,
`docs/five-lever-test-20260803-overlap-d4096.json`.

## 4. Results: zero overlap at both depths

| depth | groups | kernels | node-sum | span sum | overlap | harness W wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| d512 | 5 | 948 | 5.367 ms | 5.367 ms | 0.0% | 5.63 ms |
| d4096 | 5 | 948 | 6.113 ms | 6.113 ms | 0.0% | 6.39 ms |

Per-group, span equals node-sum to the microsecond (e.g. d512 group 0: 180.9 vs
181.0 us; group 4: 2829.7 vs 2829.8 us). The decode replay is fully serialized:
kernels execute back-to-back on the single per-device compute queue, and the
timestamp equality additionally rules out inter-kernel bubbles.

The graph groups are the JIT batches: 32/64/128/256/468 kernels at both depths
(the 948-kernel census splits the same way). GPU busy is 5.367 ms of the
5.63 ms wall at d512 (95.4%) and 6.113 ms of 6.39 ms at d4096 (95.7%); host
dispatch is ~1.0 ms across the five submits but is fully overlapped, so the
token is GPU-bound.

## 5. Why it is serialized, and the implementation choice

- llama's 22% overlap (node-sum 5.006 ms vs replay 3.89 ms) comes from
  concurrent nodes inside its CUDA graph. Our HCQGraph expresses dependencies
  but enqueues every kernel on one compute queue per device, so it cannot
  overlap independent nodes even when they exist.
- The serialization is structural, confirmed at three layers: HCQGraph creates
  one compute queue per device (`hcq.py:68`), assigns every kernel to it
  (`hcq.py:134`), and `NVComputeQueue` submits everything to the single
  `compute_gpfifo` (`ops_nv.py:205`, allocated at `ops_nv.py:633`).
  `JIT_BATCH_SIZE=0` therefore only removes graph-group boundaries: it yields
  one larger command sequence on the same compute channel and cannot create
  compute-compute overlap.
- The real implementation choice is now two options: (1) add multiple native
  NV compute channels/GPFIFOs and schedule ready DAG nodes across them with
  semaphore dependencies, or (2) run the route through DEV=CUDA and use the
  existing CUDAGraph lowerer (`ops_cuda.py:119`). If retaining the native NV
  substrate is the objective, option 1 is the grounded path. It is more
  substantial than the current scope suggests: the missing layer is native
  multi-compute-queue execution, not executable-graph replay. Dependency-driven
  batching (taskgraph scope D2) still only packs kernels; it adds no
  concurrency on this substrate.
- The scope's own caution ("our opportunity there is limited because our
  non-GEMV classes are a stream-serialized tail with real dependencies") is
  now evidence-backed: at d512, group 4 (468 kernels: rmsnorm/residual/vocab
  tail) is 2.83 ms of the 5.37 ms node-sum, and it is a dependency chain rather
  than an overlap opportunity.

## 6. Notes

- The probe's own wall is contaminated by the first-replay pass (graph
  instantiation) and is not used; the harness steady-state W rows
  (`five-lever-test-20260803-l4-fp16.json`) are the wall authority.
- PROFILE=1 adds one timestamp command pair per kernel to the captured queue;
  the measured durations are GPU time and reproduce the record's per-kernel
  shapes (q4k GEMV 9-11 us, flash score 6-7 us at d512).
