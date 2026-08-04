# NV multi-compute-queue execution scope (primitive route for decode overlap)

Date: 2026-08-03
Status: active scope. Authorizes (A) a closed-default NV device slice that persists
the handles needed to allocate additional compute GPFIFOs, and (B) a device-level
probe that answers the load-bearing hardware question: does GB202 / driver 595.84
co-schedule independent kernels submitted to two or three native compute channels?
All HCQGraph/decoder changes (D2-D4 below) are GATED on the probe result and are NOT
authorized by this document. No promotion to `dev`/`exp`/`master`; no user files;
shared runtime changes keep the AMD control (behavior at `HCQ_NUM_COMPUTE=1` must be
identical to today).

## 0. Verdict (2026-08-03, probe complete)

**GATE CLOSED: the probe FAILED on this host.** E1 (cross-GPFIFO semaphore
dependency) is correct, but E3/E4/E5 show zero overlap at full scale
(span == node-sum to <0.3%), with partial-SM grids (grid-div 2/4/8), and
across every allocatable engine type. GR1-GR7 (`engineType` 2-8) are rejected
by the RM with `NV_ERR_INVALID_ARGUMENT`; engineType 0 and 1 map to the single
GR engine, and channels on it serialize. Separate-ctxshare channels did not
execute in this slice (setup gap, not a hardware verdict). D2/D3/D4 remain NOT
authorized; lever 1 via the native multi-queue primitive is blocked on GB202 /
driver 595.84 unless RM-level channel scheduling control is found. Numbers:
`nv-multi-compute-queue-probe-measurement-record-20260803.md` and
`docs/five-lever-test-20260803-multiqueue-probe.json`.

Branch boundary: tinygrad `nvidia-bringup-20260731` at `bada37d74` (P0 status
correction). Grounding records:
`decode-replay-overlap-measurement-record-20260803.md` (0.0% overlap),
`five-lever-test-record-20260803.md` (lever 1), and
`executable-taskgraph-ir-scope-20260803.md` (the five-lever authority).

## 1. Route choice: the primitive route is native HCQ multi-queue

Two options were identified for lever 1 (graph-level overlap of independent nodes):

1. Multiple native NV compute channels/GPFIFOs, ready DAG nodes scheduled across
   them, synchronized by memory semaphores (the existing `HCQSignal` primitive).
2. Run the decode route through `DEV=CUDA` and use the existing `CUDAGraph` lowerer
   (`ops_cuda.py:119`).

Option 1 is the primitive route and stays generic; option 2 is a vendor-specific
bypass and is rejected. Reasons, grounded in the tree:

- `HCQGraph` (`tinygrad/runtime/graph/hcq.py`) is the backend-agnostic hardware
  command-queue graph shared by NV and AMD. Its primitives are `HWQueue`
  (`tinygrad/runtime/support/hcq.py:77`) and `HCQSignal`, a memory timeline
  semaphore (`wait` = acquire when value >= target, `signal` = release).
- The graph machinery is already multi-queue accounted: per-queue `out_signal`s,
  per-queue `last_j` chains (`hcq.py:154`), cross-queue waits encoded from
  range-aware buffer deps (`_resolve_deps`, `hcq.py:262-267`), demand-driven
  signal emission (a kernel emits a signal only when a later kernel waits on it),
  and `queue_access`/`dev_access` bookkeeping that removes redundant cross-queue
  waits. Copy queues already exist as N per device (`HCQ_NUM_SDMA`,
  `hcq.py:186`), and the device seam already models compute queues as a list
  (`hw_compute_queues()`, `support/hcq.py:504`).
- The semaphore primitive is already exercised across GPFIFOs in production: the
  decode graph syncs compute (`NVC56F`) with SDMA (`NVC6B5`) through memory
  semaphores. A compute-to-compute edge is a second instance of the same
  mechanism, not new hardware machinery.
- AMD has the same seam (`ops_amd.py:1109-1110` keeps a per-device compute-queue
  dict today with one ring); a generic N-queue `HCQGraph` is a shared win, and at
  `HCQ_NUM_COMPUTE=1` the code path is byte-identical to today, satisfying the
  AMD-control house rule.
- `DEV=CUDA` would abandon the native ioctl/QMD/GPFIFO substrate this fork exists
  to keep, add a CUDA-runtime dependency to the decode path, and generalize to
  AMD/Metal only through the same HCQ layer we would have had to fix anyway.

The single-queue hardwiring is three spots: `comp_queues` is one queue per device
(`hcq.py:68`), every runtime call is assigned to `self.comp_queues[enqueue_dev]`
(`hcq.py:134`), and NV has one compute GPFIFO (`ops_nv.py:633`) with
`NVComputeQueue._submit` pinned to it (`ops_nv.py:205`).

## 2. P0 probe protocol (device-level, runs now)

The load-bearing unknown is hardware behavior, not abstraction: will the RM and
GB202 actually co-schedule two or three compute channels? Until that is measured,
no HCQGraph change is justified. The probe
(`extra/llm_research/decode/nv_multi_queue_probe.py`) creates extra compute
GPFIFOs on the live `NVDevice`, runs lowered tinygrad kernels on hand-rolled
`ProbeComputeQueue` instances (one per GPFIFO), and times them with HCQ timestamp
signals (`HCQSignal.timestamp`, the same primitive the overlap measurement used).

Experiments, in one sequential GPU session (house rule):

- E1 correctness: a cross-GPFIFO memory-semaphore dependency (kernel on queue 1
  waits on a signal released by queue 0) with numeric verification against CPU.
- E2 serial calibration: dependent kernels on one queue; span must equal node-sum
  (0% overlap), validating the timestamp method.
- E3 concurrency: independent kernels on two queues; span vs node-sum.
- E4 concurrency with three queues.
- E5 compute-heavy flavor (matmul) on two queues, to separate engine
  co-scheduling from DRAM contention on the elementwise flavor.

Criteria: PASS = E1 numeric-correct AND at least one of E3/E4/E5 shows measurable
overlap (span < node-sum by >= 5%). FAIL = E1 correct but zero overlap on both the
memory-bound and compute-bound workloads (engineType=0 channels share one engine,
or the RM serializes the runlist). The probe records per-experiment span, node-sum,
overlap fraction, and kernel durations as JSON, anchored under
`docs/five-lever-test-20260803-multiqueue-*.json`.

## 3. Authorized implementation (probe substrate only)

- Persist `self.vaspace` and `self.ctxshare` on `NVDevice` (currently locals in
  `__init__`), so additional GPFIFOs can be allocated without re-creating
  contexts.
- `_new_gpu_fifo` gains `debugger: bool = True`; the debugger object and
  `debug_compute_obj`/`debug_channel` are allocated only when `debugger` is true,
  so extra compute channels do not clobber or duplicate the per-device debugger.
- The probe script under `extra/llm_research/decode/`.

No behavior change at defaults; no graph changes; no decode-path changes.

## 4. Post-probe implementation (GATED, not authorized here)

If and only if the probe passes, the next scope revision authorizes:

- D2: `HCQGraph` per-device compute-queue lists behind `HCQ_NUM_COMPUTE` (default
  1); queue picker v1 (round-robin or earliest-finish greedy) in the existing
  topological schedule loop; every cross-queue edge already encoded by
  `_resolve_deps`, so any assignment is correct and the picker only shapes
  concurrency.
- D3: `NVDevice` allocates `HCQ_NUM_COMPUTE` compute GPFIFOs; `NVComputeQueue`
  gains `queue_idx` and submits to its own GPFIFO; `hw_compute_queues()` returns N
  factories. Replay cost stays O(1) per queue per token (one GPFIFO put each).
- D4: decode-level A/B: `HCQ_NUM_COMPUTE=3` with `HCQ_GRAPH_PROFILE_JSON` at d512
  vs the 0.0% record, plus same-session wall rows, following the established
  measurement regime.

## 5. Risks and honest ceiling

- Unknown until E3-E5: number of concurrently schedulable compute channels and
  whether `engineType` variants (e.g. GR engines) are accepted for compute
  channels on this consumer part.
- Decode GEMVs are DRAM-bound at ~46% of peak bandwidth; three concurrent GEMVs
  approach saturation, which is the same class of win llama banks on, but per-item
  speedup is bounded by the bandwidth wall, not the queue count.
- The sequential tail (rmsnorm/residual/vocab chain) is a dependency chain and
  will not overlap regardless of queue count.
- Each extra GPFIFO allocates a 48 MiB error notifier; probe footprint is small on
  32 GiB.

## 6. Deliverables and bans

Deliverables: scope doc; closed-default device slice; probe script; probe
measurement record with anchored JSON; conditional next-scope revision.
Bans: no change to user files (`docs/README.md`, `docs/beating-llama-*`,
`docs/what-makes-a-token-fast-*`, `extra/llm_research/microbench/*`,
`scratchpad/t6_metal_admission_probe.py`); no `master`/`dev`/`exp` commits; no
promotion of the probe into the decode path; no concurrent GPU sessions.
