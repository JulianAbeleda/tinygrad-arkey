# NV decode overlap - exhaustive implementation scope (primitive route)

Date: 2026-08-04
Status: active implementation scope, gated phase by phase. Authorizes the
native multi-compute-queue construction fix, the D3 multi-GPFIFO compute
substrate, the D2 dependency-driven HCQGraph compute scheduling, the decode
wall A/B, and (gated) the D4 signature/reuse work. Route B (DEV=CUDA +
CUDAGraph) is analyzed and explicitly NOT authorized. No promotion to
`dev`/`exp`/`master`; no user files; shared runtime changes keep the AMD
control (behavior at `HCQ_NUM_COMPUTE=1` must be identical to today).
Branch boundary: tinygrad `nvidia-bringup-20260731` at `954e74e78` (E1-E3
records).

## 0. Authority

Authority chain: external-review scope `24deaffd7` -> amendment `39d10369b`
-> E1E3 measurement scope `fed89a201` -> E1/E2/E3 records `954e74e78` -> this
scope. The amendment's section 7 authorizes as the next artifact "a
measurement/implementation scope for D2-D4 or corrected brief"; this document
is that artifact.

`nv-multi-compute-queue-execution-scope-20260803.md` is superseded in part:
its probe verdict closed the gate on the shared-context construction only.
E3 reopened D2-D4 under the belief-flip criterion "CUDA overlap with native
serialization proves a native RM construction/scheduling gap, not a hardware
limitation" (TRUE, OBSERVED). The old scope's verdict paragraph is retained
as history; the corrected blocker statement from the amendment governs.

`executable-taskgraph-ir-scope-20260803.md` remains the D1-D5 vocabulary:
D1 explicit IR, D2 dependency-driven batching beyond consecutive calls, D3
overlap semantics/substrate, D4 signature/reuse rules, D5 non-NV lowerers.
This scope implements D3 first (substrate), then D2 inside HCQGraph (intra-
group scheduling), then the D4 reuse/census rules. D1 (explicit IR) is not
required for any phase below; it stays a separate substrate candidate.

The amendment's HARD STOPs remain in force for this document: no declaring
native overlap impossible, no composing a parity endpoint, no promoting a
route.

## 1. Measured foundation (what E1-E3 settled)

All rows OBSERVED, same RTX 5090 box, driver 595.84, one flocked GPU session
per record.

| quantity | value | record |
| --- | ---: | --- |
| llama d512 wall, opt=0 | 246.32 tok/s (median), ~0.185 ms/token delta vs opt=1 | E1 |
| llama replay span vs node-sum, opt=0 | 3.889 ms vs 5.013 ms = 22.4% below node-sum | E1 |
| llama replay span vs node-sum, opt=1 | 3.687 ms vs 4.859 ms = 24.0% | E1 |
| tinygrad d512 serialized node-sum | 5366.1 us, 948 nodes, 5 groups, 0.0% overlap | E2 |
| E2 unlimited-resource critical path | 2539.9 us = 52.7% saving | E2 |
| E2 2-queue list schedule | 3310.5 us = 38.3% saving (2.06 ms) | E2 |
| E2 3-queue list schedule | 2786.3 us = 48.1% saving (2.58 ms) | E2 |
| E2 reopen threshold | 0.8-1.1 ms; both schedules exceed it ~2x | E2 scope |
| CUDA 2-stream elementwise | 48.1% overlap, numerics ok | E3 |
| CUDA 3-stream elementwise | 65.1% overlap, numerics ok | E3 |
| CUDA 2-stream matmul 2048 | 48.4% overlap, numerics ok | E3 |
| native shared-ctx regression E3-E5 | ~0% overlap at all flavors | E3 |
| decode wall authority | 177.72 tok/s ours vs 251.8 llama = 0.704x; 5.367 ms node-sum; 95.4% GPU busy | five-lever record |
| correctness pins | token sha `9d6b3787...`, first token `151936`, decode sha `0721c16f...` | forward scope |

What is settled: (1) llama's overlap is base CUDA graph node scheduling, not
the gated QKV stream fan-out; (2) tinygrad's own DAG carries parity-scale
scheduleable parallelism; (3) the device co-schedules independent kernels when
reached through CUDA streams; (4) the native shared-context construction
serializes; (5) the separately-ctxshared native construction never executed,
so the native RM path is unproven, not disproven.

What remains open: how CUDA's stream concurrency is represented by RM channel
groups, context shares/subcontexts, runlists, and scheduling controls on this
driver; whether a corrected native construction co-schedules at all; and how
much of the E2 simulation survives DRAM-bandwidth sharing at wall.

## 2. Route choice

Route A (primitive, generic): multiple native NV compute channels/GPFIFOs with
ready DAG nodes scheduled across them, synchronized by the existing memory
semaphore primitive (`HCQSignal`, NVC56F acquire/release). This is the
primitive route because HCQGraph is backend-agnostic (NV and AMD share it),
the semaphore cross-GPFIFO mechanism is already proven in production
(compute<->SDMA, and probe E1 compute<->compute), and the seam for N compute
queues already exists on AMD (`ops_amd.py:1109-1110` keeps a per-device
compute-queue dict today, with one ring).

Route B (non-primitive): run the decode route through DEV=CUDA and the
existing CUDAGraph lowerer (`ops_cuda.py:100-119`, `graph/cuda.py:10-60`).
Not primitive: it abandons the native ioctl/QMD/GPFIFO substrate this fork
exists to keep, adds a CUDA driver dependency to the decode path, and would
generalize to AMD/Metal only through the same HCQ layer Route A fixes anyway.
It is also unproven for decode (the DEV=CUDA route was never exercised; the
fork's decode kernels live on the NV route). It is documented in section 4.5
as the fallback analysis only and is NOT authorized here.

Decision: pursue Route A. Route B work is limited to the analysis items in
section 4.5 and becomes the implementation candidate only if G1 fails and a
separate scope is written for it.

## 3. The native construction gap (exact state)

Current native topology, one of each:

- one `KEPLER_CHANNEL_GROUP_A` with `engineType=NV2080_ENGINE_TYPE_GRAPHICS`
  (`ops_nv.py:626`);
- one `FERMI_CONTEXT_SHARE_A` with `SUBCONTEXT_ASYNC` on that group
  (`ops_nv.py:632`);
- one compute GPFIFO and one DMA GPFIFO under that ctxshare (`ops_nv.py:635-
  636`), group-level `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE(bEnable=1)` once at init
  (`ops_nv.py:637`);
- `NVComputeQueue._submit` hardwired to `dev.compute_gpfifo` (`ops_nv.py:205`);
- `HCQGraph.comp_queues` one queue per device (`graph/hcq.py:68`), every
  runtime call assigned to it (`graph/hcq.py:135`); copy queues already
  multi-queue via `HCQ_NUM_SDMA` (`graph/hcq.py:71`, `:140`).

Probe state (`extra/llm_research/decode/nv_multi_queue_probe.py`,
`extra_gpfifos` at 176-207): extra channels inside the shared ctxshare
serialized at every flavor (E3-E5 ~0%); extra channels on a separate ctxshare
created per channel DID NOT EXECUTE (E1 join target stuck at 11 of 12). The
recorded cause is a setup gap: after allocating a new ctxshare and channel,
the probe re-issues only the group-level schedule; it never issues the
per-channel bind/schedule sequence CUDA's stream setup performs.

The fix hypotheses, to be resolved by Phase 0 experiments:

- H1: per-channel `NVA06F_CTRL_CMD_BIND(engineType)` +
  `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE(bEnable=1)` is required for channels whose
  ctxshare is not the boot ctxshare. Today `_new_gpu_fifo` issues those only
  when `channel_group == self.nvdevice` (`ops_nv.py:676`), which is never
  true for the group-allocated boot path either (the boot fifos get scheduled
  via the group call at 637).
- H2: runlist membership requires the group-level schedule to be re-issued
  AFTER every new channel/ctxshare is bound (the probe does this, but without
  H1 the channels may never reach a schedulable state).
- H3: the ctxshare flags/order matter: CUDA's per-stream concurrency may
  correspond to per-context-share channel groups rather than multiple
  ctxshares inside one group; if so, extra channels should be allocated under
  their own `KEPLER_CHANNEL_GROUP_A` (or a `NV_CHANNEL_GROUP_ALLOCATION_*
  variant) with their own ctxshare.
- H4: engineType acceptance is construction-dependent: GR1-GR7 were rejected
  with `NV_ERR_INVALID_ARGUMENT` in the shared-ctx construction; the same
  request under a correctly bound independent ctxshare may behave
  differently, or engineType 0/1 with independent ctxshares may co-schedule
  anyway (one GR engine is not a concurrency gate; see amendment 2.1).

## 4. Exhaustive work breakdown

Each phase has a gate; no phase starts before the previous gate passes. All
GPU work is one flocked session at a time (`flock /tmp/nv_gpu.lock`).

### 4.1 Phase 0 - corrected native construction + R1-R5 probe (device-level)

No HCQGraph, decoder, or decode-route changes. Scope: the probe script and a
closed-default device slice that supports the probe.

Work items:

1. `extra_gpfifos` gains a `per_channel_bind` mode that, for every extra
   channel, issues in order: ctxshare alloc (per-channel when
   `separate_ctxshare`) -> `_new_gpu_fifo` -> `NVA06F_CTRL_CMD_BIND` ->
   `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE` -> group-level `NVA06C` re-schedule.
   The per-channel control sequence runs on the channel's own ctxshare handle
   semantics; record every RM error per step (not just per channel) so the
   failing step is identified.
2. H3 variant: `group_per_channel` mode allocating a fresh
   `KEPLER_CHANNEL_GROUP_A` per extra channel (same device), with its own
   ctxshare and fifo, plus per-channel bind/schedule. This mirrors CUDA's
   per-context channel ownership most directly.
3. Keep the existing shared-ctxshare path as a control arm in the same
   session.

Pseudocode:

```python
def fix_channel(dev, engine_type, mode):
  if mode == "ctxshare":            # H1/H2: extra ctxshare in boot group
    cs = rm_alloc(dev.channel_group, FERMI_CONTEXT_SHARE_A,
                  params(hVASpace=dev.vaspace, flags=SUBCONTEXT_ASYNC))
    fifo = dev._new_gpu_fifo(area, cs, dev.channel_group, compute=True,
                             debugger=False, engine_type=engine_type)
    rm_control(fifo, NVA06F_CTRL_CMD_BIND, BIND_PARAMS(engineType=engine_type))
    rm_control(fifo, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, SCHED_PARAMS(bEnable=1))
  elif mode == "group":             # H3: fresh channel group per channel
    grp = rm_alloc(dev.nvdevice, KEPLER_CHANNEL_GROUP_A,
                   CHANNEL_GROUP_ALLOC_PARAMS(engineType=ENGINE_TYPE_GRAPHICS))
    cs = rm_alloc(grp, FERMI_CONTEXT_SHARE_A, ...)
    fifo = dev._new_gpu_fifo(area, cs, grp, compute=True, engine_type=engine_type)
    rm_control(fifo, NVA06F_CTRL_CMD_BIND, BIND_PARAMS(engineType=engine_type))
    rm_control(fifo, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, SCHED_PARAMS(bEnable=1))
  rm_control(dev.channel_group, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, bEnable=1)  # H2
  return fifo
```

Experiments R1-R5 (same span/node-sum criterion as the old E1-E5 probe):

- R1 cross-GPFIFO semaphore dependency: kernel on queue 1 waits on a signal
  released by queue 0; numeric check vs CPU. Must pass exactly (probe E1
  precedent).
- R2 serial calibration on one fifo: span == node-sum.
- R3 two independent fifos, elementwise, full and partial-SM (grid-div 4)
  grids.
- R4 three fifos, elementwise.
- R5 two fifos, matmul 2048.

Each R row records per-kernel HCQ timestamps, span, node-sum, overlap
fraction, and RM error list, anchored as JSON.

Gate G1 (belief-flip): PASS = R1 numeric-exact AND at least one of R3-R5 shows
overlap >= 5%. FAIL = R1 correct but zero overlap across all flavors and both
construction modes. A FAIL records the exact failing RM step and closes
Phase 1-4 of Route A; it is NOT a hardware no-concurrency verdict (amendment
HARD STOP) and it routes forward work to the Route B analysis (4.5).

### 4.2 Phase 1 - D3 substrate: multi-GPFIFO compute queues (closed default)

Gate: G1 passed. Scope: `ops_nv.py` + `graph/hcq.py` seam only; behavior at
`HCQ_NUM_COMPUTE=1` byte-identical; no scheduling policy yet.

Work items:

1. `NVDevice` allocates `compute_gpfifos: list[GPFifo]` (index 0 is today's
   `compute_gpfifo`, kept as an alias so existing references keep working).
   `gpfifo_area` grows to `0x300000 * ceil((2 + N)/3)` slots or a fresh
   contiguous area per additional fifo (offsets 0 / 0x100000 / 0x200000 are
   already taken by compute / dma / video); every fifo needs its own
   ring+gpput slot (GPFifo layout, `ops_nv.py:377-382`, `:658-685`).
2. `NVComputeQueue` gains `queue_idx: int = 0` (mirror `NVCopyQueue`,
   `ops_nv.py:215`); `_submit` routes to `dev.compute_gpfifos[self.queue_idx]`
   instead of `dev.compute_gpfifo` (`ops_nv.py:205`). At index 0 this is the
   same fifo and the same code path.
3. `NVDevice.hw_compute_queues()` returns N factories
   (`support/hcq.py:503-505` currently returns one) with distinct
   `queue_idx`; factories are constructed with the corrected construction
   from Phase 0. `_setup_gpfifos` boot sequence (`ops_nv.py:704-718`) runs the
   setup submit (compute class + shader/local-mem windows) per compute queue,
   or lazily on first factory call; choose per boot-cost measurement (each
   fifo carries a 48 MiB uncached errnotifier, `ops_nv.py:660`).
4. Env knob `HCQ_NUM_COMPUTE` (default 1) in `graph/hcq.py`, same shape as
   `HCQ_NUM_SDMA` (`:71`). `comp_queues: dict[dev] -> list[HWQueue]` of length
   N; at N=1 every existing reference (`:98`, `:135`, `:151`, `:195`, `:233`,
   `:254-256`, `:260`, `:331`) resolves to the same single queue as today.
5. AMD control: `ops_amd.py:1109-1110` already keeps `compute_queues` as a
   dict with one ring; the generic `hw_compute_queues()` change must return
   one factory on AMD at `HCQ_NUM_COMPUTE=1`, and the AMD backend must not be
   touched by Phase 1. No AMD GPU on this box: control = identical code path
   at default + review.

Gate G2: at `HCQ_NUM_COMPUTE=1`, a d512 decode run reproduces the wall
authority within measurement noise (< 1%) and the correctness pins (section
7) pass 3/3. At `HCQ_NUM_COMPUTE=2/3`, a device-level probe (R3/R4 shape)
still passes through the new factories before any scheduler exists.

### 4.3 Phase 2 - D2 dependency-driven compute scheduling in HCQGraph

Gate: G2 passed. Scope: `graph/hcq.py` scheduling only; grouping authority
unchanged (`graph_split_rewrite`, `jit.py:236`, `JIT_BATCH_SIZE`,
`helpers.py:240`).

Work items:

1. Assignment: replace the single-queue selection at `graph/hcq.py:135` with a
   deterministic picker over `comp_queues[enqueue_dev]`. Correctness
   invariant: any assignment is correct, because `_resolve_deps`
   (`hcq.py:262-267`) already encodes cross-queue edges (per-queue `last_j`
   chains, per-queue `out_signal`s, demand-driven signal emission); the
   picker shapes concurrency only.
2. Picker v1: static priority by longest-remaining-tail (from the E2
   simulator's rule), assign each node to the earliest-available queue
   (tracked by its `last_j` completion estimate), tie-break round-robin. No
   preemption; no reordering of dependent nodes; deterministic across
   replays.
3. Semantics per queue: `signals`, `last_j`, `queue_signals_to_reset`
   (`hcq.py:260`) become per compute queue; the kickoff wait
   (`:195`), the inter-device/copy waits (`:254-256`), the final per-device
   timeline signal (every compute queue signals before the device
   timeline advances), and the replay submit loop (`:331`) iterate the queue
   list. Replay cost stays O(1) per queue per token; total ring traffic is
   unchanged (one put per queue, same aggregate).
4. No change to `JIT_BATCH_SIZE`, graph group boundaries, or kernel content.
   Cross-group non-consecutive batching is Phase 4.

Pseudocode (picker + schedule loop):

```python
# per node j with enqueue_dev d:
queue = argmin(comp_queues[d], key=lambda q: q.est_finish)   # longest-tail order
est_finish[queue] = node_sum_estimate(j)                     # per-node duration
edge(j, queue):  _resolve_deps(bufs, outs, queue, d, out_signal, j, ...)
                 # existing range-aware RAW/WAR/WAW + per-queue last_j chain
```

Gate G3 (structure): one d512 token under `PROFILE=1
HCQ_GRAPH_PROFILE_JSON` with `HCQ_NUM_COMPUTE=2` shows replay span < node-sum
by >= 20% (llama's base-graph level), 3 queues >= 25%, versus 0.0% today;
correctness pins 3/3; per-class overlap matches the E2 sim's class pairs
(GEMV behind GEMV, norm behind GEMV, residual behind GEMV).

### 4.4 Phase 3 - decode wall A/B (the parity question)

Gate: G3 passed. Scope: measurement only; no code beyond the G3 state.

Protocol (one flocked session, same model Qwen3-8B-Q4_K_M, llama control
from the E1 arm-0 command family):

1. Rows: baseline `HCQ_NUM_COMPUTE=1` (today's 177.72 tok/s d512 authority),
   then 2 and 3 queues, at d512 / d2048 / d4096, same harness repetition
   protocol (nmeas=20, reps=3, median), with `HCQ_GRAPH_PROFILE_JSON` on one
   representative token per arm for span/node-sum.
2. llama same-session control at d512 (246.32 tok/s opt=0 authority) and at
   d2048/d4096 as the E1 scope's arm-0 command with matching depth.
3. Bandwidth caveat reported with the rows: the E2 simulator is an
   unbounded-resource upper bound (2-queue 2.06 ms, 3-queue 2.58 ms). Decode
   GEMVs are DRAM-bound at ~46-50% of 1792 GB/s (INFERRED accounting), so
   three concurrent GEMVs approach bandwidth saturation; the expected wall
   conversion is below the sim saving. The sequential tail
   (rmsnorm/residual/vocab chain) will not overlap regardless of queue count;
   measure per-class overlap to confirm the sim's class pairs.

Gates:

- G4 (wall value): median tok/s improvement at d512 >= 10% (~0.55 ms/token)
  with correctness pins 3/3. Below 10% the mechanism is implemented but not
  parity-scale at wall; the record says so explicitly (no promotion).
- G5 (parity direction): ratio vs the same-session llama row improves toward
  >= 1.00; PARITY-QUALIFIED only with a same-session row >= 1.00 at that
  depth (forward scope section 5; d2048/d4096 each need their own rows).

### 4.5 Route B analysis items (NOT authorized as implementation)

Kept for completeness and for the G1-fail fallback. No code, no GPU sessions.

- What exists: `CUDADevice` (`ops_cuda.py:100-119`) with `cuInit`,
  `cuCtxCreate_v2`, shared renderers (CUDARenderer/PTXRenderer/NVCCRenderer);
  `CUDAGraph` lowerer (`graph/cuda.py:10-60`) with
  `cuGraphCreate`/`cuGraphAddKernelNode`/`cuGraphAddMemcpyNode`/
  `cuGraphInstantiate_v2`/`cuGraphLaunch`, dependency edges from
  `_access_resources`, per-replay `cuGraphExecKernelNodeSetParams`. CUPTI-
  visible (proven by E1 llama trace method), which would also unblock nsys
  node tracing on our side.
- What is missing for decode: the decode route is NV-native (custom kernels
  via the NV route; `DEV=CUDA` was never exercised for decode); buffer
  allocator/kernargs/device-vars paths differ (CUDAAllocator vs NVAllocator);
  host-sync and replay semantics for the decode harness are unmeasured;
  correctness pins would need re-establishment on a second substrate.
- Cost/benefit: unblocks CUDA-grade stream concurrency and nsys tooling, but
  abandons the native ioctl/QMD/GPFIFO substrate, adds a CUDA driver
  dependency to decode, and generalizes to AMD/Metal only through HCQ anyway.
- Trigger: G1 fail. Even then this scope authorizes only the analysis above;
  any Route B implementation requires a separate scope under the amendment's
  HARD STOPs (no route promotion from this document).

### 4.6 Phase 4 - D4 signature/reuse + census + non-consecutive batching (gated)

Gate: G3 and G4 passed (overlap is real and wall-positive). Scope:
`executable-taskgraph-ir-scope` D4 section 7, plus the D2 extension section
8 (non-consecutive batching). Work items:

1. Write the reuse rule: same signature (semantic facts + symbolic var_vals +
   buffer slot identities) -> per-replay param update; changed signature ->
   re-instantiate; census records reuse hit/miss per replay so a
   re-instantiation regression is a test failure, not a mystery slowdown.
2. Audit the 5 decode graph groups (32/64/128/256/468) against
   `JIT_BATCH_SIZE` and the admission census; record which nodes are
   consecutive-chain members vs siblings with no true dependency.
3. Candidate: dependency-driven grouping of non-consecutive independent calls
   (scheduler over the captured linear, range-aware edges from
   `DepsTracker`), each candidate with its own d512 wall A/B. Cross-group
   barriers preserved.

## 5. Open RM questions (answer in Phase 0, record each answer)

- Q1 runlist membership: does re-issuing group-level `NVA06C` schedule after
  channel creation add new channels to the runlist, or is per-channel
  scheduling required first?
- Q2 bind order: is per-channel `NVA06F_CTRL_CMD_BIND` required for channels
  under a non-boot ctxshare/group (`ops_nv.py:676` currently skips it for
  group-allocated fifos)?
- Q3 ctxshare scheduling: does one `KEPLER_CHANNEL_GROUP_A` support multiple
  `FERMI_CONTEXT_SHARE_A` with `SUBCONTEXT_ASYNC` co-scheduled, or does
  concurrency need per-channel groups (H3)?
- Q4 engineType: which engineType values does the RM accept for compute
  channels on GB202 consumer under the corrected construction (GR1-GR7 were
  rejected in the shared-ctx construction)?
- Q5 submit token: is the `gpu_mmio[0x90//4]` poke per-fifo via
  `workSubmitToken` (`ops_nv.py:683`, `:139`), and does a second fifo's token
  remain valid once a third fifo is created?
- Q6 memory model: does cross-fifo compute->compute NVC56F acquire/release
  need cache flush/invalidate hints beyond what probe E1 proved (it passed
  across compute<->SDMA and compute<->compute)?
- Q7 gpfifo_area layout: alignment/offset constraints for additional
  ring+gpput slots (0x100000 stride today; any per-fifo alignment beyond
  that)?
- Q8 errnotifier: RM limit on outstanding 48 MiB uncached error notifiers
  (N queues -> N x 48 MiB on a 32 GiB box)?

## 6. Risk register

- G1 fail on both construction modes: Route A phases close; Route B analysis
  becomes the only forward; not a hardware verdict (HARD STOP).
- Bandwidth wall: E2 sim is an upper bound; concurrent GEMVs approach DRAM
  saturation at ~50% of peak; expect wall conversion well below span saving.
- Serialization regression: any shared hcq.py change must be byte-identical
  at `HCQ_NUM_COMPUTE=1`; AMD has no GPU here, so control = default code path
  identity + review (house rule).
- Boot cost: per-fifo 48 MiB errnotifier + setup submit; keep extra fifo
  creation lazy or boot-time per measured cost.
- Wrap semantics: `_submit_to_gpfifo` (`ops_nv.py:114-139`) preserves the
  cmdq wrap-drain contract; multi-fifo submission must keep
  `System.memory_barrier()` before each poke and the per-fifo `put_value`
  accounting.
- Hang diagnostics: extra fifos are created `debugger=False`;
  `on_device_hang` inspects only `debug_channel`; Phase 0 keeps R1-style
  numeric checks to detect silent mis-execution before any decode work.
- Correctness: no reordering that changes outputs; range-aware edges only
  (whole-buffer edges banned); every phase re-pins the decode sha.

## 7. Measurement conventions and correctness pins

- GPU sessions sequential, flocked (`flock /tmp/nv_gpu.lock`), same RTX 5090
  box, driver 595.84, no concurrent GPU work.
- Evidence classes OBSERVED / INFERRED per record; every wall row carries
  session provenance, commit, env, and a same-session llama control where a
  parity claim is at stake.
- Correctness pins at every phase that exercises the model route: token sha
  `9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`, first
  token `151936`, decode sha `0721c16f...`; standalone device probes carry
  their own numeric checks.
- Harness: `replay_overlap_probe.py --depth 512` (PROFILE=1,
  HCQ_GRAPH_PROFILE_JSON) for span/node-sum; the decode runtime overhead
  harness family for wall.
- Commits carry a `[prefix]` (`docs`/`runtime`/`test`); `git diff --check`
  clean; push to `nvidia-bringup-20260731` only.

## 8. Deliverables (exhaustive)

| phase | artifacts | commit |
| --- | --- | --- |
| 0 | corrected `nv_multi_queue_probe.py` construction modes; measurement record with anchored JSON (R1-R5 rows, RM errors per step); G1 verdict | `[test]` + `[docs]` |
| 1 | `ops_nv.py` compute_gpfifos + queue_idx + hw_compute_queues(N); `HCQ_NUM_COMPUTE`; G2 record (N=1 identity + decode pins) | `[runtime]` + `[docs]` |
| 2 | `graph/hcq.py` queue list + picker + per-queue signal semantics; G3 record (span vs node-sum, class pairs) | `[runtime]` + `[docs]` |
| 3 | wall A/B record d512/d2048/d4096 vs llama, per-class overlap, bandwidth caveat; G4/G5 verdict | `[docs]` |
| 4 | D4 reuse rules + census record; non-consecutive batching candidate + A/B if gated | `[runtime]` + `[docs]` |
| 4.5 | Route B analysis note (if G1 fails) | `[docs]` |

## 9. Bans and HARD STOPs

- HARD STOP: no declaring native overlap impossible; no composing a parity
  endpoint; no promoting a route. Route B implementation is NOT authorized
  by this document.
- No user files: `docs/README.md`, `docs/beating-llama-first-principles-
  20260731.md`, `docs/what-makes-a-token-fast-20260731.md`,
  `extra/llm_research/microbench/*` binaries (including the untracked
  `cuda_stream_overlap_probe` binary), `scratchpad/t6_metal_admission_probe.py`.
- No `master`/`dev`/`exp` commits; no concurrent GPU sessions.
- `HCQ_NUM_COMPUTE=1` must be byte-identical; AMD control for shared changes.
- No kernel/dtype changes, no fusion folds, no host-launch work (separate
  scopes), no whole-buffer dependency edges, no grouping changes before
  Phase 4.

## 10. Sequencing summary

| phase | work | gate | note |
| --- | --- | --- | --- |
| 0 | construction fix + R1-R5 | G1 >= 5% overlap, R1 exact | Route B analysis if fail |
| 1 | D3 substrate | G2 N=1 identity + pins | closed default |
| 2 | D2 scheduler | G3 >= 20% span saving | grouping unchanged |
| 3 | wall A/B | G4 >= 10% wall; G5 parity rows | llama same-session |
| 4 | D4 + batching | G3 + G4 | gated candidate |

## 11. One-line job

Make the native RM co-schedule independent compute GPFIFOs like CUDA streams
do (corrected per-channel bind/schedule, then N compute queues in HCQGraph
with signal-based cross-queue deps), and prove it at d512 wall against the
E2 simulation upper bound and the same-session llama row.
