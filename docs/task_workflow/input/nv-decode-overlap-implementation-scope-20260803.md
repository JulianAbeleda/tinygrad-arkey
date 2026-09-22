# NV decode overlap - exhaustive implementation scope (primitive route)

Date: 2026-08-04
Status: active amended implementation scope, gated phase by phase. Authorizes the
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

### 0.1 Review amendment incorporated here

This revision closes five implementation ambiguities found by direct source
review: (1) merged E2 rows lacked cross-group dependency evidence and are now
hypotheses; (2) Phase 0 owns the raw RM channel handle and schedules the actual
owning group; (3) multi-queue completion uses one explicit join and replicated
mutable channel state; (4) HCQGraph builds a frozen DAG before scheduling and
uses queue-local monotonic ordinals; and (5) Phase 4 full-token dependency
reconstruction is no longer blocked by the deliberately limited intra-group
wall result. These corrections are normative over older wording in the E2
record or superseded scope.

## 1. Measured foundation (what E1-E3 settled)

All rows OBSERVED, same RTX 5090 box, driver 595.84, one flocked GPU session
per record.

| quantity | value | record |
| --- | ---: | --- |
| llama d512 wall, opt=0 | 246.32 tok/s (median), ~0.185 ms/token delta vs opt=1 | E1 |
| llama replay span vs node-sum, opt=0 | 3.889 ms vs 5.013 ms = 22.4% below node-sum | E1 |
| llama replay span vs node-sum, opt=1 | 3.687 ms vs 4.859 ms = 24.0% | E1 |
| tinygrad d512 serialized node-sum | 5366.1 us, 948 nodes, 5 groups, 0.0% overlap | E2 |
| E2 intra-group critical-path sum | 4757.4 us = 608.8 us / 11.35% saving ceiling | E2 per-group rows |
| E2 merged 2-queue hypothesis | 3310.5 us = 38.3% saving (2.06 ms), cross-group edges absent from capture | E2 |
| E2 merged 3-queue hypothesis | 2786.3 us = 48.1% saving (2.58 ms), cross-group edges absent from capture | E2 |
| CUDA 2-stream elementwise | 48.1% overlap, numerics ok | E3 |
| CUDA 3-stream elementwise | 65.1% overlap, numerics ok | E3 |
| CUDA 2-stream matmul 2048 | 48.4% overlap, numerics ok | E3 |
| native shared-ctx regression E3-E5 | ~0% overlap at all flavors | E3 |
| decode wall authority | 177.72 tok/s ours vs 251.8 llama = 0.704x; 5.367 ms node-sum; 95.4% GPU busy | five-lever record |
| correctness pins | token sha `9d6b3787...`, first token `151936`, decode sha `0721c16f...` | forward scope |

What is settled: (1) llama's overlap is base CUDA graph node scheduling, not
the gated QKV stream fan-out; (2) tinygrad's captured *intra-group* DAG carries
608.8 us / 11.35% of scheduleable saving before resource contention; (3) the
device co-schedules independent kernels when reached through CUDA streams;
(4) the native shared-context construction serializes; (5) the separately-
ctxshared native construction never executed, so the native RM path is
unproven, not disproven.

The E2 merged 2/3-queue rows are a cross-group hypothesis, not a legal-schedule
measurement. `HCQ_GRAPH_PROFILE_JSON` emitted one record per HCQGraph and the
simulator can remap dependencies only inside each record; therefore
`cross_group_edges=0` means "not recorded", never "proven absent". The current
five HCQGraph calls retain full timeline barriers. No phase may quote the
2.06/2.58 ms merged savings as available until a pre-split full-token
range-aware dependency capture proves the cross-group edges and a grouping
change is separately gated.

What remains open: how CUDA's stream concurrency is represented by RM channel
groups, context shares/subcontexts, runlists, and scheduling controls on this
driver; whether a corrected native construction co-schedules at all; how much
of the 608.8 us intra-group ceiling survives DRAM-bandwidth sharing at wall;
and what the legal cross-group DAG and schedule are once captured before
`graph_split_rewrite` inserts the five execution barriers.

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

Each phase starts only after its stated prerequisites pass. Phase 4 requires
the Phase 3 record but does not require G4's >=10% classification, because the
existing group boundaries cap the Phase 2 experiment. All GPU work is one
flocked session at a time (`flock /tmp/nv_gpu.lock`).

### 4.1 Phase 0 - corrected native construction + R1-R5 probe (device-level)

No HCQGraph, decoder, or decode-route changes. Scope: the probe script and a
closed-default device slice that supports the probe.

Work items:

1. Preserve the raw RM channel handle. `_new_gpu_fifo` currently returns a
   `GPFifo` wrapper containing only ring/gpput/count/token, so a caller cannot
   legally issue a later `NVA06F` control. Add `handle: int` to `GPFifo` (index
   0 keeps otherwise identical construction), or make `_new_gpu_fifo` accept
   an explicit closed-default bind/schedule option and issue the controls
   before wrapping the handle. No `rm_control(GPFifo(...), ...)` call is
   permitted.
2. `extra_gpfifos` gains a `per_channel_bind` mode that, for every extra
   channel, issues in order: ctxshare alloc (per-channel when
   `separate_ctxshare`) -> raw channel allocation ->
   `NVA06F_CTRL_CMD_BIND(channel_handle)` ->
   `NVA06F_CTRL_CMD_GPFIFO_SCHEDULE(channel_handle)` ->
   `NVA06C_CTRL_CMD_GPFIFO_SCHEDULE(owning_group)`. Record every RM error per
   operation, handle kind, group, channel, and mode so the failing step is
   identified.
3. H3 variant: `group_per_channel` mode allocating a fresh
   `KEPLER_CHANNEL_GROUP_A` per extra channel (same device), with its own
   ctxshare and fifo, plus per-channel bind/schedule. The final group-level
   schedule targets that fresh group, never `dev.channel_group`. This mirrors
   CUDA's per-context channel ownership most directly.
4. Keep the existing shared-ctxshare path as a control arm under the same
   flock, but execute every construction mode in a fresh subprocess with a
   hard timeout. Flush partial JSON after every RM operation. An execution
   timeout or failed construction poisons only that subprocess; later arms do
   not reuse its RM objects or context.

Pseudocode:

```python
def fix_channel(dev, engine_type, mode):
  if mode == "ctxshare":            # H1/H2: extra ctxshare in boot group
    group = dev.channel_group
    cs = rm_alloc(group, FERMI_CONTEXT_SHARE_A,
                  params(hVASpace=dev.vaspace, flags=SUBCONTEXT_ASYNC))
    channel, fifo = dev._new_gpu_fifo_with_handle(area, cs, group, compute=True,
                                                  debugger=False, engine_type=engine_type)
  elif mode == "group":             # H3: fresh channel group per channel
    group = rm_alloc(dev.nvdevice, KEPLER_CHANNEL_GROUP_A,
                     CHANNEL_GROUP_ALLOC_PARAMS(engineType=ENGINE_TYPE_GRAPHICS))
    cs = rm_alloc(group, FERMI_CONTEXT_SHARE_A, ...)
    channel, fifo = dev._new_gpu_fifo_with_handle(area, cs, group, compute=True,
                                                  debugger=False, engine_type=engine_type)
  rm_control(channel, NVA06F_CTRL_CMD_BIND, BIND_PARAMS(engineType=engine_type))
  rm_control(channel, NVA06F_CTRL_CMD_GPFIFO_SCHEDULE, SCHED_PARAMS(bEnable=1))
  rm_control(group, NVA06C_CTRL_CMD_GPFIFO_SCHEDULE, bEnable=1)
  return fifo
```

Experiments R1-R5 (same span/node-sum criterion as the old E1-E5 probe):

- R1 cross-GPFIFO semaphore dependency: kernel on queue 1 waits on a signal
  released by queue 0; numeric check vs CPU. "Exact" means the anchored output
  hash and maximum-error contract recorded by the probe both match; do not
  describe `np.allclose` alone as exact.
- R2 serial calibration on one fifo: absolute `span - node_sum` and percentage
  delta are recorded; calibration passes inside a declared timestamp tolerance,
  never by floating-point equality.
- R3 two independent fifos, elementwise, full and partial-SM (grid-div 4)
  grids.
- R4 three fifos, elementwise.
- R5 two fifos, matmul 2048.

Each R row records per-kernel HCQ timestamps, span, node-sum, overlap
fraction, output hash/error, subprocess exit/timeout state, and the ordered RM
operation/error list, anchored as incrementally flushed JSON.

Gate G1 (belief-flip): PASS = R1 satisfies its hash/error contract AND at least
one of R3-R5 shows overlap >= 5%. `NO_OVERLAP` = R1 correct but overlap below
5% across all flavors and both successfully constructed modes.
`CONSTRUCTION_BLOCKED` = an RM step rejects, a queue does not execute, or an
arm times out before R1. Any non-PASS result records the exact operation or
execution boundary and closes Phase 1-4 of Route A; neither result is a
hardware no-concurrency verdict (amendment HARD STOP). Route B remains
analysis-only under 4.5.

### 4.2 Phase 1 - D3 substrate: multi-GPFIFO compute queues (closed default)

Gate: G1 passed. Scope: `ops_nv.py` + `graph/hcq.py` seam only; at
`HCQ_NUM_COMPUTE=1` the selected fifo, encoded command stream, signal order,
and observable behavior remain identical; no scheduling policy yet.

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
   fifo carries a 48 MiB uncached errnotifier, `ops_nv.py:660`). Setup
   completion uses the same join invariant required below: private per-queue
   completion followed by one timeline release, never independent writes of
   the same timeline value.
4. Env knob `HCQ_NUM_COMPUTE` (default 1) in `graph/hcq.py`, same shape as
   `HCQ_NUM_SDMA` (`:71`). `comp_queues: dict[dev] -> list[HWQueue]` of length
   N; at N=1 every existing reference (`:98`, `:135`, `:151`, `:195`, `:233`,
   `:254-256`, `:260`, `:331`) resolves to the same single queue as today.
5. AMD control: `ops_amd.py:1109-1110` already keeps `compute_queues` as a
   dict with one ring; the generic `hw_compute_queues()` change must return
   one factory on AMD at `HCQ_NUM_COMPUTE=1`, and the AMD backend must not be
   touched by Phase 1. No AMD GPU on this box: control = identical code path
   at default + review.
6. Replicate mutable compute-channel state. `_ensure_has_local_memory`
   (`ops_nv.py:725-727`) currently updates only `compute_gpfifo`; any later
   scratch/local-memory window update must be submitted to every compute fifo
   that can run a kernel and joined before the device timeline advances.
   Initial `_setup_gpfifos` setup alone is insufficient.
7. Add mock/CPU tests for factory count/indexing, RM handle/control order,
   selected-group scheduling, N=1 encoded-command equivalence, multi-fifo
   setup join, and local-memory-state replication. The live device probe does
   not replace these contracts.

Gate G2: at `HCQ_NUM_COMPUTE=1`, a d512 decode run reproduces the wall
authority within measurement noise (< 1%) and the correctness pins (section
7) pass 3/3; an encoded-queue regression test proves the N=1 command/signal
sequence. At `HCQ_NUM_COMPUTE=2/3`, a device-level probe (R3/R4 shape) still
passes through the new factories before any scheduler exists, including one
kernel that requires the replicated local-memory state.

### 4.3 Phase 2 - D2 dependency-driven compute scheduling in HCQGraph

Gate: G2 passed. Scope: `graph/hcq.py` scheduling only; grouping authority
unchanged (`graph_split_rewrite`, `jit.py:236`, `JIT_BATCH_SIZE`,
`helpers.py:240`).

Work items:

1. Build then schedule; do not assign while discovering hazards. Pass A walks
   calls in original order with a fresh range-aware `DepsTracker` whose
   dependency payload is the producer node index, producing an immutable
   predecessor/successor DAG. Pass B computes static tails and a deterministic
   ready-set list schedule. Pass C assigns queue-local sequence values and
   encodes commands/waits/signals. `_resolve_deps` consumes the frozen
   predecessor list in Pass C; it is not the source of a partially discovered
   graph during assignment.
2. Picker v1: ready node with longest remaining tail, then lowest original
   node index; assign it to the queue minimizing
   `max(queue_free[q], max(pred_end))`, then lowest queue index. The live v1
   cost proxy is `max(1, concrete KernelInfo.estimates.mem)` bytes (copy cost
   is copy bytes), with `ops`, `lds`, and node index used only as deterministic
   zero/equal-memory ties. This is intentionally a decode-memory proxy, not a
   device-time model. Record predicted order and costs. Hardcoded E2 timings
   or program-name timing tables are banned; a later learned/target cost model
   requires its own measured A/B.
3. Queue signal values are queue-local monotonically increasing ordinals,
   independent of original node `j`. Producer node -> `(queue, ordinal)` is
   retained separately for profile dependency export. Any schedule that would
   encode decreasing releases on one queue is rejected before submission.
4. Semantics per queue: `signals`, `last_j`, `queue_signals_to_reset`
   (`hcq.py:260`) become per compute queue; the kickoff wait (`:195`),
   inter-device/copy waits (`:254-256`), and replay submit loop (`:331`) cover
   all active queues.
5. Exactly one designated join queue advances the device timeline. Every
   other active compute queue releases a private completion ordinal after its
   terminal command; the join queue waits those ordinals, waits required copy
   completions, then emits the sole per-device timeline release. Inactive
   queues neither submit nor participate. Independent writes of the same
   timeline value from multiple queues are banned.
6. Replay submission cost is O(active queues) per HCQGraph and ring
   submissions/doorbells increase from one to the active-queue count. Kernel
   command payload remains approximately the same; the scope does not call
   aggregate ring traffic unchanged.
7. Add hermetic tests for DAG construction, ready-set scheduling,
   deterministic ties, dependency preservation, queue-local monotonic
   ordinals, signal elision, the all-active-queues join, an empty queue, and
   N=1 schedule/command equivalence.
8. No change to `JIT_BATCH_SIZE`, graph group boundaries, or kernel content.
   Cross-group dependency reconstruction and regrouping are Phase 4.

Pseudocode (picker + schedule loop):

```python
# Pass A: original-order hazard discovery.
preds = build_range_aware_dag(calls)

# Pass B: ready-set schedule using concrete static costs.
while ready:
  j = max(ready, key=lambda j: (remaining_tail[j], -j))
  pred_finish = max((end[p] for p in preds[j]), default=0)
  q = min(queues, key=lambda q: (max(queue_free[q], pred_finish), q.index))
  start[j] = max(queue_free[q], pred_finish)
  end[j] = start[j] + cost[j]
  queue_free[q] = end[j]

# Pass C: encode queue-local monotonic ordinals and one final join.
encode(j, q, waits=[producer_ordinal[p] for p in preds[j]])
```

Gate G3 has two separately recorded outcomes. G3-C (correctness) passes when
one d512 token under `PROFILE=1 HCQ_GRAPH_PROFILE_JSON` with
`HCQ_NUM_COMPUTE=2/3` has correctness pins 3/3, monotonic queue signals, and
all active queues joined before the sole timeline advance. G3-O (observed
structure value) passes at >=5% intra-group span saving versus 0.0% today.
Report both against the 608.8 us / 11.35% per-group no-contention ceiling. The
former 20%/25% thresholds are targets for a future proven full-token schedule,
not gates on this grouping-preserving phase. Phase 3 runs after G3-C even if
G3-O misses; wall measurement is cheap and records whether the correct
substrate has value.

### 4.4 Phase 3 - decode wall A/B (the parity question)

Gate: G3-C passed. Scope: measurement only; no code beyond the G3 state.

Protocol (one flocked session, same model Qwen3-8B-Q4_K_M, llama control
from the E1 arm-0 command family):

1. Rows: baseline `HCQ_NUM_COMPUTE=1` (today's 177.72 tok/s d512 authority),
   then 2 and 3 queues, at d512 / d2048 / d4096, same harness repetition
   protocol (nmeas=20, reps=3, median), with `HCQ_GRAPH_PROFILE_JSON` on one
   representative token per arm for span/node-sum.
2. llama same-session control at d512 (246.32 tok/s opt=0 authority) and at
   d2048/d4096 as the E1 scope's arm-0 command with matching depth.
3. Bandwidth and grouping caveats reported with the rows: Phase 2 can use
   only the 608.8 us / 11.35% summed intra-group no-contention ceiling. The
   merged 2.06/2.58 ms rows are Phase 4 hypotheses until a full-token DAG is
   captured. Decode GEMVs are DRAM-bound at ~46-50% of 1792 GB/s (INFERRED
   accounting), so concurrent GEMVs approach bandwidth saturation; the
   expected wall conversion is below the intra-group ceiling. The sequential
   tail (rmsnorm/residual/vocab chain) will not overlap regardless of queue
   count; measure per-class overlap to confirm the sim's class pairs.

Gates:

- G4 (wall value classification): median tok/s improvement at d512 >= 10%
  (~0.55 ms/token) with correctness pins 3/3 is `PARITY_SCALE_INTRA_GROUP`.
  Below 10% the mechanism is implemented but not parity-scale at wall; the
  record says so explicitly (no promotion). G4 is not a gate on the Phase 4
  dependency reconstruction because the current graph boundaries cap Phase 2.
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

### 4.6 Phase 4 - full-token DAG + D4 signature/reuse + regrouping (gated)

Gate: G1, G2, and G3-C passed. G3-O and G4 are recorded inputs, not blockers:
the current graph boundaries are the suspected cap. Scope:
`executable-taskgraph-ir-scope` D4 section 7, plus the D2 extension section
8 (non-consecutive batching). Work items:

1. Before changing grouping, capture the full token's dependency DAG from the
   original linear calls before `graph_split_rewrite`. Use one range-aware
   `DepsTracker` across all 948 calls and retain every RAW/WAR/WAW edge that
   crosses a current 32/64/128/256/468 boundary. Validate that restricting
   this DAG back to each existing group reproduces the E2 per-group edges.
   Missing exporter edges are `UNKNOWN`, never independent. Publish the
   corrected critical path and 2/3-queue schedules before authorizing a
   regrouping candidate.
2. Write the reuse rule: same signature (semantic facts + symbolic var_vals +
   buffer slot identities) -> per-replay param update; changed signature ->
   re-instantiate; census records reuse hit/miss per replay so a
   re-instantiation regression is a test failure, not a mystery slowdown.
3. Audit the 5 decode graph groups (32/64/128/256/468) against
   `JIT_BATCH_SIZE` and the admission census; record which nodes are
   consecutive-chain members vs siblings with no true dependency.
4. Candidate: dependency-driven grouping of non-consecutive independent calls
   over the proven full-token DAG, each candidate with its own d512 wall A/B.
   All true cross-group edges are preserved; only artificial group-wide
   timeline barriers may be removed. The merged E2 schedule is not reused as
   an assignment unless the corrected DAG independently reproduces it.

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
- Missing cross-group evidence: the merged E2 schedules assumed absent edges
  because the exporter could not record them. Only the 608.8 us intra-group
  ceiling governs Phase 2; Phase 4 must capture the pre-split full-token DAG.
- Bandwidth wall: even the corrected DAG simulation is a no-contention upper
  bound; concurrent GEMVs approach DRAM saturation at ~50% of peak; expect
  wall conversion below simulated span saving.
- Serialization regression: any shared hcq.py change must be byte-identical
  in selected fifo and encoded command/signal sequence at
  `HCQ_NUM_COMPUTE=1`; AMD has no GPU here, so control = default code path
  identity + review (house rule).
- Premature completion: multiple queues writing the device timeline can let
  the host recycle buffers while work remains. One join queue owns the sole
  final timeline release; this is a tested invariant, not an implementation
  suggestion.
- Queue-state skew: dynamic scratch/local-memory setup on queue 0 alone can
  fault or corrupt work dispatched to extra fifos; mutable compute state is
  replicated and joined.
- Boot cost: per-fifo 48 MiB errnotifier + setup submit; keep extra fifo
  creation lazy or boot-time per measured cost.
- Wrap semantics: `_submit_to_gpfifo` (`ops_nv.py:114-139`) preserves the
  cmdq wrap-drain contract; multi-fifo submission must keep
  `System.memory_barrier()` before each poke and the per-fifo `put_value`
  accounting.
- Hang diagnostics: extra fifos are created `debugger=False`;
  `on_device_hang` inspects only `debug_channel`; each Phase 0 arm is an
  isolated timed subprocess with incrementally flushed RM records, and R1
  detects silent mis-execution before any decode work.
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
| 0 | corrected `nv_multi_queue_probe.py` handle-owning construction modes; isolated/timeout measurement record with anchored JSON (R1-R5 rows, RM operations/errors per step); G1 verdict | `[test]` + `[docs]` |
| 1 | `ops_nv.py` compute_gpfifos + queue_idx + hw_compute_queues(N); replicated mutable state + setup join; `HCQ_NUM_COMPUTE`; mock tests; G2 record (N=1 encoded identity + decode pins) | `[runtime]` + `[test]` + `[docs]` |
| 2 | `graph/hcq.py` three-pass DAG/scheduler/encoder, queue-local ordinals, single timeline join; hermetic tests; G3-C/G3-O record | `[runtime]` + `[test]` + `[docs]` |
| 3 | wall A/B record d512/d2048/d4096 vs llama, per-class overlap, bandwidth caveat; G4/G5 verdict | `[docs]` |
| 4 | pre-split full-token dependency capture + corrected simulation; D4 reuse rules + census; non-consecutive regrouping candidate + A/B if gated | `[runtime]` + `[test]` + `[docs]` |
| 4.5 | Route B analysis note (if G1 fails) | `[docs]` |

## 9. Bans and HARD STOPs

- HARD STOP: no declaring native overlap impossible; no composing a parity
  endpoint; no promoting a route. Route B implementation is NOT authorized
  by this document.
- No user files: `docs/README.md`, `docs/beating-llama-first-principles-
  20260731.md`, `docs/what-makes-inference-fast.md`,
  `extra/llm_research/microbench/*` binaries (including the untracked
  `cuda_stream_overlap_probe` binary), `scratchpad/t6_metal_admission_probe.py`.
- No `master`/`dev`/`exp` commits; no concurrent GPU sessions.
- `HCQ_NUM_COMPUTE=1` must preserve the selected fifo and encoded
  command/signal stream; AMD control for shared changes.
- No kernel/dtype changes, no fusion folds, no host-launch work (separate
  scopes), no whole-buffer dependency edges, no grouping changes before
  Phase 4.

## 10. Sequencing summary

| phase | work | gate | note |
| --- | --- | --- | --- |
| 0 | construction fix + R1-R5 | G1 >= 5% overlap + R1 hash/error contract | isolated arms; classify blocked vs no-overlap |
| 1 | D3 substrate | G2 N=1 encoded identity + pins | closed default; state replication |
| 2 | D2 intra-group scheduler | G3-C correctness; G3-O >=5% reported separately | 608.8 us ceiling; grouping unchanged |
| 3 | wall A/B | after G3-C; G4 >=10% classification; G5 parity rows | llama same-session |
| 4 | full-token DAG + D4 + regrouping | G1 + G2 + G3-C | independent of G3-O/G4 value |

## 11. One-line job

Make the native RM co-schedule independent compute GPFIFOs like CUDA streams
do (handle-correct per-channel bind/schedule, replicated channel state, then N
compute queues in HCQGraph with a frozen dependency DAG, queue-local signals,
and one final join); prove the intra-group result at d512 wall against its
608.8 us ceiling, then capture the pre-split full-token DAG before claiming or
implementing the larger cross-group opportunity.
