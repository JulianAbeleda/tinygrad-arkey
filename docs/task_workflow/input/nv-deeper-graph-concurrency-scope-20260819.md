# NV deeper graph concurrency scope (lever 1) - 2026-08-19

Date: 2026-08-19
Branch: `nvidia-bringup-20260731`, HEAD `d14e6964e`
Status: **exhaustive scope plus queue-count test.** Adding compute queues past
the current two is a dead end on the recorded decode DAG: the ideal 3-queue
schedule beats the ideal 2-queue schedule by at most 26.4 us/token, and that
upper bound omits real cross-queue wait overhead. The remaining gap is on the
critical path, not in queue count.

## 1. Exact question

"Lever 1: deeper graph concurrency beyond the currently qualified two NV
compute GPFIFOs." Deeper concurrency can mean several different things, so the
scope splits them before deciding anything.

## 2. Mechanism decomposition

| id | mechanism | what it would change |
| --- | --- | --- |
| A | More compute GPFIFOs (3, 4, N) | More ready nodes can launch concurrently |
| B | Different cross-queue wait construction | Remove the semaphore wait boundary that currently degrades the pair |
| C | Graph restructuring (anchor+shadow) | Move support work off the critical path behind a long GEMV anchor |
| D | PDL (programmatic dependent launch) | Overlap next-kernel prologue on one GPFIFO without a second queue |
| E | Different engine family (copy/DMA) | Run tiny support kernels on a non-compute queue |
| F | Same-queue adjacency/driver overlap | Let the driver co-schedule resource-complementary neighbors |

## 3. Current baseline

The landed overlap substrate already boots two compute GPFIFOs and enables
generic readiness placement:

- `tinygrad/runtime/ops_nv.py`: `boot_compute_channels = min(2, ...)`
  (default 2).
- `tinygrad/runtime/graph/hcq.py`: `HCQ_NV_READY_PLACEMENT=1` (default).
- `tinygrad/schedule/memory.py`: reuse-lane arena coloring removes the false
  WAR/WAW edges that kept the norm/`E_*` siblings serialized.

Measured on the production route, same session, token sha identical:

| arm | tok/s | wall us/token | decode census |
| --- | ---: | ---: | ---: |
| 1 queue serial | 205.99 | 4854.8 | 32/0 |
| 2 queues landed | 212.12 | 4714.2 | 21/11 |

So two queues are already worth ~6 tok/s in the wall A/B. The question is
whether adding a third or fourth queue buys more.

## 4. The test that answers mechanism A

`extra/llm_research/decode/nv_queue_count_sweep.py` computes the
dependency-critical-path lower bound and a longest-tail list schedule over the
recorded 596-node decode DAG
(`docs/task_workflow/evidence/nv-dag-duration-head-20260812.json`).

| queues | span us | saving vs serial | slack vs critical path |
| ---: | ---: | ---: | ---: |
| 1 | 5493.27 | 0.00 | 650.89 |
| 2 | 4868.78 | 624.49 | 26.40 |
| 3 | 4842.38 | 650.89 | 0.00 |
| 4 | 4842.38 | 650.89 | 0.00 |
| 8 | 4842.38 | 650.89 | 0.00 |

The serialized node sum is 5493.27 us and the dependency critical path is
4842.38 us. Two queues capture 624.49 us of the 650.89 us schedule slack; the
entire remaining slack that any number of additional queues can capture is
26.40 us, about 0.5% of the wall and roughly 1.1 tok/s at this baseline.

This schedule deliberately omits cross-queue wait cost, so it is an upper
bound on the benefit. Real hardware must insert a signal wait for every
cross-queue dependency and can only be equal or worse. The 08-17 shape probe
already showed those wait boundaries make small-kernel shadows flat-to-negative
on the native pair, so the realized 3-queue gain is expected to be below 26.4
us, not above it.

Evidence: `docs/task_workflow/evidence/nv-queue-count-sweep-20260819.json`.

## 5. Why the wall A/B gain is much smaller than the ideal schedule saving

The schedule says two queues should save 624.49 us, but the wall A/B saved only
~140 us this session. The gap is the hardware reality documented in
`nv-overlap-substrate-build-scope-20260817.md`:

- The ready siblings are short GEMVs that contend for HBM bandwidth, so they
  do not run at full speed in parallel.
- Every cross-queue handoff pays a runqueue/wait boundary; the real decode
  support kernels (rope/norm/kv, 0.5-1.5 us) are too small to amortize it.
- The ready sets are temporally sandwiched between a primary producer and a
  primary consumer, so the auxiliary queue has little independent continuation
  to overlap.

This is exactly why more queues do not help: the schedule slack after two
queues is already only 26 us, and the hardware tax on extra queues erases that
margin.

## 6. Construction blockers for N > 2 compute GPFIFOs

Building a third queue is not just removing the `min(2, ...)` clamp:

1. The GPFIFO ring area is allocated as `0x300000` bytes and laid out as
   1 MiB strides (`offset=0x100000*i`). Two compute channels plus the DMA
   channel exactly consume the three strides. A third compute channel needs a
   larger `gpfifo_area`, a new offset layout, and revalidation of the DMA
   channel placement.
2. `boot_compute_channels = min(2, max(1, getenv("HCQ_NUM_COMPUTE", 2)))`
   deliberately fails closed above two because only two have been qualified
   on this driver/topology.
3. `hw_compute_queues()` derives `COMPUTE:{i}` from
   `len(self.compute_gpfifos)`, so the scheduler would accept more queues
   automatically, but the readiness placement and the timeline join at
   `hcq.py:349` would then serialize all of them back to the primary queue at
   the end of the graph.
4. There is no evidence the driver co-schedules more than two async
   subcontexts here; adding channels risks idle channels plus extra join
   signals, not concurrency.

## 7. Mechanism verdicts

| id | verdict | basis |
| --- | --- | --- |
| A more compute GPFIFOs | **NO-GO** | ideal upper bound +26.4 us over 2q; real wait tax likely makes it negative |
| B different wait construction | **not queue-count work** | the blocker is the wait boundary, which belongs to PDL or a no-wait substrate, not adding channels |
| C anchor+shadow restructuring | **OPEN, real concurrency lever** | current DAG's critical path is the floor; shadow composition can cut it, but it is a graph/body change |
| D PDL | **OPEN, bounded** | llama's mechanism; previously priced ~18-33 us recoverable after fusion |
| E copy/DMA engine for support | **NO-GO** | rope/norm need rsqrt/exp compute, so DMA cannot execute them |
| F same-queue adjacency/driver overlap | **FLAT/negative** | size-class probe showed tiny shadows -4% to -37% |

## 8. Remaining arithmetic after this scope

The current landed wall is ~4714 us/token. llama is ~3931 us/token. The gap is
~783 us. The queue-count sweep shows that no number of queues can remove more
than the remaining 26.4 us of schedule slack on this DAG; the other ~757 us is
critical-path time. Therefore:

- Closing lever 1 by adding GPFIFOs is **not worth building**.
- Any further overlap has to come from mechanism C (shadow work moved off the
  critical path) or D (PDL prologue overlap), both of which change what work is
  on the critical path rather than how many queues execute it.
- The non-overlap rows remain the higher-confidence remaining work: vocab top-1
  codegen (~5 us) and generalized NV flash-shape search.

## 9. Promotion decision

1. Mark mechanism A closed: do not relax the `HCQ_NUM_COMPUTE` clamp or expand
   the GPFIFO area to three compute channels.
2. Keep the landed two-queue/reuse-lane construction as the overlap baseline.
3. If deeper concurrency is revisited, the next experiment is mechanism D:
   a two-kernel PDL microbench, then a decode A/B. Mechanism C requires a
   separate graph-topology design and its own correctness gate.

## 10. Acceptance gate if mechanism A is ever retried

A third compute channel would need to beat the landed two-queue wall by more
than 50 us/token in a same-session flocked A/B with identical token sha. The
schedule upper bound is 26.4 us, so that gate cannot pass on the current DAG
without simultaneously changing the DAG itself (which would then be mechanism
C, not A).

## Evidence

- `docs/task_workflow/evidence/nv-queue-count-sweep-20260819.json`
- `docs/task_workflow/evidence/nv-overlap-substrate-reuse-lanes-20260819.json`
- `docs/task_workflow/evidence/nv-dag-duration-head-20260812.json`
- `docs/task_workflow/input/nv-overlap-substrate-build-scope-20260817.md`
- `docs/task_workflow/input/nv-substrate-exhaustive-scope-20260817.md`
