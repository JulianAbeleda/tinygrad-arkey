# Route B, B2: multi-stream CUDAGraph lowerer - measurement record (2026-08-04)

Status: G-B2 passed with a belief refinement. The capture-based multi-stream
lowerer is viable and correct, but on this driver it adds nothing over the
existing programmatic graph path: the graph scheduler already co-schedules
independent nodes on internal execution streams in ANY graph (programmatic
included). The decode graph's -4.5% serialization (B0) was a DAG-shape
artifact (memory-planner aliasing chains the frozen dependency DAG), not a
graph-mechanism limit. Evidence classes: OBSERVED / INFERRED throughout.

## 1. Scope reference

- Scope: docs/task_workflow/input/nv-decode-overlap-route-b-implementation-
  scope-20260804.md section 3 (B2, closed default).
- Gate G-B2: (1) N=1 decode run reproduces B0.2 wall (157.93 tok/s D) and
  token sha within measurement noise; (2) N=2/3 device probe shows overlap
  >= 5% with correct numerics through the new lowerer.
- Prior evidence: B1 record (nv-decode-overlap-route-b1-multi-stream-graph-
  probe-measurement-record-20260804.md) established that captured multi-
  stream graphs co-schedule decode-sized kernels (25% at 3 us kernels), and
  that the programmatic control co-scheduled only weakly (~5%).

## 2. What was built (B2 lowerer)

`tinygrad/runtime/graph/cuda.py` gains a capture-based construction path
behind `CUDA_GRAPH_STREAMS` (default 1). At the default the programmatic
path is byte-identical to pre-B2 (same `cuGraphAddKernelNode` /
`cuGraphAddMemcpyNode` construction; `__call__`, per-replay param updates,
and `cuGraphLaunch(instance, None)` unchanged). At N>1:

1. First pass freezes the range-aware dependency DAG as producer call-index
   lists (`_access_resources` with `new_dependency=j`), the same edge
   semantics the programmatic path uses and the Phase 4 tooling captures.
2. Cost proxy `max(1, mem)` from the SINK KernelInfo estimates (program) or
   copy bytes; ready-set list schedule (`plan_multi_stream`: longest
   remaining tail, queue minimizing `max(busy, pred_end)`, lowest index
   tie-break); cross-stream edges deduped (`cross_stream_edges`).
3. Capture: fork events from the launch stream to N-1 non-blocking worker
   streams, one event wait per cross-stream edge (event keyed by
   (producer, consumer-stream), recorded on the producer stream at its first
   consumer, so queue-local ordinals stay monotonic), join events back.
4. Captured nodes are matched back to calls by
   (func, grid, block, sharedMemBytes) tuple + occurrence order (CUPTI
   returns captured kernel nodes in capture order; `GetParams` v1/v2 return
   NULL pointer fields for captured nodes, so pointer matching is
   impossible - supporting probe:
   extra/llm_research/microbench/cuda_graph_node_params_probe.py).
   Memcpy nodes match on src/dst device pointers.
5. Teardown in `__del__` for capture streams, fork/join/cross events.

Hermetic tests (CPU): test/unit/test_cuda_graph_multi_stream_schedule.py -
DAG construction, determinism (canonical + random DAG), dependency
preservation via ready-set replay, stream spread for independent nodes,
chain stays on one stream, tie-breaks, exact cross-edge emission. 15 passed.

## 3. Measurement method

Probe: extra/llm_research/microbench/cuda_graph_multi_stream_tg_probe.py -
a tinygrad jit with CH independent chains of N elementwise kernels plus a
final join (decode-sized 2-4.5 us kernels), replayed REPS times through the
actual lowerer. `NO_MEMORY_PLANNER=1` is required (OBSERVED): the jit
memory planner aliases the chains' intermediates into one base buffer
(liveness-based arena reuse, `memory_plan_rewrite`), which adds WAR edges
and freezes the DAG as a strict chain (`preds [[], [0], [1], ...]` with the
default planner - the same chain that made B0's decode graph serialize).
With planning disabled and the keep list pinning each intermediate, the
frozen DAG is the intended independent-chains-plus-join shape. Chain i also
uses (i+1)*SZ floats so allocator size classes cannot merge chains.

Ground truth: `nsys profile --cuda-graph-trace=node` + SQLite export;
per-replay span and node-sum from CUPTI kernel rows grouped by graphId and
split into replay clusters by start-time gaps (>20 us); overlap =
(1 - span/node_sum). Replays: 5-7 per graph. Numerics checked vs a numpy
reference (max err 0.0 on every run below).

## 4. Results (CUPTI in-graph, per replay)

### 4.1 CH=2 (2 chains x 8 kernels + join, 17 nodes, SZ=524288 base)

| streams | overlap per replay | median | span vs node-sum |
| --- | --- | ---: | --- |
| 1 (programmatic control) | 18.4-23.4% | ~18.6% | ~52 us vs ~64-67 us |
| 2 (capture) | 18.4-19.4% | ~18.9% | ~52 us vs ~64-67 us |
| 3 (capture) | 18.5-23.5% | ~19.0% | ~52 us vs ~64-67 us |

The S=1 control already overlaps; the capture path matches it. Numerics ok
(max err 0.0) at every stream count. The S=1 timeline shows the graph
replaying on TWO internal execution streams (s13/s14) with ~2 us
interleaving between chains: the driver assigns independent nodes to
internal streams by itself.

### 4.2 CH=6 (6 chains x 8 kernels + join; jit splits into 2 graphs per replay)

| streams | graph 2 overlap | graph 5 overlap | span vs node-sum (graph 2) |
| --- | --- | --- | --- |
| 1 (programmatic control) | 51.4-58.8% | 24.4-32.4% | ~176 us vs ~370-425 us |
| 3 (capture) | 41.2-45.6% | 24.7-28.2% | ~178 us vs ~307-327 us |

At higher concurrency the programmatic control co-schedules aggressively
(52-59% on graph 2, three internal streams visible in the timeline:
s17/s18/s19); the 3-stream capture path is NOT better - if anything slightly
worse (44-46%), likely from the event fork/join structure constraining the
scheduler (INFERRED). Numerics ok (max err 0.0).

## 5. Verdict

### 5.1 Gate G-B2: PASS (both criteria), with a belief refinement

1. N=1 regression: PASS (OBSERVED). `DEV=CUDA CUDA_GRAPH_STREAMS=1`
   decode harness: D 159.72 tok/s vs B0.2 157.93 (within noise), W 177.67
   vs 177.94, token sha `55f7a13b...` identical, first token 38835,
   deterministic 6-token loop. Byte-identical programmatic path.
2. N=2/3 probe: overlap 19% >= 5% with correct numerics (OBSERVED) - but
   the N=1 control passes the same bar (18.6%), so the multi-stream
   capture mechanism is not what produces the overlap.

Belief-flip statement that DID occur: "the programmatic path cannot
co-schedule decode-sized independent kernels on this driver" (carried from
B1's Arm B, ~5%) is FALSE for tinygrad-shaped kernels. The tinygrad
programmatic graph at S=1 overlaps 19% at CH=2 and 52-59% at CH=6.

Remaining open question (INFERRED, not isolated): why B1's Arm B
programmatic control (~5% at 5.5 us, 256-thread blocks) co-schedules so
much less than the tinygrad programmatic path (~19% at 2-4.5 us,
32-thread blocks). Candidates: block count/granularity (128-256 blocks of
32 threads vs 4096 blocks of 256 threads), kernel duration, launch stream
(null vs real). Not load-bearing for Route B because the tinygrad path is
the one that matters.

### 5.2 Consequence for the route (INFERRED)

1. B0's -4.5% decode serialization was the planner-chain DAG, not a graph
   mechanism limit. Confirmed two ways: (a) the same tinygrad kernels on a
   planner-chained DAG replay -4.5% (S=1, pre-existing trace); (b) the
   same lowerer on an independent DAG replays +19% (this record), even at
   CUDA_GRAPH_STREAMS=1.
2. The multi-stream capture lowerer is viable, correct, and available
   behind `CUDA_GRAPH_STREAMS` (default off), but on driver 595.84 it is
   REDUNDANT for overlap: the graph scheduler already overlaps independent
   nodes internally. The record keeps the default as the byte-identical
   programmatic path; flipping the default is not authorized by this
   document and has no measured upside on this driver.
3. The real lever is DAG independence. B3's decode-DAG census
   (how much true independence survives the memory planner / scheduler
   rewrite in a real decode graph) is now the decisive experiment, and it
   should measure the EXISTING programmatic path, not only the capture
   path. If the decode DAG has exploitable independence, overlap appears
   without any lowerer change; if not, the lever moves to dependency/
   planner work (avoiding WAR aliasing), not to stream assignment.

## 6. Honest corrections to the scope

- Work item 3's "queue-local monotonic ordinals": implemented as per-
  (producer, consumer-stream) events recorded on the producer stream at
  first consumer (stream order preserved; no decreasing releases).
- Scope's B2 framing inherited B1's "capture structure carries the stream
  affinity the scheduler uses; the programmatic path does not reproduce
  it". This record shows that conclusion does not transfer to tinygrad's
  kernel shape; the programmatic path reproduces capture-level overlap.
- The N=2/3 gate alone would have been a false positive for the multi-
  stream mechanism; the control arm (S=1) is what exposes the redundancy.

## 7. Artifacts

- tinygrad/runtime/graph/cuda.py (lowerer, env-gated).
- test/unit/test_cuda_graph_multi_stream_schedule.py (15 hermetic tests).
- extra/llm_research/microbench/cuda_graph_multi_stream_tg_probe.py.
- extra/llm_research/microbench/cuda_graph_node_params_probe.py.
- Traces: /tmp/b2_tg_probe_s{1,2,3}.{nsys-rep,sqlite} (CH=2),
  /tmp/b2_tg_c6_s{1,3}.{nsys-rep,sqlite} (CH=6); analyzer
  /tmp/b2_tg_analyze.py (scratch, not committed).
