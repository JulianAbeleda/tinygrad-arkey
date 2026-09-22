# NV decode overlap - Route B post-B2 external review scope

Date: 2026-08-04

Status: review scope, docs only. Self-contained brief for a fresh-eyes agent.
Authorizes no code change, no GPU use by the reviewer, no route flip, no
promotion to `dev`/`exp`/`master`. Branch boundary: tinygrad
`nvidia-bringup-20260731` @ `70de5dc0f` (B2 pushed). Purpose: record the
findings through B2, state the refined design hypothesis, and get an
adversarial review of that hypothesis and the forward plan (B3) before we
spend more GPU time. Every number below is OBSERVED unless marked INFERRED;
part of the review is checking that the labels are earned.

> **Authority amendment (2026-08-04):**
> `nv-decode-overlap-route-b3-external-review-amendment-20260804.md`
> records the external-review disposition and supersedes the root-cause wording,
> route-mismatched ceiling, census decision rule, CPU-only label, and B3 gates in
> this brief. This file remains the pre-review record.

---

## 1. The ask (read this first)

Decode wall time is ~1.4-1.5x behind llama.cpp on the same box, same model
(Qwen3-8B Q4_K_M), same session (llama-bench 246.32 tok/s opt=0 vs NV-route
177.72 tok/s d512). We decomposed the gap and chased graph-level overlap
through Route B (DEV=CUDA + CUDAGraph). After B1 (mechanism proven) and B2
(lowerer built), the story changed in a way we did not predict: **the CUDA
graph scheduler already co-schedules independent nodes on this driver in ANY
graph, programmatic included; our decode graph replays serialized (-4.5%)
because the frozen dependency DAG is a chain, and the chain is caused by the
jit memory planner aliasing buffers, not by the graph mechanism.**

Your job as reviewer:

1. Read sections 2-6 and the grounding artifacts in section 9.
2. Challenge every assumption in section 7 explicitly; state which are false
   or unproven and what evidence settles each.
3. Answer the review questions in section 8.
4. Rank the three cheapest decisive experiments that would confirm or refute
   the refined hypothesis, including any you would add to section 5's plan.
5. Look for the clear reason we still cannot see: if the decode DAG turns
   out to be genuinely chain-shaped after the census, is overlap on decode
   even the right lever, or is the 1.4-1.5x gap mostly elsewhere?

Do not assume the repo docs are correct. The evidence classes are the
authors' labels; part of the review is checking whether they are earned.

## 2. Findings through B2, ordered by severity

### 2.1 The driver co-schedules independent nodes internally - even in a programmatic graph (OBSERVED, load-bearing)

The B2 device probe (tinygrad jit, CH independent elementwise chains of 8
kernels + join, 2-4.5 us kernels, CUPTI node trace) replays with real
overlap at CUDA_GRAPH_STREAMS=1, which is the plain programmatic path
(`cuGraphAddKernelNode`, no capture, no explicit streams):

| shape | overlap (median) | span vs node-sum |
| --- | ---: | --- |
| 2 chains (17 nodes) | ~18.6% | ~52 us vs ~64-67 us |
| 6 chains (49 nodes, 2 graphs) | 51-59% / 24-32% | ~176 us vs ~370-425 us |

The timeline shows the graph replaying on 2-3 internal execution streams
(CUPTI streamIds differ per node), i.e. the graph scheduler assigns
independent nodes to its own sub-streams without any stream hints from us.

### 2.2 The B2 capture-based multi-stream lowerer is redundant on this driver (OBSERVED)

The capture path (fork/join events, per-cross-edge event waits, N non-
blocking streams) replays at the same overlap as the control:

| streams | 2 chains | 6 chains (graph 2 / graph 5) |
| --- | ---: | ---: |
| 1 (programmatic control) | ~18.6% | 51-59% / 24-32% |
| 2 (capture) | ~18.9% | - |
| 3 (capture) | ~19.0% | 41-46% / 25-28% |

If anything, capture is slightly worse at 6 chains (event structure
constrains the scheduler, INFERRED). Numerics exact (max err 0.0) on every
run. The lowerer is built, hermetic-tested (15 tests), N=1 byte-identical,
and available behind `CUDA_GRAPH_STREAMS` (default 1); it is just not the
differentiator we expected.

### 2.3 B0's -4.5% serialized decode is a DAG-shape artifact, not a mechanism limit (OBSERVED + INFERRED)

The same lowerer and same kernel sizes replay -4.5% when the frozen DAG is a
strict chain and +19% when it has two independent chains. The chain itself
is caused by `memory_plan_rewrite` (tinygrad/schedule/memory.py): the jit
planner maps all intermediates of independent chains into one arena base
buffer via liveness-based TLSF reuse, and `DepsTracker` then emits WAR/WAW
edges between every consecutive call (`preds [[], [0], [1], ...]` observed
via the B2 probe's debug DAG dump). Disabling the planner
(`NO_MEMORY_PLANNER=1`, each intermediate pinned by the probe's keep list)
freezes the intended independent DAG. INFERRED part: the real decode graph
is planner-chained the same way; we have not yet frozen and counted a real
decode DAG. That is the B3 census.

### 2.4 llama's 22.4% overlap is real and reproducible (OBSERVED)

E1 re-analysis: 545/761 adjacent kernel pairs overlap, median kernel 3.1 us,
span 3890.5 us vs node-sum 5013.1 us. B1 capture arms reproduce 25-32%
overlap at 1-3 us kernels. llama's graph is capture-built with explicit
streams, but our 2.1 result shows the stream structure is not what makes it
overlap on this driver.

### 2.5 B1's "programmatic does not co-schedule" does not transfer to tinygrad's kernel shape (OBSERVED contradiction, cause INFERRED)

B1 Arm B (programmatic, 16 kernels, 256-thread blocks, 5.5 us) co-scheduled
only ~5%. The tinygrad programmatic path (32-thread blocks, 2-4.5 us) gets
~19% at the same DAG shape. Open INFERRED cause: block count/granularity
(128-256 blocks of 32 threads vs 4096 blocks of 256 threads) and/or launch
stream (null vs real). Not isolated; not load-bearing for the route, but it
should be explained before trusting any "programmatic is weak" claim.

## 3. The current design approach (pseudocode)

```python
# 3.1 THE OVERLAP MODEL (measured this campaign)
def overlap(dag: DAG, kernels) -> float:
    """Fraction of node-sum hidden by co-scheduling on driver 595.84."""
    if not has_independent_nodes(dag): return negative_pipeline_tail()  # -4.5%
    return size_dependent_co_schedule(kernels)   # ~19% at 2 chains, ~52% at 6

# 3.2 WHY OUR DECODE GRAPH HAS NO INDEPENDENCE (current hypothesis)
def frozen_dag(linear, held_bufs):
    linear = memory_plan_rewrite(linear, held_bufs)   # arena-aliases buffers
    return DepsTracker(linear).edges                   # WAR edges chain the DAG

# 3.3 ROUTE B DECISION TREE (post-B2, what B3 must decide)
if decode_dag_has_independence(census):
    measure wall at CUDA_GRAPH_STREAMS=1        # capture NOT required
    # G-B3-W: median d512 >= +5% over 157.93 tok/s OR classified not-wall-positive
else:  # chain-shaped (planner aliasing)
    dependency/planner work is the lever, not streams:
    - measure upper bound via NO_MEMORY_PLANNER=1 decode run
    - candidate fixes: held-buf liveness for graph-independent tensors,
      planner reuse policy, scheduling order
    # G-B3-P only after G-B3-C (pins 3/3) and a same-session llama row
```

Key phrases:

- `span vs node-sum`: the only honest overlap metric on this driver is
  CUPTI in-graph node timing (event-node elapsed time is invalid in capture,
  inflated ~1.4x in programmatic graphs - B1 record section 3).
- `DAG independence is the raw material`: the scheduler overlaps what the
  frozen DAG permits; nothing else matters first.
- `planner chaining is a measurement of our own making`: whether decode
  overlap is achievable at all depends on how much independence survives
  `memory_plan_rewrite`, which is ours to change.

## 4. The B3 plan (to be reviewed)

Work items (gated by the Route B implementation scope, section 4):

1. **Decode DAG independence census (new, first)**: freeze a real d512
   decode linear and count the frozen dependency structure - edges per
   call, longest chain vs node count, number of independent branches -
   using the range-aware DAG capture seam (CPU-only, cheapest) cross-checked
   by one CUPTI node trace of the existing programmatic decode graph.
   Decision rule: >= 90% strict-chain edges => chain-shaped; branches above
   the scheduler's internal-stream count => overlap should already appear
   at S=1.
2. **Numerics census**: NV route 948 kernels vs CUDA route 1021 kernels per
   token - per-class max-error checks (flash/q4k/q6k/norms) vs CPU
   references; classify as legitimate different-kernel-chain vs CUDA bug.
3. **CUDA-route re-pin**: fixed-depth protocol, 3/3 reproducibility, pins
   labeled CUDA-route pins (never presented as NV pins).
4. **Wall A/B** (same session, d512/d2048/d4096): CUDA route (S=1, and S=2/3
   if the census shows independence) vs NV 177.72 tok/s vs same-session
   llama 246.32 tok/s. Report per-class overlap, node-sum/span, bandwidth
   caveat (decode GEMVs ~50% of 1792 GB/s, INFERRED).
5. If chain-shaped: upper-bound run with the planner disabled, then a
   scoped dependency/planner candidate behind an env flag.

Gates: G-B3-C (CUDA pins 3/3 + per-class numerics), G-B3-W (median d512
wall >= +5% over 157.93 with S=2/3 OR classified not-wall-positive against
the 608.8 us ceiling), G-B3-P (same-session row vs llama >= 1.00 qualifies
that depth only; no composed endpoint).

## 5. Assumptions to review (numbered)

1. The real decode DAG is planner-chained like the probe's (INFERRED;
   settled by the census).
2. Driver co-scheduling is size-dependent with a ~few-us per-transition
   pipeline window, and decode kernels (1-6 us) are in the favorable regime
   (OBSERVED at B1/B2 probe level; not yet on a decode graph).
3. Multi-stream capture is unnecessary on driver 595.84 for overlap
   (OBSERVED at probe level, CH=2 and CH=6; not yet on a decode graph).
4. Block granularity (32 vs 256 threads) explains 2.5's contradiction
   (INFERRED, untested).
5. Decode is bandwidth-bound and GEMVs reach only ~50% of peak 1792 GB/s
   (INFERRED from earlier records).
6. The CUDA route's 1021 kernels vs NV's 948 is a legitimate different
   kernel chain, not a numeric bug (UNKNOWN; numerics census).
7. Overlap, if achieved on decode, converts into wall time (UNKNOWN; the
   replay launch overhead and the join latency are small but unmeasured on
   a 1000-node graph).
8. B0.2 (157.93 tok/s d512) is the right CUDA-route baseline floor for
   G-B3-W (OBSERVED; re-pin protocol refreshes it).

## 6. Decisive experiments (cheapest first)

1. Decode DAG census on the CPU-side capture seam: no GPU, minutes.
   Belief-flip: a branched frozen DAG flips the plan to "measure wall at
   S=1, capture not needed"; a chain-shaped DAG flips it to "planner work".
2. One CUPTI node trace of the existing decode graph at S=1 (ground truth
   for what the scheduler currently does with the real DAG; ~10 min).
3. If chain-shaped: NO_MEMORY_PLANNER=1 decode run - upper bound on
   overlap recovery and whether it converts to wall.

## 7. Review questions

1. Is the planner-chain root cause consistent with ALL evidence (B0 -4.5%,
   B2 control +19%, llama 22.4%)? What alternative explains decode
   serialization?
2. Is the census the cheapest decisive experiment, or is there a cheaper
   discriminator between chain-shaped and independent decode DAGs?
3. If the decode DAG is genuinely chain-shaped, is graph overlap even the
   right lever, or does the 1.4-1.5x gap live mostly in kernel count,
   fusion, bandwidth, or boundary time?
4. Is the "capture redundant" verdict over-fit to elementwise chains?
   Would a decode-shaped DAG (mixed GEMM/elementwise, 1000+ nodes) behave
   differently under capture vs programmatic?
5. Are any evidence labels wrong, and are there gaps in the trace-to-
   conclusion chain (clustering by start-time gap, per-graph attribution)?

## 8. Grounding artifacts

- Commits: B0 `ea4fbd439`, B1 `2ed668927`, B2 `70de5dc0f` (all pushed).
- Records: `nv-decode-overlap-route-b-viability-record-20260804.md`,
  `nv-decode-overlap-route-b1-multi-stream-graph-probe-measurement-record-
  20260804.md`, `nv-decode-overlap-route-b2-multi-stream-lowerer-
  measurement-record-20260804.md`, and the implementation scope
  `nv-decode-overlap-route-b-implementation-scope-20260804.md`.
- Probes: `extra/llm_research/microbench/cuda_graph_multi_stream_tg_probe.py`
  (B2), `cuda_graph_stream_overlap_probe.cu` (B1), `cuda_graph_node_params_
  probe.py` (node matching).
- Traces: `/tmp/b2_tg_probe_s{1,2,3}.{nsys-rep,sqlite}` (CH=2),
  `/tmp/b2_tg_c6_s{1,3}.{nsys-rep,sqlite}` (CH=6), `/tmp/b1_trace*.sqlite`
  (B1), `/tmp/e1_arm0_trace.sqlite` (llama re-analysis).

## 9. Bans and HARD STOP

- HARD STOP: no declaring hardware no-concurrency; no composed parity
  endpoint; no route promotion; reviewer makes no code changes and uses no
  GPU.
- No changes to the NV decode route, HCQGraph, or ops_nv.py.
- No user files (`docs/README.md`, `docs/beating-llama-*`,
  `docs/what-makes-inference-fast-*`, `extra/llm_research/microbench/*`
  binaries, `scratchpad/t6_metal_admission_probe.py`).

## 10. One-line job

Review whether DAG shape (memory-planner chaining) is the true reason our
decode graph does not overlap while llama's does, and whether the B3 census
plus dependency/planner plan is the right cheapest path - and tell us the
clear reason we still cannot see.
