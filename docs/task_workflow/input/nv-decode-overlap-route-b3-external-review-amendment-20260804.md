# NV decode overlap - Route B3 external-review amendment

Date: 2026-08-04

Status: review comments addressed; authoritative amendment to
`nv-decode-overlap-route-b3-external-review-scope-20260804.md`. This document
supersedes only the claims, experiment ordering, and gates named below. The
original brief remains the pre-review evidence record. Branch boundary at
amendment drafting: `nvidia-bringup-20260731` @ `9120df6ac`.

This amendment does not authorize a route flip, promotion, native NV runtime
change, HCQGraph change, or GPU use by the reviewer. It authorizes the future
B3 implementer to extend analysis/capture tooling and to run the explicitly
gated, lock-held measurements below under the existing Route B implementation
boundary. A planner/runtime candidate remains separately gated after the DAG
attribution result.

---

## 1. External-review disposition

The review accepts the direction but rejects the original B3 plan as
execution-ready for five load-bearing reasons:

1. The cited `608.8 us / 11.35%` ceiling was measured on the native NV route
   (948 kernels, five groups), while Route B is the CUDA route (1021 kernels,
   six groups). It is not a CUDA-route ceiling and cannot govern G-B3-W.
2. B2 proves memory-planner chaining on the synthetic probe, not on the real
   CUDA decode DAG. The real-decode planner root cause is still UNPROVEN.
3. A post-planner census alone cannot distinguish semantic RAW dependence from
   planner-added WAR/WAW alias protection. B3 needs aligned logical and
   physical DAG views.
4. `NO_MEMORY_PLANNER=1` changes memory topology and capacity. It is a
   planner-free feasibility extreme, not a clean overlap upper bound.
5. Historical `157.93 tok/s` is a reproducibility anchor, not the wall A/B
   denominator. G-B3-W must use a fresh same-session CUDA S=1 control.

The review also rejects the broad phrases "ANY graph", "capture is redundant
on this driver", and "the chain is caused by the planner" as route-level
findings. The evidence supports narrower probe-level statements only.

## 2. Corrected evidence statements

The following statements replace sections 1, 2.1-2.3, and 3.1-3.2 of the
pre-review brief where they conflict.

### 2.1 Programmatic co-scheduling

OBSERVED: on driver 595.84, the CUDA graph scheduler co-schedules independent
nodes in the two tested tinygrad-shaped programmatic probe families:

- CH=2, 17 nodes, 2-4.5 us elementwise kernels: about 18.6% median overlap;
- CH=6, 49 nodes split into two graphs: about 51-59% / 24-32% overlap.

UNPROVEN: that all programmatic graphs, a mixed decode-shaped graph, or the
real 1000+-node CUDA decode graph receive the same scheduler treatment.

### 2.2 Capture redundancy

OBSERVED: explicit multi-stream capture adds no overlap to the two tested
synthetic probe families on this driver and is slightly worse in one CH=6 arm.

UNPROVEN: that capture is redundant for the real mixed CUDA decode graph. It
remains a diagnostic arm only and cannot be promoted without a same-DAG wall
win over the S=1 programmatic control.

### 2.3 Decode serialization and planner attribution

OBSERVED:

- the B0 CUDA decode replay is serialized and carries about 230 us of
  inter-launch gaps (`span 5363.8 us`, `node-sum 5134.2 us`);
- the default memory planner turns the synthetic B2 independent-chain probe
  into a strict physical-buffer dependency chain;
- disabling planning and pinning the probe intermediates restores the intended
  probe DAG, which the programmatic CUDA graph overlaps.

INFERRED, not yet earned as a real-decode finding: the CUDA decode graph is
serialized because `memory_plan_rewrite` destroys its logical independence.

Correct root-cause statement:

> B0 serialization is consistent with a chain-shaped frozen CUDA DAG. The B2
> probe proves planner aliasing is one mechanism capable of producing that
> shape. B3 must determine whether it is the mechanism on real CUDA decode.

## 3. Route arithmetic and the missing strategic gate

The current anchors are:

| route | d512 tok/s | ms/token |
| --- | ---: | ---: |
| llama.cpp | 246.32 | 4.0598 |
| native NV | 177.72 | 5.6268 |
| CUDA B0.2 | 157.93 | 6.3320 |

The native-NV E2 intra-group ceiling cannot be transferred to CUDA. Even as a
counterfactual illustration, subtracting all `0.6088 ms` from CUDA B0.2 gives
about `5.7232 ms/token = 174.73 tok/s`: slightly below native NV and still
about 29% below llama.

Therefore B3 has three distinct success questions and must not collapse them:

1. **Attribution:** did planning remove valuable logical CUDA-DAG independence?
2. **Mechanism/wall:** can preserving that independence improve CUDA wall time?
3. **Route value:** does the improvement repay the CUDA route tax and then
   materially reduce the native-NV-to-llama gap?

Parity remains a separate depth-qualified result. An overlap mechanism PASS is
not a parity result and not sufficient for route promotion.

## 4. Revised B3 hypothesis and decision tree

```python
logical = dependency_dag(cuda_linear_before_memory_plan)
physical = dependency_dag(cuda_linear_after_memory_plan)
delta = classify_edges(physical - logical)  # RAW/WAR/WAW + buffer/range cause

if not logical.has_duration_weighted_parallelism():
    verdict = "OVERLAP_LEVER_CLOSED: semantic CUDA DAG is chain-dominant"
elif not delta.removes_material_parallelism():
    verdict = "PLANNER_NOT_ROOT_CAUSE: inspect graph grouping/resources/driver"
else:
    candidate = minimum_memory_unalias(delta.critical_path_edges)
    measure_same_session(default_s1, candidate_s1)
    if not candidate.correct: verdict = "CLOSED_NUMERICS"
    elif candidate.wall_gain < 5_percent: verdict = "MECHANISM_NOT_WALL_POSITIVE"
    elif candidate.wall < native_nv_wall: verdict = "CUDA_TAX_NOT_REPAID"
    else: verdict = "ROUTE_VALUE_CANDIDATE"  # still not parity/promotion
```

`has_duration_weighted_parallelism` is not an edge-count threshold. It is
reported using critical-path microseconds, ready width over time, compatible
node classes, and deterministic 2-/3-resource schedules.

## 5. B3 phase plan, amended

### Phase B3.0 - route identity and capture preflight

Before interpreting a DAG:

1. Pin the exact CUDA route, commit, driver, model hash, depth, quant route,
   graph-group sizes, kernel count, and token hash.
2. Require the CUDA route to reproduce its own fixed-depth result 3/3.
3. Label every artifact `CUDA-route`; none is an NV performance authority.
4. If call count, call identity, graph grouping, or correctness differs between
   the planner arms, stop before edge-delta attribution.

### Phase B3.1 - aligned logical/physical CUDA DAG census

Capture aligned views of the same d512 CUDA token:

- logical calls and resource dependencies before arena reuse;
- physical calls and dependencies after `memory_plan_rewrite`;
- graph-group assignment after graph splitting;
- call identity sufficient to align the views;
- edge kind (`RAW`, `WAR`, `WAW`) and exact logical/physical buffer range;
- planner alias origin: arena, offset, byte interval, and logical buffers that
  share it;
- UNKNOWN dependency count, which must be zero for a decisive verdict.

Implementation may use a single dual-snapshot capture or two controlled
default/planner-free captures. If two arms are used, accepted attribution
requires identical ordered call signatures and graph grouping before comparing
edges. A mismatch is a confounder, not evidence.

The real decode capture requires one lock-held GPU construction/run. DAG
analysis, edge comparison, scheduling simulation, and report generation are
then CPU-only. The existing `--capture` path must not be described as CPU-only.

Required metrics for each DAG:

- nodes and RAW/WAR/WAW edges;
- serialized node-sum and duration-weighted critical path;
- critical-path saving in microseconds and percent;
- ready width over time, not only maximum width;
- deterministic two- and three-resource schedule spans;
- per-class concurrent pairs and their combined bandwidth/compute character;
- graph-group-local and cross-group edges;
- arena bytes, planner-free bytes, peak allocated bytes, and context fit.

The original `>=90% strict-chain edges` rule is withdrawn. Edge count alone is
not a performance metric.

### Gate G-B3-D - DAG attribution

PASS requires all of:

1. exact route/call alignment and zero UNKNOWN dependencies;
2. a published CUDA-specific logical and physical DAG summary;
3. every material planner-added edge attributed to an arena/range reuse;
4. a CUDA-specific legal/no-contention ceiling stated as a hypothesis, not a
   wall promise;
5. the ceiling reported in microseconds and compared with both the CUDA route
   tax and the remaining native-NV-to-llama gap.

If the logical DAG is duration-weighted chain-dominant, close planner work and
route to kernel count/fusion/bandwidth/boundary analysis. If the physical DAG
retains material branches but S=1 does not overlap them, planner work is also
not the next lever; test scheduler/resource compatibility and the capture arm.

### Phase B3.2 - correctness census

Before wall ranking, settle the 1021-vs-948 chain difference:

- per-class checks for flash, q4k, q6k, norms, residual/KV/scatter, and vocab;
- end-to-end token/logit pins at fixed depths;
- deterministic 3/3 CUDA pins;
- explicit classification of legitimate lowering difference vs numeric bug.

### Gate G-B3-C - correctness

PASS requires the CUDA route and every measured planner candidate to satisfy
the fixed-depth pins and per-class checks. Failure closes wall interpretation.

### Phase B3.3 - planner counterfactuals

Rank planner-added edges by duration-weighted critical-path cost. Compute the
smallest extra-memory set that preserves the highest-value logical branches.

Arms:

1. default planner, S=1;
2. selective held-buffer/unalias candidate, S=1;
3. `NO_MEMORY_PLANNER=1`, S=1, diagnostic only and only if memory-safe;
4. explicit capture S=2/3, diagnostic only if the physical DAG retains
   material independence that S=1 does not realize.

`NO_MEMORY_PLANNER=1` is labeled `PLANNER_FREE_FEASIBILITY`, never `UPPER_BOUND`.
Every arm reports peak bytes, graph grouping, kernel identity, and context fit.
An arm that changes the call chain or cannot fit the target depth is not a clean
planner comparison.

### Phase B3.4 - same-session wall A/B

Run d512 first. Run d2048/d4096 only after d512 passes correctness and the
mechanism wall gate. In one lock-held session measure:

- fresh CUDA default-planner S=1 control;
- gated candidate arm(s);
- native NV control;
- llama control;
- CUPTI per-node span/node-sum for the CUDA arms;
- per-node duration expansion under overlap and available DRAM counters.

Historical 157.93/159.72/177.72/246.32 rows are anchors only. The same-session
rows are the ranking authority.

### Gate G-B3-M - mechanism wall conversion

PASS requires, at d512:

- G-B3-D and G-B3-C already PASS;
- candidate median wall throughput at least 5% above the same-session CUDA
  default-planner S=1 control;
- 3/3 directionally consistent fixed-depth runs;
- no correctness regression;
- overlap gain reported alongside per-node duration inflation and memory cost.

`MECHANISM_NOT_WALL_POSITIVE` is an honest closure classification, not a PASS.

### Gate G-B3-R - route value

PASS requires the candidate to match or exceed the same-session native NV
route while preserving correctness and target-depth memory fit. A candidate
that improves CUDA but remains below NV is `CUDA_TAX_NOT_REPAID`; it may remain
an analysis artifact but cannot justify route promotion.

### Gate G-B3-P - parity

Unchanged in principle: only a same-session candidate/llama ratio `>=1.00`
qualifies that measured depth. No composition, extrapolation, or endpoint claim.
G-B3-P does not authorize promotion by itself.

## 6. Corrected assumption ledger

| # | original assumption | amended verdict | settlement |
| --- | --- | --- | --- |
| 1 | real decode is planner-chained | UNPROVEN | aligned logical/physical CUDA DAG census |
| 2 | few-us kernels are favorable | PARTLY OBSERVED at probe level | real CUDA DAG trace + wall conversion |
| 3 | capture is unnecessary | UNPROVEN on decode | same-DAG S=1 vs capture diagnostic if needed |
| 4 | block granularity explains B1/B2 | UNPROVEN, non-load-bearing | controlled factorial only if still decision-relevant |
| 5 | decode is bandwidth-bound, GEMV ~50% peak | TOO BROAD / INFERRED | per-class counters and overlap duration inflation |
| 6 | 1021-vs-948 is legitimate | UNKNOWN, blocking | G-B3-C numerics census |
| 7 | overlap converts to wall | UNKNOWN | G-B3-M same-session A/B |
| 8 | 157.93 is the G-B3-W baseline | FALSE as ranking denominator | fresh same-session CUDA S=1 control |

## 7. Cheapest decisive experiments, final order

1. **Aligned logical/physical CUDA DAG capture and CPU analysis.** One
   lock-held construction/capture if required; no CUPTI needed for the initial
   dependency verdict. Belief flip: planner-added critical-path edges with
   exact alias provenance open selective planner work; their absence closes it.
2. **Offline selective-unalias simulation.** Rank the delta edges, calculate
   minimum extra bytes, and predict the CUDA-specific critical-path change.
   Belief flip: no useful span at acceptable memory closes planner work before
   another wall run.
3. **Same-session default-vs-selective CUDA wall/CUPTI A/B.** CUPTI is ground
   truth for realized scheduling, not for dependency existence. Add the
   planner-free and capture arms only as gated diagnostics.

## 8. Review-question answers adopted as authority

1. Planner chaining is consistent with the evidence but is not the only
   explanation. Alternatives are semantic RAW chains, conservative ranges,
   graph splitting, resource contention, driver heuristics, or route-specific
   lowering.
2. The census is cheapest only when it is an aligned logical/physical census.
   A post-planner edge count plus CUPTI trace is not decisive.
3. If the logical CUDA DAG is chain-dominant, overlap is the wrong immediate
   lever. Return to kernel count, fusion, GEMV achieved bandwidth, inter-launch
   gaps, and CUDA-route boundary tax.
4. The capture-redundant verdict is over-fit to synthetic elementwise probes.
   It is retained only as a probe finding.
5. Evidence corrections are section 2 above. CUPTI clustering must publish the
   graph/replay attribution rule and sensitivity to the 20 us cluster threshold.

## 9. HARD STOPs and protected files

- No declaration of hardware no-concurrency.
- No route flip or promotion to `dev`/`exp`/`master`.
- No native NV route, HCQGraph, or `ops_nv.py` change under this amendment.
- No planner/runtime candidate before G-B3-D identifies material attributed
  planner-added edges; that candidate requires its own closed-default scope.
- No CUDA wall ranking before G-B3-C.
- No reuse of the native-NV 608.8 us ceiling as CUDA authority.
- No presentation of a planner-free run as an upper bound.
- No touching user files named in the pre-review scope.

## 10. Amended one-line job

Determine whether memory planning removes valuable logical independence from
the real CUDA decode DAG, find the minimum-memory way to preserve it if so, and
prove at wall whether the recovery first repays the CUDA route tax before making
any parity or promotion claim.
