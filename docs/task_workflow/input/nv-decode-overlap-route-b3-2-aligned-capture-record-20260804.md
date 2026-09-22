# NV decode overlap - Route B3.2 aligned logical/physical capture record

Date: 2026-08-04

Authority: B3 exhaustive execution scope section 9 (Phase B3.2, aligned live
capture). Status: aligned capture completed on the real d512 CUDA decode
route; G-B3-D duration-weighted verdict still needs CUPTI timing attachment
(node durations), which this record does not substitute.

## 1. Method

Single-process dual snapshot on the live route, lock-held (`/tmp/gpu-bench.lock`):

```bash
flock -w 10 /tmp/gpu-bench.lock -c 'DEV=CUDA CUDA_GRAPH_STREAMS=1 PYTHONPATH=/home/ubuntu/tinygrad-arkey \
  .venv/bin/python extra/llm_research/decode/route_b3_dag_attribution.py \
  --capture-cuda --depth 512 --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --out docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-capture-20260804.json'
```

The seam wraps `jit_lower` (input uops), `memory_plan_rewrite` (pre-planning
linear snapshot + placement manifest snapshot), `compile_linear` (compiled
linear), and `graph_split_rewrite` (group boundaries), then:

- logical arm: pre-planning linear, logical buffer ranges, one range-aware
  tracker, write positions overlaid from compiled PROGRAM metadata;
- physical arm: compiled linear, arena-base ranges (session-stable base ids),
  best-effort logical-buffer provenance via placement manifest containment
  plus buffer lifetime;
- attribution: SEMANTIC (pair exists logically) versus PLANNER_ALIAS
  (physical-only) versus UNKNOWN (unresolved), with arena/range/kind evidence.

## 2. Results (real d512 CUDA decode, 1021 kernels / 6 groups)

| quantity | value |
| --- | ---: |
| nodes | 1021 |
| logical (semantic) edges | 1346 |
| physical edges | 355772 |
| SEMANTIC edges | 1346 (638 RAW, 708 WAR; all logical edges preserved) |
| PLANNER_ALIAS edges | 354426 (90590 RAW, 134372 WAR, 129464 WAW) |
| UNKNOWN edges / nodes | 0 / 0 |
| alignment | PASS (ordered stable signatures identical) |
| group structure | 32 / 64 / 128 / 256 / 512 / 29 (matches pinned route) |

The planner's arena reuse multiplies decode dependencies about 264x: the
physical DAG is nearly complete (355772 of 520k possible pairs), i.e. almost
every pair of decode kernels is chained by some WAR/WAW/RAW over shared arena
bytes. This is decisive edge-level evidence that the memory planner is the
dominant source of DAG structure in the CUDA decode route.

## 3. G-B3-D status and caveats

PASSED components: G-B3-0; aligned ordered signatures; zero UNKNOWN in
critical-path regions; both DAG summaries published; material
physical-minus-logical edges attributed to planner arena ranges.

Pending components:

- duration-weighted critical path and scale classification (NOT_MECHANISM_SCALE
  / MECHANISM_SCALE_ONLY / ROUTE_TAX_SCALE / PARITY_SCALE_THEORETICAL): node
  durations are not yet attached. Next step is a CUPTI/nsys node trace of the
  same route session, attached to this DAG, then the planner-delta CP and the
  route-tax comparison.
- logical-buffer provenance on physical edges is best-effort: physical edges
  carry the session-stable arena base id; per-edge logical buffer ids were not
  fully resolved because the compiled-route bases are sub-arenas of the
  planner arena (allocation-layer split). B3.4 candidate construction will use
  the logical arm's buffer/lifetime ledger plus this physical DAG.

The verdict will be either `PLANNER_CANDIDATE` (if the duration-weighted
delta is >=5% of CUDA wall) or `PLANNER_EFFECT_NOT_SCALE` (if the dense
aliasing is duration-neutral). Edge evidence alone does not decide scale.

## 4. Tooling delta since B3.1

- `capture_aligned_dags` dual-snapshot seam and `--capture-cuda` CLI in
  `extra/llm_research/decode/route_b3_dag_attribution.py`;
- write-position derivation for pre-compile SINK ASTs via STORE-target
  unwrapping (INDEX/PARAM), overlaid with compiled PROGRAM metadata;
- physical base identity via allocated Buffer base (session-stable);
- CP-impact ranking capped at the critical-path chain (or 256 by bytes) so
  dense DAGs stay tractable; impact computed exactly per candidate edge;
- 13 hermetic tests still green.

Anchored artifact: `docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-
capture-20260804.json`.
