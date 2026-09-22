# NV decode parity - E2 dependency-DAG critical-path measurement record

Date: 2026-08-03/04 (captured and simulated 2026-08-04, one flocked GPU
session for the capture; simulation is CPU-only and hermetic)
Status: measurement record for experiment E2 of
`nv-decode-parity-e1e3-measurement-scope-20260803.md`, authorized by
`nv-decode-parity-external-review-amendment-20260803.md` section 7.
Question: how much of llama's 1.116 ms missing-overlap term is even legal on
tinygrad's actual decode dependency DAG, before any implementation?
Branch: tinygrad `nvidia-bringup-20260731` at `fed89a201`. Capture and
simulation numbers OBSERVED; schedule projections are OBSERVED arithmetic on
the captured DAG.

## 1. Protocol

1. Capture: `replay_overlap_probe.py --depth 512` under `PROFILE=1
   HCQ_GRAPH_PROFILE_JSON=/tmp/replay_overlap_graph.jsonl` produced the full
   per-node dependency graph of one measured d512 decode token: 948 nodes in
   5 graph groups (32/64/128/256/468), per-node `deps`, `start`, `end`,
   `duration`, `name` (exporter: `tinygrad/runtime/graph/hcq.py`,
   `graph_profile_payload`). Group summary artifact
   `/tmp/nv_decode_d512_dag_summary.json`: node-sum 5366.1 us, span 5366.0
   us, 0.0% overlap (matches the overlap record).
2. Simulation: `extra/llm_research/decode/dag_critical_path_sim.py` (new
   [test] script) computes, per group and merged across groups (no
   cross-line edges recorded): serialized node-sum span, unlimited-resource
   critical path (earliest start = max dep end), and deterministic 2- and
   3-queue list schedules (ready set, longest-remaining-tail priority,
   earliest-available queue, no preemption).

## 2. Results

### 2.1 Per group (us)

| group | n | serialized | critical path | 2-queue | 3-queue |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 32 | 179.6 | 153.6 | 153.6 | 153.6 |
| 2 | 64 | 350.9 | 288.3 | 288.3 | 288.3 |
| 3 | 128 | 666.9 | 592.3 | 592.3 | 592.3 |
| 4 | 256 | 1340.7 | 1183.3 | 1183.3 | 1183.3 |
| 5 | 468 | 2828.1 | 2539.9 | 2539.9 | 2539.9 |

### 2.2 Merged (all 948 nodes, no cross-group edges)

| schedule | span us | saving vs serialized | saving % |
| --- | ---: | ---: | ---: |
| serialized (today) | 5366.1 | - | - |
| unlimited critical path | 2539.9 | 2826.2 | 52.7% |
| 2-queue list | 3310.5 | 2055.6 | 38.3% |
| 3-queue list | 2786.3 | 2579.8 | 48.1% |

Node-class overlap observed in the schedules: GEMV-class, flash, rmsnorm,
residual, kv, and scatter nodes all appear in overlapping class pairs;
cross_group_edges=0.

## 3. Verdict

The tinygrad decode DAG carries real, scheduleable parallelism: a
deterministic 2-queue list schedule saves 2.06 ms (38%) of GPU span on the
captured dependency graph, and 3 queues save 2.58 ms (48%). Both exceed the
scope's reopen threshold (0.8-1.1 ms) by ~2x. Even the intra-group-only
critical paths sum to ~0.61 ms of legal saving (per-group rows), above the
0.4 ms downgrade threshold.

REOPEN: graph-level overlap is parity-scale on tinygrad's own DAG topology;
the blocker is execution (one compute GPFIFO serializes every node), not
topology. Caveat: merged-schedule savings assume cross-group independence
(no recorded cross-group edges) and constant per-node durations under
overlap (no bandwidth-sharing model); the E3 record independently confirms
the device co-schedules, and the E1 record shows llama realizes 22.4% in
practice.

Artifacts: `/tmp/replay_overlap_graph.jsonl` (capture, session-scoped),
`/tmp/nv_decode_d512_dag_sim.json` (simulation output),
`/tmp/nv_decode_d512_dag_summary.json` (group summary, session-scoped).
