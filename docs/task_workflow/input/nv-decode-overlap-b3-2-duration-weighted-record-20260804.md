# NV decode overlap - Route B3.2 duration-weighted CP record (G-B3-D)

Date: 2026-08-04

Authority: B3 exhaustive execution scope section 9 (Phase B3.2, G-B3-D gate).
Status: CUPTI/nsys node durations attached to the anchored d512 CUDA decode
DAG; duration-weighted logical vs physical critical-path delta measured;
G-B3-D verdict: **NOT_MECHANISM_SCALE** (planner delta 38.6 us = 0.609% of
the 6.3319 ms/token CUDA wall anchor, far below the 5% bar; no route-tax
reach).

## 1. Findings (findings first)

- **Node durations are OBSERVED**: a fresh, lock-held nsys CUPTI node trace of
  the same d512 decode route (11 generate steps) was attached to the anchored
  DAG. All 1021 nodes across all 6 groups matched (1021/1021, aligned 6/6),
  each group on 7 steady-state CUPTI replay clusters after dropping 2 warmup
  launches; signature verification passed for every position.
- **Duration-weighted planner delta is 38.6 us = 0.609% of CUDA wall**, i.e.
  the planner's arena-reuse dependency inflation adds no meaningful serialized
  time to the decode token. Logical merged CP 3942.9 us vs physical merged CP
  3981.5 us on the whole-token DAG (per-group sums: 3947.6 vs 3982.8 us).
- **G-B3-D verdict: NOT_MECHANISM_SCALE.** The duration-weighted,
  planner-attributed CP delta is far below the 5% threshold, so planner edge
  inflation is not the mechanism that explains the observed decode wall.
  The dense PLANNER_ALIAS edge set (354426 of 355772 physical edges) is
  duration-neutral at this route's node times.
- **Overlap savings are unchanged by planner edges**: both arms show ~22%
  serialized-vs-scheduled savings (logical 22.57%, physical 22.05%), and the
  physical 2-queue schedule spans the same 3981.5 us as its CP. Planner alias
  edges add no scheduling slack recovery; they also add no duration.
- **Secondary verdict field is PLANNER_NOT_ROOT_CAUSE** (the census-style
  layered label): 2-queue savings on both arms are >= 5% (22.57% / 22.05%),
  so scheduling overlap exists, but the planner-added CP delta is not scale
  (< 5% of wall). The planner is not the root cause of the observed wall.
- **Caveat**: the merged whole-token CP (3942.9 us logical) is below the
  observed CUPTI replay span (per-token node-sum 5109.6 us, replay span from
  the B0.3 census 5363.8 us), because the merged DAG interleaves the six
  groups as one schedule while the route replays them as six launches with
  inter-launch gaps. The G-B3-D comparison deliberately uses the same delta
  convention as the anchored report (merged CPs) against the historical
  CUDA wall anchor, so the verdict is comparable with B3.2's edge-level
  arithmetic and is conservative: inter-launch gaps can only add wall, not
  planner delta.

## 2. Method

### 2.1 CUPTI trace acquisition (live, lock-held)

nsys node tracing required running the harness target without `sudo` and with
`--resolve-symbols=false`; the default symbol-download step stalls the launch
on this host (the driver's `RmProfilingAdminOnly=1` also blocks profiling as
the `ubuntu` user, and `sudo` run here stalled the launcher). The proven
invocation:

```bash
flock -w 10 /tmp/gpu-bench.lock -c 'DEV=CUDA CUDA_GRAPH_STREAMS=1 PYTHONPATH=/home/ubuntu/tinygrad-arkey \
  /usr/local/bin/nsys profile --cuda-graph-trace=node --resolve-symbols=false \
  --force-overwrite=true --output=/tmp/b3_dur_trace.nsys-rep \
  .venv/bin/python extra/llm_research/decode/cuda_duration_attach.py \
  --live-harness --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --depth 512 --warmup-decode 3 --nmeas 8'
/usr/local/bin/nsys export --type=sqlite --output=/tmp/b3_dur_trace.sqlite /tmp/b3_dur_trace.nsys-rep
```

The harness mirrors the anchored capture's d512 decode (`Transformer.from_gguf`,
`generate([1]*512, chunk_size=32, temperature=0.0)`) and keeps generating so
the decode graph replays 9 times (3 warmup + 8 measured, 2 warmup launches
dropped for steady-state medians).

### 2.2 Attachment discipline

`cuda_duration_attach.py --attach` mirrors
`cuda_route_aligned_census.align_capture`: within each group, trace kernels
are aligned to DAG nodes positionally (ordered kernel identity = name +
occurrence inside the group), the trace signature (shortName, grid, block) is
verified stable across steady-state replays, and any mismatch fails closed
(group reported unaligned, its nodes left UNKNOWN, never zero-filled). All six
groups aligned; 0 UNKNOWN nodes; 1021/1021 calls matched.

Artifacts: anchored capture
`docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-capture-manifest-
20260804.json`; CUPTI sqlite
`docs/task_workflow/output/nv-decode-overlap-b3-2-cupti-20260804.sqlite`
(sha256 `9bd5e4c44042551e880bda2f6575029d55009176d60d2ec3273ee4113e551f75`)
plus its `.nsys-rep`; report
`docs/task_workflow/output/nv-decode-overlap-b3-2-duration-weighted-
20260804.json` (schema `tinygrad.route_b3.duration_weighted.v1`).

## 3. Duration-weighted results (whole token)

| quantity | logical | physical |
| --- | ---: | ---: |
| merged critical path | 3942.9 us | 3981.5 us |
| sum of per-group CP | 3947.6 us | 3982.8 us |
| serialized (node sum) | 5109.6 us | 5109.6 us |
| 2-queue schedule | 3952.6 us | 3981.5 us |
| 3-queue schedule | 3942.9 us | 3981.5 us |
| 2-queue savings | 22.57% | 22.05% |
| nodes / edges | 1021 / 1346 | 1021 / 355772 |

Planner delta (physical - logical): **+38.6 us** merged CP
(**+35.2 us** per-group sums) = **0.609%** of the 6.3319 ms/token CUDA wall
anchor (historical B0.2 anchor; the anchored capture carries no route row).
The captured trace is from the same commit/session as the attach run
(`route.commit = e0a21494`, driver 595.84, DEV=CUDA CUDA_GRAPH_STREAMS=1).

## 4. Per-group evidence rows

| group | size | aligned | graphId | steady replays | duration sum | logical CP | physical CP | logical edges | physical edges |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 32 | yes | 2 | 7 | 157.7 us | 118.9 us | 120.9 us | 39 | 316 |
| 1 | 64 | yes | 5 | 7 | 323.5 us | 239.2 us | 244.1 us | 82 | 1365 |
| 2 | 128 | yes | 8 | 7 | 560.1 us | 416.2 us | 418.3 us | 164 | 5493 |
| 3 | 256 | yes | 11 | 7 | 1115.5 us | 835.3 us | 841.8 us | 336 | 22533 |
| 4 | 512 | yes | 14 | 7 | 2316.4 us | 1722.6 us | 1742.4 us | 674 | 89807 |
| 5 | 29 | yes | 17 | 7 | 636.4 us | 615.4 us | 615.4 us | 33 | 270 |

Every group's physical CP exceeds its logical CP by < 20 us (2.0-19.8 us), and
physical node sums equal logical node sums (durations are shared between arms
by node identity), confirming the delta comes from dependency structure only.

## 5. G-B3-D scale classification (scope section 9.3 thresholds)

Wall anchor: 6.3319 ms/token (historical B0.2 anchor).

| threshold | value | reached? |
| --- | ---: | --- |
| < 5% of CUDA wall | 316.6 us | yes (38.6 us) |
| >= 5% of CUDA wall | 316.6 us | no |
| >= route tax | 705.1 us | no |
| >= route tax + NV gap | 2272.1 us | no |

Planner delta 38.6 us / 6.3319 ms = 0.609% -> **NOT_MECHANISM_SCALE**.
Layered label (census convention): **PLANNER_NOT_ROOT_CAUSE** because both
arms exceed the 5% 2-queue savings bar (scheduling overlap exists) while the
planner-attributed CP delta does not.

## 6. Limitations

- The wall anchor is the historical 6.3319 ms/token value, not a
  same-session measured row; the captured CUPTI trace does not itself carry a
  token wall (nsys node tracing with `--resolve-symbols=false` and no sudo
  was required to unblock tracing on this host, and a same-session
  measurement would need an additional harness arm).
- Durations are per-node medians over steady-state replays (7 per group);
  the merged-DAG CP interleaves groups as one schedule, so merged CP values
  are a planner-delta metric, not a replay-span prediction.
- The trace captures the same route at the current commit
  (`e0a21494`); node identity and graph structure were verified consistent
  with the anchored capture (all six groups aligned by name + stable sig), so
  cross-session drift is bounded to unchanged kernel signatures.
- `--live-harness` was run with `warmup_decode=3 nmeas=8` (11 generate
  steps); the first 2 replay clusters per graph are dropped as warmup, and
  CUPTI's replay capture coalesces per graphNodeId, giving 9 raw clusters ->
  7 steady-state per group (>= 3 required).

## 7. Hermetic coverage

```bash
PYTHONPATH=/home/ubuntu/tinygrad-arkey .venv/bin/python -m pytest test/unit/test_cuda_duration_attach.py -q
```

8 passed (7 hermetic: two-chains fixture logical CP 35 us vs physical CP
63 us / delta 28 us; scale-threshold classification; fail-closed mismatch;
aligned medians; plus the skipif-missing integration test now runs against
the real capture + fresh trace).
