# NV Route B re-measure at HEAD - FLAT superseded (2026-08-16)

Date: 2026-08-16
Status: measurement record. The 08-15 "Route B is FLAT" verdict is **superseded at
HEAD**: the multi-stream capture-deps fix `86d653651` restored the width-4 DAG, and
`CUDA_GRAPH_STREAMS=4` now measures **+4.8% wall** (187.9 vs 179.1 tok/s) with tokens
bitwise identical. Branch `nvidia-bringup-20260731`, HEAD `3042a2a74`.

## 1. Why re-measure

The Path A/B substrate doc (`nv-path-ab-substrate-status-head-20260816.md`) still
cited the 08-15 FLAT record, but that record (c74567a24) predates the capture-deps
fix `86d653651` (fed arena bases into the dependency tracker, collapsing the planned
buffers to one shared range and degenerating the runtime DAG to a width-1 chain).
The substrate doc's own evidence for Route B was therefore stale.

## 2. Fresh measurement (DEV=CUDA, fresh process per arm, flocked)

`scratchpad/nv_route_b_wall_probe.py`, Qwen3-8B-Q4_K_M, depth 512, settle 4,
32-48 measured tokens per arm, token sha `ddf344135e...` across all arms.

| streams | median ms/token | tok/s | delta vs 1 |
| --- | ---: | ---: | ---: |
| 1 | 5.584 | 179.1 | - |
| 2 | 5.457 | 183.3 | +2.3% |
| 3 | 5.330 | 187.6 | +4.8% |
| 4 | 5.322 | 187.9 | +4.8% |

48-token bracket (1,4,1): control 5.583 ms/token, 4-stream 5.334 ms/token,
-248.7 us/token (-4.46%). Tokens bitwise identical across every arm.

Evidence: `docs/task_workflow/evidence/nv-route-b-head-cuda-streams-20260816.json`

## 3. Interpretation

- Width-4 q/k/v concurrency was NOT bandwidth-bound (the 08-15 reasoning). It was
  a capture-dependency artifact: the old lowerer never saw the sub-buffer offsets,
  so every write waited on the previous one.
- The fix `86d653651` (CUDA capture path) feeds actual sub-buffers; its own A/B
  reported the same +4.8% (179.2 -> 187.7 tok/s), independently reproduced here.
- This is CUDA-graph-capture specific. The NV HCQ production path already passes
  sub-buffers and does not use CUDA multi-stream capture, so the +4.8% does not
  directly transfer to the NV route. Route A (native multi-compute) remains
  driver-blocked and PDL remains economics-negative.

## 4. Disposition

Route B is the first measured wall-positive overlap at HEAD, but it is a CUDA-capture
substrate correction, not an NV wall win. The Path B ledger row is updated; the
"no overlap converts to wall" conclusion is retracted for Route B and kept only for
the NV production path.
