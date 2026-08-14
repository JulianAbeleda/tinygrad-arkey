# NV overlap ceiling test: Route B decision basis (2026-08-14)

Date: 2026-08-14. Target: RTX 5090, native `DEV=NV`, sm_120. This is the
arithmetic gate the user asked for before choosing whether to pursue the CUDA
overlap path (Route B). No GPU session was burned; the answer is already pinned
in committed evidence.

## 1. The mechanism is proven, but the ceiling is the DAG

Route B depends on two separate facts:

1. Does CUDA multi-stream graph capture co-schedule on this driver? Yes. B1
   measured capture-arm co-scheduling at decode-sized kernels (25-32% on 1-3 us
   kernels), PASS. (`nv-decode-overlap-route-b1-multi-stream-graph-probe-measurement-record-20260804.md`)
2. How much overlap can OUR decode DAG actually absorb? This is the binding
   question, and it is not llama's number.

## 2. The decode DAG critical path (authoritative arithmetic)

`docs/task_workflow/evidence/nv-dag-duration-head-20260812.json` is the full
pre-split decode-token DAG with per-node durations attached: 596 nodes, 110,544
dependency edges, 0 unknown-dependency nodes.

| schedule | span us | saving us | saving % | tok/s (from 193.1 baseline) |
| --- | ---: | ---: | ---: | ---: |
| serialized (current) | 5493 | 0 | 0 | ~193 |
| 2 streams | 4869 | 624 | 11.37 | ~218 |
| 3 streams | 4842 | 651 | 11.85 | ~219 |
| unlimited (critical path) | 4842 | 651 | 11.85 | ~219 |

Going past 3 streams adds nothing: 3 streams already reaches the critical path.
The overlap classes are GEMV-with-GEMV (108 pairs) and small-kernel-with-small-
kernel, i.e. pipeline slack, not one giant anchor hiding everything.

## 3. Why llama overlaps 22-24% and we cap at ~11.9%

llama's 936 us of overlap hides rms_norm (307 us), quantize_q8_1 (482 us), rope
(127 us), flash (114 us), kv (74 us), and combine (120 us) behind a single
~3239 us mmq anchor. We have no such anchor: our 253 GEMV kernels average
~16 us and the dependencies chain them, so the support kernels can only hide in
the slack between GEMV steps.

The same data reframes the whole "substrate to 240" question. Even with infinite
streams, our critical path is 4842 us. llama's whole token span is 3835 us.
There is a ~1000 us gap in the critical path itself that no overlap mechanism
can cross:

| component | us |
| --- | ---: |
| overlap recoverable (serial -> critical path) | ~651 |
| critical-path gap (our 4842 us vs llama 3835 us) | ~1007 |

The ~1007 us is kernel work on the dependency chain plus launch gaps, the same
hard per-shape territory the M1/M2/M4/DP4A work has been measuring as NO-GO.

## 4. Decision basis

- Overlap (Route A native or Route B CUDA) is bounded by the DAG, not by the
  substrate. Both top out at ~193 -> ~219 tok/s.
- Reaching 240 additionally requires shortening the critical path itself
  (kernel work + chain structure), which is independent of the overlap route.
- Therefore Route B is not the lever to parity. Its only advantage over Route A
  is implementation speed (the CUDA multi-stream lowerer already exists in
  `tinygrad/runtime/graph/cuda.py`). The native Route A is still blocked on the
  channel-activation construction documented in the CUDA-mirror probe record.

Recommendation: treat overlap as a bounded ~+25 tok/s item and stop describing
it as the path to 240. If the team wants the cheap 219, Route B is the fastest
proven substrate. If the goal is parity, the work is the critical path itself,
and no overlap route changes that arithmetic.

## Evidence

- full decode DAG with durations: `docs/task_workflow/evidence/nv-dag-duration-head-20260812.json`
- B1 probe record: `docs/task_workflow/input/nv-decode-overlap-route-b1-multi-stream-graph-probe-measurement-record-20260804.md`
- prior correction: `docs/task_workflow/input/nv-gap-audit-correction-20260814.md`
