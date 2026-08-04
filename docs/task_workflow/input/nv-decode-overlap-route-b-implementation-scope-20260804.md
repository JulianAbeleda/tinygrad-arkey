# NV decode overlap - Route B exhaustive implementation scope (DEV=CUDA)

Date: 2026-08-04
Status: active implementation scope, gated phase by phase. Authorizes the
Route B overlap mechanism probe (B1), the CUDAGraph multi-stream lowerer
(B2, closed default), and the decode-on-CUDA correctness re-pin + wall A/B
(B3). Does NOT authorize: any change to the NV decode route, promotion to
`dev`/`exp`/`master`, or a composed parity endpoint. Branch boundary:
tinygrad `nvidia-bringup-20260731` at the B0 viability commit
(`nv-decode-overlap-route-b-viability-record-20260804.md`).

## 0. Authority and trigger

Route A is closed (G1 = CONSTRUCTION_BLOCKED). B0 viability
(`nv-decode-overlap-route-b-viability-record-20260804.md`) measured:
decode runs end-to-end on DEV=CUDA; CUDA graphs replay per token; CUPTI
works; correctness pins differ (re-pin required); and the existing
single-stream CUDAGraph replays SERIALIZED (per-token span 5363.8 us vs
node-sum 5134.2 us, -4.5% overlap). E1 established that llama's 22.4%
overlap comes from graph nodes captured on multiple internal streams with
event edges. The load-bearing unknown is therefore: does a multi-stream
captured CUDA graph co-schedule independent nodes on THIS driver? B1
answers that; B2 builds the mechanism; B3 proves correctness and wall.

## 1. Measured baseline (B0, OBSERVED)

| quantity | value |
| --- | ---: |
| CUDA decode D wall | 157.93 tok/s (6.33 ms/token) vs NV route 177.72 |
| CUDA prefill W wall | 177.94 tok/s vs NV route 177.72 |
| kernels/token on CUDA | 1021 (6 graph launches: 32/64/128/256/512/29) |
| per-token node-sum / span | 5134.2 us / 5363.8 us (no overlap, -4.5%) |
| correctness pins | DIFFER from NV pins (first token 38835 vs 151936; sha 55f7a13b... vs 9d6b3787...) |
| E2 intra-group legal ceiling | 608.8 us / 11.35% no-contention |
| pre-split cross-group edges | UNKNOWN (tooling exists: `full_token_dag_capture.py`) |

## 2. B1 - multi-stream graph capture probe (device-level, runs first)

Question: on this exact driver, does a CUDA graph captured across 2-3
non-blocking streams (event-fork/join, llama's mechanism) replay with
span < node-sum, while the same kernels in one single-stream graph replay
serialized?

Protocol (one flocked GPU session; standalone CUDA probe, no tinygrad):

1. N=8 independent elementwise kernels (n=2^25) per stream, on 1/2/3
   non-blocking streams captured into one graph per arm:
   `cuStreamBeginCapture` on stream 0, record fork events, record kernels
   on each stream, join events back, `cuStreamEndCapture`,
   `cuGraphInstantiate`, launch, time with CUDA events. Numeric check vs
   host reference; hash record.
2. Arm B (control): same 16 kernels added to a PROGRAMMATIC single-stream
   graph (`cuGraphAddKernelNode`, no edges between the two independent
   halves) - expects serialized replay (B0's -4.5% class result).
3. Arm C: matmul 2048 flavor on 2 streams, same method.
4. Record per arm: span, node-sum, overlap fraction, numeric hash,
   graph node count, stream count.

Gate G-B1 (belief-flip): PASS = multi-stream capture overlap >= 5% on at
least one elementwise arm with correct numerics (mirrors E3's criterion
for streams). FAIL = all multi-stream arms ~0%: the CUDA graph scheduler
does not preserve stream concurrency on this driver, and Route B overlap
is closed (record states so; no hardware no-concurrency verdict - E3
streams still overlap outside graphs).

## 3. B2 - CUDAGraph multi-stream lowerer (closed default)

Gate: G-B1 passed. Scope: `tinygrad/runtime/graph/cuda.py` +
`tinygrad/runtime/ops_cuda.py` seams only; behavior at the default (one
stream) byte-identical to today's CUDAGraph.

Work items:

1. `CUDAGraph` gains a capture-based construction path behind
   `CUDA_GRAPH_STREAMS` (default 1): N non-blocking streams, node-to-
   stream assignment from the frozen range-aware dependency DAG
   (producer index deps, the same `_access_resources` edges; reuse the
   Phase 4 `full_token_dag_capture` edge semantics), fork events from the
   launch stream to N-1 worker streams, capture kernels per stream,
   join events back. Reuse the existing per-replay param update
   (`cuGraphExecKernelNodeSetParams`) unchanged.
2. Assignment policy v1: ready-set list schedule, longest remaining tail,
   then lowest node index; queue (stream) minimizing
   `max(stream_free, pred_end)`, then lowest stream index; cost proxy
   `max(1, KernelInfo.estimates.mem)` bytes (copy = copy bytes); no
   hardcoded timing tables.
3. Encode queue-local monotonic ordinals for event waits where the
   capture path needs explicit deps; decreasing releases on one stream
   rejected before submission.
4. One join: the launch stream waits all worker stream events before the
   graph end; per-graph replay launch is a single `cuGraphLaunch`.
5. Hermetic tests (CPU): DAG construction, stream assignment determinism,
   dependency preservation, N=1 byte-identical command/graph shape
   (programmatic path unchanged at default), and the join/event ordering.
   GPU tests (live): N=1 regression on the B0 decode harness (same wall +
   same pins as B0.2), N=2/3 on the B1 probe shape through the new
   lowerer.

Gate G-B2: N=1 decode run reproduces B0.2's wall (157.93 tok/s) and token
sha within measurement noise; N=2/3 device probe shows overlap >= 5% with
correct numerics through the new lowerer.

## 4. B3 - decode on CUDA: correctness re-pin + wall A/B

Gate: G-B2 passed. Scope: measurement + the named re-pin protocol; no
decode-route changes beyond the lowerer default flip decision recorded
here (flipping the decode route to DEV=CUDA is NOT authorized by this
document).

Work items:

1. Numerics-divergence investigation: census which kernel classes differ
   between NV route (948 kernels, fused promotions) and CUDA route
   (1021 kernels); determine whether the differing token stream is a
   legitimate different-kernel-chain result or a CUDA-specific numeric
   bug (per-class max-error checks vs CPU references on flash/q4k/q6k/
   norms). Record OBSERVED/INFERRED.
2. Re-pin protocol: new CUDA-route pins (first token, token sha, decode
   sha) from a fixed-depth protocol with the same harness/repetition
   rules; 3/3 reproducibility across sessions; the new pins are
   CUDA-route pins, never presented as the NV-route pins.
3. Wall A/B (same session, d512/d2048/d4096): CUDA route (with B2 lowerer
   at N=1 and N=2/3) vs NV route (177.72 tok/s d512 authority) vs
   same-session llama control (246.32 tok/s opt=0). Report per-class
   overlap, node-sum/span, and the bandwidth caveat (decode GEMVs ~50% of
   1792 GB/s, INFERRED).
4. Optional pre-split full-token DAG capture on the CUDA route (the
   `full_token_dag_capture.py --capture` seam) to publish the corrected
   cross-group schedule BEFORE any regrouping candidate.

Gates:

- G-B3-C (correctness): CUDA pins 3/3 under the new pins; per-class
  numeric checks pass their declared bounds.
- G-B3-W (wall value): median d512 wall improvement over the B0.2 CUDA
  baseline (157.93 tok/s) >= 5% with N=2/3, OR the record classifies the
  mechanism as implemented-but-not-wall-positive against the 608.8 us
  ceiling (no promotion either way).
- G-B3-P (parity direction): same-session row vs llama >= 1.00 at a depth
  PARITY-QUALIFIES that depth only (forward-scope rule); no composed
  endpoint.

## 5. Route selection rule

After B3, the record states which substrate (NV route vs CUDA route with
the B2 lowerer) is the decode execution path candidate. The decision to
flip the decode route is a separate promotion scope under the forward
authority (`nv-parity-and-beyond-forward-scope-20260803.md`); this scope
does not flip anything.

## 6. Bans and HARD STOPs

- HARD STOP: no declaring hardware no-concurrency; no composed parity
  endpoint; no route promotion from this document.
- No changes to the NV decode route, HCQGraph, or ops_nv.py beyond what is
  already committed.
- No user files (`docs/README.md`, `docs/beating-llama-*`,
  `docs/what-makes-a-token-fast-*`, `extra/llm_research/microbench/*`
  binaries, `scratchpad/t6_metal_admission_probe.py`).
- GPU sessions sequential, flocked; evidence classes OBSERVED/INFERRED;
  `git diff --check` clean; commits `[prefix]` on
  `nvidia-bringup-20260731` only; no `master`/`dev`/`exp`.
- The NV-route pins stay NV-route pins; the CUDA-route pins are separate.

## 7. Deliverables

| phase | artifacts |
| --- | --- |
| B1 | probe source + hermetic tests; measurement record + anchored JSON; G-B1 verdict |
| B2 | `cuda.py` multi-stream lowerer behind `CUDA_GRAPH_STREAMS`; hermetic tests; G-B2 record |
| B3 | numerics census; CUDA-route pin record; wall A/B record; G-B3 verdict; optional full-token DAG capture record |

## 8. One-line job

Prove multi-stream captured CUDA graphs co-schedule on this driver
(B1), build the capture-based multi-stream CUDAGraph lowerer (B2), and
re-pin + A/B the CUDA decode route against the NV route and llama (B3),
with the 608.8 us intra-group ceiling as the honest reference.
