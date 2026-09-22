# NV gap audit correction: the gap is overlap, not kernel arithmetic (2026-08-14)

Date: 2026-08-14. Target: RTX 5090, native `DEV=NV`, sm_120. This supersedes the
verdict in `nv-production-gap-repin-20260814.md`, which concluded "no single
buildable substrate to 240 left". That conclusion is wrong. The correct, larger
lever was sitting in the evidence and in an already-written implementation scope,
and it was mislabeled out of view.

## 1. Corrected like-for-like decomposition

Both columns below are the same measurement: nsys CUPTI kernel intervals for one
decode token. llama is one replay of graph 5; tinygrad is one replay of each of
the six sequential graphs (2/5/8/11/14/17). Sources:
`/tmp/llama_tg10_node_20260812.sqlite`, `/tmp/tg_node_20260813.sqlite`.

| metric | llama | tinygrad | delta |
| --- | ---: | ---: | ---: |
| kernels/token | 762 | 1021 | +259 |
| node sum (raw kernel work) | 4766 us | 5164 us | **+398 us** |
| interval union | 3830 us | 5164 us | +1334 us |
| overlap mass (node sum - union) | **936 us** | **0 us** | **-936 us** |
| serialized span | 3841 us | 5385 us | +1545 us |
| internal launch gaps | ~10 us | ~221 us | +211 us |
| max concurrent kernels | 4 | 1 | - |

The span delta adds up: 398 us of extra kernel work, 936 us of lost overlap, and
211 us of launch gaps. The dominant term is overlap, and tinygrad has none.

## 2. What this means in tok/s

Production wall is 193.1 tok/s (5178 us/token) vs llama 245.5 tok/s (4073 us).

| recoverable component | us | resulting tok/s |
| --- | ---: | ---: |
| baseline | 0 | 193.1 |
| llama-level overlap only (936 us) | -936 | ~236 |
| overlap + launch gaps (1147 us) | -1147 | ~248 (~= llama) |

The 398 us of raw kernel work is the hard per-shape tail that the M1/M2/M4/DP4A/
vocab/reduce-output work has been chasing. Even closing all of it is worth roughly
+15 tok/s. The overlap alone is worth roughly +43 tok/s, and it is currently zero.

## 3. Why the prior docs missed it

1. The tinygrad node ledger (`nv-tinygrad-d512-node-ledger-20260813.json`) records
   `overlap_mass_us: 0.0` next to llama's 946.4 us, but its own schema string labels
   overlap as "node sum minus all-kernel interval union; not recoverable wall
   savings." That label is backwards: llama's 936 us of overlap is exactly the
   recovered wall savings, and dismissing it steered every follow-on scope away from
   the largest lever.
2. The 08-14 re-pin compared tinygrad's DEBUG=2 kernel-sum (a single instrumented
   prime token, which can never observe overlap) against llama's nsys node-sum, then
   wrote "launch hiding exhausted (~18-33 us)". That conflates host launch overhead
   with the 936 us intra-graph overlap, which are different axes.
3. A full implementation scope for this exact work already exists:
   `nv-decode-overlap-implementation-scope-20260803.md`, plus the CUDA-mirror probe
   record `nv-decode-overlap-cuda-mirror-probe-record-20260805.md`. The session
   drifted from that substrate into per-kernel epilogue folds and never returned.

## 4. The actual blocker (already measured, not a mystery)

CUDA streams overlap on this exact hardware: the E3 records show 2-stream
elementwise 48.1%, 3-stream 65.1%, 2-stream matmul 48.4%, all with numerics clean.
So hardware concurrency is available.

The native route (Route A in the 08-03 scope, `HCQ_NUM_COMPUTE>1`) is
`CONSTRUCTION_BLOCKED`, not a no-overlap verdict:

- shared-context extra channels serialize (~0% overlap, all flavors);
- the corrected fresh-group/ctxshare topology is accepted by RM but the first
  kernel never executes (join stuck at signal 11 of expected 12);
- mirroring CUDA's own 8-channel construction (deferred group schedule, runqueue
  flag 0x10, ctx buffer registration, subctxId 63, 4-KiB notifier) still hangs;
- every trace-visible RM parameter was exhausted. The record classifies the
  remaining blocker as a driver-private native-channel activation gap.

Route B (DEV=CUDA + `CUDAGraph` with `CUDA_GRAPH_STREAMS>1`) has the overlap
substrate in-tree (`tinygrad/runtime/graph/cuda.py` `plan_multi_stream` /
`_capture_construct`). It was analyzed and deliberately not authorized in the
08-03 scope because this fork wants the native ioctl substrate. That is a
correctness/architecture decision, not an overlap-capability decision.

## 5. Conclusion

We are not clearing the gap because the token is serialized while llama overlaps
936 us of support kernels behind its anchor, and the native multi-compute-channel
construction that would fix it is blocked. The prior session optimized the smaller
398 us kernel-work term and wrote the larger term off as unrecoverable. The next
decision is strategic, not another epilogue: either crack the native channel
activation blocker, or validate Route B overlap end-to-end as a measurement
experiment to prove the 240 target is real before committing to more native work.

## Evidence

- raw traces: `/tmp/llama_tg10_node_20260812.sqlite`, `/tmp/tg_node_20260813.sqlite`
- ledgers: `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`,
  `docs/task_workflow/evidence/nv-tinygrad-d512-node-ledger-20260813.json`
- prior scope: `docs/task_workflow/input/nv-decode-overlap-implementation-scope-20260803.md`
- probe record: `docs/task_workflow/input/nv-decode-overlap-cuda-mirror-probe-record-20260805.md`
