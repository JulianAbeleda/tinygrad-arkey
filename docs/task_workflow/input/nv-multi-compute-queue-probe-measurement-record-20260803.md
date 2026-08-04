# NV multi-compute-queue probe measurement record

Date: 2026-08-03 (measured same day, one sequential flocked session, same RTX
5090 box; no concurrent GPU work)
Status: measurement record for the P0 gate in
`nv-multi-compute-queue-execution-scope-20260803.md`. Runs the device-level
probe protocol E1-E5, anchors the numbers, and records the verdict. The scope
doc's D2-D4 implementation work stays GATED: the probe did not pass, so no
HCQGraph/decoder change is authorized by this record.
Branch: tinygrad `nvidia-bringup-20260731`, worktree at the scope revision.
All numbers below carry evidence class OBSERVED unless marked INFERRED.

## 1. Question

Does GB202 / driver 595.84 co-schedule independent kernels submitted to two or
three native compute GPFIFOs? That is the load-bearing hardware question for
lever 1 (decode overlap) on the primitive route: multiple native compute
channels, ready DAG nodes scheduled across them, memory-semaphore
synchronization (the existing `HCQSignal` primitive).

PASS criterion (from the scope doc): E1 numeric-correct AND at least one of
E3/E4/E5 shows span < node-sum by >= 5%. FAIL: E1 correct but zero overlap on
both the memory-bound (elementwise) and compute-bound (matmul) workloads.

## 2. Method

`extra/llm_research/decode/nv_multi_queue_probe.py` creates extra compute
GPFIFOs on the live `NVDevice` (closed-default slice: persisted
`vaspace`/`ctxshare`, `_new_gpu_fifo(debugger=, engine_type=)`), lowers kernels
from Tensor expressions, executes them on per-GPFIFO `ProbeComputeQueue`
instances, and times with HCQ timestamp signals. Timestamp/waits/join
semantics mirror HCQGraph: per-queue monotonic signals for ordering, one
device-timeline release per batch after the first queue waits on every other
queue's last target.

Three probe-side defects were found and fixed during bring-up; they invalidate
no result below (the final numbers are from the fixed probe, 3/3 clean
reproductions):

- `HWQueue.submit` does not clear `_q`, so re-submitting a queue re-sends the
  whole accumulated command list. Stale ring entries re-execute old kernels
  and, after their signal slots were freed and reused by later experiments,
  deadlock the channel or regress the shared device timeline. Fixed by
  resetting `_q` after every submit in the probe's queue class.
- Submitting a queue with an empty command list writes a zero-length GPFIFO
  entry that stalls the frontend. Fixed by submitting each distinct queue
  exactly once per batch.
- Releasing the device timeline from every channel directly is racy: releases
  land in arbitrary cross-channel order, the last writer wins, and the value
  regresses below the last issued target, hanging `dev.synchronize()`. Fixed
  by the per-queue-signal + single-join-release design above.

## 3. Results (full size: n=33554432 elementwise, matmul 2048)

Canonical run, engines `0,0,0` (three channels, all on the default GR engine):

| exp | check | span us | node-sum us | overlap | result |
| --- | --- | ---: | ---: | ---: | --- |
| E1 | cross-GPFIFO semaphore dep, numeric | - | - | - | PASS (max err 0.00e+00) |
| E2 | serial calibration (one queue) | 478.8 | 477.2 | -0.3% | PASS (0% expected) |
| E3 | 2-queue elementwise overlap | 498.2 | 498.0 | -0.1% | FAIL |
| E4 | 3-queue elementwise overlap | 795.2 | 794.0 | -0.2% | FAIL |
| E5 | 2-queue matmul overlap + numeric | 5274.5 | 5273.8 | -0.0% | FAIL (numeric_ok=True) |

Artifact: `docs/five-lever-test-20260803-multiqueue-probe.json` (verdict FAIL).
E1 proves the semaphore primitive works across GPFIFOs with exact numerics;
E3-E5 prove the channels execute but fully serialize. The compute-bound
matmul flavor serializing as hard as the DRAM-bound elementwise flavor rules
out DRAM contention as the explanation.

Partial-SM kernels (grid-div 2/4/8, so each kernel leaves SM capacity free)
also serialize exactly (span == node-sum at every divisor), ruling out SM
saturation as the explanation. This is channel-level serialization.

## 4. Engine-type sweep

Extra channels were attempted on every GR engine type (`NV2080_ENGINE_TYPE_*`):

| engineType | meaning | result |
| --- | --- | --- |
| 0 | default (NULL/GR) | allocates; zero overlap |
| 1 | GR0 / GRAPHICS | allocates; zero overlap |
| 2-8 | GR1-GR7 | RM rejects: `NV_ERR_INVALID_ARGUMENT` |

GR1-GR7 are rejected by the RM on this consumer part, so no second physical
compute engine is addressable through native channel allocation. engineType 0
and 1 are the same physical engine.

## 5. Separate context shares (exploratory, INFERRED setup)

With `--separate-ctxshare` each extra channel gets its own
`FERMI_CONTEXT_SHARE_A` on the same vaspace, approximating CUDA's
per-context channel path. The extra channels do not execute in this slice
(E1 hang: join target 12 never released; value stuck at 11), even after
re-issuing the group-level GPFIFO schedule. The channel setup for
independently-scheduled contexts needs work beyond this probe's scope
(per-channel bind/schedule on the new ctxshare); this variant records a setup
gap, not a hardware verdict.

## 6. Verdict and consequence

**FAIL.** E1 is correct, but E3/E4/E5 show zero overlap at full scale, with
partial-SM grids, and across all allocatable engine types. GB202 (RTX 5090) /
driver 595.84 does not co-schedule independent kernels from multiple native
compute GPFIFOs through the path this fork uses: the RM exposes one GR engine
and serializes channels on it.

Consequence for the scope doc:

- D2 (HCQGraph N compute queues), D3 (NV N compute GPFIFOs), and D4 (decode
  A/B) remain NOT authorized; the gate stays closed.
- The 0.0% decode replay overlap measured earlier
  (`decode-replay-overlap-measurement-record-20260803.md`) is consistent with
  this hardware behavior, not an artifact of the graph's single-queue
  structure: even with N channels, this part would serialize the kernels.
- The primitive multi-queue route for lever 1 is blocked on this host unless
  RM-level scheduling control (channel preemption / runlist interleaving on
  the single GR engine) is found in the native path. That is outside the
  current slice's authorized surface and would need its own probe.
- `DEV=CUDA` remains the rejected vendor bypass per the scope doc; the route
  verdict stands: option 1 was the right primitive to test, and it measured
  negative on this hardware.
