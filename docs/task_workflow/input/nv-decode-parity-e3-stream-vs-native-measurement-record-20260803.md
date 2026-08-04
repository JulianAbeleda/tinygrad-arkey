# NV decode parity - E3 CUDA-stream vs native-channel measurement record

Date: 2026-08-03/04 (measured 2026-08-04, one flocked GPU session, same RTX
5090 box, no concurrent GPU work)
Status: measurement record for experiment E3 of
`nv-decode-parity-e1e3-measurement-scope-20260803.md`, authorized by
`nv-decode-parity-external-review-amendment-20260803.md` section 7.
Question: do two CUDA streams co-schedule independent kernels on this exact
device/driver while the native shared-context construction serializes?
Branch: tinygrad `nvidia-bringup-20260731` at `fed89a201`. All numbers
OBSERVED.

## 1. Protocol

CUDA leg: `extra/llm_research/microbench/cuda_stream_overlap_probe.cu` (new
[test] source; `nvcc -O3 -arch=sm_120`) launches N=8 independent kernels per
stream on 1/2/3 `cudaStreamNonBlocking` streams, elementwise (n=33554432)
and matmul (2048) flavors plus partial-SM (grid-div 4), timed with CUDA
events; span (max end - min start) vs node-sum, host-side sampled numeric
check.

Native leg: `extra/llm_research/decode/nv_multi_queue_probe.py` E1-E5
regression at full size (n=33554432, matmul 2048, engines `0,0,0`), same
span/node-sum criterion.

## 2. Results

### 2.1 CUDA streams (same device, same session)

| experiment | span us | node-sum us | overlap | numeric |
| --- | ---: | ---: | ---: | ---: |
| 1-stream calibration (elementwise) | 1398.4 | 1394.6 | -0.3% | ok |
| 2-stream elementwise | 2811.2 | 5416.6 | 48.1% | ok |
| 3-stream elementwise | 4213.6 | 12074.0 | 65.1% | ok |
| 2-stream elementwise partial-SM (grid-div 4) | 2826.3 | 5438.5 | 48.0% | ok |
| 2-stream matmul 2048 | 29165.0 | 56490.3 | 48.4% | ok |

### 2.2 Native probe regression (same device, same session)

| experiment | verdict | span us | node-sum us | overlap |
| --- | --- | ---: | ---: | ---: |
| E1 cross-GPFIFO semaphore dep | pass | - | - | numeric exact |
| E2 serial calibration | pass | 477.8 | 476.5 | -0.3% |
| E3 2-queue elementwise | FAIL | 497.3 | 497.0 | -0.1% |
| E4 3-queue elementwise | FAIL | 795.3 | 794.5 | -0.1% |
| E5 2-queue matmul | FAIL | 5283.5 | 5283.0 | ~0% (numeric ok) |

## 3. Verdict

The hardware co-schedules and the native construction does not. On this exact
RTX 5090 / driver 595.84, two CUDA streams overlap 48.1% (elementwise) and
48.4% (matmul), three streams 65.1%, with correct numerics; the native
shared-context GPFIFO construction shows ~0% at every flavor, reproducing the
original probe verdict. This proves a native RM construction/scheduling gap,
NOT a hardware no-concurrency limitation. The amendment's corrected blocker
statement is confirmed by measurement.

Per the scope's belief-flip criteria: "CUDA overlap with native serialization
proves a native RM construction/scheduling gap, not a hardware limitation" is
TRUE. The corrected native independent-context run (separate-ctxshare with
per-channel bind/schedule) remains unproven (setup gap recorded in the probe
measurement record); D2-D4 reopen under the E1E3 scope's gates.

Artifacts: `/tmp/e3_cuda_s{1,2,3}.json`, `/tmp/e3_cuda_s2_gd4.json`,
`/tmp/e3_cuda_s2_mm.json`, `/tmp/e3_native_regression.json` (session-scoped,
not committed).
