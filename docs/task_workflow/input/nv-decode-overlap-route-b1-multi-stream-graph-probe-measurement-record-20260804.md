# NV decode overlap - Route B B1 multi-stream graph probe measurement record

Date: 2026-08-04 (measured in three sequential flocked GPU sessions, same
RTX 5090 box, driver 595.84, CUDA 13.2 toolkit, sm_120; no concurrent GPU
work)
Status: measurement record for B1 of
`nv-decode-overlap-route-b-implementation-scope-20260804.md`. Branch:
tinygrad `nvidia-bringup-20260731` at `ea4fbd439` plus the new probe source
(uncommitted at record time). All numbers OBSERVED unless marked INFERRED.

## 1. Question

On this exact driver, does a CUDA graph captured across 2-3 non-blocking
streams (llama's event fork/join mechanism) replay with span < node-sum
(co-scheduling of independent graph nodes), while the same kernels in a
single-stream graph replay serialized? B0 established that our decode graph
replays serialized (-4.5%) and that llama's graph replays 22.4% below
node-sum with 762 nodes per launch. B1 isolates the mechanism with
standalone probes.

## 2. Protocol

New probe `extra/llm_research/microbench/cuda_graph_stream_overlap_probe.cu`
(`nvcc -O3 -arch=sm_120`, compiled to `/tmp/b1_probe_build/`, binary
untracked) with three arms:

- Arm A (capture): `--kernels`=8 independent elementwise kernels per stream
  (n configurable) captured into one graph per run via
  `cudaStreamBeginCapture` (ThreadLocal) on a normal capture stream, fork
  events to `cudaStreamNonBlocking` secondaries, kernels on each stream,
  join events back, `cuGraphInstantiate`, then `--reps` replays of the same
  graph exec. Streams 1 (control), 2, 3.
- Arm B (control): 16 kernels in a PROGRAMMATIC graph (`cuGraphAddKernelNode`,
  chained within each 8-kernel half, no edges between halves).
- Arm C (capture): matmul 2048 on 2 streams, same capture method.

Per-kernel timing uses CUDA events recorded inside the capture (graph event
nodes); per-replay span uses wall events around each `cuGraphLaunch`.
Numeric checks are host-side sampled (tolerance as E3) plus an FNV-1a 64-bit
hash over sampled expected/actual pairs.

Ground truth: every arm was also run under `nsys profile
--cuda-graph-trace=node`; per-kernel rows from the SQLite export
(`CUPTI_ACTIVITY_KIND_KERNEL` with a graphId) are the authoritative in-graph
timings in this record. The llama baseline re-analysis reuses
`/tmp/e1_arm0_trace.sqlite` from the E1 session (same box, same driver).

## 3. Measurement-method findings (OBSERVED, load-bearing)

1. `cudaEventElapsedTime` on event nodes created inside a stream CAPTURE
   fails on this driver with "invalid argument". The probe falls back to a
   native reference run for node-sum. That native node-sum is NOT the
   in-graph node-sum: natively the kernels run two-at-a-time contended
   (338 us each at n=2^25), while in the graph they run ~180 us each. The
   probe therefore marks capture-arm overlap as `overlap_valid: false`; the
   first probe run's "47.9%" (arm A 2-stream) and "65.0%" (3-stream) mixed
   native node-sum with graph span and are INVALID. CUPTI rows are used
   instead.
2. Arm B's programmatic event nodes do not fail, but their durations are
   inflated ~1.4x versus CUPTI (255 us vs 180 us at n=2^25; 136 us vs 92 us
   at n=2^20). Its "31.1%" overlap is also invalid. Cause unknown
   (INFERRED: event nodes measure node-to-node intervals including
   scheduling gaps, not pure kernel durations).
3. CUPTI per-kernel start/end pairs are consistent across replays and match
   the probe's wall-event spans, so the graph-replay numbers below are
   trusted.

## 4. Results (CUPTI in-graph, per replay)

### 4.1 Arm A - capture, 2 streams, elementwise, by kernel size

| n (floats) | kernel dur (median) | rep0 | rep1 | rep2 | class |
| --- | ---: | ---: | ---: | ---: | --- |
| 2^16 | 0.8 us | -11.9% | 26.5% | 29.3% | co-schedules |
| 2^18 | 1.2 us | 10.5% | 31.2% | 32.3% | co-schedules |
| 2^20 | 3.0 us | 16.6% | 25.6% | 25.3% | co-schedules |
| 2^25 | 180 us | 4.8% | 4.0% | 4.5% | pipeline-tail only |

At n=2^20 rep1: span 36.0 us vs node-sum 48.3 us (16 kernels). Numerics ok
(max err 2e-06, 32 samples, hash `0x477bb4403a2ad13a`). First replay is cold
(scheduling state); steady-state replays show the full effect.

### 4.2 Arm A - capture, 3 streams, n=2^16

rep0 7.5%, rep1 51.7%, rep2 43.7% (span 10.3-19.7 us vs node-sum ~21 us,
24 kernels). Numerics ok (hash `0x945afc9533ffc8cd`, 3 samples at the
initial 100000-stride; the re-anchored 3-stream run at n=2^20 carries hash
`0x9ac60f9fb643ac80`, 48 samples, max err 3e-06).

### 4.3 Arm B - programmatic control (16 nodes, two independent halves)

| n | kernel dur | rep0 | rep1 | rep2 |
| --- | ---: | ---: | ---: | ---: |
| 2^20 | 5.5 us | -21.6% | 5.3% | 4.8% |
| 2^25 | 180 us | 2.4% | 2.5% | 2.3% |

The programmatic graph co-schedules only weakly (~5%) at decode-sized
kernels. Capture-based structure carries the stream affinity the scheduler
uses; the programmatic path does not reproduce it.

### 4.4 Arm C - capture, matmul 2048, 2 streams (1890 us kernels)

rep0 3.7%, rep1 3.6%, rep2 3.6% (span ~29.1 ms vs node-sum ~30.3 ms).
Compute-heavy kernels behave like the 180 us elementwise class: pipeline-tail
only.

### 4.5 llama E1 trace re-analysis (same measurement method, no new GPU work)

`/tmp/e1_arm0_trace.sqlite`, launch 2 of 29: 762 kernels, median duration
3.1 us, span 3890.5 us vs node-sum 5013.1 us = 22.4%. 545 of 761 adjacent
kernel pairs overlap (negative gap, median 1.5 us, max 4.9 us). llama's
22.4% is real graph co-scheduling at decode-sized kernels, reproduced by our
capture arms (25-32% at 1-3 us kernels), and NOT reproduced by our current
decode graph (-4.5% at ~5 us kernels).

## 5. G-B1 verdict: PASS

Multi-stream captured graphs co-schedule independent nodes on driver 595.84.
The elementwise capture arm at n=2^20 (3 us kernels, decode-sized) overlaps
25.3% on warm replays with correct numerics, well above the >= 5% gate. The
belief-flip statement "the CUDA graph scheduler does not preserve stream
concurrency on this driver" is FALSE. The mechanism is size-dependent: a
roughly fixed per-transition pipeline window (a few us per node transition,
INFERRED) becomes a large overlap fraction on 1-6 us kernels and a ~4%
fraction on 180-1900 us kernels. Decode kernels are in the favorable regime.

Correction to the scope's arm-B expectation: the programmatic control does
NOT replay fully serialized; it co-schedules weakly (~5%). The interesting
difference is capture vs programmatic structure, not serial vs concurrent.

## 6. Consequence for B2 (INFERRED)

1. The capture-based multi-stream lowerer is the right mechanism; the
   programmatic path is not a substitute.
2. B0's -4.5% with ~5 us kernels means the current decode graph contains no
   exploitable independence for the scheduler (chain-like), not that the
   scheduler cannot co-schedule: the same kernel sizes overlap 25-32% when
   independent branches exist.
3. B2 must therefore assign nodes to streams from the real dependency DAG,
   and B3 must verify how much true independence the decode DAG has
   (pre-split full-token DAG capture, `full_token_dag_capture.py`, is the
   seam). If the frozen dependency DAG is chain-like, multi-stream capture
   will not help and the record must say so with the DAG-width evidence.
4. The E2 legal ceiling (608.8 us / 11.35%) remains the honest reference;
   the llama-style mechanism is now proven present on this driver, so the
   ceiling is no longer blocked by scheduler behavior.

## 7. Artifacts

- `extra/llm_research/microbench/cuda_graph_stream_overlap_probe.cu`
  (new source), `test/fixtures/cuda_graph_stream_overlap_probe_fixture.json`,
  `test/unit/test_cuda_graph_stream_overlap_probe.py` (4 hermetic tests)
- `/tmp/b1_{all,a2_n1m,a2_n256k,b_n256k,final_*}.json` (probe JSON;
  capture-arm overlap flagged invalid per section 3)
- `/tmp/b1_trace*.nsys-rep` + `/tmp/b1_trace*.sqlite` (CUPTI ground truth,
  session-scoped)
- reused `/tmp/e1_arm0_trace.sqlite` (llama baseline, session-scoped)
