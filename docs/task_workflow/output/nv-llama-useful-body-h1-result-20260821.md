# NV llama useful-body H1 result 20260821

## Verdict

H1 is **supported** (measured). Most of llama's decode-graph overlap mass is
dependency wait plus launch shadow, not simultaneous useful memory traffic.

- Useful concurrency: **4.6-8.1%** of overlap mass.
- Wait/launch shadow: **91.9-95.4%** of overlap mass.
- Applied to the authority ledger, that is roughly **52-91 us** of useful
  concurrent execution and **1042-1081 us** of shadow per token replay, out of
  the authority overlap mass of 1133.255 us.

This is the missing observable required by the 20260820 split-phase review
(H1 minimum test: isolate consumer wait-exit from kernel start). The prior
review verdict of `unmeasured` is now superseded.

## What was measured

The instrumented llama build records `%globaltimer` timestamps in a device ring
at every programmatic-launch site: `wait_exit` immediately after
`cudaGridDependencySynchronize`, and `trigger` immediately after
`cudaTriggerProgrammaticLaunchCompletion`. Per replay, the parser joins one
ring dump to the replay's CUPTI kernel intervals:

1. Clock calibration uses trigger-at-start records; per-replay offset stdev
   was 119-153 ns.
2. Each wait-exit is assigned to the newest-started matching kernel whose
   interval contains it. Trigger records are excluded from the spin anchors.
3. Per kernel, `we_lo` (earliest block exit) and `we_hi` (latest block exit)
   bound the spin phase; both are clamped into the kernel interval.
4. A time sweep decomposes overlap mass exactly with
   `overlap = useful + shadow`, where `useful(t) = max(0, U(t) - 1)` among the
   `R(t)` resident kernels and `U(t)` is the useful-phase count.
5. `we_lo` overstates useful concurrency (shadow upper bound); `we_hi`
   understates it (shadow lower bound). The pair brackets the true split, and
   H1 is judged on the lower shadow bound.

The overlap convention matches the 20260821 audit: `node_sum - union_measure`,
where dead device gaps inside the span are excluded from the union. The
authority ledger's dead gap (8.208 us) independently reproduces the measured
dead gap (8.1-8.3 us per replay).

Two captures were retained. `h1-final-capture` uses subsampled flash-kernel
ring records (lower profiling overhead) and supplies the reliable min-anchor
bound. `h1-full-sampling-capture` records every block and supplies the
authoritative max-anchor bound.

## Ledger

Sums are over seven steady decode replays; per-replay means in parentheses.

| capture | node_sum us | union us | overlap us | useful us | shadow us | shadow share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| authority, nsys no ring | 5023.823 | 3890.568 | 1133.255 | unmeasured then | unmeasured then | - |
| full-sampling ring | 38024.6 (5432.1) | 28336.6 (4048.1) | 9688.0 (1384.0) | 510.4-780.9 (72.9-111.6) | 8907.1-9177.6 (1272.4-1311.1) | 91.9-94.7% |
| final subsampled ring | 38535.1 (5505.0) | 28512.1 (4073.2) | 10023.0 (1431.9) | 458.9-743.2 (65.6-106.2) | 9279.8-9564.1 (1325.7-1366.3) | 92.6-95.4% |

The authority row comes from `nv-ledger-overlap-audit-20260821/ledger.json`.
The ring captures run under nsys plus ring instrumentation, so their absolute
mass carries roughly 8-10% profiling tax; the shadow/useful split is robust
because both captures agree.

Applying the combined measured share bracket to the authority overlap mass:

| term | share bracket | authority us per replay |
| --- | ---: | ---: |
| useful concurrency | 4.6-8.1% | 51.9-91.3 |
| wait/launch shadow | 91.9-95.4% | 1041.9-1081.3 |

## Per-replay distribution

Final capture, seven steady replays. This table keeps the span convention;
the Ledger above uses `union_measure = span - dead_gap`, where the measured
dead gap is ~8.1 us per replay, so each overlap row here sits ~8.1 us below
its union-measure twin.

| replay | overlap us | useful us | shadow us | offset stdev ns |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1423.384 | 95.8-146.3 | 1277.0-1327.6 | 126.7 |
| 1 | 1426.040 | 102.4-158.1 | 1268.0-1323.7 | 121.9 |
| 2 | 1421.172 | 112.7-169.3 | 1251.9-1308.5 | 118.7 |
| 3 | 1425.369 | 28.2-50.9 | 1374.4-1397.2 | 126.7 |
| 4 | 1424.057 | 33.0-59.2 | 1364.9-1391.1 | 125.7 |
| 5 | 1421.425 | 39.0-72.0 | 1349.4-1382.5 | 125.7 |
| 6 | 1423.837 | 47.9-87.4 | 1336.5-1375.9 | 124.1 |

## Where the shadow lives

Chosen replay per-segment kernel residence and per-kernel spin-span sums.
These spin sums are kernel-level attribution, not overlap-attributed mass.

| segment | nodes | dur us | spin us |
| --- | ---: | ---: | ---: |
| anchor (MMQ Q/O/gate/up/down) | 145 | 3449.2 | 451.0-2842.1 |
| S1 on-path support | 180 | 774.8 | 405.9-411.4 |
| S1 off-path support | 252 | 716.2 | 227.3-234.6 |
| S2 (flash/combine) | 72 | 252.7 | 104.2-105.0 |
| S4 | 70 | 241.5 | 96.4-96.7 |
| S3 (quant provider) | 36 | 46.6 | 3.6-4.3 |
| S0 | 2 | 7.0 | 2.8-2.8 |
| tail | 5 | 13.8 | 4.5-5.3 |

Top roles by spin: gate MMQ 1218 us, down MMQ 660 us, output MMQ 380 us,
query MMQ 282 us, combine 164 us. The MMQ anchors hold most dependency wait.

## Gates

| gate | result |
| --- | --- |
| G1 instrument fired | pass |
| G2 records match kernels | pass, 761/762 kernels anchored, 0 unassigned |
| G3 ledger identity closes | pass, `useful + shadow = overlap` to <0.02 us |
| G4 useful-concurrency bracket computed | pass |
| G5 H1 verdict | supported, lower shadow bound >= 0.9 |

The one unanchored kernel (`k_get_rows_float`, graph node 751) loses its
wait-exit records to the overlapping next `k_get_rows`; treating it as fully
useful is the direction that favors useful concurrency, so the shadow bound is
conservative.

## Grounding and forward direction

- The audit's serialization counterfactual (248.2 -> 193.7 tok/s when llama's
  intervals are serialized) is mostly the cost of exposing spin-wait latency,
  not lost parallel work. Fully serializing only forfeits the measured useful
  bracket, at most ~91 us per replay, roughly 5.5 tok/s at the 4028.5 us wall.
- Therefore overlap-parity with llama is a bounded target. Matching llama's
  overlap shape could recover at most the ~52-91 us useful fraction per token;
  it cannot explain or close the ~703 us wall gap. The gap lives in node mass
  and support exposure, as the ledger already showed.
- Roofline: llama runs ~1167 GB/s effective against a 1700-1792 GB/s measured
  peak. The useful-overlap measurement says the slack is not recovered by
  parallelism; it is latency and kernel-body inefficiency. Kernel-body
  efficiency, not overlap harvesting, is the lever for parity and beyond.

## Evidence

- Reconciliation JSON: `evidence/nv-llama-useful-body-h1-20260821/h1-reconciliation.json`
  and `h1-reconciliation-full-sampling.json`.
- Raw nsys reports, SQLite exports, graph dumps, and all nonempty ring files
  are retained in the same directory with SHA-256 hashes in `sha256.txt`.
- Repro note: nsys 2026.1.3 can stall after `Generating .qdstrm` while
  downloading symbols. Dropping the profiler's TCP connection to the symbol
  host lets it fail that fetch and finish writing the `.nsys-rep`.
- Parser: `extra/llm_research/decode/nv_llama_useful_body_h1.py`.
- Instrumentation: `env/llama.cpp` build `build-cuda-instrumented`, gated by
  `GGML_CUDA_PDL_TRACE`; dump env `GGML_CUDA_PDL_TRACE_DUMP`.
- Benchmarks: final capture prefill 11783.7 tok/s and decode 128.53 tok/s under
  nsys; unprofiled authority decode 248.3 tok/s (n_gen 20 avg, samples
  240.1-250.4) with prefill 14196 tok/s.

## Labels

- `observed`: ring timestamps, CUPTI intervals, ledger mass, shares, gates.
- `inferred`: the 51.9-91.3 us / 1041.9-1081.3 us authority split, obtained by
  applying measured shares to the authority overlap mass.
- `unmeasured`: the split of shadow into dependency wait versus host launch
  (trigger records are retained and could separate it); a full tinygrad-side
  wait-exit census on the decode route.

No production change or performance promotion follows from this record.

An earlier working estimate of ~97% shadow used a per-kernel sum that
double-counted records across overlapping same-name kernels. It is superseded
by the time-sliced decomposition here, whose ledger identity closes exactly.
