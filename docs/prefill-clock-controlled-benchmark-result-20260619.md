# RESULT — clock-controlled tinygrad-vs-llama prefill: tinygrad concrete ~50% of llama (NOT parity)

Executed the scope. Pinned amdgpu DPM (manual, SCLK lvl2 / MCLK lvl3) via passwordless sudo, measured both engines,
restored to auto.

## Measured (pinned DPM)
| engine | pp512 tok/s | vs llama |
|---|---:|---:|
| llama.cpp | **3086 ± 173** (±5.6%, STABLE) | 100% |
| tinygrad concrete-KV | **1551** (stable, repeatable) | **50%** |
| tinygrad symbolic (default) | 1251 | 41% |

- The pin STABILIZED llama (2430±642 unpinned -> 3086±173 pinned) -> llama's number is now trustworthy.
- The pin did NOT change tinygrad (1548->1551) -> tinygrad's throughput is stable/repeatable at ~1551, independent
  of the DPM pin.
- concrete-KV's 1.24x is intact: 41% -> 50% of llama.

## Clock-vs-efficiency: cannot be cleanly isolated (measurement limit)
Even pinned to DPM level 2, SCLK OSCILLATES 0->1992 MHz under load (the GPU idle-gates the clock during the
per-iteration host-sync gaps), and gpu_busy% is mostly 0 with bursts to 100 -> BOTH engines have gappy GPU
execution; sampling (>=1s) is far coarser than the ~300ms bursts. So we cannot finely measure the clock each engine
actually runs at. Under the SAME DPM policy, tinygrad's gappier execution (the 729-kernel graph + per-replay sync,
the busy-wait) would let its clock gate down MORE -> part of the 50% gap is tinygrad not keeping the GPU saturated
(an execution/runtime issue), part is genuine per-clock efficiency. Cannot separate the two on this box.

## CORRECTION to prior framing
The banked "PREFILL_V2 = 2486 tok/s = ~83% of llama" does NOT reproduce under these controlled measurements
(tinygrad symbolic = 1251 here, not 2486). The earlier "concrete-KV -> ~parity with llama" claim is RETRACTED.
Controlled truth: **tinygrad concrete-KV prefill = ~50% of llama**, concrete-KV improving 41%->50%. The 2486 was
likely a different harness / a warmer transient clock state / pre-regression -- not reproducible now.

## What's reliable
- concrete-KV = 1.24x over symbolic (clock-controlled, byte-identical) -- SOLID.
- tinygrad concrete prefill = ~50% of llama's ~3086 tok/s at the same DPM policy -- the defensible cross-engine number.
- The exact clock-vs-efficiency split is unmeasurable (clock oscillates with execution gaps).

## Corollary (decode, Codex's track)
Decode is MCLK/BW-bound; MCLK was already at max (1249) in all states -> decode is LESS clock-confounded than
prefill (which is SCLK/compute-bound and SCLK was the swinging variable).

## Files
`docs/prefill-clock-controlled-benchmark-scope-20260619.md`. Pin: amdgpu sysfs (root). tinygrad uses KFDIface.
