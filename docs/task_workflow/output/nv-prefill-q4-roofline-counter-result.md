# NV pp512 Q4 gate/up roofline counters

## Verdict

Representation conversion is necessary but does not put llama's Q4_K MMQ at
the analytic ceiling.  The retained complete gate/up projection lifecycle is
about 177.5 us per projection (roughly 166 us MMQ + 3.18 us DS4 quantization +
8.33 us stream-K fixup), or 145 TMAC/s.  Against the 407.7 TMAC/s analytic
INT8 ceiling this is **35.6%**.

The next limiter is a coupled residency/data-movement limit, not insufficient
CTA count.  Stream-K launches 170 CTAs for 170 SMs, but each CTA consumes 251
registers/thread and 58.9 KB shared memory.  Only one 256-thread CTA (eight
warps) resides per SM.  Measured active warps are 16.65% of the architectural
maximum and the INT8 tensor pipe is active for only 28.70% of elapsed cycles.

## Physical counter authority

The isolated kernel is the fourth Q4 MMQ in the real pp512 launch sequence,
the first gate/up projection.  NCU application replay used the exact llama
binary and model invocation, grid `(170,1,1)`, block `(32,8,1)`.

| counter | Q4_K MMQ |
| --- | ---: |
| NCU replay duration | 89.056 us |
| DRAM bytes | 11.922 MB |
| L2 requested bytes | 137.060 MB |
| L2 hit rate | 87.91% |
| shared-bank read bytes | 172.626 MB |
| shared-bank write bytes | 71.648 MB |
| INT8 tensor operations | 17.180 billion |
| IMMA warp instructions | 2,097,152 |
| tensor/IMMA active cycles | 28.70% elapsed |
| active warps | 16.65% maximum |
| long-scoreboard stalls | 0.374 inst per issue-active cycle |
| short-scoreboard stalls | 0.092 inst per issue-active cycle |
| registers/thread | 251 |
| shared memory/CTA | 58.88 KB |
| local spilling requests | 0 |

The NCU replay duration is deliberately **not** substituted for the retained
cold/lifecycle wall.  Replay leaves 87.9% L2 hits and reports only 11.9 MB of
DRAM traffic while the algorithm requests about 113.2 MB of packed tiled
weights.  Its 89 us result is a cache-conditioned upper bracket: 289.8 TMAC/s,
71.1% of the analytic ceiling.  The real cold service bracket is 145 TMAC/s,
35.6%.  Together they localize roughly half of the remaining loss to cold
weight service/cache state; even hot, residency and tensor duty leave another
29% below the arithmetic ceiling.

The L2 request count is credible for the algorithmic body: 137.1 MB is 1.21x
the 113.2 MB analytic packed-weight stream after activation, metadata, output,
and protocol traffic.  Shared traffic totals 244.3 MB.  At replay duration
these correspond to 1.54 TB/s L2 requests and 2.74 TB/s shared-bank traffic.

## Complete-chain percentages

Using the retained trace rather than replay timing:

- raw MMQ: about 25.77 GMAC / 166 us = 155.2 TMAC/s, 38.1% roof;
- quantize + MMQ + fixup: 25.77 GMAC / 177.5 us = 145.2 TMAC/s, 35.6% roof;
- conversion/fixup tax: about 11.5 us, 6.5% of the complete chain;
- Q8 producer sharing can remove only about 3.18 us for the second gate/up
  consumer and therefore cannot close the roofline gap.

The earlier 143.8 TMAC/s aggregate raw-MMQ rate is reconciled by role mix and
cold service: it includes smaller and Q6 roles and is not the same numerator
or isolated replay condition as this gate/up counter row.

## FP16 control

The corrected tinygrad gate/up control is 569.7 us for 25.77 GMAC, or 45.2
TMAC/s: **35.4%** of its measured 127.7 TMAC/s FP16 issue ceiling.  Its
algorithmic tiled FP16 weight stream is 402.7 MB, serviced at 0.707 TB/s.
Thus both implementations reach about 35% of their respective complete cold
roof, but Q4 moves 3.56x fewer weight bytes and has the higher INT8 ceiling.

A same-kernel tinygrad physical NCU row could not be obtained: the promoted
candidate is launched through tinygrad's direct NV HCQ path, outside the CUDA
API/CUPTI kernel interception used by NCU.  Two exact-name and one prefix
capture attempts completed the bit-exact projection but reported “No kernels
were profiled.”  The control values above remain analytic plus HCQ-profile
authority, not mislabeled physical counters.

## Next experiments

1. Reduce the Q4 MMQ register footprint enough to admit two CTAs/SM, then
   measure whether tensor active cycles rise without increasing L2 traffic.
2. Reduce or double-buffer the 58.9 KB shared tile and overlap packed-weight
   decode/load with IMMA; long scoreboard is visible but not dominant alone.
3. Run a cold-cache standalone extracted MMQ with real packed weights, avoiding
   NCU application-replay cache persistence.
4. Add an HCQ-compatible counter bridge for the exact tinygrad cubin before
   making any physical DRAM/shared comparison claim for the FP16 control.

No production files were edited.  Reports are retained under
`docs/task_workflow/evidence/nv-prefill-roofline-counters-20260828/`.
