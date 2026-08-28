# NV pp512 native FP16 stream-K microgate

## Verdict

The first native topology gate validates the occupancy mechanism but is not a
performance candidate.  One 170-CTA main launch plus one 32-CTA fixup is
4.93x faster than the same scalar-FMA body at the production 32-CTA topology,
but its complete 4.813 ms lifecycle is still 27.1x slower than the installed
177.3 us tensor-core candidate.  It is therefore a no-go for production.

No routing or production code changed.

## Results

Exact shape: FP16 A/B, FP32 accumulation/output, `(M,N,K) =
(512,1024,4096)`.  The native main launch assigns 170 blocks across the 32
128x128 output tiles and partitions K 5--6 ways.  A separate 32-block fixup
sums the partials.

| arm | hot R9 minimum | fresh R7 minimum | topology |
| --- | ---: | ---: | --- |
| scalar control | 23.715 ms | 23.871 ms | one 256-thread CTA per output tile, 32 CTAs |
| native stream-K complete | 4.813 ms | 4.830 ms | 170-CTA main + 32-CTA fixup |
| topology speedup | 4.93x | 4.94x | occupancy effect is real |
| installed tensor-core authority | 177.3 us | retained profiler authority | 32 CTAs |

Correctness against the unsplit scalar control is finite with maximum absolute
error `0.00192260742` and mean absolute error `0.00187767169`.  This passes the
declared `atol=0.125, rtol=0.002` contract.  The difference is the expected
change in FP32 reduction association across 5--6 K partitions.

## Occupancy and code inspection

NCU observes the requested `(170,1,1)` grid and `(256,1,1)` block.  The main
kernel uses 40 registers/thread with no spills.  Active warps are 16.67% of
peak: one 8-warp CTA per SM fills the grid but not each SM's warp slots.  SM
throughput is 11.13% and DRAM throughput only 0.27%, confirming that this
minimal scalar body is instruction-bound rather than a weight-streaming
tensor-core implementation.  NCU duration was 5.515 ms under profiling.

`ptxas` reports 37 registers/thread for control, 40 for main, and 20 for
fixup, all with zero stack and spill traffic.  A cubin was retained.  Full
SASS dumping is blocked on this host because `cuobjdump` delegates to a missing
`nvdisasm`; the failure is preserved in `artifacts/gate.sass`.  The compiler
and counter evidence is decisive enough to reject this body: it does not issue
tensor-core MMA and is over 27x behind the installed candidate.

## Consequence

The missing implementation is now tightly specified: preserve the demonstrated
170-CTA K partition and coalesced fixup, but transplant the installed
128x128x32 FP16 MMA/LDS inner loop into each K interval.  Promotion requires
the complete pair below 177.3 us.  The useful gates remain 100 us (5.57 ms
pp512 recovery), 60 us (8.45 ms), 35 us (10.25 ms), and approximately 25 us
near the FP16 roof (10.97 ms).

## Evidence

- Harness: `extra/llm_research/prefill/nv_prefill_kv_native_streamk.py`
- Hot R9: `docs/task_workflow/evidence/nv-prefill-kv-native-streamk-20260828/hot-r9.json`
- Fresh R7: `docs/task_workflow/evidence/nv-prefill-kv-native-streamk-20260828/cold-r7.json`
- Source, binary, cubin, compiler/SASS note, and NCU CSV:
  `docs/task_workflow/evidence/nv-prefill-kv-native-streamk-20260828/artifacts/`
