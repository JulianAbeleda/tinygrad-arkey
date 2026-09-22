# NV Qwen3-8B pp512 Q4 IMMA gate/up discriminator

## Verdict

The first compressed-prefill discriminator passes the physics gate and stops at
a precise tinygrad substrate wall.  Llama's production Q8_1 x Q4_K gate/up
lifecycle is about 3.2x faster than the corrected FP16-overlay population while
remaining finite in the retained production run.  It is real signed-int8 tensor
core work, not DP4A: the captured specialization issues `IMMA.16832.S8.S8`, with
no HMMA or IDP.4A in that function.

Tinygrad cannot yet produce the third arm.  Its Q4_K x Q8_1 prefill contract is
only a logical descriptor: `emit_q4k_q8_mmq_kernel` deliberately raises
`NotImplementedError`.  More fundamentally, `tinygrad/codegen/opt/tc.py` has no
CUDA signed-int8-to-int32 tensor-core descriptor (AMD has one).  This is the
first required investment; tuning the existing FP16 schedule cannot test the
compressed hypothesis.

No production route was edited.

## Exact production-shaped comparison

Shape: `(M,N,K)=(512,12288,4096)`, Qwen3-8B gate/up, RTX 5090 sm_120.

| arm | complete service | useful rate | status |
| --- | ---: | ---: | --- |
| corrected tinygrad FP16/HMMA | 569.7 us/projection (41.020 ms / 72) | 45.2 TMAC/s | finite; exact candidate output SHA equals generic FP16 SHA |
| llama Q8_1 x Q4_K IMMA | about 179.2 us/projection (12.542 ms / 70) | about 143.8 TMAC/s | retained real pp512 lifecycle; quantize + MMQ + fixup included |
| tinygrad-generated Q8_1 x Q4_K | not emitted | n/a | blocked before GPU: missing NV int8 TC lowering and MMQ emitter |

The llama total uses the 70 longest Q4 main calls (the two full-batch gate/up
projections in layers 0--34), the corresponding 70 largest Q4 Q8_1 conversions,
and 70 times the Q4 fixup mean.  This is an exact-role estimate from the serialized
trace rather than a newly instrumented standalone call; role tagging is not
present in Nsight's kernel row.  Its components are 11.671 ms main, 0.288 ms
quantization, and 0.583 ms fixup.

## Qualification and physical evidence

The isolated tinygrad candidate produced `[1,512,12288]` FP16, all finite, SHA-256
`399c3609...97c22f3`; the generic FP16 arm produced the same SHA.  The retained
corrected-candidate authority additionally covers guard/readonly and four-role
bit-exact checks.  Llama changes the numerical contract by quantizing each input
to Q8_1, so FP16 bit identity is impossible by construction; model/token quality,
not FP16 bit equality, must be the eventual admission gate.

Captured llama launch facts:

- grid 170 CTAs, 256 threads/CTA;
- 57,856 B dynamic plus 1,024 B static shared memory;
- 251 registers/thread, stack 0, local spill 0;
- captured non-fusion specialization: 256 `IMMA.16832.S8.S8`, zero HMMA,
  zero IDP.4A. (The earlier lifecycle audit's 512 count combined two variants;
  the actually captured `bool=false` function contains 256.)

Hot/cold R7/R9 could not be meaningfully run for the nonexistent native arm.
The retained R7 whole-prefill authority and fresh isolated correctness run are
sufficient for the invest/no-invest discriminator, but not for promotion.

## Recovery and roofline

Replacing the profiled full-batch gate/up population with the measured llama-like
lifecycle recovers about 28.5 profiled ms.  Scaling only as a planning estimate by
`84.025 / 95.810` gives about 25.0 ms authority-wall recovery: approximately
59.1 ms pp512 or 8.67k tok/s, before other projection families.  This is not a
promotion forecast because it changes activation semantics and includes llama's
last-layer pruning topology.

Q4_K moves 113.25 MB of analytic tiled weight payload per call versus 402.65 MB
for FP16.  At 227.6 MAC/analytic-byte and the local 1.792 TB/s DRAM peak, the
analytic packed bandwidth roof is 407.7 TMAC/s.  Llama's complete lifecycle is
at about 35.3% of that roof, leaving roughly 2.84x headroom.  These are requested
payload calculations, not physical DRAM counters; the next roofline phase must
use NCU DRAM/L2 bytes and tensor-pipe counters.

## Required implementation order

1. Add a CUDA signed-int8 input / int32 accumulator TensorCore descriptor for
   `mma.sync.aligned.m16n8k32...s8.s8.s32`, including validated lane maps.
2. Add renderer/PTX lowering and a tiny prepacked int8 IMMA canary.  Require SASS,
   no spill, guards, and FP32 int-dot exactness before Q4 grammar.
3. Implement the existing Q8_1 DS4 and Q4_K logical contracts as a research-only
   emitter: packed loads, scale/min correction, FP32 accumulation, output guards.
4. Add Q8 producer and stream-K/fixup, then run true cold/hot R7 and reverse R9
   on the full `(512,12288,4096)` lifecycle.
5. Only after that gate passes, integrate behind an opt-in research route and
   qualify whole-model logits/token quality.  Production routing remains last.

Structured evidence is in
`docs/task_workflow/evidence/nv-prefill-q4-imma-gate-20260828/discriminator.json`.
