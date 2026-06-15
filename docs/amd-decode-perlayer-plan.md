# Isolate the e2e penalty -- does the per-layer GEMV saturate cold? (scope)

Date: 2026-06-15. The kernels saturate at LARGE cold size (vdot 76%, readraw 80%), but e2e the per-layer
GEMVs run at ~13%. Two hypotheses for the ~5x penalty -- this experiment distinguishes them.

## The question
At the e2e PER-LAYER size (ffn: 12288 rows x 4096 = 28MB; attn: 4096 rows = 8MB), COLD, with launch
overhead amortized, does the GEMV saturate?
- H1 (JIT/launch): per-layer kernel saturates standalone -> the e2e 13% is the JIT-graph / per-kernel
  launch overhead across ~252 kernels -> lever = fewer/fatter launches (fusion/megakernel/persistent).
- H2 (small-kernel): per-layer kernel does NOT saturate even standalone-cold -> small kernels can't
  sustain the memory pipeline -> lever = llama.cpp's mmvq structure (more parallelism / better coalescing
  for small GEMVs).

## Method (defeat the cache, amortize launch)
The per-layer working set (8-28MB) fits in the 96MB Infinity Cache, so reps cache it (or, with few reps,
launch overhead deflates). To get the REAL cold per-layer bandwidth: allocate a LARGE backing buffer
(~2GB = 80x the per-layer region) and have each rep read a DIFFERENT region (rotate an offset scalar) ->
every rep cold, and 30+ reps amortize the launch overhead.
- Kernel: vdot_acc4 (the near-saturating one) + an `off` scalar arg; W4 = words + off + row*RW.
- Measure cold per-layer Q4-GB/s at ROWS in {4096 (attn), 12288 (ffn)}; compare to ROWS=131072 (saturating
  76%) and the e2e 13%.

## Gate
- per-layer cold ~= 70-76% -> H1: the kernel saturates at per-layer size; the e2e penalty is JIT-graph/launch
  overhead. Next: reduce launches (the per-layer GEMV is fine; the 252-launch structure is the wall) --
  back to the megakernel/fewer-launches direction, but now with the kernel proven fine.
- per-layer cold ~= 13-40% -> H2: small kernels don't sustain; the per-layer GEMV itself is the problem at
  small size. Next: study llama.cpp's mmvq structure (parallelism/coalescing for small GEMVs) and adapt it.

## Honest framing
This is the precise fork the validation re-localized. The kernel-can-saturate finding (cold/large) is
solid; this tells us WHERE the e2e 5x penalty lives -- in the per-layer kernel (small-size) or in the
graph (launches). Either is more concrete and addressable than the vague 'structural wall' we almost
closed on. llama.cpp proves a per-layer 25MB GEMV CAN sustain 57% on this GPU -> whichever hypothesis, it
is addressable, not fundamental.
