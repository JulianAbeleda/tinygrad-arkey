# Validate-before-close (V1-V3) -- the conclusion CHANGES. We were measuring wrong.

Date: 2026-06-15. Stress-tested the decode conclusion. Result: the gap is REAL, but several prior
conclusions were MEASUREMENT ARTIFACTS, and the e2e penalty is now the precise open question.

## V2 -- llama.cpp on THIS GPU: 105.7 tok/s (the bar is real, and larger than thought)
`llama-bench` on the same RX 7900 XTX, same Qwen3-8B Q4_K: **105.66 tok/s (tg64) = 57% of peak**, vs our
tinygrad ~30 (16%). The gap is REAL on our hardware -- 3.5x. Not a phantom, not a cross-machine artifact.

## V3 -- NOT degradation, NOT the clock
GPU cool (43C, no throttle). Memory clock ramps slowly in tinygrad (96->1249 over ~4s) vs llama.cpp's
immediate 1249. BUT forcing perf=high (mclk pinned 1249) gave tinygrad 24 tok/s -- still 4x below llama's
105 at the SAME clock. So the gap is real tinygrad inefficiency, not power management.

## V1 -- the GEMV DOMINATES (not Amdahl), and prior kernel measurements were CACHE/LAUNCH artifacts
- The token reads 4765 MB (= the full 4.68 GB model) at 113 GB/s = 42 ms = ~95% of the token. So the
  weight read (GEMVs) DOMINATES -- it is NOT Amdahl, the GEMV was the right target.
- BUT re-measuring the kernels with a COLD, cache-EXCEEDING working set (300 MB > 96 MB Infinity Cache,
  launch overhead amortized) OVERTURNS the earlier numbers:
  | variant | cached/small (WRONG) | cold/large (REAL) |
  |---|---|---|
  | readraw | 54% | **80.4%** |
  | fp naive | 16% | 18.8% |
  | fp_prefetch | 24% | **68.7%** |
  | fp_acc8 | 23% | 33.2% |
  | vdot_acc4 | 49.5% | **75.7%** |
  The small 9.4 MB working set was launch-overhead-DEFLATED. Properly measured, **prefetch is a 3.7x lever
  (fp 18.8% -> 68.7%)**, not the 1.45x we reported, and the kernels SATURATE (69-80%, exceeding llama's 57%).

## What changes
- CORRECTED: "fp dequant caps bandwidth at ~24%, only int-dot saturates" -- WRONG, a measurement artifact.
  Prefetch saturates fp to 69%; the dequant does NOT cap it. The MLP/prefetch root-cause was right all along.
- The kernels CAN saturate (69-80% cold) -- ABOVE llama.cpp's 57%. So the kernel is NOT the wall.
- The e2e decode runs at 13% while the saturating kernel does 69-76% -> a ~5x e2e penalty. The penalty is the
  E2E EXECUTION (per-layer GEMVs ~25-30 MB each, run in the JIT decode graph) NOT sustaining the saturation
  the large standalone kernel achieves. THIS is the precise open question, re-localized.

## So we were RIGHT not to close
The structural conclusion (e2e penalty, not kernel) holds and is sharper -- but the kernel-can't-saturate
sub-claims were artifacts. The real next question: why does the per-layer decode GEMV run at 13% e2e when a
cold large GEMV (and prefetched fp) hits 69-80%? Is it per-kernel size (25 MB too small to sustain), JIT
graph overhead, or kernel structure? llama.cpp's per-layer GEMV (same 25 MB) sustains 57% -- so it is
addressable, not fundamental. The lever: make the per-layer decode GEMV saturate in-context (prefetch
structure + whatever sustains memory for small kernels, which llama.cpp does and we don't yet).
