# CORRECTION + BLOCKER — clock-based prefill diagnosis is unreliable (manual-DPM non-reproducible); retract host/BW-bound

This turn chased "why is prefill slow" via GPU clock manipulation. It was a CONFOUND-driven detour. Honest closeout.

## What's RELIABLE
- **Auto mode prefill is REPRODUCIBLE: 1449/1449/1448 tok/s** (concrete-KV, 3 fresh runs, <0.1% spread).
- **tinygrad concrete-KV prefill = 1449 = ~47% of llama's 3086** (both auto, reproducible). [supersedes the
  manual-mode 1551/50% number, which used erratic manual DPM]
- **poll-hoist fix REFUTED** (hoisting the per-poll MMIOInterface out of HCQSignal.wait = 1455->1452, 0 change)
  -> prefill is NOT host-poll-bound.
- concrete-KV = 1.24x over symbolic (same-process, byte-identical) -- unchanged, solid.

## What's BLOCKED / RETRACTED
- **Manual-DPM pinning is NON-REPRODUCIBLE on this box**: identical config (manual, sclk=2, mclk=3) gave 1551 one
  run and ~570 another (2.6x swing). ROCm #6289 (SMU doesn't boost GFXCLK for compute in manual/auto/high) makes
  manual-mode clock erratic for compute. -> clock PINNING is not a usable diagnostic here.
- **RETRACT "prefill is host-bound"** (this turn): it rested on a manual-mode "clock-invariance" (324->2331 = flat)
  where the 324 baseline was erratic manual data. Invalid.
- **RETRACT "prefill is bandwidth-bound"** (this turn): rested on comparing 561 (manual,mclk456) vs 1455 (a
  DIFFERENT auto run) -- invalid cross-mode compare. The clean within-run MCLK sweep (manual: 456->561, 1249->579)
  is ~MCLK-INVARIANT (3%), but it's in broken manual mode so it proves nothing either way.
- Net: cannot cleanly isolate SCLK vs MCLK vs host on this box (the one knob that isolates them -- manual DPM --
  is erratic). Clock-based bottleneck diagnosis is BLOCKED.

## The reliable diagnosis (from the earlier in-model PMC ATLAS, NOT clock games)
Prefill is **compute/WMMA-bound, ~35% of WMMA peak** (matmuls L2 hit 54-87% = weights cached/reused, not
bandwidth-bound; v_wmma multi-cycle). The ~47%-of-llama gap = WMMA codegen efficiency (the SW-pipelined-K-loop, POWN
plateau 42 TFLOPS, codegen-walled) OR external Tensile (toolchain-blocked). This was established before and STANDS;
this turn's clock detour did not reliably overturn it.

## Lesson
On this box: auto-mode throughput is reproducible (use it for benchmarks); manual-DPM clock pinning is NOT (#6289
+ governor/thermal). Do clock-isolation diagnosis via tinygrad PMC (GL2C/SQ counters), never via DPM pinning.
I over-read noisy manual-mode data twice this turn -- the PMC atlas remains the trustworthy bottleneck source.

## Files
`docs/decode-bandwidth-bound-pmu-learning-20260619.md` (the PMC atlas, reliable). ROCm #6289. Clock = auto.
