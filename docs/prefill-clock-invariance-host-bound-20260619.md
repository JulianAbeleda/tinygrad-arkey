# DECISIVE — tinygrad prefill is GPU-CLOCK-INVARIANT (host-bound); the pin is irrelevant for it

Reverse-engineered "why the pin worked for llama not tinygrad" via online (ROCm #6289) + a clock-invariance test.

## ROCm issue #6289 (the mechanism for the stuck clock)
Known AMD issue: SMU firmware does NOT boost GFXCLK for compute/ROCm workloads in standard modes -> GPU stuck near
idle (~41MHz RDNA4 / ~324MHz our gfx1100) instead of ~2400 -> ~3x degradation. `auto`/`high`/**`manual` all stay
stuck**; **only `profile_peak`** (or a mem_busy daemon) forces the boost. **Our earlier pin used `manual` = the
wrong knob** (manual even "hurts" per the issue).

## The clock-invariance test (decisive)
Set `profile_peak`. tinygrad clock under sustained prefill: **median 2331 MHz, max 2334** (UNSTUCK, was ~324).
But tinygrad pp512 throughput: **concrete 1511, symbolic 1209 -- UNCHANGED** (was 1551/1251 at manual; flat/noise).
**7x clock increase -> 0x speedup.** => tinygrad prefill throughput is INDEPENDENT of GPU clock.

## Conclusion (full understanding of the pin)
- **tinygrad prefill is HOST-BOUND** (clock-invariant, GPU idle ~most of the time). The "stuck clock" (324) was a
  SYMPTOM of host-boundness (idle GPU -> SMU keeps clock idle), NOT a cause. (Confirmed real, not a sysfs artifact:
  profile_peak made sysfs read 2331 -> sysfs DOES see tinygrad's clock.)
- **llama is GPU-BOUND** (busy ~48%, dense batched GEMMs, clock auto-boosts to ~1730). Clock-sensitive -> the pin
  (and clock state) determines its speed; the pin stabilized it (2430+-642 -> 3086+-173).
- **Why the pin "worked" for llama not tinygrad: llama is clock-sensitive (GPU-bound), tinygrad is not
  (host-bound).** The pin can only help a clock-limited workload.
- **The ~50% gap is NOT kernel efficiency or clock** -- it's that tinygrad prefill is host-bound (underutilizes the
  GPU) while llama saturates it. RE-VINDICATES the earlier host/busy-wait finding (the GPU is idle; the host
  dispatch/busy-wait/fragmented 729-kernel execution is the bottleneck). The clock-invariance is the clean proof
  the ambiguous cProfile/ProfileGraphEvent measurements couldn't give.

## Lever (corrected, final)
tinygrad prefill: reduce HOST overhead / keep the GPU saturated (fewer kernels, less per-replay sync, less
dispatch). NOT clock (irrelevant), NOT matmul (exhausted), NOT kernels. concrete-KV's 1.24x fits this: it cuts
kernel count (less host dispatch). The remaining 2x to llama = closing the host-bound gap (keep GPU busy like llama).

## Files
ROCm #6289. `docs/prefill-clock-controlled-benchmark-result-20260619.md` (the 50% number). Clock restored to auto.
