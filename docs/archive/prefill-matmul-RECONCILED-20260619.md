# RECONCILED (cross-checked) — profiler durations include STALLS; matmul real-compute ~25% of wall; Tensile A/B is the trustworthy instrument

After the harness-fixed "matmul 78%/GPU-bound" claim, cross-checks show that conclusion OVER-counted. Honest close.

## The cross-check that resolves the flip-flopping
The SAME gate matmul kernel, measured 3 ways (same machine):
- isolated `_time_program` (GPU, best-of): **39.5 TF = 1.3ms**
- eager python loop (wall incl dispatch): **33.5 TF = 1.54ms** (same vs different weights IDENTICAL -> cache NOT the cause)
- in-jit full-forward (ProfileGraphEvent duration): **20 TF = 2.58ms**
Isolated + eager AGREE (~1.3-1.5ms real GPU). The in-jit profiler (2.58ms) is ~2x that -> **the ProfileGraphEvent
per-kernel duration INCLUDES inter-kernel stall/wait (~1.3ms/matmul), not just compute.** Cache refuted (same=diff),
clock refuted (profile_peak +4%), test-size refuted (full=shrunk). The only thing left that doubles it is the
profiler bracketing schedule->complete (incl dependency stall).

## What this corrects
- **RETRACT "prefill is GPU-bound 97% / matmul 78% of wall"** -- the profiler GPU-busy (355ms=97%) is inflated by
  per-kernel stalls. Real matmul GPU-compute = ~1.3ms x 72 ~ 94ms + down + attn; the 365ms wall is ~HALF real
  GPU-compute, ~half stalls/gaps/glue. Real matmul-compute ~25% of the wall (close to the original "24%" I'd
  retracted -- which was roughly right, for the wrong stated reason).
- The "in-jit matmul 2x slower" is NOT a real compute slowdown to "fix" -- it's the profiler counting the stall the
  matmul waits in (dependency chain). The matmul kernel itself runs at ~39 TF whenever it actually executes.

## The trustworthy instrument + the stable conclusion (held all session)
The e2e **Tensile A/B is the reliable instrument** (changes ONLY the matmul, 1.56x kernel-verified, measures e2e):
full 0.997x, FFN-only 0.993x. A 1.56x matmul moving the wall <1% => **matmul is NOT the e2e bottleneck**, whatever
the exact %. This held across EVERY clean e2e A/B all session; only the per-kernel decompositions flip-flopped
(profiler stalls, sub-jit inflation, .sum codegen-break -- the per-kernel timing is genuinely unreliable here).

## So the real lever (stable)
The wall is ~half stalls/dependency-chains + glue + dispatch, ~half real GPU-compute (matmul ~25%). The e2e lever
is REDUCING STALLS / kernel-count / serialization (the .contiguous()-isolated FFN chain, the symbolic-attention
kernels), NOT faster matmul. = concrete-KV (1.24x, fewer/cheaper attention kernels) -- the validated, shippable win.
The matmul TFLOPS chapter (WMMA 42, FMA, Tensile, in-jit-2x) is a RED HERRING for e2e, now triple-confirmed.

## Honest meta
Per-kernel GPU timing on this HCQ stack is NOT reliable (profiler durations include stalls; sub-jit inflates;
codegen-probes break). Trust e2e A/Bs. I over-corrected twice (host-dispatch <-> GPU-bound) by trusting per-kernel
numbers; the Tensile A/B was right throughout: matmul is not the e2e lever; kernel-count/stalls is.

## Files
/tmp/decomp_full.py, /tmp/clock_test.py, the same-vs-diff-weight + iso-vs-injit tests. Reconciles the harness-FIXED
doc (which over-counted via profiler stalls).
