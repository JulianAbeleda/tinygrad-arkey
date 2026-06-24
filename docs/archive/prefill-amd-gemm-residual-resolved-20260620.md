# Prefill AMD GEMM — Residual RESOLVED: it was host launch overhead; our kernel is GPU-faster than Tensile

Date: 2026-06-20

## The answer

The ~4% "ours behind Tensile" was **host launch overhead, not GPU execution.** Batching the launches to
amortize host overhead flips the ratio: at the **GPU level our dependency-free kernel is ~9% FASTER than the
vendored Tensile `.co`.** Reproduced 3×.

## The decisive measurement

Each kernel launched **K times back-to-back between syncs**, per-launch time = total/K. As K grows, per-call
host overhead (Python `run_linear` for ours; `tprg` kernarg-refill for Tensile) amortizes and the number
converges to **GPU execution throughput**. Pinned, each alone. `extra/qk_amd_gemm_batch_isolate.py`.

| batch K | ours TFLOPS | Tensile `.co` | ours / Tensile |
|---:|---:|---:|---:|
| 1 | ~62 | ~64 | **0.97** (host-overhead-dominated) |
| 8 | ~74 | ~68 | 1.09 |
| **32** | **~74** | **~68** | **1.09 (ours ahead)** |

- **Both saturate by K=8** (b8 ≈ b32 for each) → both are GPU-bound there; the K=32 numbers are true GPU
  throughput, not host-limited.
- **Ours jumps more** (62→74, +19%) than Tensile (64→68, +6%) when amortized → **ours had MORE per-launch host
  overhead** (the `run_linear` Python path), which is what made it *look* ~4% slower at K=1.
- Reproduced 3×: K=32 ratio **1.09–1.10**, stable.

## What this resolves

- **The residual is not a kernel deficit.** Every batch=1 "sub-parity" reading (0.92 interleaved, 0.96
  alone) was a **launch-path artifact**, not GPU work. At the GPU level **ours wins by ~9%.**
- **It also recontextualizes the PMC hint.** The earlier PMC showed ours using fewer GPU cycles than Tensile
  — consistent with ours being GPU-faster (the exact 2× from the GRBM counter was partly a foreign-`.co`
  capture artifact, but the *direction* — ours fewer cycles — was right).
- **Caveat (honest):** the host overhead is specific to the *benchmark launch path*, not the kernel —
  `run_linear` is a slow per-call Python path; real tinygrad inference launches via the JIT/scheduler
  (amortized) and pipelines many kernels back-to-back, so the **GPU-throughput (batched) number is the
  representative one for real use.** A single isolated GEMM through the slow path sees the host overhead
  (batch=1), but that's not the kernel's GPU speed.

## Corrected standing

The dependency-free kernel `build_gemm_lds2(BK=32, PAD=16, PLRA=1)` square-128 @ wg2 is, at the GPU execution
level, **~9% faster than the vendored Tensile `.co`** (ours ~74 vs ~68 TFLOPS, reproduced 3×), correct
(2.08e-4) — on a shape Tensile never tuned. The prior "~96% of Tensile" was the host-overhead-loaded batch=1
view; the GPU-execution view is **ours > Tensile.**

## Why this is the real answer to "what's causing it"

We chased the gap through every kernel lever (occupancy, bank, depth, prefetch, VALU, ds_load, tile) and found
each bounded/hidden. The reason no kernel lever explained it: **the gap wasn't in the kernel.** It was the
measurement — per-launch host overhead of the launch path, which amortizes to nothing. The batch sweep is the
one test that separates host time from GPU time, and it shows our GPU kernel is faster.

## Honesty / caveats

- GPU-throughput ratio (1.09) is the robust claim, reproduced 3×, both kernels saturated. The batch=1 ratio
  (~0.97) is host-overhead-loaded and launch-path-specific.
- Back-to-back same-buffer dispatches: both kernels measured identically (32 dispatches to the same output),
  so the ratio is fair; absolute saturated TFLOPS (~74 / ~68) are pinned-clock GPU throughput.
- Tensile `.co` is a fallback for this untuned shape; this compares our kernel to that fallback's GPU speed.

## Verdict

`RESIDUAL_WAS_HOST_OVERHEAD`. The ~4% was per-launch host overhead, not GPU execution. **At the GPU level our
dependency-free kernel beats the vendored Tensile `.co` by ~9%** (reproduced 3×), correct, zero dependencies.
We now *know* what caused the residual — and it wasn't the kernel.
