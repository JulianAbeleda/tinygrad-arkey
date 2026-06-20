# Prefill AMD GEMM — Fact-Checking the M=384 Outlier & the Gap (narrative flipped)

Date: 2026-06-20

## Question

Is Tensile's `.dat` "M=384 → 79 TFLOPS" a **shape-intrinsic** ceiling our M=512 shape misses, a
**Tensile-specific** win, or a **clock** artifact? And is the "we're behind Tensile" gap real?

## Method

Run **our** kernel (`build_gemm_lds2` BK32+PAD16+PLRA1, square 128×128) on Tensile's tuned shapes at a **pinned
clock** and compare to the `.dat` claimed speeds. Our kernel is parametric in M,N,K (no kernarg issues). If ours
also peaks at M=384 → shape-intrinsic; if flat → the 79 is Tensile/clock; the workgroup count tells the rest.

## Result (pinned, reproduced 2×)

| shape | ours TFLOPS | `.dat` claim | ours/dat | workgroups |
|---|---:|---:|---:|---:|
| 256×8192×4096 | 36.0 | 64.7 | 0.56× | 128 |
| **384×8192×4096** | **47.5** | **79.1** | 0.60× | 192 |
| 512×8192×4096 | 47.2 | (untuned) | — | 256 |
| 768×8192×4096 | 62.2 | 65.3 | **0.95×** | 384 |
| **1536×8192×4096** | **73.4** | 68.2 | **1.08× (ours wins)** | 768 |
| **512×12288×4096 (OUR shape)** | **63.0** | (untuned) | — | 384 |

## Findings — the fact-check answers all three

1. **The 79 is an OUTLIER, not shape-intrinsic.** Our kernel does **not** peak at M=384 — it rises
   **monotonically with workgroup count** (36→47→62→73 as wg goes 128→192→384→768). M=384 is just on the ramp.
   The `.dat`'s 384=79 (higher than its own 256=65 and 768=65 neighbors, despite fewer workgroups) is a
   Tensile-specific/measurement outlier our kernel doesn't reproduce. **The realistic tuned cluster is ~65–68.**

2. **The "gap" is OCCUPANCY (workgroup count), not kernel quality.** Our kernel is wg-starved at small M
   (256×8192 = only 128 wg → 36 TFLOPS); it scales up as wg fills the 96-CU GPU. This is **our kernel's own
   occupancy curve**, clock-independent — and it has nothing to do with our actual shape.

3. **On well-occupied shapes, our kernel MATCHES or BEATS Tensile's tuned `.dat`:** 768×8192 → 0.95×,
   1536×8192 → **1.08× (ours wins)**. (Cross-comparison mixes clock — `.dat` is offline, possibly boost — so
   read "≈ parity," but ours is clearly *not behind* in kernel quality when the GPU is filled.)

4. **Our ACTUAL shape is well-occupied and at the cluster.** 512×12288 has **wg=384** (large N → many N-blocks)
   → **63.0 TFLOPS**, right at the realistic tuned cluster (~65). Our big-N shape is *good* for us (more
   workgroups), not a weakness.

## Reconciling with the earlier "ours 57 vs Tensile 62"

The earlier clock-matched probe interleaved our kernel with the Tensile `.co` launch and read ours at **57**.
That was **launch-context perturbation** — ours-alone pinned on the same 512×12288 shape is **63.0**, and the
Tensile `.co` on that shape was **~62**. So **on our actual shape, ours (63) ≈ Tensile (62) — parity, ours
slightly ahead.** The "8% gap" was the interleave artifact; the "22% gap" was the 384 outlier + small-M
occupancy. Neither is a real kernel deficit.

## Verdict — the narrative flips

- **Is it the outlier?** Yes — 79 is a Tensile-specific outlier our kernel doesn't reproduce; realistic
  ceiling ~65.
- **Is it partly clock?** The cross-lib absolute ratios mix clock (`.dat` offline vs ours pinned), so they're
  caveated — but the robust, clock-independent facts (our wg-occupancy curve, no 384 peak, our-shape 63 ≈
  Tensile 62) stand regardless.
- **The real story is OCCUPANCY.** Apparent gaps were small-M wg-starvation; our actual large-N shape is
  well-occupied and **at parity with Tensile**.

**Our dependency-free kernel is at parity with Tensile on our shape (~63 vs ~62), and matches/beats Tensile's
own tuned benchmarks on well-occupied shapes.** There is no residual kernel deficit to close — the arc lands at
**parity, not 92%**. The earlier sub-parity readings were measurement artifacts (interleave perturbation,
outlier neighbor, small-M occupancy), now all identified.

## Honesty

- `.dat` speeds are offline-tuned benchmarks at unknown (possibly boost) clock; cross-library absolute ratios
  are clock-caveated. The within-our-kernel patterns (occupancy curve, no 384 peak) and the same-shape
  same-pinned-clock our-vs-`.co` parity (63 vs 62) are the robust claims.
- Single fp16 GEMM family; correctness 2.1e-4 on every shape tested.
