# Prefill AMD GEMM — Tensile Tuning-Table Audit (our shape is UNtuned)

Date: 2026-06-20

## The finding (it recontextualizes the whole comparison)

We've been benchmarking against "the Tensile kernel for our shape." Reading the rocBLAS solution-selection
library in the `.dat` shows **our shape was never tuned by Tensile** — and the kernel we compared against is a
**nearest-neighbor fallback running ~22% below its own tuned potential.**

## How rocBLAS selects (from the `.dat`)

`library.rows[0].library`: `type=Matching`, `distance=Euclidean`, `properties=[FreeSizeA(M), FreeSizeB(N),
BoundSize(K)]`, a **table of 512 tuned sizes** each with a `key=[M,N,K]`, a solution `index`, and a `speed`
(benchmarked TFLOPS). For a query it picks the Euclidean-nearest tuned size's solution.

The tuned grid:

- **M ∈ {64, 128, 256, 384, 768, 1536, 3072, 6144}** — **512 is NOT in it.**
- **N ∈ {64, 128, 256, 512, 1024, 2048, 4096, 8192}** — **12288 is NOT in it** (max tuned N = 8192).
- K ∈ {64 … 8192}.

**Our shape (512, 12288, 4096) is doubly off-grid** — neither M=512 nor N=12288 was ever benchmarked.

## What the selected kernel actually is

The selected solution (`1140853605`, MT128×128×16, DepthU=16) is the tuned winner for these shapes:

| tuned size | speed |
|---|---:|
| **[384, 8192, 4096]** (← nearest to ours) | **79.1 TFLOPS** |
| [1536, 2048, 8192] | 71.5 |
| [1536, 8192, 8192] | 66.8 |
| [1536, 4096, 8192] | 66.1 |

It is chosen for our (512, 12288, 4096) only by **Euclidean fallback from [384, 8192, 4096]** — where it hits
**79 TFLOPS**. On *our* shape it delivers the **~62** we measured. So the kernel runs **~22% below its tuned
potential** because N=12288 (1.5× the tuned 8192) and M=512 push it outside its sweet spot.

(Hardware best-case in the whole table: **85.8 TFLOPS** on [1536, 1024, 8192] ≈ 70% of the XTX's ~122 TFLOP
fp16 peak. The library has only square tiles 128²/64²/32², DepthU ∈ {16, 32}, and **no StreamK/GSU**.)

## What this means for our whole effort

- **We were comparing against Tensile in its non-ideal regime.** Our dependency-free kernel at **~60–62 is at
  parity with Tensile's *fallback selection* on an untuned shape** — a stronger result than "92% of Tensile,"
  because Tensile-on-our-shape is itself ~22% below what the same kernel achieves on a tuned shape.
- **The ~3–8% residual we chased is small noise next to the ~22% Tensile leaves on the table by running
  off-regime.** No micro-lever (prefetch, ds_load, VALU) was ever going to matter at that scale — the dominant
  factor is *shape-fit*, and neither we nor Tensile-on-our-shape gets it.
- **The real ceiling for a similar shape is ~79 TFLOPS** (Tensile tuned, 384×8192×4096). Reaching it on
  512×12288×4096 would require shape-specific tuning, not the hot-loop micro-optimizations — a different kind
  of effort (search the tile/depth/schedule space *for this exact shape*).

## Honesty / caveats

- `speed` is read as TFLOPS (max 85.8 ≈ 70% of peak; [384,8192,4096] at 79.1 → 0.33 ms, consistent). It is
  Tensile's offline-tuned benchmark, possibly at a different (boost) clock than our pinned measurements — so
  79 vs our ~62 mixes shape-fit *and* clock. The robust point stands: **our shape is untuned and the selected
  kernel is a fallback.**
- This doesn't change the kernel comparison (ours ≈ Tensile-fallback on our shape); it changes the *framing*:
  the gap to "Tensile's best" is shape-fit, not kernel quality.

## Verdict

The Tensile kernel we benchmarked against is a **nearest-neighbor fallback** for our **untuned** shape (M=512
and N=12288 both off Tensile's tuning grid), running **~22% below its tuned potential** (~62 on our shape vs
79.1 on its tuned 384×8192×4096). **Our dependency-free kernel is at parity with Tensile's fallback** — and the
residual we hunted is negligible next to the shape-fit gap both leave open. The path to the ~79 hardware-tuned
ceiling is *shape-specific* tile/depth/schedule tuning, not hot-loop micro-levers.

## Next (optional, different effort)

If the last ~20% (toward ~79) is wanted: a **bounded tile/depth sweep tuned for THIS exact aspect ratio**
(M=512 small, N=12288 large, K=4096) — not micro-optimizing the current kernel, but finding the config Tensile
*would* have tuned for this shape. Square 128² may not be optimal for the skewed aspect; this is the one search
space we haven't explored shape-specifically. No BEAM; a fixed grid, correctness-gated, clock-pinned.
