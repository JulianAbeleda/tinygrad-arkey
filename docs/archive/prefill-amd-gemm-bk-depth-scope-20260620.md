# Prefill AMD GEMM BK-Depth Push — Scope

Date: 2026-06-20

## The one question this answers

K-block depth (BK16→BK32) lifted the dependency-free LDS GEMM from ~41 to ~55 TFLOPS. **Does pushing BK
deeper (64/128) clear the 60-TFLOPS Tensile-class bar?**

**Answer: NO — the depth lever is exhausted.** Verdict `BLOCKED_GEMM_BK_DEPTH_NO_IMPROVEMENT`. `BK=32` at the
128×128 / 4-wave tile (~55.5 TFLOPS, reaching the LLVM authority ~53) is the family **sweet spot**; every
config with deeper BK either does not build or **regresses**. The dependency-free family plateaus at ~55,
~92% of the way to 60.

Bounded enumerated depth ladder, resource-fit pre-checked, correctness-gated, interleaved one-clock — **no
BEAM/search, no routing/default change**.

## Deliverables

| artifact | role |
|---|---|
| `extra/qk_amd_gemm_bk_depth_probe.py` | depth ladder, VGPR/LDS fit pre-check, correctness gate, interleaved timing vs authority |
| `bench/amd-broad-backend-roadmap/amd_gemm_bk_depth_result.json` | result (`bench/**` gitignored, reproducible) |

```bash
PYTHONPATH=. python3 extra/qk_amd_gemm_bk_depth_probe.py
```

## Results (best-of-N TFLOPS, authority shape, all correct rel RMSE 2.1e-4)

| config | BK | tile | thr | LDS B | VGPR | best | median |
|---|---:|---|---:|---:|---:|---:|---:|
| **W2×2 T4×4 BK32** | 32 | 128×128 | 128 | 16384 | 234 | **55.5** | 51.2 |
| authority_llvm | — | — | — | — | — | 53.3 | 52.0 |
| W2×4 T4×4 BK32 | 32 | 128×256 | 256 | 24576 | 226 | 49.0 | 47.7 |
| W2×2 T4×4 BK16 | 16 | 128×128 | 128 | 8192 | 218 | 42.0 | 38.5 |
| W4×2 T4×4 BK64 | 64 | 256×128 | 256 | 49152 | 250 | 39.5 | 37.9 |
| W2×4 T4×4 BK64 | 64 | 128×256 | 256 | 49152 | 250 | 39.3 | 38.1 |
| W2×4 T4×2 BK64 | 64 | 128×128 | 256 | 32768 | 154 | 32.1 | 31.4 |
| W2×2 T4×2 BK64 | 64 | 128×64 | 128 | 24576 | 170 | 31.2 | 28.2 |
| W2×2 T2×2 BK64 | 64 | 64×64 | 128 | 16384 | 106 | 25.2 | 23.2 |
| W2×2 T2×2 BK128 | 128 | 64×64 | 128 | 32768 | 138 | 24.6 | 22.5 |
| W2×2 T4×4 BK64 | 64 | 128×128 | 128 | — | **266** | **UNBUILDABLE** (VGPR overflow) | |
| global_direct (calib) | — | — | — | 0 | — | 29.0 | 26.4 |

## Why depth is exhausted (the structural wall)

`build_gemm_lds2` has a hard VGPR budget (256). Deeper BK means:
- More K-substeps per LDS load → more **cooperative-load temp VGPRs** (`CTA/CTB` scale with `loads = tile_dim
  / RSTRIDE`, and `RSTRIDE = THREADS / (BK//8)` *shrinks* as BK grows). At BK64 on the 128×128 tile this hits
  `SCR=266 > 256` → **unbuildable**.
- To make BK64 fit you must spend VGPRs *somewhere else*: shrink the tile (fewer `WM*WN` accumulators) or add
  threads (256 → larger RSTRIDE → fewer load temps). **Both regress**:
  - **Smaller tile** (T2×2, 64×64) collapses WMMA reuse per cooperative load → 25 TFLOPS.
  - **256 threads / bigger tile** (W2×4) trades occupancy and reads ~49 (BK32) / ~39 (BK64) — below the 128×128
    BK32 55.

So BK32 sits at the **reuse↔register sweet spot**: enough K-depth to amortize the barrier/staging cost, while
still affording the 16-accumulator 128×128 tile that maximizes WMMA reuse. Deeper BK can only be bought by
giving up the reuse or occupancy that made BK32 fast — a strict tradeoff on RDNA3's 256-VGPR file, not a
tunable that keeps paying.

## Significance

- The dependency-free hand-asm frontier **rests at BK32 ~55 TFLOPS**, reaching the LLVM authority (~53),
  ~1.9× global-direct, correct — confirmed again here (BK32 = 55.4/55.3/55.5 across all runs).
- It does **not** reach Tensile-class (≥60; Tensile ~66) and **depth is not the lever to get there** — the
  remaining ~10–20% is gated by the VGPR budget / occupancy, not K-block depth.
- This bounds the dependency-free ceiling for this kernel family: **~55, not 60**, via the global-direct-WMMA
  + LDS-staging approach on RDNA3.

## Honesty boundaries

- Single prefill shape (512×12288×4096); best-of-N (median has BK32 51 ≈ authority 52, parity).
- Bounded grid (10 depth/tile/wave configs + authority + global-direct) — **not BEAM**; a fundamentally
  different register-allocation or instruction schedule (not in this family) could move it, but that is the
  Tensile-class codegen wall, out of scope here.
- Clock-volatile absolute TFLOPS; ratio + power-witnessed activity (median ~50 W) is the trust basis.

## Verdict

`BLOCKED_GEMM_BK_DEPTH_NO_IMPROVEMENT` — deeper BK did not beat the BK32 frontier; BK32/128×128 is the family
sweet spot at ~55, reaching the authority but short of 60. The depth lever is exhausted.

## Next (bottleneck classification, NOT more depth/search)

The honest next step is **not** another config sweep. It is **bottleneck classification of the BK32 winner**:
why is it ~55 and not ~66 (Tensile)? Candidate levers, each its own gate:

1. **VGPR/occupancy** — BK32 uses 234/256 VGPR at occupancy ~1–2 waves/SIMD; quantify the occupancy and
   whether a register-lighter schedule (not deeper BK) lifts it.
2. **PMC/rocprof** the BK32 kernel (VALU vs WMMA-issue vs LDS-wait vs barrier stall) to name the actual stall,
   per the measured-map discipline — is it barrier-bound, issue-bound, or VGPR-occupancy-bound?
3. Only a finding there justifies more kernel work; otherwise the dependency-free frontier is **~55 / ~85% of
   Tensile**, and the only path to ~66 stays the vendored Tensile `.co` (declined) or a deeper codegen
   capability (the standing wall).
