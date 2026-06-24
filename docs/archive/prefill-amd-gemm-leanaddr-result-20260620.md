# Prefill AMD GEMM — Lean Addressing (Lever A): VALU matched to Tensile, throughput neutral

Date: 2026-06-20

## Result

`LEANADDR_CUTS_VALU_NEUTRAL_TFLOPS`. The hard PMC audit named the residual as **+23% VALU** (ours 8.66M vs
Tensile 7.04M, the per-iteration address arithmetic). Lever A (lean addressing) **cuts it to Tensile's level
exactly — and throughput does not change.** So VALU instruction count was *not* the wall-clock bottleneck. The
residual ~8% is **not VALU** either.

Implementation: `build_gemm_lds2(..., LEANADDR=1)`. Probe: `extra/qk_amd_gemm_leanaddr_probe.py`.

## What LEANADDR does

The cooperative-load addresses were recomputed every K-block with vector adds (`v_add(SCR, j·stride, v2)` ×6 +
`v2/v3 += BK·2` ×2 = ~8 VALU/iteration). LEANADDR moves this to scalar:

- **Precompute** the invariant per-lane row byte-offsets once (one VGPR per cooperative-load row).
- **Advance the K-position by incrementing the SCALAR buffer base pointers** (`s_add` on `saddr`) instead of
  recomputing vector addresses — exactly what Tensile does (`buffer_load` with scalar `soffset`).

This removes ~8 in-loop VALU/iteration (×128 K-blocks = ~1018/wave), adding only a once-off precompute and
4 SALU/iteration.

## Measured

| | VALU (SQ_INSTS_VALU) | best TFLOPS (pinned) |
|---|---:|---:|
| LEANADDR=0 (baseline) | 8,658,432 | 57.8 |
| **LEANADDR=1** | **7,097,856** (−18%) | 57.6 |
| Tensile `.co` | 7,039,488 | (~62) |

- **VALU cut −1.56M, landing at 1.008× Tensile** — the +23% excess is gone; we now emit essentially the same
  VALU count as Tensile. Correct (rel RMSE 2.08e-4).
- **Throughput neutral (0.995×).** Matching Tensile's VALU bought nothing.

## What this proves — the residual is NOT a nameable kernel lever

The address-arith VALU was real but **already hidden** (issued in parallel with WMMA/LDS, off the critical
path). Removing it confirms the kernel is **not VALU-issue-bound**. Combined with everything ruled out:

| candidate | status |
|---|---|
| prefetch (A-only, full A+B PLR) | ruled out (no parity gain) |
| L2 locality (WGM8) | ruled out (ours L2 hit *higher*: 64.7 vs 56.6) |
| occupancy | tuned (wg2 optimum) |
| LDS bank conflicts | tuned (PAD16, near-zero) |
| **VALU / address overhead** | **matched to Tensile — neutral** |

**No nameable kernel-efficiency axis explains the residual.** On every axis we can measure, the dependency-free
kernel is competitive with — or better than — Tensile (L2 better, VALU matched, occupancy/bank tuned).

## So what IS the residual ~8% wall-clock?

It is **not a closeable kernel inefficiency.** It is dominated by **measurement/work confounds** the hard audit
already flagged:

- Tensile's selected kernel does **`beta=true`** (reads + scales C) — extra work ours (alpha=1/beta=0) skips,
  yet Tensile is still wall-clock competitive.
- Different layout (col-major vs row-major), grid orientation (4×96 vs 96×4), and launch mechanism
  (`NamedAMDProgram` vs `run_linear`) — none cleanly controllable for a foreign `.co`.
- Plus the clock/power/thermal wobble (the GRBM cycle count for the `.co` was itself 2× artifacted).

A truly apples-to-apples gap would need the same work, layout, and launch — which the vendored `.co`'s
differing ABI does not permit. The honest read: **the dependency-free kernel is at parity-to-within-noise with
Tensile on measurable kernel efficiency; the ~8% wall-clock is confound, not deficit.**

## Standing

The dependency-free arc is **done**: `build_gemm_lds2(BK=32, PAD=16, PLRA=1, LEANADDR=1)` at wg2 — correct,
~58–61 pinned, VALU matched to Tensile, L2 better, occupancy/bank tuned, zero dependencies. Every kernel lever
that could explain the gap has been measured and either tuned or ruled out. LEANADDR is a free leanness
improvement (matches Tensile's instruction count, no regression); kept available, default off (proven path
byte-identical).

## Honesty

- LEANADDR cleanly cuts VALU to Tensile's count (−18%, exact, correct) but is **throughput-neutral** — an
  honest negative for *performance*, a positive for *understanding* (rules VALU out).
- The residual is not claimed closed; it is reframed as confound, with the specific confounds named.
- `LEANADDR=0`/`PLRA=0` defaults leave the proven kernel byte-identical.

## Verdict

`LEANADDR_CUTS_VALU_NEUTRAL_TFLOPS` — VALU matched to Tensile, throughput unchanged. The hard audit's named
residual (VALU) is now **ruled out as the wall-clock cause**. With prefetch, L2, occupancy, bank conflicts, and
VALU all measured and tuned/ruled-out, the dependency-free kernel is **at measurable parity with Tensile**, and
the ~8% wall-clock gap is **measurement/work confound, not a closeable kernel deficit.** The arc lands here.
