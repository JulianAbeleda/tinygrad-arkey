# Prefill AMD GEMM — Hard PMC Audit vs Tensile (naming the residual)

Date: 2026-06-20

## Why

Prefetch (A-only *and* full A+B PLR) did not close the last gap to Tensile, so the residual is *not* PLR. This
audit PMCs **both** our best kernel and the vendored Tensile `.co` (both HCQ dispatches → both capturable) to
name the residual at the counter level — replacing the murky, launch-context-confounded wall-clock comparison.

Probe: `extra/qk_amd_gemm_tensile_pmc_audit.py`. Reproduced 3×.

## A confound surfaced first — and was excluded

`GRBM_GUI_ACTIVE` (GPU active cycles) was meant to be the launch-independent "time." But for the Tensile `.co`
it **does not reconcile with its own wall-clock**:

| kernel | GRBM cycles | max TFLOPS implied @2251 MHz | wall measured |
|---|---:|---:|---:|
| ours | ~2.1 M | ~54 | ~61 (≈13% off — normal PMC slowdown) |
| **Tensile `.co`** | **~4.0 M** | **~29** | **~62 (2× off — impossible)** |

Tensile at 62 TFLOPS needs ~1.85 M cycles, but PMC reports ~4.0 M — a **foreign-`.co` PMC-capture artifact**
(the counter window over the `NamedAMDProgram` launch overcounts). **So the cycle / per-cycle-rate comparison
is invalid and is excluded.** (This is itself a finding: don't trust GRBM cycles for foreign `.co` launches.)

## The reliable counters (GRBM-independent, exact, reproduced 3×) name the residual

These don't depend on the artifacted cycle count — `VALU`/`bankconf` totals are exact instruction/event
counts, `L2 hit%` is `GL2C_HIT/(HIT+MISS)`:

| counter | ours | Tensile | reading |
|---|---:|---:|---|
| **VALU instructions (total)** | **8,658,432** | **7,039,488** | **ours +23%** (exact, deterministic) |
| L2 hit % | 64.7 | 56.6 | **ours better** (+8 pts) |
| LDS bank conflicts / active | 2.93 | 0.0 | Tensile perfect; ours small residual |

## What the residual IS — and isn't

- **It is NOT prefetch** (A and A+B PLR ruled out) — confirmed.
- **It is NOT L2 locality** — ours has the *higher* L2 hit (64.7 vs 56.6). Tensile's `WGM8` workgroup remap
  does not give it an L2 edge over us here; if anything ours is better. Rules out a whole hypothesis.
- **It is NOT a clean per-cycle compute deficit** — the cycle comparison is artifacted; on the reliable
  counters ours is competitive-to-better.
- **It IS VALU instruction overhead**: our kernel executes **+23% more vector-ALU instructions** for the same
  FLOP than Tensile — the per-iteration **address/index arithmetic** (cooperative-load index math, fragment
  address recompute, output address arithmetic). Tensile's generated code is leaner (hoisted/strength-reduced
  addressing). This is the single clean, reproducible, named difference.
- **Plus a small residual bank conflict** (ours 2.93/active vs Tensile 0): our PAD16 cut conflicts ~11× but
  not to zero; Tensile's `LdsPadB`/`LdsBlockSizePerPad` layout eliminates them.

## So the lever for the last gap is addressing, not prefetch

The named residual — **+23% VALU from address/index arithmetic** — is exactly the **"Lever A: addressing-mode
lowering"** from the original SW-pipeline charter (hoist loop-invariant address math, strength-reduce the
per-iteration index/pointer updates), now **measured-justified** rather than assumed. Our kernel recomputes
addresses each K-block / each fragment; Tensile precomputes and increments. Cutting that VALU excess is the
dependency-free path to the residual — *not* more prefetch, depth, or occupancy.

## Honesty

- The **cycle/efficiency comparison is excluded** as a foreign-`.co` PMC artifact (2× inconsistency). Only the
  GRBM-independent counters (VALU/bankconf totals, L2 hit) are used — and those are exact and reproduced 3×.
- The two kernels do slightly different work (Tensile `beta=true` reads C; col-major vs row-major; grid
  4×96 vs 96×4) — a real caveat, but the +23% VALU and the L2/bankconf readings are large and consistent
  enough to be the dominant signal, not the work difference.
- The wall-clock "~8% gap" is therefore **partly real (VALU overhead + residual bankconf) and partly
  measurement/work confound** — it does not decompose into a single dramatic kernel deficit.

## Verdict

The residual is **NOT prefetch and NOT L2** — it is **VALU/address-arithmetic instruction overhead (+23%)**
plus a small residual LDS bank conflict. The dependency-free lever is **addressing-mode leanness** (hoist +
strength-reduce the per-iteration index/pointer math), the original "Lever A", now named by counters. That is
the next concrete, bounded experiment if the last few % is pursued — and it is orthogonal to everything tried
so far (occupancy, bank-pad, depth, prefetch).

## Next (one bounded experiment)

Reduce the kernel's per-iteration VALU: precompute loop-invariant fragment/coop-load addresses once, replace
per-block `v_mul`/`v_add` index recomputation with strength-reduced increments, and re-PMC `SQ_INSTS_VALU`
(target: toward Tensile's 7.04 M) + re-time. No BEAM.
