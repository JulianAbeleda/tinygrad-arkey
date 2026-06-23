# Prefill Register-Lifetime / Pool Representation — Result (2026-06-23)

> **⚠ SUPERSEDED / RETRACTED (2026-06-23):** the `REGISTER_POOL_INSUFFICIENT_HW_LIMIT` verdict below is RETRACTED by
> `docs/prefill-adversarial-tensile-liveness-audit-result-20260623.md`. Tensile fits a pipelined 128×128 GEMM in 256
> VGPR, and `build_gemm_lds2` CAN express the deep pipeline via the 8-wave `WAVES_M=4,WAVES_N=2,WM=2,WN=4` layout (188
> VGPR, PIPELINED, correct). The prior "266>256 hardware ceiling" was an artifact of the 4-wave 4×4 layout. The real
> residual is fine instruction scheduling, not register pressure. (History preserved below.)



## Verdict: `REGISTER_POOL_INSUFFICIENT_HW_LIMIT` — prefill speed search definitively closed
The register-lifetime VGPR pool **does not unlock** the deep (A+B) software-pipelined K-loop at the occupancy-optimal
WMMA tile. The pool concept is **real and already realized** for the A-side (`PLRA`, the shipped +9-11 %). Full A+B
prefetch is **hardware-register-limited**: even an ideal liveness pool needs ~266 VGPR (> 256) at the 4×4 tile. So the
remaining ~4–5 % to Tensile is a **hardware register-pressure / occupancy ceiling**, not a missing software
representation. Stopped before building the allocator (per the scope's Phase-1 branch). No source/route/default/speed
change.

## 1. Authority — reproduced
`DBUF=1` PASS (PIPELINED, ~236 VGPR, correct); `PLRAB`(4×4) WALL (VGPR 300 > 256). Per-region VGPR layout (WM4×WN4):
reserved 0-9 · A-frag FA 10-41 (32) · B-frag FB 42-73 (32) · **ACC 74-201 (128)** · coop-temps CTA/CTB 202-233 (32) ·
SCR 234 · FB2(PLRAB) 236 → static PLRAB 300.

## 2. Liveness model (`liveness_model.json`) — the decisive analysis
**Live throughout the compute + prefetch window:**
- **accumulators** = `WM·WN·8` = **128 regs** (whole K-loop — the floor);
- **current A/B fragments** (FA,FB) = **64 regs** — live *until the WMMA completes* (RDNA3 WAR hazard: you cannot
  overwrite the WMMA inputs at issue, they must hold until the op retires);
- reserved 10.
**Dead during compute:** the coop-load temps (CTA/CTB) = **32 regs**. These are the *only* reusable registers — and
this **is** the pool. **`PLRA` already reuses exactly these 32 for the A-side prefetch** (the author's comment:
*"prefetch substep1's A fragments into the DEAD coop-load temp regs, register-lifetime overlap à la Tensile's pool …
Partial PLR (A only; B' wouldn't fit 256)"*). The achievable pool gain is already shipped.

**Ideal-pool full A+B (4×4):** A-prefetch reuses the 32 dead coop-temps (free); the B-prefetch needs **32 new** regs at
indices 234–265 → **max-live ≈ 266 > 256** (over by ~10). Squeezing to exactly 256 would leave 1 wave/SIMD —
**occupancy collapses**, so it would not help speed even if it fit.

## 3. Smaller-tile probe (ruling out the tile-reduction path)
`PLRAB` *builds* at WM4×WN2 (acc floor 64) and WM2×WN2 (acc 32) and is numerically correct (rel_rmse 2.05e-4) — but
the detector classifies them **PHASED** (0/6, 0/4 global loads in the wmma span). Reason: `PLRA/PLRAB` pipelines the
**substep LDS reads**, not the **block-level global loads** (only `DBUF` moves global loads into the span, and `DBUF`
regresses via LDS-doubling). And the smaller N-tile trades occupancy — a Phase-B-closed tradeoff for the *already
well-occupied* down/qo roles. **Tensile does both levels** (3/4 global + 76/76 ds in the span); matching both within
RDNA3's 256-VGPR limit at the occupancy-optimal tile is what the hardware does not allow.

## 4. Answers to the scope's required questions
1. **Deep pipeline fits ≤256?** No at the optimal 4×4 tile (ideal-pool 266). Yes only at smaller tiles, which are
   PHASED-on-globals + occupancy-trading.
2. **Correctness / interleave / max-live?** Smaller-tile PLRAB correct (2.05e-4) but PHASED; 4×4 ideal max-live 266.
3. **Does prefill schedule search reopen?** **No.** The pool is HW-register-limited; the A-side pool gain is already
   shipped (PLRA). `PREFILL_FULL_SPEED_SEARCH_STILL_DEFERRED` hardens to `REGISTER_POOL_INSUFFICIENT_HW_LIMIT`.

## 5. The definitive prefill conclusion
The whole arc — *Tensile wins ~4–5 %* → *K-loop software pipelining + register lifetime* → *schedule template
emittable* → **this** — terminates here: the residual is a **hardware register-pressure / occupancy ceiling** at the
occupancy-optimal WMMA tile, **not** a missing or searchable software representation. The register-pool the analysis
named is already captured (PLRA, A-side); the B-side cannot be added within 256 VGPR without collapsing occupancy.
Closing the last ~4–5 % requires **vendored Tensile** (declined) or a smaller-tile occupancy tradeoff (Phase-B-closed,
non-transferring for well-occupied roles). **Prefill is at ~96 % of Tensile / at-or-above llama, and that is the
RDNA3 ceiling for a dependency-free kernel.**

## 6. Defaults / routes changed? — NONE
No `tinygrad/` source, no model route, no default flip, no whole-prefill speed claim, no allocator built (stopped per
the scope's Phase-1 branch since ideal-packed max-live > 256). `build_gemm_lds2` and the shipped route are untouched.

## Files changed
New: this doc + 3 artifacts under `bench/qk-prefill-register-lifetime/` (authority, liveness_model, decision) + 1
project-ledger entry (now 35). The probes were ephemeral (no new tool committed — the analysis is the deliverable).

## Git status
Clean before; adds 1 doc + 3 artifacts + 1 ledger line. Defaults unchanged.
