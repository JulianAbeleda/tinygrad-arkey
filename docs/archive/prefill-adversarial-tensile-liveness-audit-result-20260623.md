# Prefill Adversarial Tensile-Liveness Audit — Result (2026-06-23)

## Verdict: `CURRENT_LDS2_REPRESENTATION_EXHAUSTED_TENSILE_PATH_UNRESOLVED` + `PREFILL_TENSILE_LIKE_PATH_REQUIRES_ASM_ALLOCATOR` — and a **retraction**
The adversarial reconciliation against Tensile **overturns my prior "hardware register-pressure ceiling."** Tensile
fits a pipelined 128×128 GEMM in **exactly 256 VGPR**, and `build_gemm_lds2` **can express the matching deep pipeline**
(8-wave layout, 188 VGPR, PIPELINED, correct) — so the "HW limit" was an artifact of analyzing the wrong wave layout.
But matching the pipeline *structure* recovers only **+0.5 % whole-prefill**, leaving ~4 % to Tensile. The true
residual is **fine-grained instruction scheduling**, below the `build_gemm_lds2` template. No default flip, no
whole-prefill speed claim, no vendored-Tensile promotion.

## ★ Retraction
`prefill-register-lifetime-pool-representation-result-20260623.md`'s **`REGISTER_POOL_INSUFFICIENT_HW_LIMIT`** is
**RETRACTED**. It concluded "266 > 256 = hardware ceiling" from `build_gemm_lds2`'s **4-wave 4×4** layout, whose
128-reg accumulator floor is **layout-specific, not fundamental**. The hardware does *not* impose that ceiling.

## 1. Tensile's real envelope (`tensile_envelope.json`)
From the `.co` (6528 kernels): the `Cijk_…MT128x128x16_MI16x16x16x1` GEMM kernels use **VGPR = 256, SGPR = 58,
scratch = 0** (162 of them), and the **MAX VGPR across *all* Tensile kernels is 256**. So Tensile achieves a
pipelined (PGR1/PLR1, DepthU=16) 128×128 macro-tile GEMM **within 256 VGPR** — the hardware allows it.

## 2. Why the prior claim was wrong (`liveness_revision.json`)
Tensile uses **256 threads (8 waves)** per 128×128 tile → **half the per-wave accumulator footprint** vs
`build_gemm_lds2`'s 4-wave 4×4 (128-reg accumulators). The prior liveness model used the 4-wave layout and concluded
"only 32 dead regs, B-prefetch overflows to 266." That floor is **not** fundamental — a different wave layout halves it.

## 3. `build_gemm_lds2` *can* express the deep pipeline (the alternative path)
The 8-wave layout `WAVES_M=4, WAVES_N=2, WM=2, WN=4` → 128×128 tile, **acc = 64** (half). Measured on a microkernel:
- **`PLRAB` (full A+B prefetch) fits at est. 188 VGPR** (vs the 4-wave 4×4's 300 wall), correct (rel_rmse 2.1e-4);
- **`DBUF + PLRAB` together → PIPELINED** (8/16 global loads + ds in the wmma span — *both* pipeline levels), 188 VGPR,
  correct.
So the deep, double-level pipeline is expressible in `build_gemm_lds2`'s **existing knobs** — `REGISTER_POOL_HW_LIMIT_RETRACTED`.

## 4. But it does not close the gap (`whole_prefill_transfer.json`)
Routed via an additive `PREFILL_GEMM_8WAVE` flag (default off), clock-pinned synced whole-prefill, 2 rounds:
| ctx | default | 8-wave deep pipeline | Δ |
|---|---|---|---|
| 512 | 3544/3540 | 3559/3560 | **+0.5 %** |
| 1024 | 3459/3460 | 3469/3470 | **+0.3 %** |
The 8-wave config is still **~3.9 % below Tensile (~3705)**. So matching Tensile's **pipeline structure** (8-wave,
DBUF + PLRAB, fits 256, PIPELINED) recovers almost nothing — **the residual is not pipeline depth or register pressure.**

## 5. The reconciled residual (`revised_decision.json`)
What's left is **fine-grained instruction scheduling** that the high-level `build_gemm_lds2` template does not express:
`s_waitcnt` placement (consumer-only), **WGM8 L2-locality traversal**, `v_wmma` issue cadence, and the exact
load/compute interleave timing (Tensile's SIA1 scheduling). Closing the last ~4 % needs an **asm-level instruction
scheduler** — *below* the current template abstraction — not register pooling, not a different tile, not a bounded
config knob. `PREFILL_TENSILE_LIKE_PATH_REQUIRES_ASM_ALLOCATOR` (precisely: an asm *scheduler*).

## 6. Honest standing
- **Hardware ceiling: refuted.** The deep pipeline fits 256 VGPR (Tensile proves it; we reproduce the structure).
- **Pipeline structure: matched, but it's not the gap.** +0.5 % only.
- **Real residual: fine instruction scheduling, asm-level.** Not searchable with the current representation; the
  `schedule_template` representation captures *which loads are in the span* but **not the cycle-level cadence/waitcnt**
  that Tensile's hand-tuned assembly schedules. That is the genuine next capability — and it is a deterministic
  asm-scheduler build, not a search.
- Prefill stays at ~96 % of Tensile / at-or-above llama; the gap is now correctly attributed to instruction-scheduling
  granularity, **not** a hardware register limit.

## Files changed
Modified (additive, default off): `extra/qk_prefill_graph_gemm_route.py` (`PREFILL_GEMM_8WAVE` flag + `PLRAB`
threaded into the builder call — research lever, +0.5 %, not shipped). New: this doc + 4 artifacts under
`bench/qk-prefill-adversarial-tensile-liveness/` + 1 ledger entry (now 36). **No `tinygrad/` source, no default flip,
no whole-prefill speed claim.** The prior register-lifetime result is superseded (retraction noted), not rewritten.

## Git status
Clean before; adds 1 doc + 4 artifacts + 1 ledger line + the additive route flag. Defaults unchanged.
