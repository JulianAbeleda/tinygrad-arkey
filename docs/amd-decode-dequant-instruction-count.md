# Phase Q — reduce the dequant instruction count (the triangulated #1 decode bottleneck)

Date opened: 2026-06-15
Goal: attack the bottleneck we triangulated (our M0 + the literature): the decode GEMV is
"memory-starved but INSTRUCTION-bound — the bottleneck is how many instructions the GPU chews through
between memory transactions" (the Q4_K dequant ALU, ~3862 vector ops/kernel, M0). The field's named
fixes all reduce dequant instructions per weight. Two scoped here, with honest placement of each.

## What we already know (so we don't re-probe blind)

- DP4A (D0): wrong axis (compute). Explicit `v_dot4` was the slowest variant.
- Latency-hiding (L0): not the constraint; forcing occupancy regresses.
- **The int-dot path (D0 `intdot`, = llama.cpp `mmvq` structure) ALREADY captures most of the dequant
  reduction**: keep weights int4, dot with int8 (q8_1) activations in int32, apply the affine ONCE
  PER GROUP (no per-weight fp dequant). It microbenched **242 Q4-GB/s on ffn_gate** (vs fp 173, +40%)
  -- BUT end-to-end it regressed to 28 tok/s because the per-layer q8_1 activation quant was an
  UNFUSED extra kernel whose batch-1 launch overhead dominated. So the ~81 tok/s ceiling that
  microbench implies is REAL but UNCAPTURED end-to-end.

## The two techniques (honest placement)

**Technique A -- fuse the q8_1 quant into the int-dot GEMV (the concrete, proven-in-microbench win).**
Not new math -- it is making the int-dot path D0 already validated actually pay off end-to-end:
- A1: fuse the activation->q8_1 quantization INTO the int-dot GEMV kernel (one launch, not two), so
  the quant overhead that killed D0's end-to-end disappears.
- A2: quantize the shared activation ONCE per layer and reuse across the linears that read it (q/k/v
  share attn-input; gate/up share ffn-input) -- amortize the quant across 2-3 GEMVs.
Pre-registered: fused int-dot reaches ~75-81 tok/s -> a real +40% decode win (58->81, 56%->78% of
llama.cpp), captured. If it stays ~58 or regresses -> the int-dot does not survive fusion either.

**Technique B -- LUT-GEMM dequant (the field's named #1, but with a real GPU caveat).**
Per group (32 weights, nibble q in 0..15), precompute a 16-entry table `lut[q] = (d*sc)*q - dmin*mn`;
each weight is `lut[nibble]` -- a lookup instead of the per-weight affine. HONEST caveat: on GPU a
per-thread runtime-indexed LUT cannot live in registers (no indexed register-file access), so it must
go in LDS (load latency + LDS pressure) or become a 16-way select tree (instructions -- defeats the
purpose). AND the int-dot path (A) already AVOIDS per-weight fp dequant entirely, so LUT optimizes a
path int-dot dominates. Pre-registered SKEPTICISM: LUT likely does NOT beat the fused int-dot (A); it
is scoped as a probe, not an assumed win.

## Phases (cheap make-or-break first; A before B)

**Q0a -- fused int-dot probe (do FIRST; most concrete).** Build the one-kernel fused
quant+int-dot decode GEMV (A1; optionally A2), wire into `tinygrad/llm/model.py` behind a flag,
measure end-to-end decode tok/s vs 58 (fp) and llama.cpp 104. Correctness-gated (q8_1 changes numerics
slightly, as in llama.cpp). Gate: >= ~75 -> capture it (productionize); ~58 or worse -> record why the
fusion does not pay off.

**Q0b -- LUT probe (only if informative).** Implement the per-group 16-entry LDS LUT dequant; measure
instruction count (vs the ~3862 baseline) AND end-to-end tok/s vs Q0a. Gate: LUT > Q0a -> real
additional lever, pursue; LUT <= Q0a -> the dequant-reduction win is captured by int-dot, LUT is not
additive on this GPU (the register-indexing caveat bites) -> record and drop.

**Q1 -- productionize the winner.** Wire the best into the decode path; correctness-gate; measure vs
llama.cpp; update the policy.

## Pre-registered ceiling + honesty

- Even the best dequant-reduction (int-dot) microbenched ~81 tok/s (~78% of llama.cpp). So Q targets a
  real +40% (58->81) that is currently UNCAPTURED, but it is NOT expected to reach PARITY (104). The
  residual 81->104 is most consistent with hand-asm efficiency (instruction scheduling), the strided
  Q4_K activation access (#2 bottleneck), and kernel-count/fusion (#3) -- NOT instruction-count alone.
- A located ceiling ~81 (~78%) with a captured +40% is a real, valuable result (the roofline
  discipline). Parity likely needs the hand-written-kernel / Writer direction the search philosophy
  avoids.
- Touch points: `extra/q4_k_gemv_primitive.py` (`q4k_q8_1_intdot_partial_kernel` + a fused-quant
  variant; a LUT variant), `extra/qk_layout.py` (`q8_1_quantize`), `tinygrad/llm/model.py`
  (`Q4KPrimitiveLinear.__call__` -- the decode GEMV dispatch + activation-quant fusion/reuse).
- This is the field-validated lever for the bottleneck we triangulated; if even it ceilings ~81, that
  is the honest end of the codegen-reachable decode story, and the gap to llama.cpp is hand-kernel
  efficiency, not a tinygrad vocabulary or a missed optimization.
