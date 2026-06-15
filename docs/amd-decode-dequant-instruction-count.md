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

### Q0a -- concrete implementation scope (the build)

Back-of-envelope first (sets the design): per layer there are 7 decode GEMVs. The fp path = 7 dot
launches. The int-dot path is 1.4x faster PER KERNEL (microbench 242 vs 173 Q4-GB/s on ffn_gate) but
needs q8_1-quantized activations. If the quant is a SEPARATE launch:
- naive (D0): 7 dots + 7 quants = 14 launches -> ~28 tok/s (measured). Quant launches dominate.
- quant-once-per-shared-activation (q/k/v share attn-input; gate/up/down share ffn-input): 7 dots + 2
  quants = 9 launches. If a small quant ~ a small dot in launch-bound cost: time ~ (7/1.4 + 2) = 7
  units vs fp 7 -> ~BREAK-EVEN (~58-65). The 2 extra quant launches eat the 1.4x dot speedup.
- FUSED (quant inside the dot kernel, no extra launch): 7 launches, each 1.4x faster -> ~81. THIS is
  the only design that captures the ceiling.

So Q0a has two steps, cheap-confirm then real-build:

**Q0a.1 -- quant-once-reuse (cheap confirm, ~30 min).** Restructure `Q4KPrimitiveLinear` dispatch in
`tinygrad/llm/model.py` so the activation is `q8_1_quantize`d ONCE per shared group (attn-input,
ffn-input) and the 2-4 sharing linears reuse (q, scales) via the existing
`q4k_q8_1_intdot_partial_kernel`. Measure end-to-end tok/s. Expectation (pre-registered): ~58-65 --
confirms the int-dot is faster per-kernel but the separate quant launch caps it. This is a cheap
sanity check of the math, not the win.

**Q0a.2 -- LDS-fused quant+int-dot kernel (the real build).** New
`extra/q4_k_gemv_primitive.py::q4k_q8_1_fused_intdot_kernel(rows,k,parts,...)`: ONE kernel taking
`(partials, words, x_fp16)`:
- Phase 1 (once per workgroup): each workgroup cooperatively loads `x` (k fp16), computes the per-32-
  block scale (`max|x|/127`), quantizes to int8, and stores `q8[k]` + `scales[k/32]` into LDS
  (`DEFINE_LOCAL`). `barrier()`.
- Phase 2 (per row): the existing int-dot body (`_q4k_group_dot_q8_1_intdot`) but reading `q8`/`scales`
  from LDS instead of global -- int4xint8 dot in int32, affine per group, accumulate.
One launch; the quant is amortized across the workgroup's rows (in LDS) with NO extra kernel. LDS
budget: `q8[4096]=4KB + scales[128]*4=512B` -- fits. Wire into `model.py` behind a flag.
Correctness-gate vs the fp matmul (`rel_err < 1e-2`; q8_1 changes numerics slightly, as in llama.cpp).
Pre-registered gate: end-to-end reaches ~75-81 -> CAPTURE the +40% (productionize, update policy);
~58-65 -> the in-kernel quant ALU offsets the dot win (record); <58 -> fusion doesn't pay off here.

Risks: (1) the in-kernel per-32-block max-reduction + quantize adds ALU to phase 1 -- but it is
amortized across rows and replaces a whole kernel launch; (2) phase-1/phase-2 barrier + LDS layout
must be correct (the W1b'/M0 lessons: stage once, don't serialize); (3) each workgroup re-quantizes x
(redundant across workgroups) -- acceptable since quant is cheap and launch-free, but if parts is high
the redundancy grows (tune parts).

**Q0a -- RESULT (2026-06-15): FAILED. The int-dot ~81 is a MICROBENCH ARTIFACT; fp (58) stays best.**
`q0a/RESULT.md`. Built the LDS-fused `q4k_q8_1_fused_intdot_kernel` (kept in `q4_k_gemv_primitive.py`
as a documented negative). Correct (rel_err 0.0073, the expected ~0.7% int8 err), but: standalone
microbench 10 Q4-GB/s (~24x SLOWER than the separate int-dot's 242); end-to-end decode 6 tok/s (vs fp
58, D0 separate-quant 28). WHY: the phase-1 quant PROLOGUE is not hoisted -- tinygrad's lowering
replicates it per OUTPUT ROW (12288 for ffn_gate) instead of once per workgroup (the recurring
fused-staging wall: W2 dequant prologue, G0''). So the D0 microbench (242 GB/s, +40%) assumed a FREE
pre-quantized activation; end-to-end the quant must be paid and BOTH strategies lose to fp -- separate
launch (28) or replicated prologue (6). The pre-registered ~75-81 gate FAILS. **fp (58 tok/s, 56% of
llama.cpp) remains the best decode kernel.** Every codegen-reachable decode lever is now NEGATIVE
end-to-end (DP4A/D0, latency/L0, lossy-quant/X0, int-dot/Q0a); the residual is the cross-layer rungs
(Mirage) or the Writer, as the "why" analysis predicted. (Q0b LUT not pursued -- it optimizes a path
the int-dot dominates, and the int-dot itself doesn't pay off end-to-end.)

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

## NEXT (the last percent, 81 -> 104): look into Mirage / multi-level superoptimization

The dequant-reduction levers (int8 int-dot, Q) cap at ~81 tok/s (~78%). The residual 81 -> 104 is the
OTHER cross-layer rungs -- algebraic reformulation, layout co-design, custom-kernel discovery,
instruction selection -- which our schedule-only search and a single dequant kernel do not touch. This
is exactly **Mirage (OSDI'25)** territory: a multi-level superoptimizer that JOINTLY searches algebraic
+ schedule transforms and DISCOVERS new custom kernels across the kernel/threadblock/thread hierarchy,
with probabilistic equivalence verification. Why it is the right tool for the last percent:
- The 81->104 gap is a CROSS-LAYER co-design problem (algorithm x layout x schedule x instructions),
  and Mirage is the one system that searches that joint space rather than one layer.
- Its uGraph hierarchy could find the decode-GEMV reformulations + custom kernels (fused
  norm+dequant+dot, better activation-access layout) that hand-tuned llama.cpp uses but tinygrad's
  schedule search cannot reach.
- It connects to our validated loop: Mirage is "machine search, but cross-layer" -- the higher rung of
  the same thesis we confirmed on the schedule axis (N1/N2).
Cheap make-or-break before adopting: take ONE decode GEMV (or the norm+GEMV pair) and ask whether
Mirage's joint search finds a reformulation/custom-kernel that beats the Q int-dot's ~81 on this GPU.
If yes -> cross-layer search closes the last percent the search way (vindicating the thesis at the top
rung); if no -> the residual is genuinely hand-asm/microarchitectural and parity needs the Writer.
Repo: github.com/mirage-project / arXiv:2405.05751 (OSDI'25). PET (OSDI'21, arXiv) is the companion for
the partial-equivalence / semantics-changing piece. This is the single most promising "close it the
search way" direction and the right thing to study next.
