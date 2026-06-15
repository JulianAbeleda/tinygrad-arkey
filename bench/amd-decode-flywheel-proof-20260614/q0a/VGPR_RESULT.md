# VGPR/occupancy measurement (2026-06-15): REFUTES the occupancy hypothesis. Measured, not inferred.

Per the rebase ("make sure it's a measurement problem"), measured the REAL VGPR/SGPR/LDS of the three
decode GEMV kernels (extracted Ops.BINARY from to_program, read .vgpr_count from the ELF AMDGPU notes):

  kernel        VGPR  SGPR  LDS(B)  e2e tok/s
  fp partial     47    18     0      58
  int-dot        68    51     0      28
  coop fused     93    52   4608     24

## The refutation
RDNA3 gfx1100: VGPR file ~196608 B/SIMD32, max 16 waves/SIMD -> full occupancy needs <= 96 VGPRs/wave.
ALL THREE kernels are <= 96 (47, 68, 93) -> ALL achieve full 16-wave occupancy. So occupancy is
IDENTICAL across fp/int-dot/coop and is NOT the differentiator. The "int-dot loses e2e because register
pressure -> low occupancy" hypothesis is REFUTED by measurement -- the exact misleading-occupancy trap
the autotuning literature (DATE'16, arXiv:1701.08547) warned about. Do NOT build a register-light DP4A
kernel on the occupancy theory.

## What the data points to instead (the corrected lever)
VGPR correlates with e2e (47->58, 68->28, 93->24) but NOT via occupancy (all full). The remaining
mechanism consistent with M0 + the literature is INSTRUCTION COUNT: the int-dot does MORE work per
weight than fp -- the int dot PLUS a separate qsum reduction PLUS per-group int->float affine. So even
at equal occupancy it is more instruction-bound. The standalone microbench (242 GB/s, tight loop that
overlaps iterations) OVERSTATED it; the e2e 136 GB/s (instruction-bound, no same-kernel loop to hide
the extra instructions) is the truth. fp's per-weight (dequant + one FMA) is SIMPLER = fewer
instructions = faster e2e.

## Implication (pragmatic, not a conclusion)
The lever is instruction count per weight, NOT occupancy. To beat fp you need GENUINELY fewer
instructions/weight than fp's dequant+FMA -- which is what llama.cpp's DP4A does (one v_dot4 = 4 MACs,
and packing folds the qsum). Our int-dot ADDED instructions (the qsum), so it lost. The next test that
would actually matter is INSTRUCTION COUNT (not VGPR): does a real DP4A-packed kernel emit fewer
instructions/weight than fp, measured? D0's explicit v_dot4 was inline-asm that blocked optimization
(slowest) -- so DP4A via the CUSTOMI escape hatch is not the way; it would need a real codegen lowering
so the compiler schedules it. So: the measurement saved a wasted register-light build, and re-pointed
the lever at instruction count -- which loops back to the DP4A-as-a-real-primitive question (Phase D),
now correctly motivated (fewer instructions, not occupancy).
