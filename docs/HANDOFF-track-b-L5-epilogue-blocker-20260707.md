# Handoff: Track B — L5 epilogue register-pressure blocker

Date: 2026-07-07. Bank point for the pure-codegen prefill-WMMA-GEMM build (replace the hand asm emitter so
`extra/qk/prefill/wmma.py` can be deleted). PROVENANCE goal, not performance: the `gen_sched` asm substrate already
delivers 5167 tok/s (unpinned) / 4413 (pinned) / ~58-68 eff TFLOPS. This work reproduces that via tinygrad codegen.

## Landed + committed (hardware-gated)
- L7 WMMA emit — 16x16x16 bit-exact (e29e970d6); R1 span-aware scheduler fix.
- K-reduction: unrolled chain (21778c090) + rolled any-K accumulator (8ad8603b8) — 16x16x64 bit-exact.
- Phase-1a fp16 16-bit global access (15ff272a8) — multi-workgroup MMU fault fixed.
- Phase-1b multi-output-tile register model (bbd7c77d1) — per-subtile LOW accumulators, keyed (id(dreg), idx.arg//8).
- Scopes/blueprint: docs/layers-3-7-completion-scope.md, track-b-phases-scope.md, the reverse-engineered hand
  blueprint (246 instr: 32 global_load_b128, 16 v_wmma 4x4, vmcnt(8), cvt+b16 epilogue, NO LDS).

## Validated on hardware
- Tiling WORKS: a 4x4 tile RAN at 10.5 TFLOPS (50x the 0.2 single-tile floor).
- Bit-exact multi-tile up to 2x2 (4 subtiles): 32x16x64, 16x32x64, 32x32x64 all rmse ~0.0016 (bisected).

## THE BLOCKER (L5): full 4x4 tile (16 subtiles) store-epilogue register pressure
64x64x64 (WM=WN=4, 16 subtiles) spills: 128 fp32 accumulator VGPRs + the store epilogue's ~128 store-address VGPRs
(all live at once) + data regs > 256, into a _vpool shrunk to ~104 by the pinned LOW accumulators -> regalloc
NotImplementedError "Inc 0: no spills". 3 approaches FAILED:
1. two-base immediate fold -> NaN (base-register aliasing / in-place +=4096 mutation).
2. single centered-base signed-13-bit immediate -> NaN at 64x64x64 AND fails loud on real shapes (512x4096 output has
   ROW-STRIDED store offsets spanning ~508000 bytes, far beyond any immediate window). Immediate-folding is a DEAD END.
3. Fable-designed epilogue serialization (thread offset_k -> store_{k-1} via an ignored trailing src so the linearizer
   interleaves offset_0,store_0,offset_1,store_1,... -> short live ranges) -> STILL SPILLS. Theoretically sound
   (Fable's linearizer-schedule analysis in tmp/) but did not reduce pressure; mechanism UNCONFIRMED (my diagnostic
   hit invalid forced-opt encodings before the rule fired).

## DRILL-DOWN PLAN (next)
Compile the DEFAULT hand_coded 64x64x64 4x4 AST (which spills) through AMDISARenderer on DEV=PYTHON (the ISA renderer
compiles/assembles WITHOUT a GPU -- see test_amd_isa_wmma.py), and INSTRUMENT regalloc (LinearScanRegallocContext,
codegen/late/regalloc.py) at the spill point to report the EXACT live-range breakdown: how many vregs are
simultaneously live and CATEGORIZED (store-offset vs cvt/accum-read data vs accumulators vs address temps). This
answers "what exactly is over the ~104-VGPR pool" -- is it the 128 offsets (serialization should fix, so why didn't
it), the 128 data regs (need separate gating), or the accumulators themselves. Only then choose the fix.

Key facts for the drill-down: no_vectorized_wmma (devectorizer.py:235-244) splits to 16 Ops.WMMA -> 128 scalar
stores; linearizer priority-toposort (linearizer.py:8-48) emits addresses-then-stores; _vpool shrunk to ~104
(amd.py:142) by pinned LOW C-accumulators (WMMA_ACC_BASE, amd.py:70,113-128). Data reg per store =
V_CVT_F2H(ACCUM_READ(v[pin])). Compile forced 4x4 via the DEFAULT matmul path (hand_coded picks 4x4), NOT hand
opts_to_apply=UPCAST (those hit AxisType.WARP and error).
</content>
