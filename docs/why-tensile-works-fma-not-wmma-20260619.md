# WHY TENSILE WORKS (verified by disassembly) — rocBLAS uses LDS-staged packed-FMA, NOT WMMA

The user pushed: "we don't fully understand why Tensile works." Correct. Disassembled the actual rocBLAS gfx1100
fp16 GEMM kernel. The premise we carried all campaign ("Tensile = WMMA at 66 vs tinygrad WMMA 42") is WRONG.

## Disassembly of TensileLibrary_Type_HS_HPA_..._Cijk_..._gfx1100.hsaco (the rocBLAS fp16 NN GEMM)
- Compute op: **`v_fma_mix_f32` x4096** (fp16-input, fp32-accumulate packed FMA on the VALU). **ZERO `v_wmma`.**
- **Heavy LDS operand-staging:** 1644 `ds_load_u16_d16`, 204 `ds_load_2addr_b32`, 144+136 `ds_store_b16`;
  group_segment up to **9216 B LDS/workgroup**; VGPR up to **185** (range 45-185 across tile variants).
- Loads: global_load_d16_b16/hi (fp16). 1432 s_waitcnt.

## So: rocBLAS does NOT use the tensor cores on RDNA3
It uses **expertly-tiled, LDS-staged, packed-VALU-FMA**. Why: RDNA3 has NO dedicated matrix units (unlike CDNA's
MFMA) -- WMMA on RDNA3 is microcoded over the SIMD/VALU. So well-tiled packed-FMA matches or beats WMMA WITHOUT
WMMA's fragment/register overhead. rocBLAS (and Tensile's tuning) exploits this.

## Therefore the "42 vs 66" comparison was apples-to-oranges
- tinygrad WMMA path = 42 TFLOPS (its best; all WMMA levers exhausted).
- rocBLAS = 66-77 via FMA+LDS (a DIFFERENT compute strategy + expert tiling).
- The "122 WMMA peak / 35%" framing is MISLEADING -- rocBLAS doesn't use WMMA. The relevant roofline is the VALU
  FMA peak (~60-70 TFLOPS region on gfx1100), which rocBLAS nearly saturates and tinygrad's WMMA (42) sits below.

## Can tinygrad replicate the rocBLAS strategy? NO (tested)
Non-TC FMA matmul + LDS(GROUP) + occupancy(LOCAL) + tile(UPCAST) configs in tinygrad: **0.3 / 1.7 / 4.3 / 2.1 /
4.8 TFLOPS** -- ~10x WORSE than rocBLAS's 66 with the SAME instruction class. tinygrad's non-TC FMA codegen cannot
tile/stage/pipeline/occupy like rocBLAS's Tensile-tuned kernel. (And the WMMA path is capped at 42, GROUP+TC is
broken.)

## ANSWER to "why past 42 is a new kernel"
YES, literally a different kernel: rocBLAS-class **LDS-staged packed-FMA**, executed at a tiling/scheduling/
occupancy quality tinygrad's codegen does NOT produce for EITHER path (WMMA capped 42; FMA <=5). This is why:
- external Tensile doesn't transfer in-model (it's a fundamentally different, hand/auto-tuned kernel, not a swap-in);
- it's NOT a bounded fix -- matching it = a Tensile-class codegen capability (operand-staging + occupancy +
  pipelining for the FMA path), a multi-month effort, OR vendoring the external kernel.

## Net (corrects the whole campaign's WMMA framing)
The prefill matmul ceiling isn't a "WMMA codegen" problem -- it's that **the winning kernel on RDNA3 is FMA+LDS,
and tinygrad's codegen produces neither a good FMA+LDS kernel (<=5) nor a WMMA kernel past 42.** rocBLAS's 66 is a
different, expertly-tuned kernel. tinygrad rests at WMMA 42 = ~47% llama; concrete-KV 1.24x is the shippable win.
The honest path past 42 is external (vendored Tensile/hipBLASLt .co, with the in-model integration problem) or a
multi-month FMA-tiling codegen capability -- neither a bounded arc.

## Files
Disasm: /opt/rocm/lib/rocblas/library/TensileLibrary_Type_HS_HPA_*_gfx1100.hsaco via llvm-objdump --mcpu=gfx1100.
Supersedes the WMMA-centric framing in wmma-both-levers-conclusion / wmma-occupancy-frontier-result.
