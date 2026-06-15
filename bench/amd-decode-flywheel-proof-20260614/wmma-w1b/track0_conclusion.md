# W1b' Track 0 -- diagnosis of the W1 fused-WMMA 28x (2026-06-15)

Rendered the W1 fused matmul (compressed Q4_K dequant -> cast f16 -> matmul, TC_OPT=2) at
rows=256, k=4096, b=64 and inspected the generated AMD source.

The slow kernel `r_32_4_2_2_2_4_4_4_2_32_2` (6553 us, **23 GFLOPS**):
- signature: `(half* out_16384, half* x_262144, unsigned char* weights_589824)` -- it reads the
  COMPRESSED Q4_K weights (256*16*144 = 589824 B) directly. The dequant is FUSED into this kernel.
- body: 640 compressed loads (data2), 258 `>>`, 192 `(half)` casts -- the full Q4_K dequant
  (bit_cast d/dmin, the affine `d*sc*q - dmin*mn`) is computed INLINE and feeds `cast0/cast4`
  straight into `__WMMA_16_16_16_half_float(...)`.

Conclusion: the recompute hypothesis HOLDS. The dequant sits on the WMMA-input critical path with
NO LDS staging / reuse, so the dequant ALU + serialization starves the tensor cores (23 GFLOPS vs
the fp16-WMMA ceiling). This is exactly the Marlin problem and validates Track A: stage the
dequanted weight tile in LDS once, reuse it across the block's WMMA ops.

Note: a separate fast elementwise kernel `E_4096_32_4` (17 us, 8603 GFLOPS) also dequants -- that is
an incidental materialize in the graph, not the bottleneck. The bottleneck is the fused reduce.
