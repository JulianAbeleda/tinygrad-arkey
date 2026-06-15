# W2 RESULT (in progress) -- parametrizing the Marlin template

## W2.0 -- grid parallelism (DONE, 2026-06-15)
Blocked the output over M-rows: one workgroup per BLOCK_M=16 tile (whole N + whole K per workgroup),
via an `AxisType.GLOBAL` block_m range. Each workgroup runs the proven W1b' fused-dequant->WMMA body.

Result: grid scaling lifts throughput ~70x vs the single-workgroup W1b' kernel.
  256x1024x512  ( 16 wg): marlin 0.46 TF | ceil 0.46 | m/c 1.01
  1024x1024x512 ( 64 wg): marlin 1.75 TF | ceil 1.74 | m/c 1.01
  4096x1024x512 (256 wg): marlin 3.31 TF | ceil 3.25 | m/c 1.02   (4.0% of 83.6 peak)
  4096x2048x256 (256 wg): marlin 2.08 TF | ceil 3.75 | m/c 0.55
  4096x1024x2048(256 wg): marlin 3.56 TF | ceil 6.91 | m/c 0.52
All correct (rel_err < 1e-2).

Findings:
1. Grid + TC composition bug (fixed): the LDS dequant-staging depends on the `block_m` GLOBAL range,
   so that range pollutes the weight operand's `.ranges`; TC's `axis=0` then picks the (size n_blocks,
   not %16) grid range -> "no tensor core available". `Opt(OptOps.TC, axis=1, ...)` selects (n,m,k)
   correctly and consistently across block counts.
2. marlin == ceiling at moderate N (1.01-1.02x) but trails at large N/K (0.52-0.55x). The
   dequant-to-LDS PROLOGUE is a fixed serial cost (BLOCK_M*K dequants, ~10 ALU each) NOT overlapped
   with the WMMA compute, so it caps throughput as the matmul grows. -> W2.1 double-buffering /
   K-tiling to overlap dequant(tile k+1) with WMMA(tile k).
3. Absolute is ~4% of the 83.6 peak: small workgroups (BLOCK_M=16), no K-tiling (K<=2048 LDS limit),
   serial prologue, no double-buffer. These are the W2.1+ levers.

## W2.1 -- K-tiling + occupancy (NEXT)
Mandatory for K=4096 (16x4096 fp16 = 128KB > 64KB LDS). Open risk (W2.1a): does the one-workgroup
K-loop compose with TC (manual K-loop accumulator is not a single Ops.REDUCE; GROUP forbidden with
TC)? Fallback W2.1b: split-K grid + partial-sum pass (each workgroup = the proven W1b' primitive).
