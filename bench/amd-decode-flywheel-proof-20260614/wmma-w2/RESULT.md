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

## W2.1 -- K-tiling (split-K) DONE + the decisive throughput verdict (2026-06-15)
`extra/qk_marlin_w2.py::marlin_splitk_kernel`, `wmma-w2/w21_summary.json`.

K-tiling via SPLIT-K works: grid over (block_m, k_block); each workgroup is the proven W1b'/W2.0
single-`Ops.REDUCE` fused-dequant->WMMA body over a BLOCK_K=2048 slice, partials summed by `.sum(0)`.
Handles real K=4096 (which W2.0's whole-K-in-LDS could not), correct (rel_err<1e-2). Chose split-K
over a one-workgroup K-loop because a manual cross-tile accumulator fights TC's ownership of the
WMMA-fragment accumulator (TC needs one REDUCE; split-K keeps each workgroup a single REDUCE).
Same `Opt(OptOps.TC, axis=1, ...)` fix as W2.0 (grid range pollutes the weight operand's ranges).

THE VERDICT (the pre-registered W2 failure mode, confirmed):
  split-K fused (reads compressed, K=4096):  2.2 - 5.0 TFLOPS  (2.7 - 5.9% of peak)
  NATIVE tinygrad fp16 matmul (same shapes): 28  - 82  TFLOPS  (33  - 98%  of peak)
The fused custom kernel is ~10x slower than native fp16 matmul, and 5-6x slower even at small-N
(N=16-64) MEMORY-BOUND decode -- where reading compressed (3.5x less weight data) should win.

ROOT CAUSE (robustly established, not a one-off):
- It is NOT the dequant: the manually-LDS-staged fp16 CEILING kernel (same structure, no dequant)
  also tops out at ~3-8% peak (W2.0). W1b' already proved fusion is ~free vs this structure.
- It IS the structure: a hand-authored custom_kernel that MANUALLY stages operands in LDS only
  applies ONE opt (TC). Native matmul applies a full schedule (TC + UPCAST*2 + LOCAL) and reaches
  98%. Appending native's exact UPCAST/LOCAL opts to the Marlin kernel barely moves it (3.0->3.7%):
  the opt machinery cannot re-tile around the hand-placed DEFINE_LOCAL + barrier. BLOCK_M sweep
  (16..128) also flat (~3%).
- FUNDAMENTAL TENSION: the manual LDS dequant-staging that makes fusion free (W1b') is exactly what
  BLOCKS the auto-tiling that reaches peak. In tinygrad's custom_kernel + opt model you can have
  fusion OR peak tiling, not both.

IMPLICATION for "machine search competitive with llama.cpp":
- A competitive FUSED quantized GEMM is NOT expressible via tinygrad custom_kernel + opts. Paths:
  (c) hand-assembly (full control of tiling AND fusion -- Marlin/rocWMMA territory), or
  matmul_decoded = a cheap separate dequant pass (Track 0: 8603 GFLOPS) writing fp16 + NATIVE matmul
  (33-98% peak). The quantization memory win comes from streaming compressed -> fp16 per tile in the
  dequant pass; the throughput comes from native matmul -- but they are SEPARATE kernels (fp16
  round-trip), not fused.
- The learned-cost-model / machine-search question is therefore meaningful on the NATIVE matmul opt
  schedule (TC+UPCAST+LOCAL, already driven to 33-98% by tinygrad's heuristic/BEAM), NOT on the
  fused custom kernel. The fused-template search space (W3/W4 as originally framed over qk_marlin)
  is MOOT: it cannot contain a competitive point.
