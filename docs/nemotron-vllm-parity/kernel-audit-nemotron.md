# Kernel audit: BoltBeam vs vLLM on every Nemotron 3 Nano 4B BF16 GEMM shape (RTX 5090)

Goal #2 on `goal-board.md` (measure first). Measure-only: no product code changed. Method: reverse engineer
(capture which kernel vLLM actually runs, per shape) -> test (same shape, same L2 conditions, both sides, ncu on both)
-> the replication spec (section 4) is the input to the "solve" half. 2026-09-26, `exp` @ b805201ce, vLLM 0.30 /
torch 2.13 cu130, all timings under `gpu-run time`.

## 1. Result in one paragraph

Every projection is slower than cuBLAS at every measured M; only ssm_out, attn_o and ffn_down at M=8 are within 6%. The gap
has two separable causes. (a) **The kernel.** The promoted candidate family (register-staged 2-buffer LDS, scalar
32-bit `LDS` fragment loads, no `ldmatrix`, no `cp.async`, static LDS <= 48 KB, K-tile 32/64) reaches ~109-158 TF per
product at prefill sizes and 0.23-0.82 TB/s at decode sizes of 64-128 rows (attn_kv, N=1024, is the low end). cuBLAS runs CUTLASS sm80 multistage
kernels (`cp.async` + `ldmatrix`, 3-6 stages, 75-99 KB dynamic smem, 4 warps) at 200-227 TF (M >= 2048) and 1.3-1.5 TB/s. ncu:
ours stalls on `mio_throttle` (23-46%) and `barrier` (13-19%) with ~50x vLLM's shared-memory bank conflicts (ssm_in, 64 rows); vLLM
stalls on `math_pipe_throttle`/`wait`, meaning the tensor pipe is saturated. (b) **hi/lo doubles the product** (2M
rows). At decode it is nearly free up to 64 rows (still memory-bound), but the doubled rows push our weaker kernel off
the memory roof from B=128 on (at B=32/64 the 2M-row product is still memory-bound; the loss there is kernel quality). At prefill it is a straight 2x on top of (a); prefill routes also stop at 512 hi/lo
rows, so a 1024-token piece runs 4 chunked launches of the 512-row kernel. GEMMs are 51% of our B=32 step (6.8 of
13.3 ms; vLLM 4.8 ms) and 67% of our 10k prefill (1110 of 1650 ms; vLLM 286 ms).

**Top-5 levers** (ms saved per step if matched to vLLM; details and specs in section 4):

| # | Lever | B=32 step | B=64 | B=128 | 10k prefill |
|---|---|---|---|---|---|
| 1 | Prefill substrate: all roles at 512-8192 rows (a cuBLAS-class compute tile + routes above 512 rows) | - | - | - | **824 ms** (50% of 1.65 s) |
| 2 | ssm_in decode (N=17504, K=3136, 21x/step) | 0.42 | 1.58 | 3.27 | (in 1) |
| 3 | ffn_up + ffn_down decode (12544x3136 / 3136x12544, 17x each) | 0.89 | 2.07 | 3.94 | (in 1) |
| 4 | Narrow-N decode: ssm_out + attn_o (N=3136; too few CTAs, split-K too shallow) | 0.31 | 0.93 | 1.74 | (in 1) |
| 5 | lm_head at B>=64 (N=131072) + q/k/v fusion (ours: 3 GEMMs; vLLM: 1) | 0.36 | 0.77 | 1.78 | (23 ms, in 1) |

Total decode GEMM gap: 1.97 ms at B=32 (15% of our step), 5.36 ms at B=64 (25%), 10.7 ms at B=128 (30%).

## 2. Method

- **vLLM side** (`bench/audit/vllm_gemm.py`): vLLM 0.30's unquantized linear on CUDA is
  `torch.nn.functional.linear` (`model_executor/layers/utils.py: default_unquantized_gemm`), bf16 in/out, so the
  microbench calls exactly that per logical op. vLLM fuses q/k/v into one `QKVParallelLinear` (N=7168), so `qkv` is
  measured as well as q and kv alone. Times are CUPTI kernel durations (torch.profiler), including `splitKreduce`.
  **Validated against in-situ vLLM traces** (`vllm-bench/runs/prof_default_*`): same kernels and grids, and per-call
  times within 3% (ssm_in B=64 74.1 vs 74.3 us; B=128 91.4 vs 88.3; ffn_up B=128 68.8 vs 67.1). vLLM's 10k prefill
  runs the same two CUTLASS kernels as the microbench (`prefill10k_kern.csv`: 64x256_32x4 and 128x64_64x3, 298 ms
  total GEMM vs 286 ms estimated here).
- **BoltBeam side** (`bench/audit/ours_gemm.py`): the production path, `_Linear.__call__` on bound weights: <=16 rows
  fp32 matvec; 17-128 rows routed hi/lo (`route_bound`, the route at 2M rows); prefill rows inside
  `route_scratch()` exactly as `NemotronHPrefill` (piece 1024; the scratch path needs a route at 2x1024 rows, none
  exists, so it falls to `route_dense_bf16` in 512-row chunks). Totals include the hi/lo split, split-K sum and
  hi+lo sum kernels (6% of the op at decode, 14% at prefill); "GEMM" columns are the candidate kernel alone. Per-kernel
  device times from the HCQ graph profile, `HCQ_NV_READY_PLACEMENT=0` (one queue: independent copies otherwise overlap
  and per-kernel durations become noisy; with it runs repeat to 0.1%). Prefill M > 1024 is M/1024 pieces of 1024
  (production never issues more rows per call). **Validated in-situ** (`bench/spec/ours_prof.sh 64`): ssm_in 127 vs
  141 us (microbench 10% pessimistic), ffn_up 104.7 vs 105.7, ffn_down 101.8 vs 109.8, ssm_out 71.2 vs 67.9, lm_head
  968 vs 984.
- **L2**: the 5090 has a 96 MB L2 and several weights fit; the model streams 42 layers of distinct weights. Both
  sides rotate over R distinct weight copies totalling >= 384 MB; ncu runs use `--cache-control all`.
- **ncu**: vLLM's cuBLAS kernels directly (`vncu.sh`). tinygrad's NV backend bypasses the CUDA driver, so ncu cannot
  see it; `ours_ncu.py` runs `DEV=CUDA` with the route table admitted for that device. The rendered GEMM source is
  **byte-identical** to NV's except the name hash (diffed), and time is within 4% (141.6 vs 136.7 us). Unmodified
  clocks (`--clock-control none`). SASS of our ssm_in kernel: `nvcc -arch=sm_120` + `cuobjdump`.
- cuBLAS kernel names contain `gemm_relu`; that is the CUTLASS kernel family name, `F.linear` applies no activation, so both sides do the same epilogue work (store only).
- In prose, TB/s figures are bytes/time from 5A unless marked "ncu DRAM" (measured, from 5B).
- Roofline: compute 250 TF (measured bf16 TC peak ~248-252), memory = bf16 bytes (W + X + Y) / 1.69 TB/s, the max of
  the two for the logical op. "% hi/lo roof" uses 2x the FLOPs (the product we actually run).
- Raw ncu reports (`vncu.ncu-rep` 32 MB, `oncu.ncu-rep` 19 MB, `vop`/`oop` opcode reports) and raw CSVs are kept out
  of git in `~/scratchpad/audit/`; timing JSONs are in `bench/audit/results/`.

## 3. What differs, kernel anatomy (ssm_in as the example; the other roles follow the same pattern)

| | vLLM / cuBLAS | BoltBeam promoted candidate |
|---|---|---|
| decode <=16 rows | CUTLASS `wmma 16x16x128`, 2 stages, 1 warp/CTA, 1096 CTAs, 90% of BW | fp32 matvec (`r_*`, no tensor core), 66% of BW at 8 rows, **39% at 16** (grid.y=2; ~1.6x the weight bytes read) |
| decode 32-128 rows | `128x64_64x3` (Nout 128 x Mtok 64 x K 64, 3 stages), 4 warps, 158 regs, 75 KB; split-K 3-6 + `splitKreduce` for N=3136 | hi/lo -> route at 64-256 rows: tiles 64x64x64 .. 128x128x32 (Mtok x Nout x K), 4-8 warps, 2 LDS buffers, 38-42 KB, split-K 1-4 then a separate sum |
| prefill | `64x256_32x4` or `128x64_64x3`, 4 warps, 230/158 regs, 83/75 KB, 1 CTA/SM, tensor pipe 87-96% | 512-row route chunks, 128x128x32 or 64x64x64, tensor pipe 57-71% |
| global->LDS | `LDGSTS` (cp.async) multistage ring | `LDG.128` -> registers -> `STS.128`, 2 buffers (register-staged) |
| LDS->fragments | `LDSM` (ldmatrix): ssm_in M=64: 648k LDSM + 252k LDS | scalar 32-bit `LDS`: 48 LDS per 32 HMMA in the loop body; 5.16M LDS for 3.44M HMMA |
| bank conflicts (ssm_in M=64) | 35k | 1.74M (M=512 chunk: 6.9M) |
| top stalls | math_pipe_throttle, wait (tensor pipe busy) | math_pipe_throttle + **mio_throttle 23-46%** + barrier 13-19% |

The commit log already has the one staging-only data point: `fda18e7ab` (cp.async 3-stage 128x128x32 at ssm_in
M=4096, bf16, not a promoted route) gave 141 TF vs 136 TF for the 2-buffer schedule. So staging alone moves ~4%. The
dominant difference in ncu is fragment-load issue pressure (scalar LDS + conflicts -> `mio_throttle`), not DRAM
latency. The goal board's "BoltBeam 141 TF" is that experimental kernel; the production ssm_in path measures 118 TF
per product (about 51 TF useful after hi/lo and the aux kernels).

## 4. Replication spec for the top-5 levers (what BoltBeam must search/emit)

Common to all five: the search space needs (i) `ldmatrix` (LDSM.x4/x2) fragment loads with an XOR-swizzled LDS layout
in place of padded-stride scalar loads, (ii) a `cp.async` ring of 3-6 stages with a dynamic-smem arena above 48 KB
(the renderer has this since `fda18e7ab`; the dense candidate set's `static_constraints.max_lds_bytes` is 49152),
(iii) grids of at least one full wave on 170 SMs via split-K where N is narrow, with the split-K reduction and the
hi+lo sum fused into the epilogue or a single pass. The gates are vLLM's measured kernel times in section 5A.

1. **Prefill substrate (all roles, 512-8192 rows; -824 ms of 1650).** Target vLLM's two tiles: `64x256x32`, 4
   stages, 4 warps, 230 regs, 83 KB (ssm_in, ffn_up, ffn_down at M >= 2048; q/o at 512-4096), and `128x64x64`, 3
   stages, 4 warps, 158 regs, 75 KB (ssm_out, attn_o, q at 8192). 1 CTA/SM, tensor pipe >= 90%, 205-227 TF. Promote
   routes at the rows prefill actually issues (2048 hi/lo rows per 1024 piece, or 1024 bf16 rows) so a piece is one
   launch, not 4. Decide hi/lo vs plain bf16 for prompts: the bf16 route is already 2.1x faster at M=1024 (ssm_in
   1050 vs 2223 us) and hi/lo can never beat 2x cuBLAS time. Gate per role: M=1024 piece <= 2x vLLM's M=1024 time with
   hi/lo, <= 1.1x without.
2. **ssm_in decode, 32-128 rows (N=17504, K=3136).** vLLM: `128x64x64` 3-stage, 4 warps, splitK 1, 137 CTAs (<= 64
   rows) or 274 (128 rows), 1.51-1.52 TB/s at 32-64 rows, 1.26 TB/s at 128. Ours: 1.19 TB/s at 32 rows (the m=64
   route is close), **0.75 at 64, 0.47 at 128** (the 128/256-row routes, 128x128x32 8 warps, are compute/issue-bound
   under hi/lo). Emit Nout-128 x Mtok-(64..128) tiles, 4 warps, 3 stages, LDSM. With hi/lo at 256 rows the product
   is compute-bound (floor 112 us at 250 TF, ~134 us at cuBLAS-grade 210 TF vs vLLM 91), so B=128 cannot reach parity
   with hi/lo stacked on M. Gate: <= 1.1x vLLM at B=32/64, <= 1.5x at B=128.
3. **ffn_up / ffn_down decode.** ffn_up: vLLM uses `64x64x32` 6-stage splitK 5 (1000 CTAs) at 32 rows,
   `128x64x64` 3-stage at 64, `256x64x32` 4-stage splitK 3 at 128. Ours: one 64x64x64 2-buffer tile everywhere, 196
   CTAs at 32 rows (1.21 TB/s), 392-784 CTAs at 64-128 rows but 0.53-0.82 TB/s (mio_throttle). ffn_down (K=12544):
   vLLM `128x64x64` 3-stage **splitK 6** (192 CTAs) + splitKreduce at 32/64 rows, 1.50 TB/s. Ours at 64 rows:
   128x128x32 split-K 4 = **100 CTAs** (< 1 wave), 0.69 TB/s (ncu DRAM: 0.64). Search split-K depth jointly with tile so CTAs >= 170.
4. **Narrow-N decode: ssm_out (3136x7680) and attn_o (3136x5120).** Same story as ffn_down: vLLM `128x64x64` 3-stage
   splitK 6 (192 CTAs) at 32-64 rows, splitK 3 at 128, 1.3-1.4 TB/s. Ours: 100-150 CTAs, 0.43-1.07 TB/s. Because N=3136
   pads to 3200, Nout-64 tiles give 50 columns and need splitK >= 4 for one wave.
5. **lm_head at B >= 64 (N=131072), and q/k/v fusion.** lm_head: vLLM `128x64x64` 3-stage, 1024-2048 CTAs, 1.56 TB/s
   at 64 rows, 1.43 at 128 (92% of roofline). Ours at 128/256 hi/lo rows: 128x128x32 8 warps, **mio_throttle 39-46%**,
   0.95 TB/s, plus a 35-90 us hi+lo sum pass over 131072-wide fp32 rows. q/k/v: ours runs q (5120) + k (1024) + v
   (1024) as three GEMMs, where k and v sit at 10-15% of roofline (too few CTAs). vLLM runs one 7168-wide GEMM:
   32.8 us at B=32 vs our 97 us per attention layer. Fusing the three projections (the model-graph change vLLM makes)
   removes 2 launches per layer and lets the ssm_in-class tile serve them. Gate: <= 1.1x vLLM's qkv time.

Smaller items seen on the way: the 16-row fp32 matvec takes 1.4-2.0x the 8-row time (attn_kv 1.0x; grid.y=2, ssm_in reads ~1.6x the weight bytes);
routing B=16 through the existing 32-row hi/lo routes would cut ssm_in 168 -> ~90 us (the 64-row route measures 93.5
us). The hi/lo split is recomputed per chunk at prefill (4 `E_64_49` launches per 1024 piece).

## 5. Tables (generated by `bench/audit/audit_table.py` from the JSON results and the ncu CSVs)

### 5A. Time per call (useful FLOPs = 2MNK; roofline = max(2MNK/250 TF, bf16 bytes/1.69 TB/s); * = derived from the M=1024 piece)


#### ssm_in (N=17504, K=3136; 21x per step)

| M | bound | ours us | ours TF useful (per-product GEMM) | ours GB/s | ours % roof | ours % hi/lo roof | vLLM us | vLLM TF | vLLM GB/s | vLLM % roof | ours/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | mem | 98.3 | 8.9 (-) | 1120 | 66% | 66% | 72.2 | 12.2 | 1526 | 90% | 1.36x |
| 16 | mem | 167.7 | 10.5 (-) | 659 | 39% | 39% | 72.8 | 24.1 | 1518 | 90% | 2.30x |
| 32 | mem | 93.5 | 37.6 (81) | 1189 | 70% | 70% | 73.5 | 47.8 | 1512 | 89% | 1.27x |
| 64 | mem | 149.4 | 47.0 (100) | 752 | 45% | 45% | 74.1 | 94.9 | 1518 | 90% | 2.02x |
| 128 | mem | 246.9 | 56.9 (119) | 466 | 28% | 46% | 91.4 | 153.8 | 1259 | 75% | 2.70x |
| 512 | comp | 1019.1 | 55.2 (117) | 128 | 22% | 44% | 292.1 | 192.4 | 448 | 77% | 3.49x |
| 1024 | comp | 2223.4 | 50.6 (118) | 68 | 20% | 40% | 546.9 | 205.6 | 278 | 82% | 4.07x |
| 2048* | comp | 4446.8 | 50.6 (118) | 44 | 20% | 40% | 1055.4 | 213.0 | 184 | 85% | 4.21x |
| 4096* | comp | 8893.6 | 50.6 (118) | 31 | 20% | 40% | 2069.1 | 217.3 | 135 | 87% | 4.30x |
| 8192* | comp | 17787.2 | 50.6 (118) | 25 | 20% | 40% | 3957.4 | 227.3 | 113 | 91% | 4.49x |

#### ssm_out (N=3136, K=7680; 21x per step)

| M | bound | ours us | ours TF useful (per-product GEMM) | ours GB/s | ours % roof | ours % hi/lo roof | vLLM us | vLLM TF | vLLM GB/s | vLLM % roof | ours/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | mem | 46.2 | 8.3 (-) | 1046 | 62% | 62% | 44.1 | 8.7 | 1096 | 65% | 1.05x |
| 16 | mem | 84.0 | 9.2 (-) | 577 | 34% | 34% | 44.3 | 17.4 | 1095 | 65% | 1.90x |
| 32 | mem | 47.9 | 32.2 (70) | 1020 | 60% | 60% | 35.1 | 44.0 | 1393 | 82% | 1.37x |
| 64 | mem | 74.1 | 41.6 (91) | 669 | 40% | 40% | 36.0 | 85.7 | 1378 | 82% | 2.06x |
| 128 | mem | 112.4 | 54.8 (119) | 453 | 27% | 44% | 40.7 | 151.7 | 1253 | 74% | 2.77x |
| 512 | comp | 383.0 | 64.4 (140) | 155 | 26% | 52% | 139.4 | 177.0 | 425 | 71% | 2.75x |
| 1024 | comp | 794.4 | 62.1 (140) | 89 | 25% | 50% | 304.9 | 161.8 | 231 | 65% | 2.61x |
| 2048* | comp | 1588.8 | 62.1 (140) | 58 | 25% | 50% | 495.7 | 199.0 | 187 | 80% | 3.21x |
| 4096* | comp | 3177.7 | 62.1 (140) | 43 | 25% | 50% | 986.5 | 200.0 | 139 | 80% | 3.22x |
| 8192* | comp | 6355.3 | 62.1 (140) | 35 | 25% | 50% | 1842.6 | 214.2 | 122 | 86% | 3.45x |

#### attn_q (N=5120, K=3136; 4x per step)

| M | bound | ours us | ours TF useful (per-product GEMM) | ours GB/s | ours % roof | ours % hi/lo roof | vLLM us | vLLM TF | vLLM GB/s | vLLM % roof | ours/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | mem | 42.4 | 6.1 (-) | 761 | 45% | 45% | 23.5 | 10.9 | 1373 | 81% | 1.80x |
| 16 | mem | 60.1 | 8.6 (-) | 539 | 32% | 32% | 23.6 | 21.8 | 1375 | 81% | 2.55x |
| 32 | mem | 43.0 | 23.9 (53) | 759 | 45% | 45% | 24.2 | 42.5 | 1351 | 80% | 1.78x |
| 64 | mem | 41.5 | 49.5 (110) | 799 | 47% | 47% | 25.2 | 81.6 | 1316 | 78% | 1.65x |
| 128 | mem | 63.2 | 65.1 (148) | 542 | 32% | 52% | 24.3 | 169.0 | 1407 | 83% | 2.60x |
| 512 | comp | 232.0 | 70.9 (158) | 175 | 28% | 57% | 82.1 | 200.2 | 494 | 80% | 2.82x |
| 1024 | comp | 512.4 | 64.2 (157) | 96 | 26% | 51% | 163.2 | 201.5 | 300 | 81% | 3.14x |
| 2048* | comp | 1024.8 | 64.2 (157) | 64 | 26% | 51% | 326.3 | 201.6 | 202 | 81% | 3.14x |
| 4096* | comp | 2049.6 | 64.2 (157) | 49 | 26% | 51% | 642.4 | 204.8 | 155 | 82% | 3.19x |
| 8192* | comp | 4099.1 | 64.2 (157) | 41 | 26% | 51% | 1222.5 | 215.2 | 137 | 86% | 3.35x |

#### attn_kv (N=1024, K=3136; 8x per step)

| M | bound | ours us | ours TF useful (per-product GEMM) | ours GB/s | ours % roof | ours % hi/lo roof | vLLM us | vLLM TF | vLLM GB/s | vLLM % roof | ours/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | mem | 38.1 | 1.3 (-) | 170 | 10% | 10% | 7.5 | 6.8 | 861 | 51% | 5.05x |
| 16 | mem | 38.2 | 2.7 (-) | 172 | 10% | 10% | 19.1 | 5.4 | 343 | 20% | 2.00x |
| 32 | mem | 27.2 | 7.6 (17) | 246 | 15% | 15% | 8.0 | 25.7 | 838 | 50% | 3.41x |
| 64 | mem | 30.8 | 13.4 (30) | 226 | 13% | 13% | 8.1 | 50.6 | 857 | 51% | 3.79x |
| 128 | mem | 32.6 | 25.2 (60) | 230 | 14% | 20% | 9.5 | 86.5 | 788 | 47% | 3.43x |
| 512 | comp | 71.9 | 45.7 (109) | 149 | 18% | 37% | 23.5 | 139.9 | 455 | 56% | 3.06x |
| 1024 | comp | 140.5 | 46.8 (114) | 106 | 19% | 37% | 43.1 | 152.6 | 347 | 61% | 3.26x |
| 2048* | comp | 281.0 | 46.8 (114) | 83 | 19% | 37% | 83.1 | 158.3 | 282 | 63% | 3.38x |
| 4096* | comp | 562.0 | 46.8 (114) | 72 | 19% | 37% | 164.2 | 160.2 | 247 | 64% | 3.42x |
| 8192* | comp | 1123.9 | 46.8 (114) | 66 | 19% | 37% | 269.2 | 195.4 | 277 | 78% | 4.18x |

#### attn_o (N=3136, K=5120; 4x per step)

| M | bound | ours us | ours TF useful (per-product GEMM) | ours GB/s | ours % roof | ours % hi/lo roof | vLLM us | vLLM TF | vLLM GB/s | vLLM % roof | ours/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | mem | 32.7 | 7.9 (-) | 987 | 58% | 58% | 30.9 | 8.3 | 1044 | 62% | 1.06x |
| 16 | mem | 56.7 | 9.1 (-) | 571 | 34% | 34% | 31.1 | 16.5 | 1042 | 62% | 1.83x |
| 32 | mem | 33.7 | 30.5 (69) | 969 | 57% | 57% | 24.6 | 41.9 | 1329 | 79% | 1.37x |
| 64 | mem | 58.2 | 35.3 (80) | 570 | 34% | 34% | 25.7 | 79.9 | 1290 | 76% | 2.26x |
| 128 | mem | 85.6 | 48.0 (105) | 400 | 24% | 38% | 28.7 | 143.3 | 1193 | 71% | 2.98x |
| 512 | comp | 251.2 | 65.5 (143) | 161 | 26% | 52% | 98.2 | 167.5 | 413 | 67% | 2.56x |
| 1024 | comp | 529.8 | 62.1 (141) | 93 | 25% | 50% | 181.3 | 181.4 | 270 | 73% | 2.92x |
| 2048* | comp | 1059.7 | 62.1 (141) | 62 | 25% | 50% | 334.9 | 196.4 | 197 | 79% | 3.16x |
| 4096* | comp | 2119.3 | 62.1 (141) | 47 | 25% | 50% | 653.3 | 201.3 | 153 | 81% | 3.24x |
| 8192* | comp | 4238.6 | 62.1 (141) | 39 | 25% | 50% | 1236.3 | 212.8 | 135 | 85% | 3.43x |

#### ffn_up (N=12544, K=3136; 17x per step)

| M | bound | ours us | ours TF useful (per-product GEMM) | ours GB/s | ours % roof | ours % hi/lo roof | vLLM us | vLLM TF | vLLM GB/s | vLLM % roof | ours/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | mem | 73.5 | 8.6 (-) | 1074 | 64% | 64% | 49.8 | 12.6 | 1585 | 94% | 1.48x |
| 16 | mem | 115.0 | 10.9 (-) | 688 | 41% | 41% | 49.9 | 25.2 | 1586 | 94% | 2.30x |
| 32 | mem | 84.1 | 30.0 (63) | 948 | 56% | 56% | 53.1 | 47.4 | 1499 | 89% | 1.58x |
| 64 | mem | 112.3 | 44.9 (95) | 719 | 43% | 43% | 53.5 | 94.1 | 1508 | 89% | 2.10x |
| 128 | mem | 165.0 | 61.0 (128) | 501 | 30% | 49% | 68.8 | 146.4 | 1202 | 71% | 2.40x |
| 512 | comp | 616.5 | 65.3 (138) | 154 | 26% | 52% | 211.1 | 190.8 | 449 | 76% | 2.92x |
| 1024 | comp | 1337.2 | 60.2 (140) | 83 | 24% | 48% | 408.5 | 197.2 | 271 | 79% | 3.27x |
| 2048* | comp | 2674.5 | 60.2 (140) | 53 | 24% | 48% | 786.1 | 205.0 | 182 | 82% | 3.40x |
| 4096* | comp | 5348.9 | 60.2 (140) | 39 | 24% | 48% | 1497.6 | 215.2 | 138 | 86% | 3.57x |
| 8192* | comp | 10697.9 | 60.2 (140) | 31 | 24% | 48% | 2848.6 | 226.3 | 118 | 91% | 3.76x |

#### ffn_down (N=3136, K=12544; 17x per step)

| M | bound | ours us | ours TF useful (per-product GEMM) | ours GB/s | ours % roof | ours % hi/lo roof | vLLM us | vLLM TF | vLLM GB/s | vLLM % roof | ours/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | mem | 73.5 | 8.6 (-) | 1075 | 64% | 64% | 70.9 | 8.9 | 1113 | 66% | 1.04x |
| 16 | mem | 135.3 | 9.3 (-) | 585 | 35% | 35% | 70.9 | 17.7 | 1116 | 66% | 1.91x |
| 32 | mem | 74.3 | 33.9 (72) | 1073 | 63% | 63% | 52.9 | 47.6 | 1507 | 89% | 1.40x |
| 64 | mem | 117.0 | 43.0 (92) | 689 | 41% | 41% | 53.8 | 93.6 | 1501 | 89% | 2.18x |
| 128 | mem | 194.2 | 51.9 (109) | 426 | 25% | 41% | 58.9 | 170.9 | 1403 | 83% | 3.29x |
| 512 | comp | 645.8 | 62.4 (132) | 147 | 25% | 50% | 194.2 | 207.4 | 488 | 83% | 3.33x |
| 1024 | comp | 1311.6 | 61.4 (134) | 84 | 25% | 49% | 393.0 | 205.0 | 282 | 82% | 3.34x |
| 2048* | comp | 2623.1 | 61.4 (134) | 54 | 25% | 49% | 802.0 | 200.9 | 178 | 80% | 3.27x |
| 4096* | comp | 5246.2 | 61.4 (134) | 39 | 25% | 49% | 1570.2 | 205.2 | 132 | 82% | 3.34x |
| 8192* | comp | 10492.5 | 61.4 (134) | 32 | 25% | 49% | 3080.4 | 209.2 | 109 | 84% | 3.41x |

#### output (N=131072, K=3136; 1x per step)

| M | bound | ours us | ours TF useful (per-product GEMM) | ours GB/s | ours % roof | ours % hi/lo roof | vLLM us | vLLM TF | vLLM GB/s | vLLM % roof | ours/vLLM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 8 | mem | 544.4 | 12.1 (-) | 1514 | 90% | 90% | 490.2 | 13.4 | 1682 | 100% | 1.11x |
| 16 | mem | 1072.6 | 12.3 (-) | 770 | 46% | 46% | 493.9 | 26.6 | 1673 | 99% | 2.17x |
| 32 | mem | 621.9 | 42.3 (88) | 1336 | 79% | 79% | 523.9 | 50.2 | 1586 | 94% | 1.19x |
| 64 | mem | 1022.4 | 51.5 (107) | 821 | 49% | 49% | 529.8 | 99.3 | 1584 | 94% | 1.93x |
| 128 | mem | 1974.7 | 53.3 (112) | 434 | 26% | 43% | 553.1 | 190.2 | 1548 | 92% | 3.57x |

### 5B. Kernel structure and ncu (GEMM kernel only; vLLM tile from the CUTLASS name = Nout x Mtok x Ktile, cuBLAS column-major)


#### ssm_in

| M | vLLM kernel | vLLM grid / warps / regs / smem | vLLM ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls | ours route | ours kernel grid / warps / regs / smem | ours ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls |
|---|---|---|---|---|---|---|
| 8 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 137, 1) / 1 / 124 / 18KB | 9%, 12%, 1.52, long_scoreboard 81%, wait 9%, math_pipe_throttle 5% | fp32 matvec (no TC), `nn.Linear` path | (2188, 1, 1) / 4 / 56 / 5KB | 62%, 0%, 1.22, long_scoreboard 94%, lg_throttle 2%, wait 2% |
| 16 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 137, 1) / 1 / 124 / 18KB | 9%, 12%, 1.51, long_scoreboard 81%, wait 9%, math_pipe_throttle 4% | fp32 matvec (no TC), `nn.Linear` path | (2188, 2, 1) / 4 / 54 / 5KB | 72%, 0%, 1.03, long_scoreboard 93%, wait 2%, mio_throttle 2% |
| 32 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (8, 18, 1) / 4 / 158 / 75KB | 8%, 46%, 1.50, math_pipe_throttle 34%, wait 29%, long_scoreboard 22% | hi/lo 64 rows -> route m=64: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (274, 1, 1) / 4 / 126 / 38KB | 13%, 48%, 1.46, long_scoreboard 31%, math_pipe_throttle 27%, barrier 17% |
| 64 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (8, 18, 1) / 4 / 158 / 75KB | 8%, 46%, 1.47, math_pipe_throttle 33%, wait 28%, long_scoreboard 23% | hi/lo 128 rows -> route m=128: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 2 | (137, 2, 1) / 8 / 122 / 42KB | 25%, 57%, 0.90, math_pipe_throttle 46%, mio_throttle 31%, wait 9% |
| 128 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (16, 18, 1) / 4 / 158 / 75KB | 8%, 74%, 1.11, math_pipe_throttle 51%, wait 42%, long_scoreboard 4% | hi/lo 256 rows -> route m=256: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 1 | (137, 2, 1) / 8 / 124 / 42KB | 26%, 56%, 0.48, math_pipe_throttle 45%, mio_throttle 30%, barrier 13% |
| 512 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (64, 18, 1) / 4 / 158 / 75KB | 8%, 87%, 0.38, math_pipe_throttle 51%, wait 43%, long_scoreboard 2% | hi/lo 1024 rows -> route m=512, 2 chunks: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 1 | (137, 4, 1) / 8 / 124 / 42KB | 29%, 57%, 0.53, math_pipe_throttle 44%, mio_throttle 32%, barrier 14% |
| 1024 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (128, 18, 1) / 4 / 158 / 75KB | 8%, 93%, 0.23, math_pipe_throttle 52%, wait 44%, long_scoreboard 2% | hi/lo 2048 rows -> route m=512, 4 chunks: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 1 | (137, 4, 1) / 8 / 124 / 42KB | 29%, 57%, 0.53, math_pipe_throttle 44%, mio_throttle 32%, barrier 14% |
| 2048 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (64, 35, 1) / 4 / 230 / 83KB | 8%, 95%, 0.15, math_pipe_throttle 53%, wait 45%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 4096 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (128, 35, 1) / 4 / 230 / 83KB | 8%, 96%, 0.12, math_pipe_throttle 53%, wait 45%, short_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 8192 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (256, 35, 1) / 4 / 230 / 83KB | 8%, 96%, 0.10, math_pipe_throttle 53%, wait 45%, short_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |

#### ssm_out

| M | vLLM kernel | vLLM grid / warps / regs / smem | vLLM ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls | ours route | ours kernel grid / warps / regs / smem | ours ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls |
|---|---|---|---|---|---|---|
| 8 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 25, 1) / 1 / 124 / 18KB | 2%, 8%, 0.98, long_scoreboard 60%, wait 22%, math_pipe_throttle 10% | fp32 matvec (no TC), `nn.Linear` path | (784, 1, 1) / 4 / 54 / 5KB | 37%, 0%, 0.77, long_scoreboard 82%, lg_throttle 13%, wait 2% |
| 16 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 25, 1) / 1 / 124 / 18KB | 2%, 8%, 0.98, long_scoreboard 60%, wait 21%, math_pipe_throttle 10% | fp32 matvec (no TC), `nn.Linear` path | (784, 2, 1) / 4 / 48 / 5KB | 66%, 0%, 0.26, long_scoreboard 66%, lg_throttle 28%, mio_throttle 3% |
| 32 | `mma.sync 128x64x64`, 3 stages, splitK 6 +splitKreduce | (8, 4, 6) / 4 / 158 / 75KB | 8%, 45%, 1.39, long_scoreboard 30%, math_pipe_throttle 29%, wait 25% | hi/lo 64 rows -> route m=64: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (50, 2, 1) / 4 / 122 / 38KB | 8%, 36%, 1.07, math_pipe_throttle 37%, wait 27%, long_scoreboard 15% |
| 64 | `mma.sync 128x64x64`, 3 stages, splitK 6 +splitKreduce | (8, 4, 6) / 4 / 158 / 75KB | 8%, 47%, 1.41, long_scoreboard 31%, math_pipe_throttle 29%, wait 26% | hi/lo 128 rows -> route m=128: tile 128x64x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 3 | (50, 3, 1) / 8 / 76 / 32KB | 17%, 43%, 0.68, math_pipe_throttle 35%, mio_throttle 30%, wait 12% |
| 128 | `mma.sync 128x64x64`, 3 stages, splitK 3 +splitKreduce | (16, 4, 3) / 4 / 158 / 75KB | 8%, 76%, 1.07, math_pipe_throttle 49%, wait 42%, long_scoreboard 4% | hi/lo 256 rows -> route m=256: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 3 | (25, 2, 3) / 8 / 124 / 42KB | 17%, 50%, 0.43, math_pipe_throttle 43%, mio_throttle 36%, wait 12% |
| 512 | `mma.sync 128x64x64`, 3 stages, splitK 4 | (64, 4, 4) / 4 / 158 / 75KB | 8%, 81%, 0.35, math_pipe_throttle 48%, wait 41%, long_scoreboard 6% | hi/lo 1024 rows -> route m=512, 2 chunks: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 3 | (25, 4, 3) / 8 / 124 / 42KB | 27%, 61%, 0.29, math_pipe_throttle 45%, mio_throttle 29%, barrier 15% |
| 1024 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (128, 4, 1) / 4 / 158 / 75KB | 8%, 76%, 0.20, math_pipe_throttle 53%, wait 44%, long_scoreboard 1% | hi/lo 2048 rows -> route m=512, 4 chunks: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 3 | (25, 4, 3) / 8 / 124 / 42KB | 27%, 59%, 0.28, math_pipe_throttle 45%, mio_throttle 29%, barrier 15% |
| 2048 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (256, 4, 1) / 4 / 158 / 75KB | 8%, 91%, 0.15, math_pipe_throttle 53%, wait 44%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 4096 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (512, 4, 1) / 4 / 158 / 75KB | 8%, 92%, 0.17, math_pipe_throttle 53%, wait 44%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 8192 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (1024, 4, 1) / 4 / 158 / 75KB | 8%, 96%, 0.30, math_pipe_throttle 53%, wait 44%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |

#### attn_q

| M | vLLM kernel | vLLM grid / warps / regs / smem | vLLM ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls | ours route | ours kernel grid / warps / regs / smem | ours ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls |
|---|---|---|---|---|---|---|
| 8 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 40, 1) / 1 / 124 / 18KB | 4%, 10%, 1.27, long_scoreboard 64%, wait 19%, math_pipe_throttle 9% | fp32 matvec (no TC), `nn.Linear` path | (640, 1, 1) / 4 / 56 / 5KB | 31%, 0%, 0.74, long_scoreboard 96%, wait 2%, lg_throttle 1% |
| 16 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 40, 1) / 1 / 124 / 18KB | 4%, 10%, 1.25, long_scoreboard 62%, wait 20%, math_pipe_throttle 9% | fp32 matvec (no TC), `nn.Linear` path | (640, 2, 1) / 4 / 54 / 5KB | 62%, 0%, 0.71, long_scoreboard 94%, wait 2%, mio_throttle 1% |
| 32 | `mma.sync 128x64x64`, 3 stages, splitK 4 +splitKreduce | (8, 5, 4) / 4 / 158 / 75KB | 8%, 42%, 1.34, long_scoreboard 39%, math_pipe_throttle 25%, wait 23% | hi/lo 64 rows -> route m=64: tile 64x64x32 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (80, 2, 1) / 4 / 94 / 22KB | 8%, 27%, 0.85, long_scoreboard 51%, math_pipe_throttle 15%, wait 14% |
| 64 | `mma.sync 128x64x64`, 3 stages, splitK 4 +splitKreduce | (8, 5, 4) / 4 / 158 / 75KB | 8%, 43%, 1.30, long_scoreboard 37%, math_pipe_throttle 27%, wait 24% | hi/lo 128 rows -> route m=128: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (80, 2, 1) / 4 / 122 / 38KB | 8%, 57%, 0.82, math_pipe_throttle 36%, wait 26%, long_scoreboard 24% |
| 128 | `mma.sync 64x64x64`, 6 stages, splitK 1 | (16, 10, 1) / 4 / 96 / 99KB | 8%, 75%, 1.02, math_pipe_throttle 45%, wait 39%, long_scoreboard 5% | hi/lo 256 rows -> route m=256: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 2 | (40, 2, 2) / 8 / 124 / 42KB | 17%, 56%, 0.44, math_pipe_throttle 45%, mio_throttle 33%, wait 12% |
| 512 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (16, 10, 1) / 4 / 230 / 83KB | 8%, 86%, 0.34, math_pipe_throttle 52%, wait 44%, long_scoreboard 2% | hi/lo 1024 rows -> route m=512, 2 chunks: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 2 | (40, 4, 2) / 8 / 124 / 42KB | 28%, 66%, 0.27, math_pipe_throttle 46%, mio_throttle 27%, barrier 15% |
| 1024 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (32, 10, 1) / 4 / 230 / 83KB | 8%, 88%, 0.20, math_pipe_throttle 52%, wait 44%, long_scoreboard 1% | hi/lo 2048 rows -> route m=512, 4 chunks: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 2 | (40, 4, 2) / 8 / 124 / 42KB | 28%, 67%, 0.27, math_pipe_throttle 46%, mio_throttle 27%, barrier 15% |
| 2048 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (64, 10, 1) / 4 / 230 / 83KB | 8%, 90%, 0.13, math_pipe_throttle 52%, wait 44%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 4096 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (128, 10, 1) / 4 / 230 / 83KB | 8%, 91%, 0.10, math_pipe_throttle 52%, wait 45%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 8192 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (1024, 5, 1) / 4 / 158 / 75KB | 8%, 94%, 0.11, math_pipe_throttle 52%, wait 44%, long_scoreboard 2% | pieces of 1024 | (as M=1024) | (as M=1024) |

#### attn_kv

| M | vLLM kernel | vLLM grid / warps / regs / smem | vLLM ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls | ours route | ours kernel grid / warps / regs / smem | ours ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls |
|---|---|---|---|---|---|---|
| 8 | `wmma 16x16x128`, 2 stages, splitK 13 +splitKreduce | (8, 8, 13) / 1 / 124 / 18KB | 9%, 7%, 0.89, long_scoreboard 72%, wait 11%, short_scoreboard 6% | fp32 matvec (no TC), `nn.Linear` path | (128, 1, 1) / 4 / 56 / 5KB | 8%, 0%, 0.17, long_scoreboard 97%, wait 2%, no_instruction 0% |
| 16 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 8, 1) / 1 / 124 / 18KB | 2%, 2%, 0.32, long_scoreboard 60%, wait 21%, math_pipe_throttle 9% | fp32 matvec (no TC), `nn.Linear` path | (128, 2, 1) / 4 / 54 / 5KB | 13%, 0%, 0.17, long_scoreboard 97%, wait 2%, no_instruction 0% |
| 32 | `wmma 32x32x128`, 2 stages, splitK 5 +splitKreduce | (8, 4, 5) / 4 / 84 / 35KB | 8%, 14%, 0.94, long_scoreboard 60%, wait 15%, short_scoreboard 6% | hi/lo 64 rows -> route m=64: tile 16x64x32 (Mtok x Nout x K), 2 warps, 2 LDS bufs, splitK 2 | (16, 4, 2) / 2 / 76 / 14KB | 4%, 8%, 0.28, long_scoreboard 67%, wait 14%, math_pipe_throttle 12% |
| 64 | `mma.sync 64x64x32`, 6 stages, splitK 7 +splitKreduce | (8, 2, 7) / 4 / 88 / 50KB | 8%, 26%, 0.90, long_scoreboard 26%, math_pipe_throttle 25%, wait 24% | hi/lo 128 rows -> route m=128: tile 32x64x32 (Mtok x Nout x K), 2 warps, 2 LDS bufs, splitK 2 | (16, 4, 2) / 2 / 102 / 16KB | 4%, 15%, 0.28, long_scoreboard 50%, math_pipe_throttle 23%, wait 21% |
| 128 | `mma.sync 64x64x64`, 6 stages, splitK 5 +splitKreduce | (16, 2, 5) / 4 / 96 / 99KB | 8%, 43%, 0.76, math_pipe_throttle 30%, wait 29%, long_scoreboard 16% | hi/lo 256 rows -> route m=256: tile 64x64x32 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (16, 4, 2) / 4 / 96 / 22KB | 8%, 29%, 0.28, long_scoreboard 44%, math_pipe_throttle 19%, wait 19% |
| 512 | `mma.sync 64x64x64`, 6 stages, splitK 1 | (64, 2, 1) / 4 / 96 / 99KB | 8%, 62%, 0.32, math_pipe_throttle 46%, wait 40%, long_scoreboard 5% | hi/lo 1024 rows -> route m=512, 2 chunks: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (16, 8, 1) / 4 / 122 / 38KB | 8%, 45%, 0.25, math_pipe_throttle 35%, long_scoreboard 26%, wait 26% |
| 1024 | `mma.sync 64x128x64`, 3 stages, splitK 1 | (64, 2, 1) / 4 / 158 / 75KB | 8%, 67%, 0.24, math_pipe_throttle 51%, wait 43%, long_scoreboard 3% | hi/lo 2048 rows -> route m=512, 4 chunks: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (16, 8, 1) / 4 / 122 / 38KB | 8%, 44%, 0.24, math_pipe_throttle 36%, wait 26%, long_scoreboard 26% |
| 2048 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (64, 2, 1) / 4 / 230 / 83KB | 8%, 70%, 0.20, math_pipe_throttle 52%, wait 44%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 4096 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (128, 2, 1) / 4 / 230 / 83KB | 8%, 72%, 0.18, math_pipe_throttle 52%, wait 44%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 8192 | `mma.sync 128x256x32`, 3 stages, splitK 3 | (256, 1, 3) / 8 / 222 / 75KB | 17%, 83%, 0.19, math_pipe_throttle 72%, wait 21%, long_scoreboard 2% | pieces of 1024 | (as M=1024) | (as M=1024) |

#### attn_o

| M | vLLM kernel | vLLM grid / warps / regs / smem | vLLM ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls | ours route | ours kernel grid / warps / regs / smem | ours ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls |
|---|---|---|---|---|---|---|
| 8 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 25, 1) / 1 / 124 / 18KB | 2%, 7%, 0.95, long_scoreboard 59%, wait 22%, math_pipe_throttle 10% | fp32 matvec (no TC), `nn.Linear` path | (784, 1, 1) / 4 / 54 / 5KB | 37%, 0%, 0.84, long_scoreboard 83%, lg_throttle 10%, wait 2% |
| 16 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 25, 1) / 1 / 124 / 18KB | 2%, 7%, 0.96, long_scoreboard 59%, wait 22%, math_pipe_throttle 10% | fp32 matvec (no TC), `nn.Linear` path | (784, 2, 1) / 4 / 48 / 5KB | 70%, 0%, 0.31, long_scoreboard 76%, lg_throttle 18%, mio_throttle 2% |
| 32 | `mma.sync 128x64x64`, 3 stages, splitK 6 +splitKreduce | (8, 4, 6) / 4 / 158 / 75KB | 8%, 43%, 1.33, long_scoreboard 34%, math_pipe_throttle 28%, wait 25% | hi/lo 64 rows -> route m=64: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (50, 2, 1) / 4 / 122 / 38KB | 8%, 32%, 1.02, math_pipe_throttle 36%, wait 26%, long_scoreboard 15% |
| 64 | `mma.sync 128x64x64`, 3 stages, splitK 6 +splitKreduce | (8, 4, 6) / 4 / 158 / 75KB | 8%, 42%, 1.30, long_scoreboard 34%, math_pipe_throttle 29%, wait 26% | hi/lo 128 rows -> route m=128: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 4 | (25, 4, 1) / 8 / 122 / 42KB | 17%, 31%, 0.51, math_pipe_throttle 43%, mio_throttle 35%, wait 12% |
| 128 | `mma.sync 128x64x64`, 3 stages, splitK 3 +splitKreduce | (16, 4, 3) / 4 / 158 / 75KB | 8%, 70%, 1.00, math_pipe_throttle 47%, wait 41%, long_scoreboard 6% | hi/lo 256 rows -> route m=256: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (50, 4, 2) / 4 / 122 / 38KB | 15%, 54%, 0.42, math_pipe_throttle 41%, mio_throttle 24%, wait 17% |
| 512 | `mma.sync 256x64x32`, 4 stages, splitK 3 | (64, 2, 3) / 4 / 230 / 83KB | 8%, 81%, 0.33, math_pipe_throttle 50%, wait 43%, long_scoreboard 4% | hi/lo 1024 rows -> route m=512, 2 chunks: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (50, 8, 2) / 4 / 122 / 38KB | 15%, 65%, 0.27, math_pipe_throttle 41%, mio_throttle 26%, wait 16% |
| 1024 | `mma.sync 128x64x64`, 3 stages, splitK 2 | (128, 4, 2) / 4 / 158 / 75KB | 8%, 84%, 0.21, math_pipe_throttle 50%, wait 42%, long_scoreboard 4% | hi/lo 2048 rows -> route m=512, 4 chunks: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (50, 8, 2) / 4 / 122 / 38KB | 15%, 66%, 0.27, math_pipe_throttle 41%, mio_throttle 26%, wait 16% |
| 2048 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (256, 4, 1) / 4 / 158 / 75KB | 8%, 90%, 0.14, math_pipe_throttle 52%, wait 44%, long_scoreboard 2% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 4096 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (128, 7, 1) / 4 / 230 / 83KB | 8%, 90%, 0.12, math_pipe_throttle 53%, wait 45%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 8192 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (1024, 4, 1) / 4 / 158 / 75KB | 8%, 95%, 0.27, math_pipe_throttle 52%, wait 44%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |

#### ffn_up

| M | vLLM kernel | vLLM grid / warps / regs / smem | vLLM ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls | ours route | ours kernel grid / warps / regs / smem | ours ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls |
|---|---|---|---|---|---|---|
| 8 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 98, 1) / 1 / 124 / 18KB | 9%, 13%, 1.53, long_scoreboard 83%, wait 9%, math_pipe_throttle 4% | fp32 matvec (no TC), `nn.Linear` path | (1568, 1, 1) / 4 / 56 / 5KB | 69%, 0%, 0.91, long_scoreboard 95%, lg_throttle 2%, wait 1% |
| 16 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 98, 1) / 1 / 124 / 18KB | 9%, 12%, 1.49, long_scoreboard 81%, wait 9%, math_pipe_throttle 4% | fp32 matvec (no TC), `nn.Linear` path | (1568, 2, 1) / 4 / 54 / 5KB | 71%, 0%, 0.68, long_scoreboard 94%, wait 2%, mio_throttle 1% |
| 32 | `mma.sync 64x64x32`, 6 stages, splitK 5 | (8, 25, 5) / 4 / 88 / 50KB | 16%, 46%, 1.39, long_scoreboard 36%, barrier 24%, math_pipe_throttle 19% | hi/lo 64 rows -> route m=64: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (196, 1, 1) / 4 / 126 / 38KB | 10%, 41%, 1.21, math_pipe_throttle 31%, long_scoreboard 26%, wait 19% |
| 64 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (8, 13, 1) / 4 / 158 / 75KB | 8%, 46%, 1.41, math_pipe_throttle 46%, wait 38%, long_scoreboard 9% | hi/lo 128 rows -> route m=128: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (196, 2, 1) / 4 / 122 / 38KB | 15%, 57%, 0.82, math_pipe_throttle 40%, mio_throttle 23%, wait 17% |
| 128 | `mma.sync 256x64x32`, 4 stages, splitK 3 | (16, 7, 3) / 4 / 230 / 83KB | 8%, 68%, 1.00, math_pipe_throttle 45%, wait 40%, long_scoreboard 8% | hi/lo 256 rows -> route m=256: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (196, 4, 1) / 4 / 122 / 38KB | 15%, 68%, 0.53, math_pipe_throttle 41%, mio_throttle 24%, wait 17% |
| 512 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (64, 13, 1) / 4 / 158 / 75KB | 8%, 87%, 0.36, math_pipe_throttle 51%, wait 43%, long_scoreboard 3% | hi/lo 1024 rows -> route m=512, 2 chunks: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (196, 8, 1) / 4 / 122 / 38KB | 16%, 71%, 0.34, math_pipe_throttle 42%, mio_throttle 25%, wait 16% |
| 1024 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (32, 25, 1) / 4 / 230 / 83KB | 8%, 88%, 0.21, math_pipe_throttle 52%, wait 45%, long_scoreboard 1% | hi/lo 2048 rows -> route m=512, 4 chunks: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (196, 8, 1) / 4 / 122 / 38KB | 16%, 71%, 0.33, math_pipe_throttle 42%, mio_throttle 25%, wait 16% |
| 2048 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (256, 13, 1) / 4 / 158 / 75KB | 8%, 92%, 0.14, math_pipe_throttle 52%, wait 44%, long_scoreboard 2% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 4096 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (512, 13, 1) / 4 / 158 / 75KB | 8%, 94%, 0.11, math_pipe_throttle 52%, wait 44%, long_scoreboard 2% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 8192 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (256, 25, 1) / 4 / 230 / 83KB | 8%, 96%, 0.10, math_pipe_throttle 53%, wait 45%, short_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |

#### ffn_down

| M | vLLM kernel | vLLM grid / warps / regs / smem | vLLM ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls | ours route | ours kernel grid / warps / regs / smem | ours ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls |
|---|---|---|---|---|---|---|
| 8 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 25, 1) / 1 / 124 / 18KB | 2%, 8%, 1.02, long_scoreboard 61%, wait 21%, math_pipe_throttle 10% | fp32 matvec (no TC), `nn.Linear` path | (784, 1, 1) / 4 / 54 / 5KB | 37%, 0%, 0.78, long_scoreboard 89%, lg_throttle 7%, wait 2% |
| 16 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 25, 1) / 1 / 124 / 18KB | 2%, 8%, 1.01, long_scoreboard 61%, wait 21%, math_pipe_throttle 10% | fp32 matvec (no TC), `nn.Linear` path | (784, 2, 1) / 4 / 48 / 5KB | 66%, 0%, 0.22, long_scoreboard 61%, lg_throttle 34%, mio_throttle 2% |
| 32 | `mma.sync 128x64x64`, 3 stages, splitK 6 +splitKreduce | (8, 4, 6) / 4 / 158 / 75KB | 8%, 49%, 1.50, math_pipe_throttle 31%, long_scoreboard 28%, wait 26% | hi/lo 64 rows -> route m=64: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (50, 2, 1) / 4 / 122 / 38KB | 8%, 35%, 1.12, math_pipe_throttle 37%, wait 27%, long_scoreboard 18% |
| 64 | `mma.sync 128x64x64`, 3 stages, splitK 6 +splitKreduce | (8, 4, 6) / 4 / 158 / 75KB | 8%, 50%, 1.50, math_pipe_throttle 32%, wait 27%, long_scoreboard 26% | hi/lo 128 rows -> route m=128: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 4 | (25, 4, 1) / 8 / 122 / 42KB | 17%, 36%, 0.64, math_pipe_throttle 47%, mio_throttle 31%, wait 13% |
| 128 | `mma.sync 128x64x64`, 3 stages, splitK 3 +splitKreduce | (16, 4, 3) / 4 / 158 / 75KB | 8%, 78%, 1.12, math_pipe_throttle 51%, wait 43%, long_scoreboard 3% | hi/lo 256 rows -> route m=256: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (50, 4, 2) / 4 / 122 / 38KB | 15%, 60%, 0.50, math_pipe_throttle 43%, mio_throttle 23%, wait 18% |
| 512 | `mma.sync 128x64x64`, 3 stages, splitK 5 | (64, 4, 5) / 4 / 158 / 75KB | 8%, 87%, 0.40, math_pipe_throttle 49%, wait 42%, long_scoreboard 5% | hi/lo 1024 rows -> route m=512, 2 chunks: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (50, 8, 2) / 4 / 122 / 38KB | 16%, 70%, 0.34, math_pipe_throttle 44%, mio_throttle 24%, wait 17% |
| 1024 | `mma.sync 128x64x64`, 3 stages, splitK 5 | (128, 4, 5) / 4 / 158 / 75KB | 8%, 87%, 0.24, math_pipe_throttle 49%, wait 42%, long_scoreboard 5% | hi/lo 2048 rows -> route m=512, 4 chunks: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 2 | (50, 8, 2) / 4 / 122 / 38KB | 16%, 70%, 0.34, math_pipe_throttle 44%, mio_throttle 24%, wait 17% |
| 2048 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (256, 4, 1) / 4 / 158 / 75KB | 8%, 92%, 0.20, math_pipe_throttle 53%, wait 44%, long_scoreboard 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 4096 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (512, 4, 1) / 4 / 158 / 75KB | 8%, 93%, 0.30, math_pipe_throttle 53%, wait 44%, dispatch_stall 1% | pieces of 1024 | (as M=1024) | (as M=1024) |
| 8192 | `mma.sync 64x256x32`, 4 stages, splitK 1 | (256, 7, 1) / 4 / 230 / 83KB | 8%, 92%, 0.54, math_pipe_throttle 54%, wait 45%, dispatch_stall 0% | pieces of 1024 | (as M=1024) | (as M=1024) |

#### output

| M | vLLM kernel | vLLM grid / warps / regs / smem | vLLM ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls | ours route | ours kernel grid / warps / regs / smem | ours ncu: achieved occ, tensor-pipe %, DRAM TB/s, top-3 stalls |
|---|---|---|---|---|---|---|
| 8 | `wmma 16x16x128`, 2 stages, splitK 1 | (8, 1024, 1) / 1 / 124 / 18KB | 10%, 13%, 1.68, long_scoreboard 85%, wait 8%, math_pipe_throttle 4% | fp32 matvec (no TC), `nn.Linear` path | (16384, 1, 1) / 4 / 56 / 5KB | 73%, 0%, 1.51, long_scoreboard 92%, barrier 4%, wait 1% |
| 16 | `wmma 16x16x128`, 1 stages, splitK 1 | (8, 1024, 1) / 1 / 96 / 10KB | 20%, 12%, 1.67, long_scoreboard 68%, lg_throttle 13%, mio_throttle 12% | fp32 matvec (no TC), `nn.Linear` path | (16384, 2, 1) / 4 / 54 / 5KB | 74%, 0%, 1.52, long_scoreboard 93%, barrier 3%, wait 2% |
| 32 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (8, 128, 1) / 4 / 158 / 75KB | 8%, 46%, 1.57, long_scoreboard 35%, math_pipe_throttle 25%, wait 21% | hi/lo 64 rows -> route m=64: tile 64x64x64 (Mtok x Nout x K), 4 warps, 2 LDS bufs, splitK 1 | (2048, 1, 1) / 4 / 126 / 38KB | 16%, 46%, 1.58, long_scoreboard 46%, barrier 19%, math_pipe_throttle 18% |
| 64 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (8, 128, 1) / 4 / 158 / 75KB | 8%, 46%, 1.56, long_scoreboard 36%, math_pipe_throttle 25%, wait 21% | hi/lo 128 rows -> route m=128: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 1 | (1024, 1, 1) / 8 / 122 / 42KB | 32%, 51%, 0.94, mio_throttle 39%, math_pipe_throttle 34%, barrier 13% |
| 128 | `mma.sync 128x64x64`, 3 stages, splitK 1 | (16, 128, 1) / 4 / 158 / 75KB | 8%, 88%, 1.43, math_pipe_throttle 49%, wait 42%, long_scoreboard 5% | hi/lo 256 rows -> route m=256: tile 128x128x32 (Mtok x Nout x K), 8 warps, 2 LDS bufs, splitK 1 | (1024, 2, 1) / 8 / 124 / 42KB | 32%, 52%, 0.96, mio_throttle 46%, math_pipe_throttle 31%, barrier 13% |

### 5C. Plain-bf16 routed variant (route_dense_bf16, no hi/lo; not production) vs production hi/lo vs vLLM, us

| role | M=32 bf16 / hilo / vLLM | M=64 bf16 / hilo / vLLM | M=128 bf16 / hilo / vLLM | M=512 bf16 / hilo / vLLM | M=1024 bf16 / hilo / vLLM |
|---|---|---|---|---|---|
| ssm_in | 127 / 93 / 73 | 95 / 149 / 74 | 154 / 247 / 91 | 523 / 1019 / 292 | 1050 / 2223 / 547 |
| ssm_out | 64 / 48 / 35 | 48 / 74 / 36 | 73 / 112 / 41 | 189 / 383 / 139 | 370 / 794 / 305 |
| attn_q | 39 / 43 / 24 | 42 / 42 / 25 | 38 / 63 / 24 | 118 / 232 / 82 | 231 / 512 / 163 |
| attn_kv | 25 / 27 / 8 | 25 / 31 / 8 | 29 / 33 / 9 | 35 / 72 / 23 | 76 / 140 / 43 |
| attn_o | 35 / 34 / 25 | 34 / 58 / 26 | 55 / 86 / 29 | 141 / 251 / 98 | 266 / 530 / 181 |
| ffn_up | 95 / 84 / 53 | 81 / 112 / 54 | 110 / 165 / 69 | 306 / 616 / 211 | 637 / 1337 / 408 |
| ffn_down | 109 / 74 / 53 | 74 / 117 / 54 | 115 / 194 / 59 | 316 / 646 / 194 | 623 / 1312 / 393 |
| output | 899 / 622 / 524 | 591 / 1022 / 530 | 986 / 1975 / 553 | - | - |

### 5D. Lever ranking (per step; gap = ours - vLLM)


RL decode step, B=32 (per step, ms; ours step 13.3 ms RolloutSampler): GEMM total ours 6.81 ms (51% of step) vs vLLM 4.83 ms; gap 1.97 ms

| role | ours ms | vLLM ms | gap ms | gap % of our step |
|---|---|---|---|---|
| ffn_up | 1.429 | 0.903 | 0.525 | 4.0% |
| ssm_in | 1.963 | 1.543 | 0.420 | 3.2% |
| ffn_down | 1.263 | 0.899 | 0.364 | 2.7% |
| ssm_out | 1.006 | 0.736 | 0.269 | 2.0% |
| attn_qkv (ours q+k+v; vLLM fused qkv) | 0.390 | 0.131 | 0.258 | 1.9% |
| output | 0.622 | 0.524 | 0.098 | 0.7% |
| attn_o | 0.135 | 0.098 | 0.037 | 0.3% |

RL decode step, B=64 (per step, ms; ours step 21.6 ms): GEMM total ours 10.26 ms (47% of step) vs vLLM 4.90 ms; gap 5.36 ms

| role | ours ms | vLLM ms | gap ms | gap % of our step |
|---|---|---|---|---|
| ssm_in | 3.138 | 1.555 | 1.582 | 7.3% |
| ffn_down | 1.990 | 0.914 | 1.075 | 5.0% |
| ffn_up | 1.908 | 0.910 | 0.999 | 4.6% |
| ssm_out | 1.555 | 0.755 | 0.800 | 3.7% |
| output | 1.022 | 0.530 | 0.493 | 2.3% |
| attn_qkv (ours q+k+v; vLLM fused qkv) | 0.412 | 0.135 | 0.277 | 1.3% |
| attn_o | 0.233 | 0.103 | 0.130 | 0.6% |

RL decode step, B=128 (per step, ms; ours step 35.2 ms): GEMM total ours 16.48 ms (47% of step) vs vLLM 5.77 ms; gap 10.72 ms

| role | ours ms | vLLM ms | gap ms | gap % of our step |
|---|---|---|---|---|
| ssm_in | 5.185 | 1.919 | 3.266 | 9.3% |
| ffn_down | 3.301 | 1.002 | 2.299 | 6.5% |
| ffn_up | 2.805 | 1.169 | 1.636 | 4.6% |
| ssm_out | 2.361 | 0.854 | 1.507 | 4.3% |
| output | 1.975 | 0.553 | 1.422 | 4.0% |
| attn_qkv (ours q+k+v; vLLM fused qkv) | 0.513 | 0.154 | 0.360 | 1.0% |
| attn_o | 0.342 | 0.115 | 0.228 | 0.6% |

RL decode step, B=8 (per step, ms; ours step 8.8 ms): GEMM total ours 6.68 ms (76% of step) vs vLLM 5.23 ms; gap 1.45 ms

| role | ours ms | vLLM ms | gap ms | gap % of our step |
|---|---|---|---|---|
| ssm_in | 2.065 | 1.516 | 0.550 | 6.2% |
| ffn_up | 1.249 | 0.847 | 0.402 | 4.6% |
| attn_qkv (ours q+k+v; vLLM fused qkv) | 0.474 | 0.120 | 0.354 | 4.0% |
| output | 0.544 | 0.490 | 0.054 | 0.6% |
| ssm_out | 0.970 | 0.926 | 0.044 | 0.5% |
| ffn_down | 1.249 | 1.205 | 0.043 | 0.5% |
| attn_o | 0.131 | 0.123 | 0.007 | 0.1% |

10k prefill (ms; ours 1650 ms; ours = 10000/1024 pieces of 1024, vLLM = 10000/8192 x the M=8192 kernel; lm_head excluded): GEMM total ours 1110.32 ms (67% of step) vs vLLM 286.03 ms; gap 824.29 ms

| role | ours ms | vLLM ms | gap ms | gap % of our step |
|---|---|---|---|---|
| ssm_in | 455.971 | 101.446 | 354.525 | 21.5% |
| ffn_up | 222.002 | 59.114 | 162.888 | 9.9% |
| ffn_down | 217.739 | 63.925 | 153.815 | 9.3% |
| ssm_out | 162.918 | 47.234 | 115.684 | 7.0% |
| attn_qkv (ours q+k+v; vLLM fused qkv) | 30.991 | 8.271 | 22.720 | 1.4% |
| attn_o | 20.696 | 6.037 | 14.660 | 0.9% |
| output | 0.000 | 0.000 | 0.000 | 0.0% |

## 6. Reproduce

All from a checkout of `exp` with a BoltBeam checkout next to it (or `BOLTBEAM_ROOT`), each GPU step under the
exclusive lock (`~/scratchpad/bin/gpu-run time ...`):

```
B=docs/nemotron-vllm-parity/bench/audit
/home/ubuntu/vllm-bench/.venv/bin/python $B/vllm_gemm.py OUT/vllm_gemm.json         # vLLM-side timings
$B/ours_sweep.sh OUT/ours.json                                                     # ours, prod + bf16 modes
$B/vncu.sh OUT/vncu ssm_in,ssm_out,qkv,attn_q,attn_kv,attn_o,ffn_up,ffn_down,output 8,16,32,64,128,512,1024,2048,4096,8192
$B/oncu.sh OUT/oncu ssm_in,ssm_out,attn_q,attn_kv,attn_o,ffn_up,ffn_down,output 8,16,32,64,128,512,1024
python3 $B/audit_table.py OUT tinygrad/llm/generated/dense_bf16_sm120_candidate_set.json
```

**NCU is BoltBeam's** (2026-09-26, Julian): `vncu.sh`/`oncu.sh` are thin callers of `boltbeam ncu-collect` (one ncu
command line with the section list incl. `InstructionStats`, `--cache-control all --clock-control none`, the
memory-capped `sudo systemd-run --scope -p MemoryMax=20G`; raw export and a unit-aware import into
`boltbeam.ncu_kernel_counters.v1`, launches labelled by NVTX `role/M` or the target's `SHAPE role M n` lines).
`audit_table.py` parses through BoltBeam's `parse_kernel_row` (its tables are byte-identical to the original run).
BoltBeam's own side-by-side with the gap decomposition and research queue is `boltbeam ncu-audit`; the whole scan
(strategy table + NCU audit + lifecycle) is `docs/nemotron-vllm-parity/bench/scan.sh OUTDIR`.  tinygrad keeps only
the exporter: `extra/llm_research/decode/nv_cubin_capture.py` writes the cubin + launch spec
(`tinygrad.nv_cubin_capture.v1`, `--run SCRIPT ARGS` for any program); BoltBeam replays it under ncu
(`boltbeam.collectors.cubin_launch`, which `nv_cubin_ncu_launcher.py` now forwards to).  The 2026-09-26 captures are
imported in BoltBeam `evidence/ncu/nemotron_kernel_audit_20260926/` (with a GPU proof of the ssm_in M=128 row);
`results/ncu-kernel-counters-20260926.json` is the first conversion, kept as history -- its shared-memory fields have
the static/dynamic units swapped (ncu reports static in byte/block, dynamic in Kbyte/block).
The whole audit is ~40 min of GPU time.
