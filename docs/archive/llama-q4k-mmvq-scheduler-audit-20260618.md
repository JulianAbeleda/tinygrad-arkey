# llama Q4_K MMVQ scheduler / work-decomposition audit (2026-06-18)

Audit-only (no kernels). The inner-loop math was audited before; **this audits the SCHEDULER — and it overturns
the prior "53% ceiling / backend territory" verdict.** llama reaches 70% via a work decomposition tinygrad has
**not** tried: ~128 threads cooperate on ONE row (K-blocks parallelized across threads, not serial), reduced
in-kernel by warp-shuffle + small shared-mem, one write. RX 7900 XTX (gfx1100/RDNA3), Qwen3-8B-Q4_K_M.

## Phase 0 — exact llama runtime path

| stage | file | function / template | constants (Q4_K, ncols_dst=1, RDNA3) |
|---|---|---|---|
| MMVQ launch+kernel | `ggml/src/ggml-cuda/mmvq.cu:477` | `mul_mat_vec_q<type, ncols_dst, has_fusion, small_k>`, `__launch_bounds__(nwarps*warp_size,1)` | **nwarps=4** (GENERIC table; RDNA3 ∉ GCN/RDNA4) → **128 threads/block**; **rows_per_cuda_block=1** |
| vecdot | `vecdotq.cuh:864 / 505` | `vec_dot_q4_K_q8_1` → `_impl_vmmq` | VDR=2, QR4_K=2 |
| q8_1 quant | `quantize.cu` | `quantize_q8_1` (once per activation) | QK8_1=32 |
| dp4a | `common.cuh:694` | `ggml_cuda_dp4a` → `__builtin_amdgcn_sdot4` | (tinygrad matches via inline-asm `v_dot4_i32_iu8`) |
| warp reduce | `common.cuh:447` | `warp_reduce_sum` (`__shfl_xor_sync`) | width=warp_size |

## Phase 1 — inner-loop math (refresh; matched by tinygrad `_sdot4`)

`(v>>4i)&0x0F0F0F0F` packed extract → dp4a dot (`vec_dot_q4_K_q8_1_impl_vmmq`) → `dp4a(0x01010101,u)` qsum →
per-group `sc`/`m` → block `dm` once. ~4 dp4a + ~few fp per 256-wt block. **tinygrad's `_sdot4` path reproduces
this exactly (native v_dot4) — the math/dot is NOT the gap.** (See `llama-q4k-mmvq-inner-loop-audit-20260618.md`.)

## Phase 2 — llama scheduler / work decomposition (the new audit)

```
mul_mat_vec_q (Q4_K, ncols_dst=1, RDNA3):
  block = 4 warps x 32 lanes = 128 threads; computes rows_per_cuda_block = 1 output row
  grid.x = nrows / 1
  tid = 32*threadIdx.y + threadIdx.x          # threadIdx.y = warp, threadIdx.x = lane
  float tmp[1][1] = 0                          # ONE register accumulator per thread
  for kbx = tid/(qi/vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter:   # K-blocks STRIDED across all 128 threads
      kqs = vdr*(tid % (qi/vdr))               # within-block sub-chunk for this thread (coalesced)
      tmp += vec_dot_q4_K_q8_1(vx, &y[kby], row*stride + kbx, kqs)
  # REDUCTION, fully in-kernel:
  __shared__ tmp_shared[nwarps-1][1][1][warp_size]
  if warp>0: tmp_shared[warp-1][lane] = tmp; __syncthreads()        # cross-warp -> shared (3 values)
  if warp==0: for l in 0..nwarps-2: tmp += tmp_shared[l][lane]      # warp 0 sums other warps
             tmp = warp_reduce_sum(tmp)                              # within-warp __shfl_xor (32->1)
             if lane < 1: dst[row] = d*tmp ...                      # ONE output write
```

| feature | llama behavior | evidence | relevance to tinygrad |
|---|---|---|---|
| row mapping | **1 row per block, 128 threads cooperate** | rows_per_cuda_block=1, nwarps=4 | tinygrad coop uses only **8** threads/row |
| K split | **K-blocks parallelized across 128 threads** (strided, no serial block loop) | `kbx += blocks_per_iter` | tinygrad coop: 8 lanes, **serial blk REDUCE loop** over all 16 blocks |
| within-block | qi/vdr lanes split one block's quant-words (consecutive→coalesced) | `kqs = vdr*(tid%(qi/vdr))` | matched (8-lane within-block) |
| accumulator | **1 register per thread** (`tmp`) | `float tmp[1][1]` | matched (register) |
| reduction | **warp-shuffle (within warp) + shared (cross 4 warps), IN-KERNEL** | `warp_reduce_sum` + `tmp_shared` | tinygrad: **global partials + EXTERNAL stage-2 .sum** (the 10% drag) |
| output write | **one write per row** (lane 0) | `if lane<rows: dst[...]` | tinygrad: 8 partials/row + stage-2 |
| q8 reuse | y loaded per block, reused across rows_per_block | `&y[kby]` shared | n/a at rows=1 |
| occupancy | 128 threads/row × many rows → saturates CUs | launch geom | tinygrad coop: 8 threads/row → **16× less K-parallelism** |

### Why ~70%
Dominant: **16× more threads per row** (128 vs 8) parallelizing the K-block dimension (no serial loop) → far
better coalescing + occupancy + latency hiding; plus **in-kernel warp-shuffle reduction** (no global partials
round-trip, no external stage-2). NOT primarily "less math" (the dot is matched by `_sdot4`).

## Phase 3 — tinygrad scheduler audit

| variant | row mapping | K mapping | reduction | scale decode | coalescing | % peak | bottleneck |
|---|---|---|---|---|---|---|---|
| base fp | 1 thread/row (LOCAL:64 rows) | serial, whole row | none | per-thread | **no** | 40 | uncoalesced |
| fp coop | **8 lanes/row** | 8 within-block lanes, **serial blk loop** | global partials + **stage-2 .sum** | per-lane | yes | 48 (partial-alone 53) | few threads/row + stage-2 |
| `_sdot4` | 8 lanes/row | same + native dot4 | partials + stage-2 | per-lane | yes | 49 | same (dot4 wasn't it) |
| opaque asm | 1 thread/row, hand-packed | serial | in-thread | per-thread | partial | 52 | one-thread serial |

## Phase 4 — side-by-side gap

| feature | llama | tinygrad best | status |
|---|---|---|---|
| dot4 | native dp4a | `_sdot4` native v_dot4 | **already matched** |
| packed extract | `&0x0F0F0F0F` | same | **matched** |
| threads per row | **128** | **8** | **MISSING (the key gap), expressible** |
| K-block parallelism | across threads (no serial loop) | serial blk REDUCE | **MISSING, expressible** |
| reduction | warp-shuffle + shared, in-kernel | global partials + stage-2 | **MISSING but expressible** (`extra/amd_warp_reduce.py` provides `ds_bpermute` warp_reduce_sum) |
| one output write | yes | no (partials) | MISSING, follows from in-kernel reduce |
| register accumulator | yes | yes | matched |

**Conclusion: llama wins from WORK DECOMPOSITION, not math.** The gap is the 128-threads-per-row +
K-block-parallel + warp-shuffle-reduce structure, which tinygrad has **not tried** (the prior fused-coop-row arc
tested only 8 threads/row + LDS → 53%). The enabling primitives exist (`_sdot4`, `amd_warp_reduce.warp_reduce_sum`
via `ds_bpermute`, `MV_THREADS_PER_ROW`/GROUP for the standard path). See
`q4k-mmvq-missing-quadrant-design-20260618.md`.
