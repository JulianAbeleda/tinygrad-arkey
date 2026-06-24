# llama.cpp decode primitive audit (HIP/gfx1100) — 2026-06-17

Read-only source audit of `/home/ubuntu/env/llama.cpp` (HIP backend reuses the CUDA kernels in
`ggml/src/ggml-cuda/`, compiled via HIP; `dp4a` → `__builtin_amdgcn_sudot4` on RDNA3). Every claim cites a
file/function. Hypotheses marked HYP. Goal: what makes llama's single-token decode efficient.

## Primitive table

| primitive | file:function | what's specialized / fused |
|---|---|---|
| Q4_K/Q6_K matvec (decode) | `mmvq.cu mul_mat_vec_q<type,ncols_dst,has_fusion>`; dispatch `ggml_cuda_should_use_mmvq()` on `ne11==1` | dedicated batch-1 (vector) path; **dequant inlined into the dot** |
| Q4_K dot | `vecdotq.cuh vec_dot_q4_K_q8_1_impl_vmmq` (~line 505) | 2× `ggml_cuda_dp4a` (int8 4-way dot) per block; `scale*(quant_dot − min_dot)` fused in registers; **no separate dequant kernel** |
| Q6_K dot | `vecdotq.cuh vec_dot_q6_K_q8_1_impl_mmvq` (~624) | 1× dp4a/block; extract q6 hi/lo, −32, scale — fused |
| warp config | `mmvq.cu calc_nwarps` device table (~397) | Q4_K ncols=1: 8 warps (RDNA3); 1–2 rows/block |
| RMSNorm | `norm.cu rms_norm_f32<block,do_multiply,do_add>` (~77-158) | **fused scale × weight + residual-add**: `dst = scale*x*mul + add` (~147) — 1 kernel does norm+scale+add |
| RoPE | `rope.cu rope_norm<forward,has_ff,T,D>` (~44) | **fused view + set_rows**; cos/sin computed inline (no LUT) |
| decode attention | `fattn-vec.cuh flash_attn_ext_vec<D,ncols=1,...>` | dedicated **1-query flash** kernel (128 threads); QK·softmax·V fused; KV can be quantized; no GEMM fallback at batch-1 |
| FFN SwiGLU | `mmvq.cu has_fusion` branch (~526-667) | **gate matmul + up matmul + silu·mul fused in one mmvq kernel**; else `unary.cu swiglu_oai_kernel` |
| residual/elementwise | `binbcast.cu add_f32` / mostly folded into `norm.cu do_add` | residual add fused into the following RMSNorm |
| scheduler / dispatch | `ggml-cuda.cu` graph capture (~3296-4490), `cudaGraphLaunch` (~4484); fuse detector `ggml_cuda_try_fuse` (~4424) | **captures the decode graph → 1 launch**, amortizing per-kernel host launch. HYP: graph may be **disabled on AMD/HIP** (CC≥AMPERE check ~3497) — if so llama eats per-launch host cost on ROCm |
| memory | `ggml-alloc.c` graph allocator | activation buffer reuse |
| logits/sampling | `softmax.cu soft_max_f32`; last-token logits at decode | decode computes logits for the last token only |

## Per-token kernel-launch estimate (n_layers=36)

Naive per-layer ≈ 11 ops (rmsnorm, 3 qkv matvec, rope, attn, o-proj, residual, rmsnorm, ffn gate/up/down,
residual) → ~400. **With llama fusion** (rmsnorm+scale+add, gate*up*silu, rope+view) ≈ **~7/layer → ~260 total**
(HYP: exact count depends on fusion enablement). With CUDA graph → effectively **1 host launch**.

## Top 3 things llama does that a naive per-op dispatch (tinygrad-like) would not

1. **Multi-op fusion** — RMSNorm+scale+residual (`norm.cu:147`), gate*up*silu (`mmvq.cu:526-667 has_fusion`),
   RoPE+view+set_rows (`rope.cu:79-84`): ~2–3× fewer kernels, fewer memory round-trips.
2. **Inline-dequant int8 matvec** — `vecdotq.cuh` dp4a dot folds dequant into the dot (no dequantized
   intermediate in DRAM). (tinygrad's QK primitive already does this — int-dot GEMV, ~76% HBM — so this gap is
   ~closed.)
3. **Graph capture** — `ggml-cuda.cu:4484 cudaGraphLaunch` collapses the decode graph to one launch.
   (tinygrad's TinyJit HCQ graph also batches — so this is NOT a tinygrad disadvantage; both amortize launch.)

## So where can the gap be? (answered by measurement in the gap-plan doc)

Since (2) and (3) are ~matched by tinygrad, llama's remaining edge is **(1) fusion** — the surrounding small ops
(norm/rope/silu/residual) that tinygrad runs as separate kernels (~21% of decode GPU per the census) and a
~6.5 ms copy (~17%). Plus a **dedicated 1-query flash decode attention** (`fattn-vec.cuh`) that degrades
gracefully at long context, where tinygrad's baseline SDPA decays 3.4× (and tinygrad's own flash-decode recovers
1.73× @4096). See `docs/llama-vs-tinygrad-primitive-gap-plan-20260617.md`.
