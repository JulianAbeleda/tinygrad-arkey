# llama.cpp ROCm decode-attention audit — why llama is context-flat (2026-06-17)

Audit only (no tinygrad kernel built, no llama code copied). Goal: identify llama's actual ROCm decode-attention
primitive and explain why llama decode is ~context-flat (99.5→92.2 tok/s, −7%) while tinygrad decays −43%.
Every llama claim cites file/function; **measured** vs **inferred** tagged.

## Phase 0 — version pin [measured]

- llama.cpp **build b9592**, commit `ac4cddeb0`, ROCm **7.2.4**, `AMDGPU_TARGETS=gfx1100`, Release.
- Flags (`build/CMakeCache.txt`): `GGML_HIP=ON`, `GGML_CUDA_FA=ON`, `GGML_HIP_GRAPHS=ON`,
  **`GGML_HIP_ROCWMMA_FATTN=OFF`**, `GGML_HIP_MMQ_MFMA=ON`, `GGML_CUDA_FA_ALL_QUANTS=OFF`.
- Bench: `llama-bench -m Qwen3-8B-Q4_K_M.gguf -ngl 99 -p 0 -n 128 -d 0,512,1024,2048,4096 -r 3`.
- GPU: RX 7900 XTX (gfx1100), 24 GB. Artifact: `bench/llama-rocm-attention-audit/provenance.json`.

## Phase 1 — which kernel decode uses [measured: source dispatch]

Dispatch chain (`ggml/src/ggml-cuda/fattn.cu`):
`ggml_cuda_flash_attn_ext` → `ggml_cuda_get_best_fattn_kernel(device, dst)` → switch.

For Qwen3-8B decode (head dim `DKQ=DV=128`, `gqa_ratio = 32/8 = 4`, fp16 K/V, `Q->ne[1]==1` = batch-1):
- `gqa_opt_applies = true` (`gqa_ratio≥2 && mask && max_bias==0 && KV % FATTN_KQ_STRIDE == 0`) — fattn.cu:~440.
- `should_use_wmma_fattn(cc)` → **false** (`fattn-wmma-f16.cuh:26`: HIP without `GGML_HIP_ROCWMMA_FATTN` returns
  false) → **WMMA fattn unused**.
- RDNA3 has no MFMA (`amd_mfma_available`=false); the `amd_wmma` MMA branch needs `ne[1]*gqa_eff > 8`
  (batch-1 → 1·4 = 4) → skipped.
- Final block (fattn.cu:~523): `can_use_vector_kernel` true, K/V not quantized, `ne[1]==1`, but the VEC return
  is guarded by `if (!gqa_opt_applies)` — which is **false** here → falls through to **`return
  BEST_FATTN_KERNEL_TILE`**.

→ **Decode attention = the TILE kernel** (`fattn-tile.cu` / `fattn-tile.cuh`), *not* vec, *not* WMMA.

## Phase 2 — runtime confirmation [measured: AMD_LOG_LEVEL=3]

Dispatch-name trace of a real decode run (`-d 1024 -n 8`):

| kernel | role |
|---|---|
| **`flash_attn_tile`** | the decode attention kernel (confirms the source trace) |
| **`flash_attn_stream_k_fixup_general`** | stream-K split partial-reduction fixup (`fattn-common.cuh:804`) |
| **`flash_attn_combine_results`** | merge split-KV partials (`fattn-common.cuh:913`) |
| `flash_attn_ext_f16` | the **prefill** (d1024 prompt) path, not decode |

rocprofv3 is installed but produced no usable trace in this environment; AMD_LOG_LEVEL dispatch names are the
runtime evidence.

## Phase 3 — the TILE primitive structure [measured: source] vs tinygrad hoisted flash

llama tile config for `(DKQ,DV,ncols) = (128,128,4)` on RDNA (`fattn-tile.cuh` `..._get_config_amd_rdna`):
**nthreads=128, occupancy=8, nbatch_fa=64, nbatch_K=64**. (`ncols = gqa_ratio = 4` — the GQA group is the tile
width.) Structure (`fattn-tile.cuh`/`fattn-common.cuh`):

| feature | llama.cpp tile (ROCm) | tinygrad hoisted flash | impact | m/i |
|---|---|---|---|---|
| kernels / layer (decode attn) | **1** tile + stream-K fixup + combine (≈3, mostly the 1 tile) | **6** (`flash_max/prob/partial_v2/gmax/den/combine`) + score matmul | launch + intermediate traffic | measured |
| GQA handling | **group batched as `ncols=4`** — K/V tile staged once, reused across the 4 query heads | **query head `h` is a GLOBAL axis** → V[h//G] re-read **4× (per head)** | **4× V traffic** in tinygrad | measured (source) |
| K/V staging | cooperative **`__shared__` tile_KV**, vectorized **half2 16-byte** coalesced copies (`load_tile_*`) | per-thread strided global/IC reads in `flash_partial_v2` | effective BW | measured (source) |
| occupancy over KV | **stream-K**: KV split across enough blocks to fill all CUs (occupancy=8/CU) + fixup/combine | fixed-L split `S=cdiv(KV,L=128)` workgroups (context-dependent block count) | full GPU fill at any ctx | measured |
| softmax | online, register/shared state, `nbatch_fa=64` KV/iter | 2-pass (max → prob → weighted-sum), exp hoisted to `flash_prob` | passes/state | measured |
| KV dtype | fp16 (default) | fp16 cache | — | measured |
| graph | HIP graph (`GGML_HIP_GRAPHS=ON`) — fixed launch overhead | TinyJit HCQ graph (also batched) | both amortize launch | measured |
| effective attn BW | high (not the decode bottleneck) | **~33 GB/s** (decode-block map) — occupancy/issue-bound | the slope driver | measured |

## Why context-flat (llama −7%) vs decaying (tinygrad −43%) [the answer]

**Decode is weight-bandwidth-bound.** Both read the full 4.68 GiB Q4 model per token (fixed, context-
independent): llama 99.5 tok/s ⇒ ~10 ms/token ⇒ ~470 GB/s effective on weights. Attention adds the K/V read,
which grows with context: at ctx4096, K+V = ~16 MB fp16.

- **llama:** the tile kernel makes that 16 MB cost *small and efficient* — stream-K fills the GPU at any KV
  length, the GQA group reuses one staged K/V tile (no redundancy), vectorized coalesced fp16 LDS loads hit high
  BW. So attention is a thin marginal add on top of the dominant fixed weight read → **−7%** (10.0→10.8 ms).
- **tinygrad:** attention is a *large, inefficient, growing* fraction. `flash_partial_v2` reads V **4×
  redundant** (GQA not reused) at **~33 GB/s** effective (per-thread strided, not vectorized-LDS), 6 kernels,
  fixed-L split. Its share goes **13%@ctx512 → 47%@ctx4096** (decode-block map) → **−43%** slope. On top of a
  base decode already ~2.3× slower than llama (the structural GEMV/program-granularity gap).

**Reconciliation with the v3 refutation:** v3 found *naive* LDS/WMMA single-query staging 0.5–0.77× (slower).
llama proves LDS staging *wins* — but only with the **missing ingredients**: stream-K occupancy + GQA-batched
tile width (ncols=4) + vectorized coalesced loads. So LDS itself was never refuted; *naive LDS without stream-K
and GQA-batching* was. The audited primitive names exactly what to add.

**Ceiling caveat:** even a perfect (flat) attention only removes the *slope*. tinygrad would then be ~flat at its
base-decode rate (~43 tok/s) — still ~44% of llama, because the **base-decode 2.3× gap (GEMV + ~780 progs/token
vs llama's fused)** is the larger structural limiter. Attention is the *long-context* lever (~+60% @ctx4096),
not the headline gap.

## Unknowns / not measured
- Exact achieved BW of the llama tile kernel (no working rocprof here) — inferred from the −7% slope + weight-BW
  arithmetic, not counter-measured.
- Per-kernel time split inside llama decode (weights vs attention) — inferred from context slope, not profiled.
