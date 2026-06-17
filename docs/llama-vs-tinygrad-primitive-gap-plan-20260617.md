# llama.cpp vs tinygrad decode — primitive gap analysis & ranked plan (2026-06-17)

Settles *why* llama.cpp decodes faster, by primitive + measurement (not vibes). Sources: llama source audit
(`docs/llama-primitive-audit` findings, citing `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/*`), tinygrad
per-token census (`bench/qk-decode-primitive-census/`), llama-bench (same model/GPU, ROCm), long-context curve
(`bench/qk-long-context-20260617/`).

## Measured baselines (Qwen3-8B-Q4_K_M, RX 7900 XTX / gfx1100)

| metric | llama.cpp (ROCm) | tinygrad | ratio |
|---|---:|---:|---:|
| decode short (tg128) | ~80–100 tok/s (thermal-noisy) | 54 (no-demote) / ~64 (banked) | ~0.6–0.8× |
| prefill (pp512) | 2742–3104 tok/s | ~2486 (PREFILL_V2) | ~0.8× |
| decode @ ctx4096 | (flash, graceful) | 11.4 baseline / **19.7 flash-decode** | — |
| GPU programs / token | ~260 fused (audit estimate) | **780** | 3× |

## Primitive comparison

| primitive | llama.cpp (file:fn) | tinygrad | measured tinygrad cost | gap type | next action |
|---|---|---|---|---|---|
| Q4_K/Q6_K GEMV | `mmvq.cu mul_mat_vec_q` + `vecdotq.cuh vec_dot_q4_K_q8_1` (dp4a/`__builtin_amdgcn_sudot4`), inline-dequant, ne11==1 vec path | `extra/q4_k/q6_k_gemv_primitive` int-dot GEMV | ~62% of decode GPU; ~76% HBM standalone | **~none** (competitive) | leave; already int-dot, near-peak |
| QKV projections | 3× mmvq (or fused) | 3 QK GEMVs | attn_q/o 10.6%, attn_k/v 1.9% | minor | (covered by GEMV) |
| FFN gate/up/down | gate+up+silu **fused** in one mmvq (`has_fusion`, mmvq.cu:526-667); down separate | 3 separate QK GEMVs + separate silu/mul | ffn_down 19.9%, ffn_gate/up 15.2% (+ silu in small-ops) | **missing fusion** | fuse gate*up*silu (codegen) |
| RMSNorm | `norm.cu rms_norm_f32<do_multiply,do_add>` — **fused scale + residual-add** | separate norm + mul + add kernels | inside nonGEMV 20.6% | **missing fusion** | fuse norm+scale+residual |
| RoPE | `rope.cu rope_norm` — **fused view+set_rows**, inline cos/sin | separate rope + cat kernels | inside nonGEMV 20.6% | **missing fusion** | fuse rope path |
| decode attention / KV | `fattn-vec.cuh flash_attn_ext_vec<D,ncols=1>` — dedicated 1-query flash, 128 threads | SDPA (baseline) or `qk_flash_decode` (FLASH_DECODE) | dominates long ctx; baseline decays 3.4× @4096 | **kernel/occupancy** at long ctx | **flash-decode default for long ctx** |
| residual / small elementwise | mostly fused into norm | separate add/mul/cast kernels | inside nonGEMV 20.6% (580 kernels) | **missing fusion** | scheduler fusion |
| lm_head | mmvq last token | q6k_gemv 151936×4096 | 14.6% (biggest single kernel) | extra at prefill | last-token-only logits at prefill |
| sampling/logits | last-token only at decode | — | — | — | (n/a decode) |
| **scheduler / dispatch** | **CUDA/HIP graph capture** (`ggml-cuda.cu:4484 cudaGraphLaunch`) — 1 launch (HYP: maybe disabled on AMD) | **TinyJit HCQ graph** — also batches (`batched N`) | host launch amortized BOTH | **~none** (both graph) | — |
| memory / copies | `ggml-alloc.c` graph allocator, buffer reuse | — | **one ~6.5ms copy = 17% of decode GPU** | **extra copy / round-trip** | identify + eliminate the copy |

## The four required answers

1. **Is llama winning on a better GEMV, or because surrounding primitives waste less?** → **Surrounding
   primitives.** tinygrad's Q4_K/Q6_K GEMVs are competitive (int-dot, ~76% HBM standalone, ~62% of decode GPU,
   reading the same quantized bytes; llama uses the same dp4a int8-dot idea). The loss is the **~38% of decode
   GPU spent outside the GEMVs** — 580 unfused small kernels (~21%) + a large copy (~17%) — that llama **fuses
   or avoids** (RMSNorm+scale+add, gate*up*silu, RoPE+view). And both runtimes graph-batch launches, so it is
   NOT per-kernel host launch overhead.
2. **Largest measured tinygrad gap?** → short context: the **non-GEMV overhead (~38% of decode GPU): 580 small
   kernels + one ~6.5 ms copy**. Long context: the **attention/KV read** (baseline decode decays 3.4× to ctx
   4096).
3. **Highest-value, lowest-risk next?** → **Make flash-decode the default for long context** — already built
   (`FLASH_DECODE`), measured **1.73× @ ctx 4096**, zero new kernel, growing benefit with context. Then
   **identify the ~6.5 ms copy** (17% of decode GPU in a single kernel — likely an avoidable round-trip).
4. **Does long context change the priority?** → **Yes.** Short context → small-op fusion. Long context →
   attention, where flash-decode (already built) is the clear win. Flash-decode is the cross-cutting lever.

## Ranked next workstreams (measured value × success probability × low risk)

1. **Flash-decode default/auto-enable above a context threshold.** Value: HIGH (1.73× @4096, measured). Success:
   HIGH (built). Risk: LOW (gated; quality already validated as exact-ish). Helps: long-context decode, all
   models. **Do first.** (Just needs a default policy + a short-ctx guard so it doesn't regress small ctx.)
2. **Identify & eliminate the ~6.5 ms copy/gather** (17% of decode GPU, one kernel). Value: MED–HIGH. Success:
   MED (need to identify it — likely the lm_head logits realize, embedding gather, or a KV copy). Risk: LOW
   (diagnostic first, no kernel). Helps: all decode. **Do second (diagnostic).**
3. **lm_head last-token-only at prefill.** Value: MED (lm_head is the biggest single kernel; prefill computes
   logits for all positions). Success: HIGH. Risk: LOW–MED ([nn], gated). Helps: prefill.
4. **Decode small-op fusion (RMSNorm+scale+add, gate*up*silu, RoPE+view).** Value: MED–HIGH (the ~21% non-GEMV
   tail), but it's the structural gap llama closes. Success: MED. Risk: MED–HIGH (scheduler/codegen — does
   tinygrad's fuser already do some? needs investigation). Helps: short decode + prefill, all models.
5. **Model/config matrix** (4B/14B/32B × demotion × flash-decode). Value: MED (does the gap hold/scale?).
   Success: HIGH. Risk: LOW (measurement only).
6. **VRAM-frugal PREFILL_V2.** Value: MED (enables bigger models / longer ctx). Risk: MED.
7. *(De-prioritized by prior evidence)* symbolic-KV TC attention (refuted in-model), fused flash/SHAPED_WMMA
   (stale idiom, codegen-deep), per-stream ring/timeline APIs (decode HBM-bound, ~1.2× ceiling). Don't reopen
   without a new reason.

## Bottom line

llama's decode edge is **fusion + fewer/cheaper surrounding ops + a dedicated decode-flash attention**, NOT a
better quantized GEMV — tinygrad's GEMV already matches. The cheapest real wins are **(1) flash-decode by default
for long context (measured 1.73× @4096, already built)** and **(2) killing the ~6.5 ms copy**. The structural
prize is **decode-path small-op fusion** (the ~21% non-GEMV tail), which is exactly what llama's
`has_fusion`/`do_add` kernels do — higher effort, the real long-term lever. No speculative kernels built here;
this is the measured map to pick from.
