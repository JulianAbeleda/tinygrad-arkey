# CORRECTED: the decode bottleneck is the Q6_K matmuls (no primitive), not the Q4_K GEMV

Date: 2026-06-15. This SUPERSEDES NARROW_RESULT.md, which measured the wrong kernel (a non-decode-enabled
fallback reduce) and wrongly concluded the in-graph Q4_K GEMV runs at 12%. A proper per-kernel profile of a
real decode token corrects the whole picture.

## What the real decode path actually does (DEBUG=2 per-kernel, full clock)
The Q4_K primitive kernels run WELL — near llama.cpp:
| kernel (decode)                  | us   | GB/s | %peak |
|----------------------------------|-----:|-----:|------:|
| q4k_gemv_partial_4096_4096 (q,o) |  32  | 262  |  31%  |
| q4k_gemv_partial_12288_4096(g,u) |  54  | 463  |  54%  |
| q4k_gemv_partial_4096_12288(down)|  ~18 | fast |   —   |
The Q4_K GEMV is NOT the wall (it's ~llama quality; our standalone int-dot is better still, 76%).

## The real decode-token breakdown (one clean decode graph, PROFILE, merged intervals)
- GPU-busy ≈ **20 ms/token**; wall ≈ **45 ms/token (22 tok/s)** → ~**25 ms (55%) is HOST/sync** (JIT-replay
  launch + per-token `.item()` sync + argmax sampling over 151936 vocab). (The earlier "99% GPU-busy" was a
  prefill-contaminated cross-token sum; the single-graph number is the truth.)
- Of the 20 ms GPU work, ONE kernel family `r_32_32_4_48` = **11.8 ms = 59%**, plus `r_1187` (lm_head) = 2.6 ms.

## Root cause (confirmed from the GGUF + install debug)
This is mixed-quantization Q4_K_M. Per role across 36 layers:
```
attn_q/k/output, ffn_gate/ffn_up : Q4_K (all 36)         -> primitive, fast
ffn_down : Q6_K x18, Q4_K x18      attn_v : Q6_K x18, Q4_K x18      output(lm_head): Q6_K
```
`_install_q4k_primitives` installs **Q4_K only** (debug: `by_kind=Q4K`, no Q6_K). `_q6k_policy` exists but no
Q6_K primitives are wired. So **every Q6_K matmul falls back to the slow generated fp-dequant reduce**:
- the 18 Q6_K `ffn_down` (25 MB each, the biggest matmul) = `r_32_32_4_48` (out 4096 = 32·32·4, reduce over
  12288 = 48 Q6_K blocks) → 11.8 ms = **59% of GPU decode work**, running ~38 GB/s (~4% peak).
- the 18 Q6_K `attn_v` + the Q6_K `lm_head` (`r_1187`, 2.6 ms) also fall back.

Ablation confirming the diagnosis: forcing the Q4_K primitive onto attn_k/attn_v (`Q4K_COVER_KV`) moved
tok/s only 23.3→23.7 — because attn_k is tiny and attn_v is half Q6_K (uncoverable by the Q4_K path). The
cost is the **Q6_K ffn_down**, which no primitive touches. (Flag reverted; null.)

## The two real levers (both concrete, neither is "a better Q4_K kernel")
1. **Q6_K decode GEMV primitive** (biggest GPU win): build/wire a Q6_K analog of the Q4_K primitive for
   ffn_down (+ attn_v, lm_head). Converts the 59% slow reduce (~38 GB/s) to primitive speed (~50%+). Est:
   GPU-busy 20 ms → ~10–12 ms. The int-dot/decode-GEMV machinery already exists for Q4_K; Q6_K needs its
   own dequant (6-bit) packing + kernel.
2. **Host/sync overhead** (~25 ms/token, 55% of wall): the per-token `.item()` sync + sampling + replay
   launch. llama.cpp's tight loop has ~0 here. Levers: async/on-GPU sampling, avoid the per-token CPU
   round-trip, fold argmax into the graph.

## Honest status
- The standalone-kernel WIN (Q4_K int-dot 76% > llama 57%) stands and is independent.
- NARROW_RESULT's "in-graph GEMV is 12%" was WRONG (wrong kernel). The Q4_K in-graph GEMV is 31–54%, fine.
- The decode gap is: (1) Q6_K matmuls have no primitive (59% of GPU work), (2) ~55% of the wall is host
  overhead. Both are addressable and neither was the thing we'd been optimizing (the Q4_K GEMV kernel).

Repro: `extra/qk_decode_breakdown.py` (per-token split, exclude prefill graphs); the single-graph dump;
GGUF types via `gguf_load_with_metadata`. Q4K_PRIMITIVE_DEBUG=1 shows `by_kind=Q4K` (no Q6_K installed).
