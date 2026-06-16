# S3 — batched Q4_K/Q6_K GEMM primitives: kernels built + verified; integration blocked by symbolic prefill

Date: 2026-06-15. Goal (from B5/S0): make the K-token verify fast (it was 74 ms = 4× the decode, because the
primitives are batch-1 and the verify falls to the slow generic prefill reduces). The fix is a batched GEMM
primitive that reads each weight ONCE and reuses it across the K activation columns.

## Done (durable, committed, tested)
- **`q4k_gemm_kernel`** (raw-words layout — same storage as the decode GEMV primitive) and **`q6k_gemm_kernel`**
  (uint16-halfs layout) — both dequant-GEMM kernels where the batch axis `bb` is UPCAST'd so tinygrad hoists
  the dequant: each weight is dequantized once and multiplied across all K columns. This is exactly the
  pattern `qk_gemm_b1` previously validated as "beats fp16 dense at small batch."
- **Correctness verified** vs the fp dequant reference: q4k err 3.8e-06, q6k err 7.7e-06
  (`test/external/test_qk_gemm_batched.py`, 2 passed).
- **Wired** into `Q4KPrimitiveLinear`/`Q6KPrimitiveLinear.__call__`: K=1 → decode GEMV (unchanged); concrete
  1<K≤32 → batched GEMM (with `UPCAST:1:min(K,16)`); else fallback. Gated `Q4K_BATCHED` (default OFF).
- **No regression**: default decode 52.7 tok/s, output byte-identical (the batched path is off by default and
  the symbolic prefill falls back regardless).

## The blocker (why it isn't an e2e win yet)
The decode primitives need a **concrete** K to generate the per-K kernel. But tinygrad's prompt prefill is
**symbolic** — the token count is a UOp variable (`toks`, 1..32) bound at run time, so `x.shape[-2]` is a UOp,
not an int, and the dispatch correctly falls back. The GEMM only fires for a **concrete-K forward**, which a
quick test confirmed (`q4k_gemm_4096_4096_8`, `q6k_gemm_4096_12288_8`, etc. all run) — but routing a concrete-K
forward through the model means a dedicated **verify path** (its own JIT keyed on the fixed K), not the
symbolic `prefill_jit`. That path is not built.

## What remains for the speculative win (S1)
1. A **concrete-K verify forward** on the target model: run K candidate tokens at `start_pos`, return
   `logits[K]`, JIT-keyed on the fixed K, using the batched primitives (now built). The symbolic prefill stays
   as-is for variable-length prompts.
2. The **speculative loop**: 1.7B draft proposes K → verify → greedy accept/reject → advance both KV caches
   (the fiddly, correctness-critical part). Greedy accept makes the output exactly equal to greedy 8B.
3. Measure effective tok/s; S0's bound says ~1.3× once the verify is fast (draft-cost-limited).

## Honest status
The kernel foundation — the part that needed real engineering and is reusable — is **done and verified**. The
remaining work is integration (a concrete-K verify forward + the speculative loop), which is a self-contained
next step, not a kernel problem. The batched GEMM also amortizes prompt-prefill in principle, but the symbolic
prefill would need a fixed-chunk concrete path to benefit — same blocker.

Repro: `test/external/test_qk_gemm_batched.py` (correctness); concrete-K activation via a direct
`model(concrete_K_tokens, concrete_start_pos, temp)` forward with `Q4K_BATCHED=1` (DEBUG=2 shows the
`q4k_gemm_*`/`q6k_gemm_*` kernels).
