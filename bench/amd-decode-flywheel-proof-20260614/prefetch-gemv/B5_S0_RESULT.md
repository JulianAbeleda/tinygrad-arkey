# B5 / S0 — speculative is NO-GO as-is; the prerequisite is a BATCHED primitive (S3), which is itself a win

Date: 2026-06-15. S0 is the pre-registered go/no-go for speculative decoding: does a K-token verify amortize
to ~1 weight-read pass? Measured on RX 7900 XTX, full clock, per-shape-warmed (the JIT-compile and
eager-overhead confounds controlled).

## Measurement
```
8B decode(1):    18.3 ms/tok   (fast primitive path)
8B verify(K=4):  74.5 ms total  (18.6 ms/verify-tok)   ratio to decode 1.02
8B verify(K=8):  73.2 ms total  ( 9.2 ms/verify-tok)   ratio 0.50
8B verify(K=16): 74.5 ms total  ( 4.7 ms/verify-tok)   ratio 0.25
1.7B draft decode: 7.5 ms/tok
```

## Finding: verify amortizes PERFECTLY across K — but at 4× the decode rate
`verify(K)` is **flat at ~74 ms** for K=4..16 → the K-token forward reads the 8B weights ONCE for all K
(perfect token-amortization). But 74 ms is **4× the 18 ms primitive decode**, because the Q4_K/Q6_K decode
primitives are **batch-1 only** — the K-token verify is prefill, which falls to the slow generic dequant
reduces (the exact problem decode had *before* the Q6_K fix). Pre-registered gate (verify(4) ≲ 1.5× decode)
**FAILS at 4.1×.**

## Speculative break-even (why no-go as-is)
Round = verify(74 ms) + K·draft(7.5 ms); yields m+1 accepted tokens. Beats the 54.7 tok/s baseline only if
`(m+1) > (74 + 7.5K)/18.3 = 4.0 + 0.41K`:
- K=4: need m+1 > 5.7 — impossible (max K+1=5).
- K=8: need m+1 > 7.3 — needs ≥7/8 accepted (≥87% per-token).
- K=16: need m+1 > 10.6 — but geometric accept caps E[accepted] ≈ 1/(1−p) (~4 at p=0.75) regardless of K.
With a 1.7B-drafts-8B accept rate (~0.7–0.8 → E[accepted]≈4), every K is at or below break-even. **No-go.**
The two killers: the verify is 4× too slow (prefill path, not primitives), and the autoregressive draft costs
K·7.5 ms.

## Redirect: S3 (batched primitive) is the prerequisite — and a standalone prefill win
The fix is to extend the Q4_K/Q6_K decode primitives to a small **batch dimension (K rows)** so the verify/
prefill uses the FAST primitive (one weight read at ~20 ms, not 74 ms). This is **dual-purpose**:
1. **Prefill / prompt-processing (TTFT) ~4× faster** — a real user-facing win, independent of speculative.
   (Prompt processing currently runs the slow generic reduces; it has the same untapped 4× the decode had.)
2. **Unlocks speculative**: with verify ≈ 20 ms, K=8, draft 60 ms → round 80 ms, m+1≈6 → ~75 tok/s = 1.3×.
   Modest, capped by the autoregressive draft cost; a smaller/faster draft or self-speculative heads (Medusa,
   no draft model) would lift it further.

## Honest verdict
Speculative on the current code is a **no-go** — correctly caught by S0 before any build. The valuable result
is the diagnosis: **the primitive path stops at batch-1, so all batched work (prefill, verify) runs 4× slow.**
The next lever is therefore **S3, the batched primitive** — it pays for itself on prefill/TTFT alone, and is
the gate speculative must pass through. Whether to then pursue speculative (modest ~1.3×, draft-limited) vs.
the exact attention lever (P2) is a separate call once S3 lands.

Repro: `/tmp/s0c.py` style — per-shape-warmed `time-to-first-token` for K-token prompts minus one decode;
1.7B draft decode rate. Models: `Qwen3-8B-Q4_K_M.gguf` (target), `Qwen3-1.7B-Q8_0.gguf` (draft).
