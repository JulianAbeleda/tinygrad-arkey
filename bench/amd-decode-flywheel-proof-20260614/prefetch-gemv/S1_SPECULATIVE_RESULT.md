# S1 — speculative decoding: EXACT and correct, but net-slower (draft-bound), confirming S0

Date: 2026-06-15. Built greedy speculative decoding (1.7B drafts → 8B verifies a batch via the S3 GEMM →
greedy accept). `extra/qk_speculative.py`.

## It works and is exact
- **Output is byte-identical to greedy 8B** across all runs (`EXACT vs greedy 8B: True`). The algorithm, the
  batched verify, and the manual KV-cache management (re-process the last accepted token at `start_pos`, then
  advance) are all correct.
- **The verify is fast — the S3 batched GEMM paid off**: ~40 ms/round for K+1=7 tokens (≈5.7 ms per
  verify-token, well under the 18 ms decode). This is exactly what S3 was built for, and it delivered.

## But it does not speed up decode (draft-bound)
| K | tok/s | mean accept | draft % | verify % |
|---|------:|------------:|--------:|---------:|
| 4 | 4.9 | 1.29/5 | 70% | 15% |
| 6 | 7.0 | 2.58/7 | 76% | 11% |
The **draft dominates (70–95%)**. Two layers of why:
1. **Implementation overhead**: my manual draft loop runs ~47 ms/draft-token vs the 1.7B's true ~7.5 ms
   (separate JIT, per-token `.item()` sync, full-vocab argmax each step). Fixable, but —
2. **Fundamental**: even at the draft's true 7.5 ms/token, a K=6 round = ~52 ms draft + 40 ms verify = 92 ms
   for ~2.6 accepted = **35 ms/token = 28 tok/s, still below the 53 baseline.** The 1.7B draft is only ~2.6×
   cheaper than the 8B target (memory-bound: 1.8 GB vs 4.68 GB/token), so drafting K of them costs more than
   the ~2–4 tokens it buys. Speculative needs a draft ~10× cheaper (a 0.5B model, or self-speculative heads).

This **confirms S0's pre-registered prediction exactly**: the verify amortizes (S3 fixed that), but the
autoregressive draft cost on this 1.7B/8B pair makes speculative net-negative.

## What this establishes
- The S3 batched GEMM primitive is **validated e2e** — the verify uses it and is fast (5.7 ms/verify-token).
  That foundation is real and reusable (e.g. for prompt-prefill, or a cheaper-draft speculative).
- Greedy speculative is implemented and exact — a correct, reusable harness.
- Speculative is **not a win on this hardware with the available draft** (1.7B too expensive). It would need
  a much smaller draft (≤0.6B, not on disk) or self-speculative/Medusa heads (training required). Honest stop.

## Honest verdict
S0 said "no-go without a fast verify; even then draft-limited (~1.3×)." S1 built the fast verify (S3) and the
exact loop, and measured the draft limit directly: net-negative with 1.7B/8B. The kernel/integration work is
sound; the lever is blocked by draft economics, not by tinygrad. Not pursuing further without a smaller draft.

Repro: `DEV=AMD Q4K_PRIMITIVE=1 Q4K_BATCHED=1 PYTHONPATH=. .venv/bin/python extra/qk_speculative.py --k 6`.
