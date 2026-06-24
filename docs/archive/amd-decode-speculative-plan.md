# B5 — speculative decoding: break the batch-1 weight-read floor (scope)

Date: 2026-06-15. State: per-kernel GEMV is closed at its batch-1 ceiling (B1, ~52% ≈ llama 57%). The only
quality-preserving way past the weight-read roofline (4.68 GB/token → 183 tok/s ceiling) is to **emit more
than one token per 8B weight-read pass.** That is speculative decoding.

## Why this is the right next lever
- **Biggest ceiling**: throughput ≈ (tokens accepted per verify) × (per-pass rate). With a good draft this is
  multiplicative — 53 → ~130–160 tok/s is the target, clearly past llama's 105.
- **Zero quality cost**: greedy speculative is EXACT — the accepted prefix + the target's correction is
  byte-identical to pure greedy 8B (like the Q6_K win). Easy to verify, nothing to trade.
- **Achievable now**: target = `Qwen3-8B-Q4_K_M` (our fast 53 tok/s path); draft = `Qwen3-1.7B-Q8_0`
  (already on disk, same Qwen3 vocab/tokenizer, ~1/3 the weight bytes). Verify = the existing prefill path.
- **Honest caveat**: llama.cpp also supports speculative decoding, so this is not "beyond llama" in the
  machine-search sense (that was the kernel/coverage win, already proven). It is "beyond llama's single-
  stream rate." Report both framings: us+spec vs llama (no spec) = clear win; us+spec vs llama+spec = tests
  our implementation.

## Mechanism (greedy, exact)
1. **Draft**: the 1.7B autoregressively proposes K candidate tokens t[1..K] from the current context (cheap:
   ~1.8 GB read × K draft steps; 1.7B is ~2.6× fewer weight bytes than the 8B).
2. **Verify**: ONE 8B forward over the K candidates at positions p..p+K-1 (a prefill-style batched call) →
   logits[1..K]. Reads the 8B weights **once** for all K.
3. **Accept**: greedy — accept t[i] iff argmax(logits[i-1]) == t[i], for the longest matching prefix m; the
   token at the first mismatch is replaced by the target's argmax (a free correct token). So each verify
   yields **m+1** real 8B tokens (1 ≤ m+1 ≤ K+1).
4. **Advance**: both KV caches by m+1; the draft re-syncs from the corrected token.

## Feasibility crux — does the verify amortize? (S0, do FIRST)
The whole win rests on: **verify(K tokens) costs ≈ one weight-read pass, not K passes.** A K-token 8B forward
is weight-bound only if the matmuls read the weights once for the K-row activation. The decode primitives are
batch-1 (`x.shape[0]==1`), so the K-token verify falls to the **generic prefill matmuls** (the `r_32_32_*`
reduces we profiled at ~1480 µs for 32-token chunks). Those read weights once per call — good — but are tuned
for 32-wide, so small K may be inefficient.
- **S0 measurement**: time the 8B forward at batch/seq K ∈ {1, 2, 4, 8, 16}. Compute verify(K)/verify(1).
- **S0 gate**: verify(4) ≲ ~1.5× verify(1) → it amortizes (weight-bound) → speculative pays → proceed.
  verify(4) ≈ 4× verify(1) → compute-bound at small K, no amortization → speculative won't help via the
  prefill path → first fix the batched verify (extend Q4_K/Q6_K primitives to batch>1, a sub-project) or stop.
- This single measurement decides whether B5 is viable on the current code. Cheapest, run before any build.

## Build (only if S0 passes)
- **S1 — minimal greedy spec, K=4.** Two `Transformer`s (8B target, 1.7B draft), separate KV caches. Draft
  loop → verify forward → greedy accept/reject → advance both caches. Measure: effective 8B tok/s, mean
  accept length (m+1), and **confirm byte-identical output** vs pure greedy 8B (the exactness guarantee).
  Gate: effective tok/s > 53 (baseline) with identical output.
- **S2 — tune the frontier.** Sweep K ∈ {2,4,6,8} and draft ∈ {1.7B-Q8, 4B-Q4_K}. The optimum trades draft
  cost vs accept rate: bigger draft → higher accept, slower draft. Report the speed/accept frontier and the
  best effective tok/s vs llama's 105.
- **S3 — verify-path efficiency (if S0 showed weak amortization).** Extend the Q4_K/Q6_K decode primitives to
  a small batch dimension (K rows), so the verify uses the FAST primitives instead of the generic prefill
  reduce. This is the kernel piece — reuses the q4k/q6k packed layouts with an outer K loop. Biggest if the
  prefill path is the bottleneck.

## Correctness / KV management (the risk)
- Greedy accept makes the output provably identical to greedy 8B — the test is a direct token-sequence match
  vs the baseline (must be exact, not just coherent).
- The fiddly part is the KV cache: on partial accept, advance by m+1 and DISCARD the draft's speculative KV
  past the accept point (both models). `Transformer.generate`/`get_start_pos` already track `start_pos` and
  `_cached_tokens`; the rewind must reset both caches to p+m+1. Get this exactly right or the next step
  corrupts. Unit-test the rewind on a fixed sequence.

## Measurement & honesty
- Primary metric: **effective 8B tok/s** (real tokens / wall), at full clock, sustained, vs the 53 baseline
  and llama 105. Secondary: mean accept length, draft/verify time split.
- Exactness: token-sequence identical to greedy 8B (assert, don't eyeball).
- Report us+spec vs llama-no-spec AND note llama+spec exists. Do not overclaim "beyond llama" — claim
  "beyond our own single-stream and past llama's single-stream rate, quality-preserving."
- ±6% e2e noise floor (from B1) — speculative's win should be >> that, so it's resolvable; still repeat runs.

## Why not the others (this round)
- **P2 (attention fusion)**: exact but parity-only (~15% of token, caps below llama). Good later, smaller.
- **B3 (sub-4-bit adaptive quant)**: the mission-purest beyond-llama lever, but the achievable form (demote
  Q6→Q4) trades quality, and true equal-quality-fewer-bytes needs sub-Q4 kernels + fp16 weights to re-quant.
  Bigger and quality-entangled — scope it after B5 (and it pairs with R6, the flywheel re-score).
- **B2 (overlap)**: caps at the weight-read time (~90 tok/s) — can approach but not clearly exceed llama, and
  the layer dependencies make it hard. Lower ceiling than B5.

## One-line plan
S0 (verify amortization measurement — the go/no-go) → S1 (minimal greedy spec, exact, >53) → S2 (tune
K/draft, target >105) → S3 (batched-primitive verify if the prefill path is the bottleneck).

Anchors: `amd-decode-beyond-llama-roadmap.md` (the lever map), `B1_INTDOT_RESULT.md` (why per-kernel is
closed), models on disk: `Qwen3-8B-Q4_K_M.gguf` (target), `Qwen3-1.7B-Q8_0.gguf` / `Qwen3-4B-Q4_K_M.gguf`
(draft candidates).
