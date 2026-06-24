# Speculative decoding — gate verdict + integration scope (2026-06-17)

Gate question: can a cheap draft produce enough accepted tokens for Qwen3-8B to make spec decoding worth
building? **Acceptance: YES, excellent. Speed: marginal (~1.3×) with the only local draft (1.7B); the draft is
too expensive — a 0.6B draft (not local) would make it strong (~1.7×).** Offline, gate-only, no integration
built. `extra/qk_spec_decode_acceptance_gate.py`, `bench/qk-spec-decode-acceptance/`.

## 1. Draft & tokenizer
Target Qwen3-8B-Q4_K_M; draft **Qwen3-1.7B-Q8_0** (smallest local Qwen3). Tokenizer **verified compatible**
(encode matches across chat/code/QA strings, same eos 151645). No Qwen3-0.6B local.

## 2. Prompts
16 real prompts, 4 categories (chat, code, factual QA, reasoning/math), committed
(`bench/qk-spec-decode-acceptance/prompts.jsonl`).

## 3–4. Acceptance + speed (greedy, K sweep)

| K | accepted tokens / target pass | est. speedup vs 55 tok/s |
|---:|---:|---:|
| 2 | 2.39 | **~1.32× (optimal)** |
| 4 | 3.26 (16 prompts, every prompt ≥1.6) | ~1.25× |
| 8 | 4.44 | ~1.05× |

Per-position accept (K=4): 79 / 60 / 48 / 40% — 40% of passes accept all 4. Acceptance is **excellent** (≥2.0
bar) and low-variance (per-prompt min 1.6).

**Speed model:** target decode 55 tok/s (18.2 ms/tok), draft 137 tok/s (7.3 ms/step), verify ≈ one target
weight-read (~18.2 ms for the T=K+1 prefill). Per pass = K·7.3 + 18.2 ms → speedup = A(K)·18.2 /
(K·7.3+18.2). **Optimal K=2 → ~1.32×.** Higher K accepts more but the sequential draft cost dominates (K=8 →
~1.05×).

## UPDATE — 0.6B draft fetched, gate PASSES both criteria (the integrate trigger)

Per the engineering sequence (get 0.6B → re-run exact gate → integrate iff accepted/pass >2 AND draft >200
tok/s): downloaded `Qwen3-0.6B-Q8_0.gguf` (tokenizer verified compatible). **Draft 273 tok/s (>200 ✓).**
Re-ran the exact K=4 gate (same 16 prompts): **accepted/pass = 2.844 (>2 ✓)**, every prompt ≥1.6, per-pos
69/52/36/28%. Refined speed model (draft 3.66 ms/step, target+verify 18.2 ms): optimal K≈3 → **~1.60×** (K2
1.58, K4 1.58) — vs the 1.7B's ~1.3×. The smaller draft trades a little acceptance (2.84 vs 3.26) for 2× the
speed → net better. **BOTH gate criteria met → INTEGRATE (with the 0.6B draft, not 1.7B).**

## 5. Verdict
- **Acceptance gate: PASSES excellent** (3.26 @K4 ≫ 2.0). Acceptance is NOT the bottleneck.
- **Speed: bucket B (marginal), ~1.3×** with the 1.7B draft — it's only 2.5× faster than the 8B target, so K
  sequential draft steps cost as much as the target pass they save. **Draft cost is the bottleneck.**
- **Highest-leverage next step: fetch Qwen3-0.6B** (same tokenizer). At ~250 tok/s it would give ~1.7× (strong)
  for the same ~3.3 acceptance. Worth doing before integration if a bigger win is wanted.

**Integration-worthy?** Yes, but modest (~1.3×) on local hardware as-is. ~1.3× beats every refuted kernel arc
(GEMV final-mile, small-op fusion — all ≤0/<5%), so it's the best remaining 8B decode lever. But it's an
algorithmic win requiring a real generation-loop change; recommend doing it **only with the 0.6B draft** (→~1.7×)
or accepting ~1.3× with 1.7B.

## Integration scope (only if approved)

- **Hook:** the generation loop (`extra/llm_generate` / `tinygrad.llm.cli` generate path), NOT the decode kernels.
  A `SPEC_DECODE=1` env flag (default off) selects a speculative generate loop; normal decode untouched.
- **Models:** load target + draft (shared tokenizer); `DRAFT_MODEL` env for the draft path. Both via
  `load_model_and_tokenizer`. VRAM: 8B Q4 (5 GB) + 1.7B Q8 (1.8 GB) = ~7 GB (or 0.6B ~0.4 GB) — fits 24 GB.
- **Loop per step:** draft proposes K (greedy, draft decode jit, T=1 ×K, its own KV cache); target verifies the
  K+1 positions in **one T=(K+1) prefill** (a single fixed shape → compiles once, reused) using the target KV
  cache; accept matching prefix + 1; advance both caches to the accepted length; on mismatch, the draft's
  speculative KV beyond the accepted prefix is discarded (re-decode from the accepted position — start_pos
  rollback, no recompile).
- **KV cache:** the verify writes K+1 entries; on partial accept, only the accepted entries are kept (target
  cache position = ctx + accepted). Draft cache similarly rolled back. This is the main correctness surface.
- **Verification batching:** the single T=(K+1) prefill is the fixed-shape kernel (set K at load to keep one
  compiled shape). K=2 is the speed-optimal default; expose `SPEC_K`.
- **Correctness / sampling:** greedy (temperature 0) is exact-equivalent (accept iff draft==target argmax). For
  temperature>0, sampling correction (the standard accept/resample rule) is required and NOT in this gate —
  scope as a follow-up; ship greedy first.
- **First performance gate:** measured decode tok/s with `SPEC_DECODE=1` (K=2) must beat the `=0` baseline by
  ≥1.2× on the prompt set, output sane (greedy stream identical to non-spec greedy — spec decoding is exact at
  T=0), no regression with the flag off.

## Files
`extra/qk_spec_decode_acceptance_gate.py`, `bench/qk-spec-decode-acceptance/{prompts.jsonl,result.json}`.
Gate-only; no model/decode changes; nothing default-on.
