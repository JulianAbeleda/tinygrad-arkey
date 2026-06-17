# Short-context decode exhaustion report (2026-06-17)

Exhausting the remaining short-context decode gap vs llama.cpp before moving to 14B. Four audits + a measured
refutation. **Conclusion: short decode is exhausted — no remaining LOCAL fix clears the gate; the only fat
non-GEMV (attention) is already handled by flash-decode-auto. Move to 14B.**

## 1. Programs/token breakdown (Phase 1, `bench/qk-decode-layer-census/`, ctx 512)

780 programs/token. By region GPU time (eager DEBUG=2 relative proxy; input-upload sync artifact excluded):

| region | kernels | % decode GPU |
|---|---:|---:|
| attention (SDPA, grows with ctx) | 270 | 32.1% |
| ffn_down GEMV | 36 | 19.3% |
| lm_head GEMV | 1 | 14.8% |
| ffn_gate/up GEMV | 72 | 14.8% |
| attn_q/o GEMV | 72 | 10.2% |
| elementwise (rope/residual/cast) | 128 | 2.8% |
| RMSNorm | 73 | 2.2% |
| reduce (other) | 109 | 2.2% |
| attn_k/v GEMV | 18 | 1.8% |

## 2. Per-layer average
**~21.5 programs/layer × 36 = 774 per-layer; only 6 outside-layer** (lm_head, input, output_norm). vs llama's
~7 fused/layer — the 3× program-count gap is real, but the extra kernels are attention reduces (→ flash-decode)
and many tiny ALU/reduce ops the scheduler already handles.

## 3. Largest repeated non-GEMV buckets
All attention: `r_*_513` reduces (KV-length 513 = sp+1 at ctx 512) — scores/softmax/@V, ~7/layer. These are the
fat non-GEMV tail and they GROW with context. Everything else per-layer is ≤0.5 ms.

## 4. RoPE / KV-write / layout verdict (Phase 2) — **A: already cheap**
Source audit (model.py): RoPE = `apply_rope` (chunk/mul/sub/add) + a `.cat()` to rejoin the non-rotated tail
(Qwen3 partial rope, rope_dim < head_dim, so the cat is **design-necessary**). KV write = lazy `stack`→`store`→
slice (no forced copy, no extra kernel). q/k/v reshape+transpose = lazy views (no copies). Total ≈ the 2.8%
elementwise + part of reduce-other — small, and nothing forces an avoidable materialization. **Not worth
optimizing; the cat is necessary by design.**

## 5. Norm/residual fusion verdict (Phase 3) — **REFUTED (below gate)**
RMSNorm 2.2%, residual/norm-scale/cast 2.8%, SwiGLU 0.1% (already scheduler-fused). The one concrete local
candidate — removing the FFN `silu().contiguous()` (model.py:704, marked `# TODO`) — was **tested**:
programs/token 780→744 (−36, one/layer), **decode tok/s 55.42→55.29 (0%)**, argmax unchanged. **36 < 72-kernel
gate, 0% < 3% gate → reverted.** The block residual `.contiguous()` (line 719) is a `@function` boundary
(load-bearing, high risk). tinygrad already fuses adjacent elementwise. **No classic small-op fusion clears the
gate; not worth a dedicated arc.**

## 6. lm_head / logits verdict (Phase 4) — **A: necessary, already minimal**
lm_head is a Q6_K GEMV (14.8%, the biggest single kernel) — irreducible (must read the 510 MB output weight).
At T=1 decode it already runs on ONE token (no all-positions waste; the `[:,-1,:]` slice matters only for
prefill T>1). Sampling is Gumbel-max (`forward()`): ~3–4 fused elementwise + argmax kernels, no host
materialization, no copies — near-optimal. The outside-layer tail is just 6 programs. **No local fix.**

## 7. Changes made and measured impact
None kept. The FFN `silu().contiguous()` removal was measured (0% tok/s, −36 kernels) and **reverted** per the
gate. The flash-decode-auto win (prior task) stands as the shipped short→long improvement.

## 8. Remaining gap to llama after these audits
tinygrad ~54–64 vs llama ~80–100 tok/s short decode. **The gap is NOT a single fixable primitive** — it's the
sum of: (a) ~3× more kernels each slightly less efficient (the scheduler fuses the cheap elementwise but emits
many small reduces), (b) a competitive-but-not-faster GEMV (llama's inline-dequant dp4a may hold a small edge),
(c) per-kernel overhead within the batched graph. No remaining LOCAL lever; closing it further needs
codegen-level fusion (broad, sub-3% gains for the small ops) or accepting it.

## 9. Is short decode exhausted enough to move to 14B? **YES.**

## Final ranking

| candidate | measured cost | expected gain | risk | decision |
|---|---|---|---|---|
| flash-decode auto (long ctx) | attention 32%@512, grows | 1.25–1.73× @≥1024 | low | **FIXED (shipped)** |
| FFN silu .contiguous() removal | 36 kernels, ~0.1% | 0% (measured) | med (other paths) | **REFUTED (tested, reverted)** |
| norm+residual fusion | ~5% (2.2+2.8) | <3%, mostly already fused | med | refuted (below gate) |
| RoPE .cat() | part of 2.8% | <1% (cat design-necessary) | low | not worth it |
| KV-write / q-k-v layout | ~0 (lazy) | 0 | — | not worth it (already cheap) |
| block residual .contiguous() | part of 2.8% | <1% | high (@function boundary) | not worth it |
| lm_head GEMV | 14.8% (necessary) | 0 (irreducible @T=1) | — | not worth it |
| sampling / logits tail | 6 progs, tiny | ~0 | — | not worth it (already minimal) |
| broad program-count fusion | 780 vs 260 | sub-3% for small ops | high (codegen) | deferred (broad rewrite) |

**Bottom line:** every local short-decode lever is fixed (flash-decode), refuted (norm/residual, FFN contig), or
necessary/already-minimal (RoPE, KV, lm_head, sampling). The residual gap is structural (program granularity +
GEMV parity), not a local fix. Short decode is exhausted — next work is 14B (out of scope here).
