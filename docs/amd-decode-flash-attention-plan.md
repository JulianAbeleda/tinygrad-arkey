# P2 — Flash-Decoding for long-context decode attention (full scope)

Date: 2026-06-15. P2_ATTENTION_RESULT found decode collapses 54→14 tok/s from context 8→3072 — the attention
is occupancy-bound at long context (batch-1, one workgroup per head ≈ 32 of 96 CUs; the work is neither
compute- nor memory-bound). The established fix is **Flash-Decoding** (Tri Dao et al., Oct 2023).

## Reference algorithm (Flash-Decoding)
Standard FlashAttention parallelizes over (batch, query-length) only → at batch=1, query=1 (decode) it uses
<1% of the GPU. Flash-Decoding adds a **third parallel dimension: the KV sequence length**. Two kernels:
1. **Per-split partial attention** (the win): split the KV cache into S chunks (zero-cost views). For each
   (head, split), run online-softmax FlashAttention over that chunk → store a **partial output** (Hd vector,
   unnormalized) + **one log-sum-exp scalar** (the softmax max+sum for that split). Now there are
   H×S workgroups instead of H → the GPU saturates even at batch 1.
2. **LSE reduction**: combine the S partial outputs per head, reweighting by `exp(m_i − m_global)` and
   dividing by `Σ l_i·exp(m_i − m_global)` (softmax-combine across splits). Tiny.
Result: attention time stays ~constant up to 32k+ context (vs growing/collapsing). Up to 8× e2e on long-ctx.

## Our setup
RX 7900 XTX (96 CUs), Qwen3-8B: **GQA** n_heads=32, n_kv_heads=8 (g=4 queries/kv head), head_dim=128, fp16
KV cache, max_context=4096. Decode is batch=1, T=1. Current SDPA: materialized [32, Tc] scores + several
small low-occupancy kernels/layer → the 4× long-context slowdown. Exact (no quality cost) — this is purely a
scheduling/occupancy fix.

## Approach A — Tensor-level Flash-Decoding (try FIRST; cheap, no custom kernel)
Express the split + online-softmax + reduction in tinygrad Tensor ops with an explicit split dim S, so the
scheduler's global dims become [KvH, g, S] and it emits KvH·g·S workgroups (8·4·S) — S=4 → 128 wg ≈ saturates
96 CUs. Sketch (decode, T=1, GQA-grouped so k/v read once per kv head):
```
# q:[B,KvH,g,Hd]  k,v:[B,KvH,Tc,Hd]  ->  Tc = S*L
k = k.reshape(B,KvH,1,S,L,Hd); v = v.reshape(B,KvH,1,S,L,Hd)
s   = (q.reshape(B,KvH,g,1,1,Hd) * k).sum(-1) / Hd**0.5      # [B,KvH,g,S,L] scores
m   = s.max(-1)                                              # [..,S]  per-split max
p   = (s - m[...,None]).exp()                                # [..,S,L]
l   = p.sum(-1)                                              # [..,S]
o   = (p[...,None] * v).sum(-2)                              # [..,S,Hd] partial (unnormalized)
gm  = m.max(-1)                                              # [..]     global max
w   = (m - gm[...,None]).exp()                               # [..,S]   per-split reweight
out = (o * w[...,None]).sum(-2) / (l*w).sum(-1)[...,None]    # [..,Hd]
```
- The split S is an explicit non-reduce dim → tinygrad should parallelize over it (more workgroups). Tune S
  to target ~128–256 wg. Handle Tc not divisible by S (pad last split / mask).
- Causal mask for decode (T=1) is trivial: the single query attends to all valid 0..start_pos (no triangular
  mask needed). For prefill (T>1) keep the existing SDPA path; this is a decode-only kernel.
- **Gate A**: long-context decode (ctx 3072) speeds up materially AND output stays byte-exact vs SDPA. If
  yes → ship (gated, then default). If the scheduler doesn't parallelize over S (no speedup) → Approach B.

## Approach B — custom 2-kernel flash-decode (if A's scheduler won't cooperate)
Two `custom_kernel`s with explicit workgroup-per-(head, split):
1. partial: workgroup = (q_head, split); threads cooperate over head_dim/L; online softmax over the split's L
   keys → write partial_out[head, split, Hd] + lse[head, split].
2. reduce: per head, combine S partials via the LSE formula → out[head, Hd].
Full control over occupancy; mirrors the FlashAttention-2 decode kernel. More effort + correctness-critical
(online softmax numerics, the cross-thread reductions). Only if A is insufficient.

## Split-size heuristic
Choose S so H·(g)·S (or per-(kv,q-group)·S) ≈ 2–4× the 96 CUs (≈128–384 workgroups), capped so each split L
is large enough to amortize (L ≳ 128). For Tc up to 4096 and the GQA factor, S∈{4,8,16}; sweep. At small
context (Tc < ~512) fall back to non-split (S=1) — the short-context path is already fine (54 tok/s).

## Validation
- **Exactness**: byte-identical (or <1e-3) tokens vs the SDPA path on a fixed prompt — it's algebraically the
  same softmax, just reassociated. (Online softmax is exact up to fp reassociation.)
- **The headline metric**: decode tok/s as a function of context (8 / 1024 / 3072) — target ~flat attention
  time (Flash-Decoding's property) instead of the 4× collapse. Compare to SDPA at each context.
- Per-kernel: attention us/layer at ctx 3072 should drop from ~1.4 ms toward the ~31 µs KV-read floor.

## Stages
- S0: implement Approach A behind `FLASH_DECODE` (default off), decode-only (T=1), GQA-grouped. Verify exact.
- S1: sweep S at ctx {1024, 3072}; measure tok/s vs SDPA; pick S (or a Tc-based heuristic).
- S2: if A wins → default-on for decode (keep SDPA for prefill); update the context-sweep numbers in the arc.
- S3 (only if A flat-lines): Approach B custom kernels.

## Honest framing / risks
- **Exact, no quality cost** — unlike B3 (lossy). Pure scheduling win.
- The whole bet is that tinygrad parallelizes over the explicit S dim. If its codegen folds S into a reduce
  or serializes it, Approach A won't help and B is required (bigger). S0's gate decides this fast.
- Short-context decode (the arc's headline numbers) is unaffected (S=1 fallback) — this is a long-context
  lever, the biggest remaining for real usage.

## S0 GATE RESULT (2026-06-15): Approach A FAILS → Approach B required
Microbenchmark of the Tensor-level split at concrete Tc (exact, max_err 1.8e-4):
| Tc=3072 | sdpa | flash S=4 | S=8 | S=16 | S=32 |
|---|---|---|---|---|---|
| GPU us | **198** | 385 | 652 | 2235 | 4712 |
**The explicit split is SLOWER and worse with more splits** — tinygrad's codegen does not map the split dim to
occupancy; it generates worse kernels (e.g. `r_2_8_16_4_4_384_8` at 540µs). Approach A is dead.

Re-localized the real long-context cost (profile, decode @ctx~3000): the attention `r_*start_pos` kernels
(`r_4_2_8_16_4_28start_pos` 6.0ms, `r_8_4_28start_pos` 3.0+1.5ms, ...) are the **dominant** GPU cost of the
token — confirming P2's premise (the standalone-SDPA microbenchmark's 198µs was unrepresentative; the real
decode attention over the symbolic-length KV cache is far heavier). So: the attention IS the long-context
bottleneck, AND the cheap Tensor-level fix doesn't work → **Approach B (custom 2-kernel flash-decode) is the
only remaining path.** It is a substantial, correctness-critical build (online-softmax numerics, cross-thread
reductions, the symbolic split count from `start_pos` at runtime). Decision pending before committing to it.

## Sources
- Flash-Decoding for long-context inference — PyTorch blog: https://pytorch.org/blog/flash-decoding/
- Princeton NLP / Tri Dao et al.: https://princeton-nlp.github.io/flash-decoding/
- FlashDecoding++ (MLSys 2024): https://proceedings.mlsys.org/paper_files/paper/2024/file/5321b1dabcd2be188d796c21b733e8c7-Paper-Conference.pdf
- FlashInfer (customizable attention engine): https://arxiv.org/pdf/2501.01005
