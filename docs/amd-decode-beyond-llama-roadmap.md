# Beyond llama.cpp — the decode roadmap past parity

Date: 2026-06-15. State: default decode **53.5 tok/s = ~29% of HBM peak** (RX 7900 XTX, Qwen3-8B Q4_K_M),
up from 23 before the Q6_K work. llama.cpp = 105.7 tok/s = 57%.

## The roofline (why "beyond llama" is real, not aspirational)
Batch-1 decode reads ~4.68 GB of weights/token. At HBM peak (859 GB/s) the floor is **5.45 ms = 183 tok/s
(100% of peak)**. So:

```
peak 183 tok/s |==========================================| 100%
llama.cpp 105  |=========================|                  57%   <- a FIXED hand-tuned scheme
us now    53.5 |=============|                               29%
                ^------ parity gap ------^------- BEYOND room (43% of peak still on the table) ------^
```

There is **as much headroom above llama as between us and llama.** llama leaves 43% of peak unused because it
is a fixed scheme (fixed quant per type, fixed kernels, per-layer-sequential). The machine-search edge is an
*adaptive per-tensor policy* — and we already proved that wins (mixed-quant coverage: 23→53 tok/s). The
levers below extend the policy's decision space (kernel, bit-width, schedule, sparsity) — each one is a place
a search can beat a fixed reference.

## Parity levers (close the last of the gap to llama, ~29%→57%)
- **P1 — lm_head primitive: DONE** (Q6K_COVER_MORE default-on; part of 53.5).
- **P2 — attention reduces (~3.2 ms/token, the `r_*..start_pos` sdpa over the KV cache).** Today it is a
  generic materialized softmax over the cache. Parity = a fused flash-attention-style kernel (no materialized
  scores, online softmax) to match llama's fused attention. *Scope below.*

## Beyond-llama levers (surpass 57%) — each ties to the policy/primitive frame
Ranked by (ceiling × feasibility). Roofline deltas are per-token, stacking on the current 18.7 ms.

- **B1 — in-graph int-dot GEMV (read weights FASTER than llama).** Our standalone int-dot kernel sustains
  **76% of peak vs llama's mmvq 57%.** We already beat llama at the kernel; the only gap is the unsolved
  in-graph integration (amortized q8 activation quant feeding all linears, the D1/E0 problem). If it
  translates, the GEMV drops 10.4 ms → ~7.1 ms. This is the single most direct beyond-llama win because the
  kernel advantage is already proven (`KERNEL_BEATS_LLAMACPP.md`). Highest priority.
- **B2 — overlap non-GEMV behind the weight stream.** Today token = GEMV + non-GEMV (sequential, ~sum). The
  48% non-GEMV (attention/norms/lm_head) can be pipelined to run *while* the next layer's weights stream from
  HBM → token = max(GEMV, non-GEMV) not sum. llama is largely per-layer-sequential too, so a deeply pipelined
  decode beats it structurally. Stacked with B1: token → max(7.1, ~5) ≈ 7 ms = ~140 tok/s.
- **B3 — per-tensor sub-4-bit policy (read FEWER bytes than llama).** llama reads the full Q4_K_M (4.5 b/wt).
  The policy already decides *per tensor* (which kernel); extend it to decide *bit-width* — push tolerant
  tensors to 3-bit/2-bit where a search shows the per-tensor error is acceptable (the inverse of mixed-quant:
  go LOWER where robust, not just higher where sensitive). ~15–20% fewer weight bytes → GEMV 7.1 → ~5.9 ms.
  This is the purest "machine search beats a fixed scheme" lever and directly reuses the cost-model/flywheel.
- **B4 — sparse / compressed-KV attention (DeepSeek-DSA style).** The attention reads the full KV cache every
  token; top-k / compressed-KV reads only the relevant slots → less bandwidth as context grows. Beyond
  llama's dense attention; biggest at long context. (Was flagged earlier in the session as on-hardware-relevant.)
- **B5 — multi-token / self-speculative (amortize the read).** Emit >1 token per 4.68 GB weight pass via MTP
  / Medusa-Eagle heads (no separate draft model). Amortizes the dominant cost across tokens. (Partly a llama
  feature via draft models; self-speculative heads are the beyond version.)

### Stacked beyond-llama ceiling
B1 (int-dot) + B2 (overlap) + P2/B4 (cheap attention) + B3 (sub-4-bit): token ≈ max(5.9 ms GEMV, ~4 ms
non-GEMV) ≈ **6 ms = ~165 tok/s ≈ 1.6× llama.** Not all-or-nothing — B1 alone (with the non-GEMV already
shrinking from P1/P2) plausibly clears llama's 105.

## Scope: lever P2 (attention) — the immediate next concrete step
1. **Identify** the attention kernels precisely (the `r_*start_pos*` reduces): what they read (KV-cache
   bytes vs scores), their per-token cost as a function of context length, and whether scores are materialized.
2. **Parity**: a fused online-softmax attention (flash-style) — no materialized N-wide scores; one pass over
   the cache. Measure vs the generic reduce.
3. **Beyond (B4)**: once fused, add top-k / compressed-KV selection — read only the relevant cache slots.
4. **Gate**: attention/token drops materially AND output stays coherent (attention is lossy under sparsity —
   verify perplexity/coherence, unlike the exact Q6_K win).

## The thesis (why this is the machine-search mission, not just kernel hacking)
llama.cpp is the strong *fixed* baseline. Every beyond-lever is the policy gaining a new degree of freedom:
- coverage (which kernel) — DONE, won.
- kernel choice (int-dot vs fp-dequant) — B1.
- bit-width (per-tensor sub-4-bit) — B3.
- schedule (overlap) — B2.
- sparsity (KV) — B4.
A hand-tuned reference picks one good fixed point in this space. A search picks per-tensor, per-shape,
per-context. That is the structural reason search can go beyond llama — and the roofline says there is 43% of
peak waiting to prove it.

Anchors: `KERNEL_BEATS_LLAMACPP.md` (int-dot 76%), `Q6K_FIX_RESULT.md` (coverage win), `amd-decode-arc-
synthesis.md` (the primitive frame), `amd-decode-measurement-confounds.md` (how to measure any of this).
