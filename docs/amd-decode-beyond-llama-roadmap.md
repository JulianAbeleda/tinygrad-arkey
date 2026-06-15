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

- **B1 — in-graph int-dot GEMV. TESTED → DECISIVE NEGATIVE (2026-06-15, `B1_INTDOT_RESULT.md`).** The
  standalone int-dot is 76% vs fp 56%, but in-graph it runs 28.5 µs vs fp 32 µs — only 1.12×, both ~34% of
  peak. The same kernel hits 64% amortized-standalone → the gap is **single-shot occupancy** (the attn GEMV
  is 64 workgroups, too few to fill 96 CUs), not compute. int-dot's compute win is invisible because the
  in-graph GEMV is occupancy-bound, not compute-bound; the q8 quant overhead then cancels the tiny gain →
  null. split-K (more wg) is within noise; fusion hurts. **The per-layer GEMV is already at its batch-1
  ceiling (~50–55% ≈ llama's 57%).** A faster per-kernel GEMV is NOT the path beyond llama. This also
  down-grades R3/R4/R5 (also per-kernel GEMV levers → likely within-noise for the same reason).
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

### Stacked beyond-llama ceiling (revised after B1 negative)
B1 is out — the per-kernel GEMV is already at ceiling, so the path is **change the work, not the kernel**:
B3 (sub-4-bit: fewer bytes) + B2 (overlap: token = max not sum) + B4/P2 (cheap attention) + B5 (amortize
across tokens). B5 is the multiplier: even at our current ~52% per-kernel rate, emitting 2 tokens per weight
pass ≈ doubles throughput. The realistic beyond-llama route is **B5 × (B3 + B2)**, not a faster GEMV.

## Scope: lever P2 (attention) — the immediate next concrete step
1. **Identify** the attention kernels precisely (the `r_*start_pos*` reduces): what they read (KV-cache
   bytes vs scores), their per-token cost as a function of context length, and whether scores are materialized.
2. **Parity**: a fused online-softmax attention (flash-style) — no materialized N-wide scores; one pass over
   the cache. Measure vs the generic reduce.
3. **Beyond (B4)**: once fused, add top-k / compressed-KV selection — read only the relevant cache slots.
4. **Gate**: attention/token drops materially AND output stays coherent (attention is lossy under sparsity —
   verify perplexity/coherence, unlike the exact Q6_K win).

## REVISIT — closures measured against the Q6_K-bottlenecked baseline (likely false nulls)
The Q6_K discovery invalidates the baseline. Every e2e experiment closed *before* the flag was on ran with
the Q6_K `ffn_down` fallback as a fixed 59%-of-token noise floor. A lever improving the Q4_K path (then ~40%
of the token, now 49% of a shorter token) moved e2e by only `~0.4 × win` — often below noise → **null**. With
Q6_K fast, the same levers should now register. Re-run each against the new 53.5 tok/s baseline:

- **R1 — D1/E0/A0: in-graph int-dot + amortized quant (= B1). HIGH.** Closed as "vdot e2e == fp e2e == 30",
  but that null was dominated by the Q6_K 59%. The Q4_K GEMV is now ~half the token and the int-dot kernel
  is proven 76% standalone → the win that was masked should surface. This is the single most important
  revisit; it *is* B1.
- **R2 — amortized q8 quant (A0/E0 mechanism).** The quant overhead "didn't pay" because the GEMV it sped up
  was a small, diluted fraction. Now the GEMV is the dominant remaining cost → re-evaluate the quant-cache
  break-even (it shares one quant across q/k/v + gate/up per layer).
- **R3 — horizontal fusion (`Q4K_FUSE`, q/k/v→1, gate/up→1).** Fewer/fatter Q4_K launches mattered little
  when Q6_K dominated; now the Q4_K GEMVs are the bulk of GPU work, so fusion's occupancy/launch win is worth
  re-measuring.
- **R4 — per-kernel opts / warm-start on the Q4_K GEMVs.** The opt sweeps were judged against the bottlenecked
  e2e; the Q4_K `q4k_gemv_partial` opts (LOCAL/parts/UPCAST) now move a 49% slice — re-sweep.
- **R5 — Q4_K attn_k coverage (`Q4K_COVER_KV`, closed null 23.3→23.7).** That ablation ran with Q6_K off;
  re-test against the fast baseline (small expected, but cheap and now measurable).
- **R6 — the generated policy / flywheel cost-model (`QK_GENERATED_POLICY`).** The whole machine-search policy
  was fit/scored against the Q6_K-bottlenecked regime; its per-tensor opt picks may be stale. Re-score on the
  new baseline — this is the flywheel's own dogfood and ties directly to the mission.
- **R7 — re-audit the "structural / occupancy e2e wall" conclusion.** We nearly closed the whole investigation
  on "the standalone→e2e gap is a structural occupancy wall." That was **wrong** — it was Q6_K coverage. Any
  downstream reasoning that assumed an intrinsic e2e cap is suspect and should be re-derived.

**Stays closed (not baseline-dependent, re-confirm only if cheap):** RDNA3 WMMA/tensor-cores need fp16 +
concrete dims (decode runs fp32, batch-1 — architectural, unchanged by Q6_K); BEAM hangs gfx1100 (hardware
fact); the cold/clock-controlled *standalone* kernel numbers (measured correctly, independent of the e2e
baseline). The int-dot-beats-llama standalone result is solid regardless.

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
