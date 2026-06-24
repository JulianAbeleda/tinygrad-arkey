# Beyond llama.cpp — the decode roadmap past parity

Date: 2026-06-15, **updated 2026-06-16** (post-capstone: ffn_down demotion, P2 flash-decode shipped,
default-on flip, prefill diagnosed). State: default decode **60.9 tok/s = ~33% of HBM peak = 58% of
llama.cpp** (RX 7900 XTX, Qwen3-8B Q4_K_M), up from 23 before the Q6_K work. llama.cpp = 105.7 tok/s = 57%.
Out-of-the-box (default-on primitives) is now **55 tok/s** with no flags (was 12).

> Scope: this doc is **decode** (batch-1 token generation). Prefill is a separate, worse gap with a
> different root cause — see `docs/amd-decode-prefill-plan.md` (~2% of llama; it's an LDS-cache-blocking
> codegen problem, not a coverage/policy problem). Don't conflate the two.

## The roofline (why "beyond llama" is real, not aspirational)
Batch-1 decode reads ~4.68 GB of weights/token. At HBM peak (859 GB/s) the floor is **5.45 ms = 183 tok/s
(100% of peak)**. So:

```
peak      183 tok/s |==========================================| 100%
llama.cpp 105.7     |=========================|                  57%   <- a FIXED hand-tuned scheme
us now    60.9      |==============|                             33%   (58% of llama)
                     ^---- parity gap ----^------- BEYOND room (43% of peak still on the table) ------^
```

There is **as much headroom above llama as between us and llama.** llama leaves 43% of peak unused because it
is a fixed scheme (fixed quant per type, fixed kernels, per-layer-sequential). The machine-search edge is an
*adaptive per-tensor policy* — and we already proved that wins (mixed-quant coverage: 23→53 tok/s; ffn_down
Q6→Q4 demotion: +14% free). The levers below extend the policy's decision space (kernel, bit-width, schedule,
sparsity) — each one is a place a search can beat a fixed reference.

## ⚠ The key reframe (2026-06-16): the parity gap is the NON-GEMV tax, not a slow GEMV
A clean post-fix profile (`amd-decode-arc-synthesis.md` §6) makes the deciding factor explicit:

- The **in-graph GEMV is already at ~52% of peak ≈ llama's 57% per-kernel rate.** A GEMV-only token would run
  at **~80–95 tok/s (≈80–90% of llama)**. The per-kernel GEMV is **not** what keeps us below llama.
- We land at 60.9 (58%) because the token is **sequential**: `token = GEMV + non-GEMV (≈ sum)`. The non-GEMV
  work (attention reduces, norms, rope, lm_head sampling) is the **tax** that drops e2e from ~85 → 60.9.
- llama reaches its GEMV ceiling end-to-end, which means **llama is NOT paying our non-GEMV tax** (it fuses /
  has near-zero framework non-GEMV). So **closing/overlapping that tax is the parity path** — not a faster
  GEMV.

**Consequence for the categorization:** the lever that closes parity is **make non-GEMV cheap (P2/flash, done)
+ overlap non-GEMV behind the weight stream (B2)**. B2 was previously filed under "beyond"; it is promoted to
the **lead parity lever** below. The faster-GEMV levers (B1/R1/R3/R4/R5) are out — see B1.

## Parity levers (close the gap to llama, 33% → 57%)
- **P1 — lm_head primitive: DONE** (`Q6K_COVER_MORE` default-on; eliminated the 2.7 ms `r_1187`).
- **P2 — flash-attention decode: SHIPPED & EXACT (2026-06-16).** `FLASH_DECODE=1`, byte-exact vs SDPA
  (`extra/qk_flash_decode.py`, 5 single-accumulator UOp kernels). It is a **long-context** parity lever:
  SDPA's dense KV read collapses 6.0× by ctx 3072; flash flattens that to 2.1×. Measured (Q4K_PRIMITIVE=1):
  ctx 8 `56.2→47.5` (0.84×), ctx 1024 `27.6→34.3` (1.24×), ctx 3072 `9.4→22.7` (**2.41×**). Crossover ~ctx
  400, so **default-off** (5 extra kernels/layer cost ~15% at short ctx) — enable for long-context serving.
  At short context P2 is spent: the residual non-GEMV is diffuse (norms 1.3 ms, sampling 0.85 ms, small sdpa)
  and the win is now **B2 overlap**, not a faster attention kernel.
  - **Now CONTEXT-AWARE (2026-06-16, `amd-decode-flash-threshold-20260616.md`):** the flag is a searched
    threshold — `FLASH_DECODE_THRESHOLD=384` uses SDPA below ctx 384 and flash above, so one serving run
    gets the long-ctx win with **zero short-ctx regression**. 8B crossover searched on the scaffold (Track
    2 dogfood); exact; default-off. The flash lever is now exhausted for 8B.
- **B2 — overlap non-GEMV behind the weight stream. ← THE LEAD PARITY LEVER (promoted from "beyond").**
  Today `token = GEMV + non-GEMV` (sequential, ≈ sum). The ~29% non-GEMV (attention/norms/lm_head) can be
  pipelined to run *while* the next layer's weights stream from HBM → `token = max(GEMV, non-GEMV)` not sum.
  Since GEMV alone ≈ 80–95 tok/s, overlapping the ~4–5 ms non-GEMV recovers most of the gap to llama by
  itself (→ ~80 tok/s ≈ 75–80% of llama) and **keeps stacking past it**. This is structural, not a kernel
  tweak; it is both the parity finish and a beyond-llama multiplier. **Build this next.**
  - **Status 2026-06-16: validated + GREENLIT for build** (`amd-decode-overlap-derisk-20260616.md`).
    Spike confirmed HBM ~58% idle during the GEMV (+38% ceiling); a concurrent-decode capacity test
    reclaimed +32% (the AMD hardware genuinely overlaps two streams) → realizable **~+25–32%**. Needs a
    second compute queue in `runtime/graph/hcq.py` (the device already exposes the queue primitive; the
    graph uses one today). The norm-fusion warm-up was **refuted** (single-accumulator kernel blocks the
    RMS reduction; non-exact int8 round; norm already lazily fused) — overlap is the path, not fusion.
  - **Milestone 0 (two-queue probe, `amd-decode-two-queue-probe-20260616.md`): scope escalated.**
    tinygrad's AMD backend has **one** hardware compute ring (`ops_amd.py:1001`, `_submit` hardcodes
    `dev.compute_queue`), so two queue objects serialize (measured 1.0×, invariant to kernel shape).
    Hardware overlap is real (cross-process +32%), so the build now needs a **`[runtime]` 2nd compute
    AQL ring + per-ring submit routing** *before* the cross-layer scheduler — deeper than first
    estimated. Decision pending: invest in the backend surgery for ~+30%, or pivot to **B3** (fewer
    bytes, no runtime surgery). `extra/qk_two_queue_probe.py` is the gate that re-fires once a 2nd
    ring exists (A‖B should jump > 1.2×).

## Beyond-llama levers (surpass 57%) — each ties to the policy/primitive frame
Ranked by (ceiling × feasibility). The path is **change the work, not the kernel.**

- **B1 — faster in-graph int-dot GEMV. TESTED → DECISIVE NEGATIVE (`B1_INTDOT_RESULT.md`).** Standalone
  int-dot is 76% vs fp 56%, but in-graph it runs 28.5 µs vs fp 32 µs — only 1.12×, both ~34% of peak. The
  gap is **single-shot occupancy** (the attn GEMV is 64 workgroups, too few to fill 96 CUs), not compute, so
  int-dot's compute win is invisible and the q8 quant overhead cancels it → null. split-K is within noise;
  fusion hurts. **The per-layer GEMV is already at its batch-1 ceiling (~50–55% ≈ llama's 57%).** A faster
  per-kernel GEMV is NOT the path. This also closes R1/R3/R4/R5 (all per-kernel GEMV levers, same reason).
  The only way to make the GEMV launch "bigger" is to give it more work per pass → **B5**.
- **B3 — per-tensor sub-4-bit policy (read FEWER bytes than llama).** llama reads the full Q4_K_M (4.5 b/wt);
  we already beat that once by demoting ffn_down Q6→Q4 (+14%, free). Extend the policy to decide *bit-width*
  per tensor — push tolerant tensors to 3-bit/2-bit where a search shows acceptable per-tensor error (the
  inverse of mixed-quant). ~15–20% fewer weight bytes → GEMV time drops proportionally. The purest "machine
  search beats a fixed scheme" lever; directly reuses the cost-model/flywheel + the existing `qk_quantize`
  quantizer. (Near-term sub-task: cache the requant to kill the ~3 min load cost.)
  - **Status 2026-06-16: Q6→Q4 demotion search DONE — frontier mapped, lever tapped at the quality budget**
    (`amd-decode-demotion-search-20260616.md`). First real run of the `qk_search_spec` scaffold (spec →
    isolated tok/s+dNLL runner → quality gate → AcceptedPolicy). Result: ffn_down (shipped) + attn_v accepted
    (~64 tok/s, 63% llama, dNLL within budget); **lm_head rejected** — fastest at 75 tok/s / 74% llama but
    dNLL +0.051 (the gate caught a tempting-but-degrading config). So the in-pattern Q6→Q4 frontier is ~64
    tok/s. **Bigger bytes need true sub-4-bit (Q3/Q2 on the Q4 bulk) = a new quantizer + new GEMV kernel
    (dangerous-power surface) — deferred**, same gate as the overlap 2nd-ring build.
- **B4 — sparse / compressed-KV attention (DeepSeek-DSA style).** Builds on the shipped flash kernel: instead
  of reading the full KV cache every token, read only top-k / compressed slots → bandwidth grows sub-linearly
  with context. Beyond llama's dense attention; biggest at long context (where flash already pays). Lossy →
  gate on perplexity/coherence, unlike the exact Q6_K/flash wins.
- **B5 — multi-token / self-speculative (amortize the read). THE MULTIPLIER.** Emit >1 token per 4.68 GB
  weight pass via MTP / Medusa-Eagle heads (no separate draft model). Even at our current ~52% per-kernel
  rate, emitting 2 tokens per weight pass ≈ doubles throughput. NB: the **draft-model** form was tested and
  is **net-negative** (`B5_S0`/`S1_SPECULATIVE`: a 1.7B draft is too costly); the self-speculative *heads*
  form is the viable version. The verify path is already fast (S3 batched GEMM primitive, built + dormant).

### Stacked beyond-llama ceiling
B1 is out (per-kernel GEMV at ceiling). The realistic route is **B5 × (B3 + B2)**:
B2 (overlap: token = max not sum) + B3 (sub-4-bit: fewer bytes) + B4/flash (cheap attention) + B5 (amortize
across tokens). B5 is the multiplier; B2 is the structural unlock that makes everything else stack instead
of sum.

## Scope: the immediate next concrete steps (revised)
1. **B2 (overlap)** — pipeline the non-GEMV behind the next layer's weight stream so `token = max(GEMV,
   non-GEMV)`. This is the lead parity-and-beyond lever now that flash has flattened long-ctx attention and
   the GEMV is at its per-kernel ceiling.
2. **B3 (sub-4-bit)** — extend the per-tensor policy from "which kernel" to "which bit-width"; reuse the
   `qk_quantize` quantizer + flywheel cost-model; cache the requant. Free-quality demotions first (the
   ffn_down win generalizes), then search the tolerant tensors.
3. **B4 (sparse KV)** — once long-context serving uses flash, add top-k/compressed-KV on top; gate on
   coherence.
4. **B5 (self-speculative heads)** — the throughput multiplier; the draft-model form is already refuted, the
   verify primitive (S3) already exists.

## REVISIT — closures measured against the Q6_K-bottlenecked baseline (resolved)
The Q6_K discovery invalidated the pre-fix baseline; every e2e experiment closed *before* the flag was on ran
with the Q6_K `ffn_down` fallback as a fixed 59%-of-token noise floor, so Q4_K-path levers moved e2e by only
`~0.4 × win` → often null. Status after the post-fix re-derivation:

- **R1 (= B1, in-graph int-dot): re-tested → still null**, but for the *right* reason now (occupancy ceiling,
  not Q6_K dilution). See B1.
- **R3/R4/R5 (horizontal fusion / per-kernel opts / attn_k coverage):** all per-kernel GEMV levers; B1's
  occupancy finding predicts within-noise. Cheap to re-confirm but low priority — the GEMV is at ceiling.
- **R6 (generated policy / flywheel cost-model):** was fit against the Q6_K-bottlenecked regime; re-scored on
  the fast baseline. Generated policies now win on 14B/32B (see `amd-decode-current-verdicts.md`).
- **R7 (the "structural occupancy e2e wall" conclusion): RESOLVED — it was Q6_K coverage, not an intrinsic
  cap.** The post-fix wall is the **non-GEMV sequential tax** (the reframe above), which B2 attacks.

**Stays closed (architectural, not baseline-dependent):** RDNA3 WMMA/tensor-cores need fp16 + concrete dims
(decode is fp32, batch-1); BEAM hangs gfx1100 (hardware fact); the cold/clock-controlled *standalone* kernel
numbers (measured correctly). The int-dot-beats-llama standalone result (76% vs 57%) is solid regardless.

## The thesis (why this is the machine-search mission, not just kernel hacking)
llama.cpp is the strong *fixed* baseline. Every lever is the policy gaining a new degree of freedom:
- coverage (which kernel) — **DONE, won** (Q6_K + COVER_MORE).
- bit-width (per-tensor sub-4-bit) — **B3** (ffn_down demotion already proved it).
- schedule (overlap) — **B2** (the parity unlock).
- sparsity (KV) — **B4** (on top of shipped flash).
- amortize-across-tokens — **B5** (self-speculative heads).
- kernel choice (int-dot vs fp-dequant) — **B1, refuted** (occupancy-bound, not the path).

A hand-tuned reference picks one good fixed point in this space. A search picks per-tensor, per-shape,
per-context. That is the structural reason search can go beyond llama — and the roofline says there is 43% of
peak waiting to prove it. The corrected insight from the arc: **the next wins are scheduling and bytes, not a
faster kernel** — the kernel is already at its batch-1 ceiling.

Anchors: `KERNEL_BEATS_LLAMACPP.md` (int-dot 76%), `Q6K_FIX_RESULT.md` (coverage win),
`B3_DEMOTE_RESULT.md` (sub-4-bit demotion), `amd-decode-capstone.md` (the 60.9 ledger),
`amd-decode-arc-synthesis.md` (the primitive frame + §6 non-GEMV breakdown),
`amd-decode-flash-attention-plan.md` (P2 SHIPPED section), `amd-decode-prefill-plan.md` (separate prefill
gap), `amd-decode-measurement-confounds.md` (how to measure any of this).
</content>
</invoke>
