# LEARNING + SCOPE — prefill: matmul is NOT the lever (exhausted); symbolic-KV attention IS; + explicit TC attention

Exhausts "why the 1.4x gate/up kernel didn't win e2e" and scopes the symbolic-start_pos mechanism + the explicit
TC-attention path. The prefill bottleneck is now precisely understood.

## EXHAUSTED — why no matmul-kernel improvement wins prefill e2e
2x2 A/B (gate/up old/new x start_pos symbolic/concrete, clock-controlled, forced-high clock):
| | concrete | symbolic |
|---|---:|---:|
| old gate/up | 1539 tok/s | 1244 |
| new gate/up | 1542 (+0.2%) | 1238 (-0.5%) |
- **gate/up new schedule = ~0% e2e in BOTH concrete AND symbolic** → NOT masked by attention, genuinely irrelevant.
- Joins Tensile (0.999x), transpose-free (0.997x): **ALL FOUR matmul-kernel wins → ~1.00x e2e.**
- **Mechanism:** at warm/full clock the matmuls (compute, TC) are fast → a SMALL fraction of the wall; the wall is
  set by work that does NOT scale with core clock — memory-bound attention + per-kernel overhead. So "63 TFLOPS
  matmul kernel" is real but irrelevant: the matmul isn't the critical path. (The cold-capture 42% gate/up share
  inverts at warm clock.) **Prefill is NOT matmul-bound at warm clock.**

## SCOPE 1 — symbolic-start_pos mechanism (the real bottleneck; 1.24x lever)
`_attention` (model.py:798) slices the KV cache to `0:start_pos+T` (symbolic length when start_pos is a bound UOp,
used so ONE prefill jit replays across chunks). The SDPA reduction is then over a **symbolic KV dimension** → the
optimizer can't tile/TC it → it lowers to a slow generic reduce.
- Measured penalty (concrete vs symbolic capture diff): symbolic has `r_2_512_(start_pos+512)_8_4_4_16` = 75ms @
  **1451 GFLOPS** (the dominant attention reduce) — GONE/folded in concrete; and `r_16_32_..._128` runs **4415
  (symbolic) -> 6491 (concrete) GFLOPS (1.47x)**.
- **Concrete start_pos=0 -> KV=0:512 concrete -> reduce tiles -> 1.24x e2e, byte-identical** (validated 2x).
- Symbolic is needed ONLY for chunked prefill (start_pos>0). **Single-chunk (prompt <= 512, start_pos=0) gets the
  1.24x for free** (one cached concrete-0 jit). Chunked: a concrete jit per start_pos (recompile, amortized over
  server reuse) or keep symbolic.

## SCOPE 2 — explicit TC attention (Option B), stacks on concrete KV
`extra/qk_prefill_tc_wr_softmax_probe.py` `_explicit`: explicit Q@Kᵀ (TC, fp16) -> materialized fp16 scores ->
softmax -> P@V (TC), GQA via BROADCAST (K/V per kv-head expanded over the G group dim, no repeat_interleave).
- **2.56x over SDPA standalone (concrete KV)**; **0.79-0.92x in-model (symbolic KV -> TC can't fire)** -> the
  symbolic KV is the ONLY blocker (`amd-prefill-tc-attention-probe-20260617.md`).
- **On the concrete-KV path, the TC fires** -> Option B should win, stacking on top of the 1.24x reduce-tiling.
  Wiring: in `_attention`'s prefill_v2 branch, when start_pos is concrete, call `_explicit(q,k,v,mask,KV)` instead
  of `q.scaled_dot_product_attention(...)`. fp16 scores: confirm dNLL <= 0.01 (the probe's smoke was byte-identical).

## The prefill lever (ranked; matmul de-prioritized as exhausted)
1. **Concrete-start_pos prefill** (single-chunk, start_pos=0): 1.24x, byte-identical, one cached jit. SHIPPABLE.
2. **Explicit TC attention on the concrete path**: stacks (2.56x standalone on the ~25% attention) -> potentially
   another ~1.1-1.2x e2e. Needs dNLL gate (fp16 scores).
3. ~~matmul kernel (Tensile / gate-up schedule / transpose-free)~~ — EXHAUSTED, ~1.00x e2e, do not pursue.

## Caveats
Cold-clock captures (% split valid, absolute ms ~2x inflated). JIT-replay per-kernel uncapturable (capture-run is
the workaround). The 1.24x is the clean clock-controlled interleaved e2e number. Chunked-prefill concrete-jit
recompile cost is the open question for prompts > 512.

## Files
2x2 + A/Bs inline / `extra/qk_gateup_sched_ab.py`; captures `/tmp/concrete_cap.txt` vs `/tmp/tuned_cap.txt`.
Prior: `prefill-l1-l2-result-20260619.md`, `prefill-exact-split-result-20260619.md`,
`amd-prefill-tc-attention-probe-20260617.md`, probe `extra/qk_prefill_tc_wr_softmax_probe.py`.
