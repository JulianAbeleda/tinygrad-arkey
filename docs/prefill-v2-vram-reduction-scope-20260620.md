# Phase 5 (DESIGN ONLY): reduce PREFILL_V2 +14GB VRAM so 16GB cards can use it

Date: 2026-06-20. Scope: `docs/prefill-policy-integration-scope-20260620.md` Phase 5. **Design/scope only — do NOT
implement without owner approval (higher risk).** Context: `docs/prefill-default-policy-evaluation-result-20260620.md`.

## Problem

`PREFILL_V2` realizes an fp16 copy of every covered linear (FFN gate/up/down + attn q/k/v/o) and keeps it ALONGSIDE
the Q4_K decode storage → ~+14GB for 8B (5GB → 19GB). On 24GB it fits; on 16GB it OOMs, so the auto-policy disables
it there and those cards are stuck on the slow universal path. The cost is **duplication**: both Q4 (decode) and
fp16 (prefill) of the same weights resident at once. `realize_prefill_v2_weights()` (model.py) is the source:
`lin._pf16_w = lin.weight.cast(float16).contiguous().realize()` for every covered linear.

## Candidate approaches (ranked by risk/reward)

### A. Stream/per-layer fp16 realize during prefill (recompute, don't store) — best fit, hardest
Don't keep all fp16 weights resident. Realize each layer's fp16 weight just-in-time for its prefill matmul, free it
after. Peak extra VRAM ≈ a few layers' worth (~hundreds of MB) instead of 14GB.
- **Pro:** could bring PREFILL_V2 peak to ~Q4 + small → fits 16GB.
- **Con:** re-dequant Q4→fp16 per prefill forward = extra compute each call; the realized-buffer win (a clean TC
  GEMM input) must survive. Prefill is matmul-bound, so per-call dequant may erode the speedup. Needs the graph to
  realize→use→free per layer without the JIT pinning all buffers. **Hardest; measure the recompute cost first.**

### B. Lazy realize only the layers a given prefill actually touches — partial
For short prompts / few chunks, realize fp16 weights on first use and cache; cap resident set. Bounds VRAM by
working-set, not whole model. Simpler than A (no per-call free) but only helps when not all layers are hot (rare —
every prefill hits every layer). **Low reward for full-prompt prefill; skip.**

### C. Drop the Q4 source for prefill-covered tensors, keep ONLY fp16 — risky
If a tensor is fp16-realized for prefill, free its Q4 storage and serve DECODE from the fp16 too (dequant-free).
Removes the duplication (one copy, fp16). But: decode currently relies on the Q4 int-dot GEMV kernel (76% HBM peak)
— serving decode from fp16 would change decode perf/quality and the whole decode path. **Couples prefill VRAM to
decode correctness/throughput — high risk, likely a decode regression. Do not pursue without a decode re-eval.**

### D. fp8 / lower-precision prefill weights instead of fp16 — halves the cost, quality-gated
Realize the covered weights as fp8 (e4m3) not fp16 → +7GB instead of +14GB → fits some 16GB cards. RDNA3 WMMA
supports fp8 (`HIPRenderer` type_map has hip_fp8). But: prefill numerics change (fp8 GEMM) → must pass the dNLL +
greedy-exact gate; the loop-found TC warmstart schedule is fp16-specific (would need an fp8 variant). **Medium risk,
quality-gated; a real measurement project, not a quick win.**

## Recommended sequencing (if owner approves implementation)
1. **Measure the per-call re-dequant cost first** (Approach A's gating risk): time a prefill forward that
   re-realizes fp16 weights per layer vs the resident-fp16 baseline. If the recompute erodes the prefill win below
   ~the symbolic path, A is dead → only D (fp8) remains, behind a quality gate.
2. If A's recompute is cheap (prefill is matmul-bound, dequant may hide): build per-layer streaming realize, gate
   peak VRAM (target: fits 16GB) + byte-identical + synced prefill not materially slower.
3. D (fp8) is the fallback if A fails — but it is a quality-gated numerics change, scope it separately.

## Gate (any implementation)
Peak VRAM measured (target ≤ ~14GB for 8B so 16GB cards fit) + rel_RMSE/dNLL/greedy-exact unchanged + synced
prefill within ~10% of resident-fp16 + decode UNCHANGED (esp. Approach C/D). Default-off; gfx1100.

## Verdict
**Do not implement in this policy pass.** The auto-policy (Phase 1) already makes PREFILL_V2 safe on 24GB+ and a
no-op elsewhere — 16GB support is a nice-to-have, not a blocker. If pursued, **Approach A (per-layer streaming
realize), gated by a recompute-cost measurement first**, is the only one that both fits 16GB AND keeps decode
untouched. C is risky (decode coupling); D is a separate quality-gated numerics project.
