# RESULT — transpose-free Tensile FFN: CORRECT but 0.997× → Tensile offers NO in-model advantage (lever REFUTED)

Built the transpose-free column-layout Tensile FFN (`prefill-tensile-transpose-free-scope-20260619.md`):
`route_pf16_col` (column in/out, no transposes) + `_ffn_tensile_col` (model.py, keeps gate/up/down in [feature,T],
one transpose at FFN entry + exit). Measured via the clean clock-controlled interleaved A/B.

## Result [M] (clean interleaved A/B, reproduced)
- **correctness rel_err(ON vs OFF) = 0.00000** — the column FFN is byte-identical to the WMMA path (the transpose
  identity C=W@xᵀ=(x@Wᵀ)ᵀ holds).
- **OFF (fp16-WMMA) 2675 tok/s vs ON (transpose-free Tensile) 2666 → 0.997×.** route={qo:72,gateup:72,down:36}
  (FFN gate/up/down now via `route_pf16_col`, attn-qo still per-linear).
- **Eliminating the transposes did NOT help.** So the diagnostic's transpose-tax hypothesis, while real for the
  *old* route, was NOT the e2e bottleneck.

## What this proves — tinygrad's in-model WMMA already ≈ Tensile (the "66 vs 41" was a category error)
The OFF (WMMA) path never had the route's transposes (those were Tensile-route-specific; `x.linear(w.T)` fuses).
So transpose-free ON (Tensile gateup = 810µs/63 TFLOPS, measured) ≈ OFF ⟹ **tinygrad's in-model warmstart-WMMA
gateup already runs at ~810µs (~63 TFLOPS) — the same speed as Tensile.** The earlier hierarchy "Tensile 66 >
tinygrad 41" compared **Tensile kernel-TFLOPS vs tinygrad e2e-effective-TFLOPS** (2486 tok/s × 16.4 GFLOP) — a
category error. The matmul KERNEL is ~63 TFLOPS in-model; the e2e 41 is the **non-matmul dilution** (attention,
norms, transposes, residuals, layout). warmstart-TC lifts WMMA from eager-untuned 20 TFLOPS to ~63 in-model.

## Verdict — Tensile prefill lever REFUTED (both as-built and transpose-free)
- **Tensile gives no in-model prefill win** — tinygrad's warmstart-WMMA matmul already matches it. Frontier #2
  (Tensile fp16) is dead at the in-model level. (The deps/TPE-0 policy question is moot — there's nothing to land.)
- This is the THIRD convergent instance of the campaign meta-pattern, now with the sharpest twist: not only do
  isolated kernel wins fail to transfer in-model — here **tinygrad's own in-model kernel was already at the
  "winning" kernel's speed**, so the isolated comparison was measuring the wrong thing entirely.
- **The real prefill gap** = e2e-effective 41 vs matmul-kernel ~63 TFLOPS = the **~35% non-matmul overhead**
  (attention, RMSNorms, residuals, the activation transposes/casts, lm_head). THAT is where any prefill e2e win
  lives — NOT the matmul, and NOT Tensile.
- prefill rests at PREFILL_V2 fp16-WMMA (~82% llama). The transpose-free column FFN is correct + kept
  research-flagged (`PREFILL_TENSILE_GEMM=0`); it's the "right" route but e2e-neutral, so not landed.

## Files
`extra/qk_tensile_inmodel.py` (`route_pf16_col`), `tinygrad/llm/model.py` (`_ffn_tensile_col` + flag branch),
`extra/qk_tensile_ab_measure.py` (now with correctness check). Diagnostic: `prefill-tensile-diag-result-20260619.md`.
Map: `inference-perf-measured-map-20260619.md`.
