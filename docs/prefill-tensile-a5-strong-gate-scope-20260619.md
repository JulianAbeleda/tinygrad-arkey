# Scope - A5: push the in-model Tensile prefill route to the strong gate (>=1.35x)

A4 measured **1.27x** warm pp512 (PASS_RESEARCH) routing ffn_gate/up/down through the extracted Tensile kernel. Two
bounded levers remain to reach the strong gate (>=1.35x pp512), neither needing new capability:

1. **Route attn_q/o** (the ~17% of the matmul bucket left on PREFILL_V2). attn_q + attn_output are `(4096,4096)`,
   already on the `_pf16` path (`_PREFILL_V2_LINEARS`), and the kernel is extracted (TPE-5 attn_q_o, 58.9 TFLOPS =
   1.40x tinygrad). Just add `(4096,4096)->"qo"` to `ELIGIBLE` + a `TensileRunner` for it. Low-risk.
2. **Drop the per-linear x/out transposes** via a `[feature,T]` FFN-block restructure (gate/up output `[FF,T]` ->
   silu*mul `[FF,T]` -> down `[IN,T]`, intermediate stays transposed; only block entry/exit transpose). Higher value
   (transposes are on every routed linear) but more invasive (touches `FFNBlock.__call__` + residual layout).

## Phases
- **A5-1 (do first, low-risk):** add attn_q/o routing (role "qo", `(4096,4096)`); re-measure warm pp512 + dNLL.
  Gate: speedup rises and dNLL still <=0.01; decode/fallback unchanged.
- **A5-2 (only if <1.35x after A5-1):** quantify the transpose overhead (graph attribution / count), then if it is
  material, restructure the FFN block to `[feature,T]` to remove per-matmul transposes. Re-measure.
- **A5-3:** final verdict — PASS_STRONG_POLICY_GATED if >=1.35x (pp512), else PASS_RESEARCH stands with the new
  (higher) pp512 number and the transpose overhead documented.

## Gates / constraints
- correctness rel<=2e-2 per routed linear; dNLL<=0.01; fallback (flag off / ineligible) == PREFILL_V2; decode
  untouched; research-only, default off, external HSACO artifact only when `PREFILL_TENSILE_GEMM=1`.
- KILL a lever if it regresses dNLL, breaks fallback, or doesn't move pp512.

## Non-goals
No default/ship; no new extraction (attn_q/o already extracted); no pure-codegen; no attn_k/v (low EV) unless A5-2
still short. TPE-0 artifact policy stays open.

## Deliverables
attn_q/o eligibility in `extra/qk_tensile_inmodel.py` (+ optional `[feature,T]` FFN restructure in model.py, flag-
gated), updated `bench/qk-tensile-extraction/inmodel_measurement.json`, result appended to
`prefill-tensile-inmodel-measurement-result-20260619.md`.
