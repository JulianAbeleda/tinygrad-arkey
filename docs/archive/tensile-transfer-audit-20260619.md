# AUDIT — Tensile DOES transfer in-model (1.56x kernel / 1.39x predicted); the 0.997x is route_pf16 TRANSPOSES (fixable)

Audited the "external Tensile doesn't transfer in-model (0.997x)" verdict. It is WRONG as a wall -- the win is real
and destroyed by a known, fixable routing overhead.

## The evidence (decisive)
- **Clean interleaved A/B (clock-fair, reproduced): 0.997x** e2e with route applied {qo:72,gateup:72,down:36},
  correctness exact. (NOTE: this A/B was always clock-fair -- my earlier "0.999x is clock-confounded/suspect" claim
  was WRONG; the interleaved round-robin hits both arms equally.)
- **Shape-matrix (Tensile kernels launched at the EXACT in-model shapes via HCQ, isolated): Tensile is FASTER:**
  - ffn_gate_up: 65.6 TF = **1.56x tinygrad** | ffn_down: 69.8 TF = **1.66x tinygrad** (rel_err ~3e-4, verified).
  - Weighted model prediction: gate_up 1.183x, +down 1.313x, +attn_q_o **1.393x** e2e.

## The contradiction -> the cause
Tensile is 1.56-1.66x at the in-model shapes ISOLATED (predicts 1.39x e2e), but routed THROUGH THE MODEL = 0.997x.
So it is NOT a shape mismatch (fast at the exact shapes) and NOT kernel-not-selected (route applied) and NOT a
wall. The win evaporates in the ROUTE. Cause found in code:
- **`route_pf16` (model.py:63, the DEFAULT/measured route) adds TWO transposes+copies per linear:**
  `x2.transpose().contiguous()` (activation -> [in,T]) and `out_t.transpose()` (output). These copies cost ~as much
  as the matmul saves -> net 1.0x.

## The fix (already coded, NEVER measured e2e)
`route_pf16_col` / `_ffn_tensile_col` (model.py:70, qk_tensile_inmodel.py:78) = TRANSPOSE-FREE: keep the whole FFN
in [feature,T] (column) layout so the per-linear in/out transposes CANCEL across gate->up->down; Tensile's native
output is already [out,T]. If the transposes were the 1.39x->1.0x destroyer, the column route should recover most of it.

## Corrected verdict + solution
The vendor-.co solution is VIABLE, not walled: a real ~1.39x e2e prefill win exists, blocked by a FIXABLE routing
overhead (route_pf16 transposes), with the fix (route_pf16_col) coded but unmeasured. This REOPENS Option B from the
solution scope as the bounded path to ~1.39x prefill (42->~58 effective, ~80% llama).
- NEXT (P0): wire/measure `_ffn_tensile_col` (transpose-free) e2e in the clean interleaved A/B. Gate: >=1.25x e2e.
- If it recovers the win -> the bounded solution is: transpose-free column route + extend to attn_q_o + handle the
  FFN-boundary transposes (enter/leave column layout once, amortized). Dependency = the vendored Tensile .co.
- If it does NOT recover (transposes weren't the whole story) -> PMC the in-model Tensile kernel: does it hit 65 TF
  IN the model graph, or does graph context (occupancy/contention/clock) slow it vs the isolated HCQ launch.

## Files
extra/qk_tensile_shape_matrix.py (in-model-shape Tensile bench), qk_tensile_ab_measure.py (e2e A/B),
qk_tensile_inmodel.py:62/78 (route_pf16 vs route_pf16_col), model.py:62-74. Supersedes the "doesn't transfer" framing
in prefill-tensile-land-result + the (now-corrected) clock-confound claim in rocblas-quality-solution-scope.
