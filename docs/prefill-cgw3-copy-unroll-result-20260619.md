# Result — CG-W3 copy vectorization via reduce-axis UNROLL (SHIPPED, +3.7% prefill, dependency-free)

The cheap "option 1" test (extend the hand-coded warmstart opt tuple) — and it WORKS, modestly.

## What shipped
`model.py` `_prefill_v2_opts`: added `Opt(OptOps.UNROLL, 0, 8)` to the warmstart tuple
`(TC, UPCAST(0,4|2), UPCAST(1,4))` → `(…, UNROLL(0,8))`. Gated behind PREFILL_V2 (default off), prefill-only,
decode untouched.

## Mechanism (corrects the CG-W1.5 pessimism)
CG-W1.5 feared "no contiguous copy axis exists (the global read is strided by the transpose)". FALSE — **unrolling
the reduce (K) axis by 8 makes each thread's per-iteration copy loads contiguous in K**, so they fold (via
`fold_expanded_index`) from per-element `global_load_d16` (each needing a `v_mov` register-init) to wide
`global_load_b128`. The ffn matmul ISA:
- baseline (CG-W1.5): `v_mov`=127, `d16`=16, `b128`≈2, `wmma`=16 → ~8 `v_mov`/WMMA, per-element d16 loads.
- +UNROLL(0,8): `d16`=**0**, `b128`=159, `v_mov` amortized over `wmma`=64 → **~2 `v_mov`/WMMA**, no per-element d16.

## Gates (all pass)
| gate | result |
|---|---|
| copy vectorized (ISA) | d16→b128, v_mov/WMMA 8→2 ✓ |
| pp512 (fair, min-of-20, reproduced ×2) | **1170 → 1213 tok/s = +3.7%** ✓ |
| VGPR spill | **0** (UNROLL,4 spilled 362 → rejected; UNROLL,8 clean) ✓ |
| dNLL (vs baseline) | **−0.00013** (≤0.01) ✓ |
| warmstart apply / error | 5 / 0 (applies to all linears) ✓ |
| decode | untouched (`_prefill_v2_opts` is prefill-only) ✓ |
| dependency / BEAM | none (pure tinygrad, no runtime search) ✓ |

## Honest sizing
+3.7% is **small** — it re-confirms gfx1100 is Infinity-Cache-served: vectorizing the copy (wide loads + fewer
`v_mov`) helps, but the copy was never the dominant cost, and the wide-load bandwidth win is muted by the IC. This
is NOT the 48→66 (1.37×) Tensile gap; that gap is broader (WMMA issue density / the full Tensile schedule), still
only reachable via the external rocBLAS route (1.41× llama, dependency) or a deeper codegen change.

## What this closes
- **Option 1 (extend the opt tuple) is settled: it works** — a contiguous copy axis exists via reduce-unroll, the
  copy vectorizes, and it yields a real (if modest) dependency-free win. SHIPPED.
- The two-stage-copy scheduler change (Route A deeper form) is **not needed** for the vectorization itself (UNROLL
  achieves it); it would only matter for chasing the larger Tensile gap, which is IC-limited anyway.
- Net prefill state: PREFILL_V2 + UNROLL ≈ +3.7% over prior PREFILL_V2; still ~0.8× llama; Tensile (dep) the only
  ≥llama option.

## Provenance
`bench/qk-codegen-wmma/inmodel_matmul.json`; the CG-W2/W2b refutations + CG-W1.5 baseline; ISA via llvm-objdump;
pp512 `/tmp/cgw_measure.py` & `/tmp/cgw_verify.py` (min-of-20, ×2 each); dNLL `extra/qk_prefill_v2_nll_eval.py`
(window 0: baseline 2.58238 / v2 2.58224).
