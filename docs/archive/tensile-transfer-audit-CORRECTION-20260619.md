# CORRECTION — Tensile does NOT transfer in the JITted graph (the transpose hypothesis is REFUTED)

The audit (`tensile-transfer-audit-20260619.md`) concluded "Tensile transfers at the kernel level (1.56x); the
0.997x is route_pf16 TRANSPOSES; fix = column route." Tested it -> REFUTED.

## Tests
1. The column route (`_ffn_tensile_col`) is ALREADY the active path under PREFILL_TENSILE_GEMM (model.py:740) -> the
   0.997x A/B already used it. "Unmeasured fix" was wrong.
2. **FFN-only e2e (qo routing DISABLED via QK_SKIP_ROUTE_PF16, only FFN column routed): 0.993x** -- STILL no win.
   So transposes/qo are NOT the canceller.
3. The isolated FFN-block "1.228x" I measured was an **EAGER-mode artifact**: eager FFN block = 31ms vs ~6ms/layer
   in the JITted forward (eager is dispatch-bound, where Tensile's fewer ops help; the JITted graph is not).

## Conclusion (honest)
**The external Tensile kernel does NOT transfer to the JITted model graph**, even FFN-only, even transpose-free.
The shape-matrix's 65 TF / 1.39x prediction comes from ISOLATED HCQ kernel launches (captured kernarg, no
surrounding work) and does NOT materialize when the same kernel runs via custom_kernel inside the 729-kernel JITted
HCQ graph (0.993-0.997x). By Amdahl, if the in-graph FFN matmul actually ran at 1.56x it would give ~1.36x e2e;
getting ~1.0x means the in-graph Tensile matmul achieves ~0 speedup over in-graph fp16-WMMA -> the isolated 65 TF
is not realized in-graph.

## So the vendor-.co path is NOT bounded
The "doesn't transfer in-model" verdict STANDS (now triple-confirmed: full-route 0.997x, FFN-only 0.993x, + the
campaign-wide meta-pattern). The blocker is NOT transposes (audit hypothesis refuted) -- it's that the isolated
kernel doesn't deliver its speed inside the JITted graph (clock/occupancy/dispatch context, or the isolated HCQ
measurement was optimistic). Diagnosing WHY the in-graph kernel underperforms its isolated launch is the remaining
question, but it is NOT a quick transpose fix.

## Honest meta-note
This audit made two premature claims, both refuted by measurement: (a) "0.997x is clock-confounded" (the
interleaved A/B is clock-fair), (b) "the fix is the unmeasured column route / transposes" (column route was
already active; FFN-only is still 0.993x). The reliable result: Tensile isolated = fast (65 TF at in-model shapes),
in JITted graph = no transfer. Prefill rests at PREFILL_V2 WMMA ~42 / ~47% llama + concrete-KV 1.24x shippable.

## Files
qk_tensile_ab_measure.py (e2e A/B), qk_tensile_shape_matrix.py (isolated 65 TF), model.py:740 (_ffn_tensile_col
already active). Supersedes the optimistic tensile-transfer-audit-20260619.
