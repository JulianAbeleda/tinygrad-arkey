# NV Q/K norm completion launch-elimination scope (2026-08-22)

Status: implementation scope; schedule fusion only after the isolated gate.

1. Row and edge: Q norm `reduce_output_rmsnorm_32_128` (36 x 2.560 us) and K
   norm `reduce_output_rmsnorm_8_128` (36 x 2.496 us). Edge: norm -> rope ->
   completion -> projection consumer.
2. Dominant term: `D + R` (launch + residual). Body is BODY_PARITY (Q 1.190,
   K 1.196 vs llama 1.15/1.13). Legal recoverable term is D+R, not P.
3. Code paths: schedule fusion of Q completion+norm+rope and K
   completion+norm+rope/store into one support kernel per side.
4. Legality: generic Q/K head-norm completion shapes; no quant-route change.
5. Fallback: two separate shapes (Q, K), not one combined; if the body moved
   into the survivor is slower, stop.
6. Contract: exact norm/rope output; token SHA identical.
7. Arms: isolated = D+R decomposition (retained `phase3/`); installed =
   reverse wall bracket.
8. Census gate: 72 norm nodes -> 36 fused survivors; verify no copy added.
9. Reverse wall bracket, +50 us promotion bar.
10. Rollback: revert schedule; non-regression on non-NV targets.
11. Projected ceiling 96.1 us, labelled unmeasured.
12. Prohibited: model-name or block-list dispatch.
