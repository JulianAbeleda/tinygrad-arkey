# NV vocab tail reduction-topology scope (2026-08-22)

Status: implementation scope; reduction topology only. Top-1 fusion into the
current main route is closed (wall-negative) and must not be repeated.

1. Row and edge: `r_32_4_1187` (39.296 us), `r_128_16_8_1187` (11.040 us),
   and `E_1187_32_4` (7.936 us) after the vocab GEMV. Edge: logits -> argmax
   -> sampler feedback.
2. Dominant term: BODY. `r_32_4_1187` is a single-warp (block [32,1,1])
   reduction of 151,936 fp32 logits at 16.0 GB/s effective; `r_128_16_8_1187`
   is 2048-thread at 58.3 GB/s.
3. Code paths: vocab argmax/top-1 reduction schedule; two-stage or wider-block
   reduction; no change to the vocab GEMV itself.
4. Legality: generic reduction over 151,936 logits to the existing sampler
   contract; target-derived shape only.
5. Fallback: two-stage reduction; preserve exact sampler feedback order.
6. Contract: exact argmax/top-k output; token SHA identical.
7. Arms: isolated = B/C retained (`phase8/`); installed = reverse bracket.
8. Census gate: tail node count may shrink; sampler feedback node unchanged.
9. Reverse wall bracket, +50 us promotion bar.
10. Rollback: revert reduction schedule; non-regression on non-NV targets.
11. Projected ceiling 58.3 us, labelled unmeasured.
12. Prohibited: model-name or block-list dispatch.
