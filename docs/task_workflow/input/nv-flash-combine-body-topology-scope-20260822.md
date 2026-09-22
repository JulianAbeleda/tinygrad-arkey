# NV flash combine body-topology scope (2026-08-22)

Status: implementation scope; combine topology only. Coarse S=4 and S=2
splits are closed candidates.

1. Row and edge: `flash_fused_gmax_combine_f16_32_128` (36 x 2.880 us). Edge:
   flash score -> combine -> O input.
2. Dominant term: BODY. 2.304 us body versus llama 1.024 us (2.25x);
   single-warp, latency-bound (0.79% SM, 2.08% warps).
3. Code paths: flash combine kernel topology; preserve online max/sum
   stability; no coarse split.
4. Legality: generic attention combine; target-derived f16/32x128 shapes.
5. Fallback: body-parity target (1.024 us); if exactness breaks, stop.
6. Contract: exact combine output; token SHA identical.
7. Arms: isolated = NCU retained (`phase6/`); installed = reverse bracket.
8. Census gate: 36 nodes unchanged; no copy added.
9. Reverse wall bracket, +50 us promotion bar.
10. Rollback: revert combine; non-regression on non-NV targets.
11. Projected ceiling 46.1 us, labelled unmeasured.
12. Prohibited: model-name or block-list dispatch.
