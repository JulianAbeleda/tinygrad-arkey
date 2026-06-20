# BB-5a.10 P8 TTA Completion Scope

Date: 2026-06-20

## Verdict

`PASS_BB5A10_P8_TTA_COMPLETION_SCOPE_READY`

TTA is now scoped through completion. The remaining work is no longer an open-ended launch-mapping blob; it is a fixed sequence from the P7d tile to P8 timing and then P9 q8 reopen.

## Address Contract

- Authority shape: `M=512,N=12288,K=4096`
- TTA1 correctness bridge: `16x16x4096`, grid `(768,32,1)`, local `(32,1,1)`
- TTA3 performance target: `128x128x4096`, grid `(96,4,1)`, local `(128,1,1)`
- K-loop: `256` WMMA K steps, `32` bytes per step

Global byte formulas:

- A: `(row_base + lane_or_row_fragment) * K * 2 + k_iter * 32`
- Bt: `(col_base + lane_or_col_fragment) * K * 2 + k_iter * 32`
- C: `((row_base + output_row) * N + (col_base + output_col)) * 2`

## Completion Sequence

- `TTA1`: full-grid correctness bridge. Extend P7d to `gidx0/gidx1`, prove sampled deterministic correctness, no timing.
- `TTA2`: full authority launch sampled correctness. Verify first/middle/last row and column tiles, no timing.
- `TTA3`: selected-compatible `128x128` macro-tile candidate. Prove resource metadata and reject scratch/private spill.
- `TTA4`: P8 timing gate. Time the exact candidate that passed TTA2/TTA3; pass requires `>=60 TFLOPS`.
- `P9`: q8 reopen decision. Starts only after P8 passes.

## Non-Negotiable Checks

- The TTA1 `16x16` bridge cannot satisfy P8 performance.
- TTA2 full-launch correctness must precede timing.
- TTA3 must reject scratch/private spill before P8.
- P8 must time the exact candidate that passed TTA2/TTA3.
- P9 remains blocked until P8 passes.

## Next

Implement `TTA1`: `extra/qk_amd_bb5a10_p8_tta1_full_grid_correctness.py`.
