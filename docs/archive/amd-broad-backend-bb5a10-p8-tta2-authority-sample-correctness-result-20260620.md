# BB-5a.10 P8 TTA2 Authority Sample Correctness Result

Date: 2026-06-20

## Verdict

`PASS_BB5A10_P8_TTA2_AUTHORITY_SAMPLE_CORRECTNESS`

TTA2 passes. The authority-shape launch is not a narrow-grid shortcut: it runs the full `M=512,N=12288,K=4096` grid and verifies sampled first, middle, and last row/column tiles.

## What Passed

- Authority shape: `M=512,N=12288,K=4096`
- Full grid: `(768,32,1)`
- Sample coverage:
  - first row tile
  - middle row tile
  - last row tile
  - first column tile
  - middle column tile
  - last column tile
- Max sampled relative RMSE: `0.00022756120597478002`
- Gate: `<= 0.001`

## Boundary

TTA2 proves authority launch correctness for the `16x16` bridge. It is still not the P8 performance candidate.

Next is TTA3: build the selected-compatible `128x128` macro-tile candidate and prove resource metadata with scratch/private `0`.
