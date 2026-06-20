# BB-5a.10 P8 TTA1 Full-Grid Correctness Result

Date: 2026-06-20

## Verdict

`PASS_BB5A10_P8_TTA1_FULL_GRID_CORRECTNESS`

TTA1 passes. The proven P7d `16x16x4096` K-loop now maps across the full authority grid with `gidx0/gidx1`.

## What Passed

- Authority shape: `M=512,N=12288,K=4096`
- Tile: `16x16x4096`
- Grid: `(768,32,1)`
- Local size: `(32,1,1)`
- LDS path: `ds_store_b64 -> ds_load_b128 -> WMMA`
- K-loop: full `4096 / 16 = 256` WMMA K steps
- Sampled tiles: first, last, and middle row/column tiles
- Max sampled relative RMSE: `0.00022756120597478002`
- Gate: `<= 0.001`

## Boundary

This is still a correctness bridge, not a P8 performance candidate. The `16x16` full-grid launch proves address mapping, output placement, and K-loop coverage. P8 timing remains blocked until TTA2 full-launch sampled correctness and TTA3 `128x128` macro-candidate resource gates pass.

## Next

Implement TTA2: full authority launch sampled correctness with no narrow-grid shortcut.
