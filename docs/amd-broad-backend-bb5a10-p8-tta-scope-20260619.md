# BB-5a.10 P8 TTA Scope

Date: 2026-06-19

## Verdict

`PASS_BB5A10_P8_TTA_SCOPE_READY`

TTA means tile-to-authority launch mapping: the work needed to take the proven P7d `16x16x4096` tile and map it into the full authority `M=512,N=12288,K=4096` launch.

## Key Boundary

TTA has two different jobs and they must stay separate:

- correctness bridge: `16x16` tile over the full authority grid, useful to prove `gidx0/gidx1`, base addresses, K-loop coverage, and output mapping
- performance candidate: selected-compatible `128x128` macro tile, required before the `>=60 TFLOPS` P8 gate is meaningful

The `16x16` bridge is not a valid performance candidate.

## Phases

- `TTA0` freeze contract: `M=512,N=12288,K=4096`, row-major A, row-major Bt, fp16 C
- `TTA1` single-wave full-grid correctness bridge: grid `(768,32,1)`, one `16x16` tile per workgroup
- `TTA2` authority-shape sampled correctness: full launch with deterministic sampled output checks
- `TTA3` selected macro-tile performance candidate: `128x128`, grid `(96,4,1)`, scratch/private `0`
- `TTA4` P8 timing gate: same candidate that passed correctness reaches `>=60 TFLOPS`

## Missing Items

- `gidx0/gidx1` tile offsets
- C output base by global row/column tile
- A/Bt global base formulas
- sampled full-authority correctness harness
- selected-compatible `128x128` macro-tile candidate
- resource and spill proof for the timed candidate

## Next

Implement `TTA1`: extend the P7d K-loop with `gidx0/gidx1` full-grid mapping and prove deterministic correctness before any timing.
