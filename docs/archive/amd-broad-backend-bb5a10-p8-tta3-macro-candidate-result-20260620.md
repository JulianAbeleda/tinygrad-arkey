# BB-5a.10 P8 TTA3 Macro Candidate Result

Date: 2026-06-20

## Verdict

`BLOCKED_BB5A10_P8_TTA3_SELECTED_COMPATIBLE_MACRO_CANDIDATE`

TTA3 found the expected blocker. The repo has a `128x128` LDS macro helper with the right authority launch shape, but it does not satisfy the selected-compatible LDS store contract.

## Candidate Found

- Shape: `M=512,N=12288,K=4096`
- Macro tile: `128x128x4096`
- Grid: `(96,4,1)`
- Local size: `(128,1,1)`
- LDS bytes: `8192`
- Scratch/private: `0`
- WMMA instructions: present
- `ds_load_b128`: present

## Blocker

- `ds_store_b64`: `0`
- `ds_store_b128`: `4`

The current macro helper uses `ds_store_b128` for cooperative LDS stores. BB-5a.10 selected authority evidence requires the first macro candidate to stay compatible with the proven `ds_store_b64 -> ds_load_b128 -> WMMA` path.

## Next

Implement TTA3a: convert the `128x128` macro candidate cooperative LDS stores from `ds_store_b128` to selected-compatible `ds_store_b64`, then rerun TTA3.
