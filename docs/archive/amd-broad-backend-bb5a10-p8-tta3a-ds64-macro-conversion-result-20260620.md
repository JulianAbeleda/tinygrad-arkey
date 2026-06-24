# BB-5a.10 P8 TTA3a DS64 Macro Conversion Result

Date: 2026-06-20

## Verdict

`PASS_BB5A10_P8_TTA3A_DS64_MACRO_CONVERSION`

TTA3a converted the `128x128` macro helper cooperative LDS stores from `ds_store_b128` to selected-compatible `ds_store_b64`.

## Conversion

- Original instruction count: `706`
- Converted instruction count: `710`
- Replaced `ds_store_b128`: `4`
- New `ds_store_b64`: `8`
- Remaining `ds_store_b128`: `0`
- Branch repatched: yes
- Branch offset: `-103` dwords

## Converted Candidate

- Shape: `M=512,N=12288,K=4096`
- Macro tile: `128x128x4096`
- Grid: `(96,4,1)`
- Local size: `(128,1,1)`
- LDS bytes: `8192`
- Scratch/private: `0`
- `ds_load_b128`: `16`
- WMMA: `16`

## Follow-Up

After TTA3a, TTA3 reruns against the converted stream and passes as `PASS_BB5A10_P8_TTA3_MACRO_CANDIDATE`.
