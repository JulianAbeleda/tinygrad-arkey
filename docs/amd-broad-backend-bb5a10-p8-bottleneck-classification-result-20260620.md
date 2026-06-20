# BB-5a.10 P8 Bottleneck Classification Result

Date: 2026-06-20

## Verdict

`PASS_BB5A10_P8_BOTTLENECK_CLASSIFIED_LDS_STAGING_FAMILY`

The P8 blocker is classified. The converted `ds_store_b64` macro candidate is not slow because of correctness, launch mapping, scratch/private spill, or the DS64 conversion itself. It is slow because it is in the wrong performance family: multi-wave LDS staging with a global→LDS→WMMA round trip and barriers.

## Measurements

Command:

```bash
CNT=10 python3 extra/qk_amd_bb5a10_p8_bottleneck_classification.py
```

| candidate | best TFLOPS | median TFLOPS | LDS stores |
|---|---:|---:|---|
| original macro | `21.47` | `18.96` | `4 x ds_store_b128` |
| converted macro | `20.93` | `18.77` | `8 x ds_store_b64` |

The converted DS64 candidate is `97.5%` of the original B128 candidate, so the DS64 conversion is not the primary bottleneck.

## Classification

Primary bottleneck: `LDS_STAGING_FAMILY_BOTTLENECK`

Not primary:

- correctness: sampled P8 correctness passes
- scratch/private spill: both are `0`
- launch mapping: TTA1/TTA2/TTA3 prove the full authority grid and macro shape
- DS64 conversion: B128 and DS64 variants are both in the same slow LDS-staged family

## Decision

Stop optimizing this LDS-staged macro as the P8 path. Next work should reopen a selected-compatible global-direct / Infinity-Cache-served WMMA candidate, or explicitly classify why the selected Tensile layout cannot transfer without the LDS round trip.
