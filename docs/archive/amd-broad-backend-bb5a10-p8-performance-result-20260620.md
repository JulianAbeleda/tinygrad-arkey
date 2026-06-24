# BB-5a.10 P8 Performance Result

Date: 2026-06-20

## Verdict

`BLOCKED_BB5A10_P8_PERFORMANCE_GATE_NOT_MET`

P8 ran the converted selected-compatible `128x128` macro candidate. It is correct on sampled authority tiles and scratch/private free, but it misses the `>=60 TFLOPS` gate.

## Measurement

- Command: `CNT=30 python3 extra/qk_amd_bb5a10_p8_performance.py`
- Shape: `M=512,N=12288,K=4096`
- Macro tile: `128x128x4096`
- Grid: `(96,4,1)`
- Local size: `(128,1,1)`
- Best: `18.383176771855297 TFLOPS`
- Median: `17.372618335970028 TFLOPS`
- Gate: `>=60 TFLOPS`
- LDS bytes: `8192`
- Scratch/private: `0`

## Correctness

- Sampled correctness: pass
- Max sampled relative RMSE: `0.00022195265046320856`

## Next

P9/q8 transfer remains blocked. Next work is P8 bottleneck classification for the converted DS64 macro candidate before changing more layout or scheduler code.
