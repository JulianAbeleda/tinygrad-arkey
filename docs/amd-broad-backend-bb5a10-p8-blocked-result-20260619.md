# BB-5a.10 P8 Blocked Result

Date: 2026-06-19

## Verdict

`BLOCKED_BB5A10_P8_FULL_AUTHORITY_LAUNCH_MAPPING_REQUIRED`

P8 correctly refuses to time the P7d single-tile smoke as the authority performance gate.

## Why

P7d/P7e prove a correct executable candidate over full `K=4096`, but only for one `16x16` output tile. The P8 gate is broader: pure tinygrad authority prefill must reach `>=60 TFLOPS` on the `M=512,N=12288,K=4096` launch without scratch/private spill.

## Next

Implement full-authority launch mapping for the proven P7d K-loop:

- map grid/workgroups over `M=512,N=12288`
- preserve selected-compatible LDS staging
- keep output mapping correct across all tiles
- preserve scratch/private `0`
- then run:

```bash
CNT=30 K=4096 python3 extra/qk_amd_bb5a10_p8_performance.py
```
