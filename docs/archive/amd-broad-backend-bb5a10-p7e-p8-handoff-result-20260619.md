# BB-5a.10 P7e P8 Handoff Result

Date: 2026-06-19

## Verdict

`PASS_BB5A10_P7E_P8_HANDOFF_PACKAGE`

P7e is complete. The P8 handoff now has a reproducible correctness artifact, source hash, resource summary, and exact P8 entry command.

## Handoff Candidate

- Candidate: `bb5a10_p7d_authority_k4096_single_tile`
- Source: `extra/qk_amd_bb5a10_p7d_authority_correctness.py`
- Correctness result: `PASS_BB5A10_P7D_AUTHORITY_SUBSET_CORRECTNESS`
- Shape proved: `16x16x4096`
- Authority contract represented: `M=512,N=12288,K=4096`
- Relative RMSE: `0.00019154782057739794`
- LDS path: `ds_store_b64 -> ds_load_b128 -> WMMA`
- Runtime LDS allocation: `2048` bytes
- Scratch/private: `0`

## P8 Command

```bash
CNT=30 K=4096 python3 extra/qk_amd_bb5a10_p8_performance.py
```

## Boundary

P7e does not claim performance. It explicitly carries forward that full authority launch mapping is not yet proved. P8 must either map the proven P7d K-loop into `M=512,N=12288,K=4096` or block before timing.
