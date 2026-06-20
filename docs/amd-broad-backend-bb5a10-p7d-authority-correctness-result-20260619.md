# BB-5a.10 P7d Authority-Subset Correctness Result

Date: 2026-06-19

## Verdict

`PASS_BB5A10_P7D_AUTHORITY_SUBSET_CORRECTNESS`

P7d is no longer blocked. The selected-compatible LDS path passed a full authority-K correctness smoke without making a performance claim.

## What Passed

- Authority contract target: `M=512, N=12288, K=4096`
- Tested subset: one `16x16` output tile with full `K=4096`
- LDS path: `ds_store_b64 -> ds_load_b128 -> WMMA`
- K-loop: present, branch-controlled, accumulates over `4096 / 16 = 256` WMMA K steps
- Output: fp16 store with numpy fp32 reference comparison
- Relative RMSE: `0.00019154782057739794`
- Gate tolerance: `<= 0.001`

## Artifact

- Script: `extra/qk_amd_bb5a10_p7d_authority_correctness.py`
- Result: `bench/amd-broad-backend-roadmap/bb5a10_p7d_authority_correctness_result.json`

## Remaining Boundary

This proves full authority-K accumulation and selected-compatible LDS staging for a single output tile. It does not yet prove full authority launch mapping across `M=512, N=12288`, nor does it time the candidate.

Next valid phase is P7e: package the correct executable candidate with source/ISA/resource metadata and the exact P8 timing command. P8 remains blocked until P7e exists.
