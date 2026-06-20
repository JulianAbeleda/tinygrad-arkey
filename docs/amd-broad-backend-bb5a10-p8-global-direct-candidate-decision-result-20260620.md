# BB-5a.10 P8 Global-Direct Candidate Decision Result

Date: 2026-06-20

## Verdict

`PASS_BB5A10_P8_GLOBAL_DIRECT_CANDIDATE_DECISION`

The existing in-repo global-direct WMMA candidates were tested on the authority shape. They are correct, no-LDS, and scratch/private free, but none reaches the `60 TFLOPS` P8 gate.

## Best Existing Candidate

- Candidate: `global_direct_pipe_T4x2`
- Tile: `64x32x4096`
- Grid: `(384,8,1)`
- LDS bytes: `0`
- `ds_load_b128`: `0`
- `ds_store_*`: `0`
- WMMA: `32`
- Sampled correctness: pass
- Best: `17.881236827506367 TFLOPS`
- Median: `17.48027945227987 TFLOPS`

## Decision

Do not reopen q8 and do not continue tuning the LDS-staged macro. The current global-direct candidates are not sufficient either. Next work is to reconcile P8 timing authority against the prior `~43 TFLOPS` global-direct artifact, then decide whether a new global-direct scheduling/ILP candidate is justified.

P9/q8 remains blocked.
