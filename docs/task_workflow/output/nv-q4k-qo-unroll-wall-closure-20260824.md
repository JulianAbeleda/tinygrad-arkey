# Q4_K Q/O grouped-block wall closure

## Decision

Do not promote four-block unrolling of the vector Q4_K attention-Q/output GEMV.
The construction was bit-exact and slightly faster in an isolated cold launch, but it
lost in the clean full-token reverse bracket. The production wall remains the authority.

## Admission funnel

| Stage | Observation | Decision |
|---|---|---|
| two-block microgate | bit-exact; slower both hot and cold | close |
| four-block microgate | bit-exact; hot essentially flat/slower; cold about 0.13 us/call faster | admit one token-path test |
| production lease | all 72 model Q/O sites found; shared-Q ownership may bypass leased Q sites | scope valid |
| reps-9 A/B/A wall | candidate 4.268175 ms/token; interpolated control 4.259529 ms/token | no-go, 8.646 us/token loss |

All three production arms produced the same token-stream hash. GPU memory and SM clocks
were the same in the authoritative bracket. The two controls moved in the favorable
direction from A to C, so midpoint interpolation did not hide a candidate win.

## Accounting lesson

The cold counter improvement was real but did not compose into a faster token. Grouping
reduced instructions, while increasing register use and leaving streamed weight bytes
essentially unchanged. The isolated gain therefore remained on the kernel side of the
ledger; scheduling, cache state, and the surrounding token path erased it. This is a
measured wall result, not an information wall, so the mechanism is closed rather than
promoted for more investment.

Evidence:

- `docs/task_workflow/evidence/nv-q4k-qo-wall-first-20260824/unroll-microgate.json`
- `docs/task_workflow/evidence/nv-q4k-qo-wall-first-20260824/unroll-wall-reps9.json`
