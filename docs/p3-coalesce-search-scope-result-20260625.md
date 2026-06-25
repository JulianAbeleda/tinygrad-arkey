# P3 coalesce search scope/result

Date: 2026-06-25
Verdict: `P3_COALESCE_STATIC_SEARCH_SLICE_PASS`

## Scope executed

The M-E gate passed with `lane_partition` at ~99.7-100.0% of owned and `tokens_match=True`, so P3 is allowed to start.
This commit implements the first bounded P3 slice:

- Add `OptOps.COALESCE` as a default-off beam-visible marker.
- Add a static q4k coalescing scorer that ranks candidate thread maps before timing.
- Prove the scorer selects the hand-found `lane_partition_q4k` candidate without timing.
- Prove the static choice agrees with the latest measured M-E artifact.

## Explicit non-scope

`Ops.LAYOUT_TRANSFORM` is not landed in this slice. The M-E win did not require a storage-permutation op, and adding one
would broaden shape/simplifier behavior before the static predicate gate is proven. It remains the next gated P3 item.

Generic `add_gpudims` REDUCE substitution is also not changed. The safe semantic bridge is still `LanePartitionReduce`.

## Next P3 step

Implement a real `LAYOUT_TRANSFORM` movement op only if it can pass these gates:

- survives shape/simplifier/rangeify unchanged unless explicitly consumed,
- preserves byte-exact correctness on the q4k lane-partition probe,
- lets the beam path reproduce the coalesced candidate without a q4k-specific environment flag.
