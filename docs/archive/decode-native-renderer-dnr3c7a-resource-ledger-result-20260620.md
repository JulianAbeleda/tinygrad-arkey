# Decode Native Renderer DNR-3C7A Resource Ledger Result - 2026-06-20

## Verdict

`PASS_DNR3C7A_RESOURCE_LEDGER_BUILT_BLOCKED_ON_PMC_AND_ORACLE_RESOURCE_GAPS`

DNR-3C7A builds the native/C4/oracle resource ledger requested by the issue/resource attribution scope.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c7a_resource_ledger.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c7a_resource_ledger_result.json
```

## Native vs C4

| row | allocated VGPR/workitem | unique VGPR | max SGPR | private | LDS | launch waves/wg |
|---|---:|---:|---:|---:|---:|---:|
| native DNR-2 | `56` | `34` | `22` | `0` | `16` | `4` |
| DNR-3C4 | `96` | `48` | `22` | `0` | `16` | `4` |

Key delta:

| item | value |
|---|---:|
| allocated VGPR delta | `+40` |
| allocated VGPR ratio | `1.714x` |
| unique VGPR delta | `+14` |
| best static movement | `8.346us` |
| remaining gap to oracle | `69.637us` |

## Interpretation

What the ledger rules out:

- native and DNR-3C4 do not spill private/scratch;
- native and DNR-3C4 use the same tiny `16` byte LDS allocation;
- launch wave shape is unchanged by the static C4 rewrite.

What the ledger suggests:

- DNR-3C4 raises allocated VGPR/workitem from `56` to `96`;
- the extra `v[80:95]` preload band is a plausible resource-pressure reason that static count wins do not translate
  into oracle-like timing.

Why this is still not enough:

- the best static variant still improves native by only single-digit microseconds;
- oracle VGPR/SGPR/live-range data is missing from current artifacts;
- no counter evidence yet connects VGPR pressure, memory wait, or issue occupancy to the remaining gap.

## Oracle Gap

Oracle metadata is still partial:

| field | value |
|---|---:|
| local size | `[32, 4, 1]` |
| group segment | `16` |
| private segment | `0` |
| kernarg size | `40` |

Missing oracle resource data:

- VGPR count;
- SGPR count;
- live intervals;
- occupancy estimate from artifact metadata.

## Next

DNR-3C7B should run a same-harness PMC counter ladder for native vs DNR-3C4/best-static:

1. SQ counters: busy, wait, VALU, SALU if available;
2. GL2C counters: hit/miss/read behavior;
3. SQC/LDS counters: LDS active/conflict direction;
4. normalize by dispatch and keep correctness gates.

No renderer defaults changed.
