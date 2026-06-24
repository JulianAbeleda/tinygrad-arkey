# Decode Native Renderer DNR-3C6 Attribution Scope Result - 2026-06-20

## Verdict

`BLOCKED_DNR3C6_STATIC_LADDER_REFUTES_LOCAL_COUNT_ATTRIBUTION`

DNR-3C6 scoped and executed the first attribution gate after DNR-3C5. The goal was to decide whether local static feature
matching still plausibly explains the oracle gap.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c6_attribution_scope.py --warmups 2 --iters 6
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c6_attribution_scope_result.json
```

## Timing Ladder

| variant | median |
|---|---:|
| native DNR-2 | `171.523us` |
| b128 loads only | `164.670us` |
| b128 loads + markers | `166.624us` |
| b128 loads + `ds_load_b128`, no markers | `163.177us` |
| b128 loads + `ds_load_b128` + markers | `164.579us` |
| hipcc/LLD oracle | `93.54us` |

Best local static-feature movement is `8.346us`, from `171.523us` to `163.177us`. That is below the `30us` attribution
threshold and leaves roughly `69.637us` to oracle.

## Scope Decision

| phase | question | status |
|---|---|---|
| DNR-3C6A same-harness static-feature ladder | do load shape, vector LDS reduction, and markers explain a material fraction of the oracle gap? | executed |
| DNR-3C6B marker placement attribution | are marker counts helpful, neutral, or harmful? | deprioritized by ladder |
| DNR-3C6C issue/resource attribution | is the remaining gap from issue interleaving, VGPR/resource occupancy, or branch/wait control? | next |
| DNR-3C6D branch/wait experiment | only if attribution points there, build semantic branch/wait control | blocked on C6C |

## Interpretation

The local static-count path is now refuted as the main explanation. DNR-3C2 through DNR-3C4 made the native stream look
much more oracle-shaped, but timing barely moved. Adding branch or wait instructions just to match counts is not
justified.

The next valid step is issue/resource attribution or a pause on native decode renderer work:

1. inspect issue/resource behavior, live ranges, or occupancy enough to identify a credible `>=30us` lever;
2. construct a schedule with changed interleaving/resource behavior, not just changed counts;
3. if that attribution is not available, keep the q8 artifact oracle path as the practical decode route.

No renderer defaults changed.
