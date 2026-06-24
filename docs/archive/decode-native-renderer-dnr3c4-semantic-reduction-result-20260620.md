# Decode Native Renderer DNR-3C4 Semantic Reduction Result - 2026-06-20

## Verdict

`PASS_DNR3C4_SEMANTIC_REDUCTION_CORRECT_BLOCKED_ON_BRANCH_WAIT_TIMING`

DNR-3C4 resolves the DNR-3C3 LDS blocker. The fix is not deleting cross-wave partial reads; it is replacing four scalar
LDS reads with one vector LDS read of the same four wave partials.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c4_semantic_reduction_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c4_semantic_reduction_result.json
```

## Result

| gate | result |
|---|---:|
| candidate launches | pass |
| candidate correct | pass |
| global-load budget closed | pass |
| DS budget closed | pass |
| marker counts match oracle | pass |
| dot4 preserved | pass |
| branch policy matches oracle | fail |
| waitcnt count matches oracle | fail |

Correctness:

| output | max abs | mean abs |
|---|---:|---:|
| gate | `0.00048828125` | `0.00018835067749023438` |
| up | `0.000274658203125` | `0.0001456737518310547` |

Grouped counts:

| grouped count | DNR-3C4 candidate | hipcc/LLD oracle |
|---|---:|---:|
| dot4 | `16` | `16` |
| global load | `10` | `11` |
| ds | `7` | `7` |
| global store | `1` | `1` |
| shuffle | `5` | `5` |
| branch | `0` | `5` |
| waitcnt | `10` | `20` |
| `s_clause` | `3` | `3` |
| `s_delay_alu` | `30` | `30` |

## Reduction Model

Before:

```text
5 ds_bpermute + 1 ds_store_b32 + 4 ds_load_b32 = 10 ds ops
```

After:

```text
5 ds_bpermute + 1 ds_store_b32 + 1 ds_load_b128 = 7 ds ops
```

The four wave partials remain present in LDS. DNR-3C4 vectorizes the cross-wave read instead of removing dataflow.

## Next

DNR-3C5 should time this candidate before adding branch/control flow. At this point branch and waitcnt counts differ
from the oracle, but the core movement budgets are closed and correctness is preserved.

No renderer defaults changed and no performance claim is made here.
