# Decode Native Renderer DNR-3C3 Compound Shape Result - 2026-06-20

## Verdict

`BLOCKED_DNR3C3_COMPOUND_SHAPE_NEEDS_SEMANTIC_BRANCH_REDUCTION_MODEL`

DNR-3C3 tested the remaining compound-shape pieces after DNR-3C2 closed the coalesced load/dataflow primitive.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c3_compound_shape_probe.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c3_compound_shape_result.json
```

## What Passed

The marker-count candidate launches and remains correct while preserving the DNR-3C2 coalesced-load win:

| grouped count | DNR-3C3 marker candidate | hipcc/LLD oracle |
|---|---:|---:|
| dot4 | `16` | `16` |
| global load | `10` | `11` |
| ds | `10` | `7` |
| branch | `0` | `5` |
| waitcnt | `10` | `20` |
| `s_clause` | `3` | `3` |
| `s_delay_alu` | `30` | `30` |

Correctness:

| output | max abs | mean abs |
|---|---:|---:|
| gate | `0.00048828125` | `0.00018835067749023438` |
| up | `0.000274658203125` | `0.0001456737518310547` |

This proves marker insertion is not blocked mechanically. It does not prove the marker placement is a performance policy.

## What Failed

A static `ds=7` candidate was tested by keeping the five wave-local `ds_bpermute` ops, the LDS store, and only one
cross-wave `ds_load`. It reaches the oracle static DS budget, but it is numerically wrong:

| grouped count | naive DS7 candidate | hipcc/LLD oracle |
|---|---:|---:|
| ds | `7` | `7` |
| branch | `0` | `5` |
| global load | `10` | `11` |

Correctness:

| output | max abs | mean abs |
|---|---:|---:|
| gate | `585.2509765625` | `541.2554321289062` |
| up | `218.06051635742188` | `147.89602661132812` |

So the remaining issue is not "delete three LDS loads." The four unconditional cross-wave LDS loads are carrying the
four wave partials. Matching the oracle DS count requires real branch/exec lane-role semantics.

## Next Blocker

DNR-3C4 must build the semantic branch/exec reduction model:

1. model which lanes/waves own cross-wave partial loading, final accumulation, and global store;
2. emit branch or exec-mask control flow that preserves exactly one full row sum per output;
3. replace the four unconditional cross-wave `ds_load`s with the oracle-shaped controlled reduction;
4. revalidate synthetic gate/up correctness with load shape and markers still enabled;
5. then time the compound candidate against the q8 oracle.

No renderer defaults changed and no performance claim is made here.
