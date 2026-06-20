# Decode Native Renderer DNR-3C5 Timing Result - 2026-06-20

## Verdict

`BLOCKED_DNR3C5_C4_IMPROVES_BUT_REMAINS_BEHIND_ORACLE`

DNR-3C5 timed the correctness-preserving DNR-3C4 compound candidate against the DNR-2 native baseline in the same
process, on the real GGUF gate/up tensors.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3c5_timing_probe.py --warmups 3 --iters 8
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3c5_timing_result.json
```

## Timing

| candidate | median |
|---|---:|
| DNR-2 native | `171.607us` |
| DNR-3C4 compound | `169.253us` |
| hipcc/LLD oracle | `93.54us` |

DNR-3C4 improves by only `2.354us` and remains `75.713us` behind the oracle.

## Static Shape

| grouped count | DNR-2 native | DNR-3C4 | hipcc/LLD oracle |
|---|---:|---:|---:|
| dot4 | `16` | `16` | `16` |
| global load | `22` | `10` | `11` |
| ds | `10` | `7` | `7` |
| global store | `1` | `1` | `1` |
| shuffle | `5` | `5` | `5` |
| branch | `0` | `0` | `5` |
| waitcnt | `17` | `10` | `20` |
| `s_clause` | `0` | `3` | `3` |
| `s_delay_alu` | `0` | `30` | `30` |

## Interpretation

DNR-3C4 closes the obvious static movement budgets, but the timing barely moves. That means the remaining decode gap is
not explained by local static count matching alone.

The next step is attribution, not another branch-count patch:

1. determine whether oracle branch/wait differences are causal or incidental;
2. check whether marker placement is helping, neutral, or hurting;
3. inspect issue/resource behavior enough to explain why closing global-load and DS counts only moved about `2us`;
4. only then build a branch/exec policy or promotion path.

No renderer defaults changed.
