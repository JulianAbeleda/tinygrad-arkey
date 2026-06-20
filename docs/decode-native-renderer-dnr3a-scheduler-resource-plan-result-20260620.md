# Decode Native Renderer DNR-3A Scheduler/Resource Plan Result - 2026-06-20

## Verdict

`PASS_DNR3A_PLAN_STRUCTURAL_BLOCKED_ON_COMPOUND_EMITTER_AND_ATTRIBUTION`

DNR-3A turns the scheduler/resource gap into a first-class native plan. It does not emit a new kernel and does not claim
performance. It records the exact resource deltas and the policies a native emitter must satisfy before we can compare
against the hipcc/LLD oracle.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_native_renderer_dnr3a_scheduler_resource_plan.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_native_renderer_dnr3a_scheduler_resource_plan_result.json
```

## What Is Now Represented

| area | native | oracle | required policy |
|---|---:|---:|---|
| global loads | `22` | `11` | coalesced load lowering from Q4_K/q8 semantics |
| LDS/ds ops | `10` | `7` | reduction/resource policy |
| waitcnt | `17` | `20` | edge-driven wait policy |
| branch | `0` | `5` | lane-role branch/exec policy |
| `s_clause` | `0` | `3` | semantic marker insertion |
| `s_delay_alu` | `0` | `30` | latency/resource marker insertion |
| time | `166.649us` | `93.54us` | compound scheduler/resource candidate |

## What This Closes

The DNR-3 blocker is no longer vague. The repo now has:

- DNR-2 correct native lowering;
- DNR-3 scheduler/resource scope;
- DNR-3A scheduler/resource plan object and structural gate.

## What Is Still Blocked

DNR-3B is the next implementation wall:

1. lower the DNR-2 instruction stream through the DNR-3A plan;
2. apply coalesced load, wait, marker, branch, and register policies together;
3. launch the compound candidate;
4. pass gate/up correctness;
5. time against the q8 oracle;
6. attribute enough movement to continue or kill native decode scheduler work.

BEAM/search still stays blocked until a DNR-3B candidate exists and passes correctness.
