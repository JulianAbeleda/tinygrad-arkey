# Decode Route-Level Primitive Ledger Result - 2026-06-20

Verdict: `PASS_DECODE_ROUTE_LEVEL_PRIMITIVE_LEDGER_READY`

This is the local-progress half of the dual track. It reconciles the route-level decode options after DNR4-T3:
native local schedule rewrites are parked, but route-level q8 work can still progress.

Run:

```bash
PYTHONPATH=. python3 extra/qk_decode_route_level_primitive_ledger.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_route_level_primitive_ledger_result.json
```

## Route Ledger

| route | decision | why |
|---|---|---|
| current default decode | keep | banked W==D authority baseline |
| q8 FFN handwritten artifact | keep hardened opt-in | quality passes, default-off policy passes, W==D min speedup is about `1.05x` |
| imported llama Q4 graph route | reject as speed route | `attn_output` speedup `0.763x`, gate/up speedup `0.744x` |
| native local MMVQ schedule edits | park | T3 is correct but not material; counters did not form a search objective |
| owned q8 lifecycle successor | scope next | only route with plausible local progress that is not blocked by ATT |

## Current Promotion Boundary

Promotable now:

- `Q8_FFN_HANDWRITTEN=1` as a hardened opt-in candidate only.

Not promotable now:

- q8 artifact default-on;
- imported llama Q4 graph route as a speed path;
- native local MMVQ schedule edits;
- BEAM/search over decode schedules.

## Owned Successor Contract

The next route-level primitive should not be another local schedule patch. It should be a metadata/contract object for
an owned q8 lifecycle successor:

- q8 producer/cache lifecycle;
- gate/up consumer reuse policy;
- supported model, shape, and device coverage;
- dNLL quality gate and fallback behavior;
- same-harness W==D timing gate;
- artifact parity target before native lowering/search.

## Decision

Do both tracks:

- continue route-level decode locally by scoping the owned q8 lifecycle successor;
- continue ATT separately as a tooling unblock for native PC/stage attribution.

Do not block route-level progress on ATT. Do not resume native local count-matching until ATT or a route-level
primitive gate gives a real objective.
