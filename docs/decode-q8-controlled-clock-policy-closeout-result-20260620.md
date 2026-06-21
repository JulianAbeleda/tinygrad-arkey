# Decode q8 Controlled-Clock Policy Closeout Result

Date: 2026-06-20

## Verdict

`PASS_DECODE_Q8_CONTROLLED_CLOCK_RESEARCH_ROUTE_POLICY`

Command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_controlled_clock_policy_closeout.py
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_q8_controlled_clock_policy_closeout_result.json
```

## Decision

The q8 route has a solution path, but it is **not** default-on and not auto-authority:

```text
Q8_FFN_HANDWRITTEN=1 + manual_peak = controlled-clock research route
```

The route remains default-off. Auto/user-realistic authority remains blocked.

## Evidence

| authority | median lifecycle | target pass sessions | decision |
|---|---:|---:|---|
| auto | `121.92us` | `2/5` | blocked user-realistic authority |
| manual_peak | `58.04us` | `9/10` | controlled-fast research authority |

The existing q8 artifact promotion also remains passed:

```text
PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN
```

## Policy

Accepted:

- default-off hardened opt-in;
- controlled-clock research route;
- supported route flag: `Q8_FFN_HANDWRITTEN=1`;
- clock authority: `manual_peak` only;
- rollback: unset `Q8_FFN_HANDWRITTEN` and restore GPU perf level to `auto`.

Rejected:

- default-on promotion from controlled-clock evidence;
- reporting `manual_peak` speed as auto-session speed;
- spending next work on primitive/kernel rewrites for this q8 route while the controlled-clock path is unresolved at
  policy level.

## What This Solves

This closes the immediate "what happened?" loop:

```text
auto session band: uncontrolled and often slow
manual_peak band: controlled fast enough for the q8 route
```

The primitive is not the next blocker. The blocker is policy: whether controlled-clock authority is acceptable for the
intended use.

## Next

If controlled-clock runs become routine, add a small command wrapper that:

1. sets `manual_peak`;
2. runs the q8 benchmark/route with `Q8_FFN_HANDWRITTEN=1`;
3. restores `auto` in a `finally` path;
4. labels the result as controlled-clock, not auto.

If auto is required, q8 stays blocked.

## Boundary

No decode default changed.
