# Decode q8 Controlled-Clock Policy Closeout Scope

Date: 2026-06-20

## Goal

Convert the clock-authority finding into a concrete q8 decode policy, without changing defaults.

Inputs:

- q8 artifact promotion already passed as hardened opt-in;
- auto clock authority fails the lifecycle target;
- `manual_peak` controlled-clock authority passes median-of-10 with `9/10` target-pass sessions.

## Command

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_controlled_clock_policy_closeout.py
```

## Gate

Pass only if:

- q8 artifact promotion remains `PASS_Q8_FFN_ARTIFACT_PROMOTION_TO_HARDENED_OPT_IN`;
- auto median lifecycle is above `115.24us`;
- `manual_peak` confirmation median lifecycle is at or below `115.24us`;
- no default behavior changes.

## Decision Shape

If the gate passes, the only accepted solution is:

```text
Q8_FFN_HANDWRITTEN=1 + manual_peak = controlled-clock research route
```

Auto/user-realistic authority remains blocked. Controlled-clock numbers must not be reported as auto-session speed.
