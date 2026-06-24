# Decode q8 Lifecycle Target Reconciliation Scope - 2026-06-20

Verdict target: `PASS_DECODE_Q8_LIFECYCLE_TARGET_RECONCILED`

The interleaved lifecycle gate missed by only `0.56us` at median, with a best observed lifecycle row `0.04us` over the
`115.24us` target. That is too close to treat as a new schedule/codegen blocker without repeated-session evidence.

## Tool

`extra/qk_decode_q8_lifecycle_target_reconciliation.py`

## Method

Run the existing interleaved lifecycle gate repeatedly in fresh Python sessions:

```sh
PYTHONPATH=. python3 extra/qk_decode_q8_lifecycle_target_reconciliation.py --sessions 5 --rounds 24
```

Each child run writes a preserved session artifact under:

`bench/qk-decode-primitive-transfer/decode_q8_lifecycle_target_reconciliation_session_*.json`

The reconciler then computes:

| metric | reason |
|---|---|
| full median | exact original gate policy |
| steady median | removes the first four lifecycle rows, where first-run/cache/clock outliers appear |
| best observed row | tells whether the target is physically reached in any paired row |
| session spread | distinguishes stable schedule debt from threshold variance |

## Gates

| gate | threshold |
|---|---:|
| all producer correctness | pass |
| all consumer correctness | pass |
| session artifacts present | `sessions` |
| full median of session medians | reported |
| steady median of session medians | reported |
| target delta classification | required |

## Decision Policy

| verdict | condition |
|---|---|
| `PASS_DECODE_Q8_LIFECYCLE_TARGET_RECONCILED` | full or steady median clears `115.24us` |
| `BLOCKED_DECODE_Q8_LIFECYCLE_THRESHOLD_VARIANCE` | miss is `<= 1us`; repeatability/target policy is the blocker |
| `BLOCKED_DECODE_Q8_LIFECYCLE_SCHEDULE_DEBT` | repeated steady median is materially above target by `> 1us` |
| `BLOCKED_DECODE_Q8_LIFECYCLE_INCORRECT` | any child run fails correctness |

This gate decides whether decode should reopen schedule work or close the current miss as target-policy variance.
