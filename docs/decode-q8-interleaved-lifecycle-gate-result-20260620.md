# Decode q8 Interleaved Lifecycle Gate Result - 2026-06-20

Verdict: `BLOCKED_DECODE_Q8_INTERLEAVED_LIFECYCLE_STILL_SLOW`

Command:

```sh
PYTHONPATH=. python3 extra/qk_decode_owned_q8_interleaved_lifecycle_gate.py --rounds 24
```

## Result

| row | median us |
|---|---:|
| producer only | 25.20 |
| lifecycle producer | 25.02 |
| lifecycle consumer | 90.76 |
| lifecycle total | 115.80 |
| target lifecycle | 115.24 |

The gate misses by `0.56us` at median. The best observed lifecycle row was `115.28us`, only `0.04us` over the target.

## Gates

| gate | result |
|---|---|
| rows present | pass |
| producer correctness | pass |
| consumer correctness | pass |
| lifecycle total <= target | fail |

Correctness passed before and after the interleaved timing loop:

| check | max abs |
|---|---:|
| producer fp | 4.768e-7 |
| q8 dequant | 0.01165 |
| gate output | 9.537e-7 |
| up output | 1.431e-6 |

## Interpretation

This run refutes the earlier broad producer-context blocker under paired/interleaved ordering. Producer-only and
lifecycle-producer medians are effectively the same: `25.20us` vs `25.02us`.

The remaining lifecycle gap is no longer the earlier `~8-9us` miss. It is a `0.56us` median miss against a very tight
`115.24us` target, with first-run outliers but a stable steady band around `115.28-116.56us`.

The artifact recorded clock provenance:

| sample | sclk |
|---|---|
| before | level 1, 1362 MHz |
| after | level 1, 95 MHz |

Because the gate is interleaved, the producer/consumer split is still useful. Absolute pass/fail at this margin should
be reconciled with repeated paired sessions before declaring a new native schedule/codegen blocker.

## Next Step

Run a repeated paired-session target reconciliation:

1. Repeat this gate across multiple fresh sessions under the same one-clock discipline used for prefill.
2. Report median, min, p10, and post-warm steady-window medians separately.
3. Decide whether the target should remain `115.24us`, use a steady-window policy, or reopen schedule work only if the
   repeated steady median remains materially above target.

Until that reconciliation is done, the decode q8 route is blocked by threshold attribution, not by q8 addressing,
producer correctness, q4 residency, or a proven missing primitive.
