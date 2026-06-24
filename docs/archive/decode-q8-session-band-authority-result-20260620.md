# Decode q8 Session-Band Authority Result

Date: 2026-06-20

## Verdict

`BLOCKED_DECODE_Q8_SESSION_BAND_NO_AUTHORITY_POLICY`

Initial matrix command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_session_band_authority.py
```

Confirmation command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_session_band_authority.py \
  --sessions 5 \
  --protocols consumer_warm,lifecycle_warm \
  --out bench/qk-decode-primitive-transfer/decode_q8_session_band_authority_confirm_result.json
```

Outputs:

```text
bench/qk-decode-primitive-transfer/decode_q8_session_band_authority_result.json
bench/qk-decode-primitive-transfer/decode_q8_session_band_authority_confirm_result.json
```

## Initial Matrix

The bounded 3-session search found apparent passing warm policies:

| protocol | median lifecycle | fast sessions | slow sessions | consumer | producer |
|---|---:|---:|---:|---:|---:|
| cold | `121.06us` | `0` | `2` | `100.74us` | `20.12us` |
| producer warm | `121.86us` | `1` | `2` | `101.60us` | `20.20us` |
| consumer warm | `104.06us` | `3` | `0` | `87.68us` | `16.46us` |
| lifecycle warm | `108.08us` | `3` | `0` | `90.76us` | `17.34us` |
| producer then consumer warm | `558.18us` | `1` | `2` | `207.14us` | `155.94us` |

The initial result was therefore `PASS_DECODE_Q8_SESSION_BAND_WARM_POLICY_FOUND`, with `consumer_warm` as best protocol.
Because this was only 3 sessions and one protocol had pathological high-clock slowdowns, it required confirmation.

## Confirmation

The 5-session confirmation on the two apparent pass protocols failed:

| protocol | median lifecycle | best session | worst session | fast sessions | slow sessions | consumer | producer |
|---|---:|---:|---:|---:|---:|---:|---:|
| consumer warm | `122.10us` | `108.24us` | `122.26us` | `2` | `3` | `101.60us` | `20.48us` |
| lifecycle warm | `122.04us` | `108.26us` | `122.14us` | `1` | `4` | `101.50us` | `20.52us` |

Both protocols still produce occasional fast sessions, but neither survives median-of-5 fresh-session authority. The
confirmation verdict is `BLOCKED_DECODE_Q8_SESSION_BAND_NO_AUTHORITY_POLICY`.

## Interpretation

The fast q8 lifecycle band is real:

- best confirmed `consumer_warm`: `108.24us`;
- best confirmed `lifecycle_warm`: `108.26us`;
- both clear the `115.24us` target when the session lands fast.

But none of the bounded warm-state policies forces that band. Most confirmation sessions fall back to the known slow
shape:

```text
producer ~20.5us + consumer ~101.5us = lifecycle ~122us
```

The coarse `rocm-smi --showgpuclocks` boundary samples do not provide a usable authority. Fast and slow sessions both
show low sampled SCLK values at block boundaries; the samples are too sparse/coarse to explain the band.

## Decision

q8 decode remains research-only. Promotion is blocked by session-band authority, not by:

- q8 producer correctness;
- q8 producer `NT=1024` thread shape;
- fused consumer static resource shape;
- producer->consumer adjacency;
- a simple warmup policy.

Do not rewrite the consumer kernel body based on the `~101us` slow sessions. The current missing piece is a stronger
session/perf-state authority mechanism.

## Next

The next useful work is not another warmup permutation. It is one of:

1. a stronger telemetry path that samples relevant GPU state during the timing block, not only before/after;
2. an explicit power/perf-state control experiment if acceptable on this host;
3. a q8 promotion policy that intentionally uses best-stable-session or steady-fast-session authority instead of
   median-of-fresh-sessions.

Until one of those is accepted, the honest result is `NO_AUTHORITY_POLICY`.

## Boundary

No decode default changed.
