# Decode q8 Lifecycle Band Attribution Result

Date: 2026-06-20

## Verdict

`BLOCKED_DECODE_Q8_LIFECYCLE_SESSION_STATE`

## Current Status

This artifact is still valid as a raw isolated-lifecycle measurement, but it is superseded as the top-level decode
decision authority by the later clock-authority and in-model route timing audits.

| question | lifecycle artifact answer | newer artifact answer | current decision |
|---|---|---|---|
| Is the q8 lifecycle slow because producer->consumer adjacency is bad? | No. Prebuilt, immediate-after-producer, and after-dummy consumers followed the same session band. | Still no. Clock control changes the band without changing producer/consumer adjacency. | Do not start fused producer+consumer work from this evidence. |
| Is the isolated q8 lifecycle intrinsically limited to `~122us`? | No final authority. `auto` median was `121.74us`, with one fast `108.22us` session. | No. `manual_peak` confirmation reached `58.04us` median, `9/10` pass, with producer `~9.44us` and consumer `~48.61us`. | Treat `auto` as user-realistic blocked, `manual_peak` as controlled-fast research authority. |
| Does lifecycle explain the whole decode gap to llama? | Not by itself; this artifact only priced the isolated FFN q8 lifecycle. | No. In-model q8 route is only `~1.06x` faster, lands around `70-72 tok/s`, and has `0.0%` host-sync residual. | Llama gap is broader than q8 lifecycle. Move to role/tensor/kernel attribution. |
| Should q8 become default-on? | Blocked under uncontrolled fresh-session policy. | Still not default-on. The controlled-clock route is research-only; whole-model gain is stable but small. | Keep q8 opt-in/default-off. |
| What is stale? | The raw rows are not stale. | The old conclusion that session-band policy is the next decode blocker is stale for whole-model work. | Use this artifact only as local q8 context, not as the next-build authority. |

Superseding references:

- `docs/decode-q8-clock-authority-result-20260620.md`
- `docs/decode-q8-model-route-timing-audit-result-20260620.md`

Command:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_lifecycle_band_attribution.py --sessions 5
```

Output:

```text
bench/qk-decode-primitive-transfer/decode_q8_lifecycle_band_attribution_result.json
```

## Result

Correctness passed in all child sessions. The `NT=1024` producer remains recovered, but lifecycle still misses the
`115.24us` target under median-of-session-medians policy.

| row | median of session medians | best session | worst session |
|---|---:|---:|---:|
| prebuilt-q8 consumer | `101.20us` | `90.84us` | `101.54us` |
| lifecycle steady producer | `20.18us` | `17.44us` | `20.44us` |
| lifecycle steady consumer | `101.44us` | `90.78us` | `101.52us` |
| lifecycle steady total | `121.74us` | `108.22us` | `121.96us` |
| after-dummy consumer | `101.42us` | `90.68us` | `101.54us` |
| after-dummy total | `121.80us` | `107.98us` | `121.90us` |

Session pattern:

| session | prebuilt consumer | steady consumer | steady total | after-dummy consumer | verdict |
|---:|---:|---:|---:|---:|---|
| 0 | `90.84us` | `90.78us` | `108.22us` | `90.68us` | fast/pass |
| 1 | `97.02us` | `101.44us` | `121.94us` | `101.54us` | mixed/slow |
| 2 | `101.20us` | `101.44us` | `121.96us` | `101.40us` | slow |
| 3 | `101.54us` | `101.52us` | `121.74us` | `101.42us` | slow |
| 4 | `101.50us` | `101.40us` | `121.68us` | `101.44us` | slow |

## Interpretation

This pass does **not** support a producer->consumer adjacency-specific blocker. In slow sessions, all consumer forms are
slow:

- prebuilt-q8 consumer;
- consumer immediately after producer;
- consumer after an extra dummy consumer dispatch.

In the one fast session, all of those are fast and the lifecycle clears target (`108.22us`). That points to whole-session
state or timing policy, not the fused consumer body and not producer->consumer ordering.

The prior `decode_q8_consumer_band_attribution` result showed steady consumer rows can reconcile fast under a different
session/protocol mix. This lifecycle pass shows the same consumer can also remain in the slow band for an entire fresh
session. The unresolved issue is therefore the session-band authority: which timing state is admissible for promotion,
and how to force or classify it reproducibly.

The sampled `rocm-smi --showgpuclocks` values do not explain the split by themselves. The fast session and slow sessions
all report low SCLK samples around the measured region, so a stronger clock/perf-state probe is needed before changing
policy.

## Decision

Do **not** rewrite the q8 consumer kernel body yet.

Next bounded step: build a session-band authority probe that tries to force or classify the fast/slow state before q8
promotion is discussed. It should run the same lifecycle attribution under explicit GPU state controls or stronger
telemetry:

1. warm-state ladder: cold, compile-only, producer-only warmup, consumer-only warmup, lifecycle warmup;
2. clock/perf-state authority: sample `rocm-smi` repeatedly during timing or use a stronger ROCm/PMU source if available;
3. policy matrix: median-of-fresh-sessions, best-stable-session, and steady-after-warmup must be reported separately.

Until that exists, q8 decode remains opt-in/research-only and blocked by session-band policy, not kernel correctness.

## Boundary

No decode default changed.
