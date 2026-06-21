# Decode q8 Clock Authority Result

Date: 2026-06-20

## Verdict

`PASS_DECODE_Q8_CLOCK_AUTHORITY_CONTROLLED_FAST`

The tooling exists and is sufficient to audit the session band. The result argues **against** a primitive rethink as the
next move: controlled DPM state can force the q8 lifecycle into a fast median band.

## Commands

Inventory/control smoke:

```bash
PYTHONPATH=. python3 extra/qk_prefill_clock_dpm_authority.py inventory
```

Full lane matrix:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_clock_authority.py \
  --lanes auto,high,profile_peak,manual_peak \
  --sessions 5 \
  --rounds 256 \
  --warmups 32 \
  --out bench/qk-decode-primitive-transfer/decode_q8_clock_authority_result.json
```

Focused confirmation:

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_clock_authority.py \
  --lanes manual_peak \
  --sessions 10 \
  --rounds 256 \
  --warmups 32 \
  --out bench/qk-decode-primitive-transfer/decode_q8_clock_authority_manual_peak_confirm_result.json
```

## Tooling Audit

Available on this host:

- `rocm-smi`: present;
- `amd-smi`: absent;
- sysfs DPM files: present;
- `gpu_busy_percent` / `mem_busy_percent`: present;
- hwmon power/temp: present;
- passwordless sudo for DPM controls: present.

The audit script sets one lane per child session, samples sysfs telemetry during the timing window, restores `auto`, and
records per-dispatch medians for the q8 lifecycle.

## Lane Matrix

| lane | sessions | median lifecycle | pass sessions | median consumer | median producer | notes |
|---|---:|---:|---:|---:|---:|---|
| auto | `5` | `121.92us` | `2/5` | `101.52us` | `20.28us` | reproduces slow median |
| high | `5` | `56.76us` | `4/5` | `47.64us` | `9.16us` | fast median, one huge outlier |
| profile_peak | `5` | `331.10us` | `2/5` | `235.16us` | `10.16us` | unstable/bad despite high clocks |
| manual_peak | `5` | `99.30us` | `4/5` | `88.98us` | `10.04us` | controlled fast candidate |

The `manual_peak` confirmation is stronger:

| lane | sessions | median lifecycle | best | worst | pass sessions | median consumer | median producer |
|---|---:|---:|---:|---:|---:|---:|---:|
| manual_peak | `10` | `58.04us` | `57.52us` | `1862.60us` | `9/10` | `48.61us` | `9.44us` |

## What Happened

The previous `auto` measurements were not exposing an intrinsic q8 primitive limit. They were sampling uncontrolled GPU
session/perf state:

```text
auto median:        ~122us
manual_peak median:  ~58us
target:             115.24us
```

The fast band is not just `~108us`; under `manual_peak`, the common case is much faster:

```text
producer ~9.4us + consumer ~48.6us = lifecycle ~58us
```

So the right explanation is:

```text
q8 lifecycle performance is DPM/perf-state sensitive; auto-session authority was pessimistic and unstable.
```

## Caveat

This does not yet mean q8 should become default-on. `manual_peak` had one pathological outlier in the 10-session
confirmation. Continuous sysfs telemetry is enough to prove lane control affects the band, but it is still too coarse for
short decode kernels to explain every outlier. Most child sessions only captured one active telemetry sample.

Also, `profile_peak` being worse than `manual_peak` shows that "higher reported clock" is not sufficient authority by
itself; the chosen lane must be empirically gated.

## Decision

Do **not** pivot back to primitive/kernel redesign yet. The immediate solution path is a controlled-clock q8 research
authority:

1. Treat `manual_peak` as the q8 controlled timing authority candidate.
2. Keep `auto` as the user-realistic baseline.
3. Report both numbers separately.
4. Do not promote q8 default behavior unless the owner accepts a controlled-lane policy or a user-realistic policy.

## Next

The next bounded implementation is a q8 promotion-policy closeout:

- `auto` verdict: blocked (`~122us` median);
- `manual_peak` verdict: controlled-fast (`~58us` median, `9/10` pass);
- policy question: is controlled-clock authority acceptable for the route, or must q8 pass under auto?

If the policy requires auto, q8 remains blocked. If controlled-clock authority is acceptable, q8 can move from
`NO_AUTHORITY_POLICY` to `CONTROLLED_FAST_RESEARCH_ROUTE`, still default-off.

## Boundary

No decode default changed.
