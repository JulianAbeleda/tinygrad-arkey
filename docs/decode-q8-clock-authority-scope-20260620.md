# Decode q8 Clock Authority Scope

Date: 2026-06-20

## Goal

Audit whether existing clock/DPM tooling can turn the q8 fast band into controlled authority. This tests the hypothesis
that session-band variance is perf-state/clock-state driven rather than a primitive/kernel-body issue.

## Command

```bash
PYTHONPATH=. python3 extra/qk_decode_q8_clock_authority.py \
  --lanes auto,high,profile_peak,manual_peak \
  --sessions 5
```

## Controls

Use existing sysfs controls already proven by the prefill clock tooling:

- `/sys/class/drm/card0/device/power_dpm_force_performance_level`
- `/sys/class/drm/card0/device/pp_dpm_sclk`
- `/sys/class/drm/card0/device/pp_dpm_mclk`
- `gpu_busy_percent`, `mem_busy_percent`, hwmon power/temp

Each child process sets one lane, samples telemetry continuously during the q8 timing block, and restores auto.

## Gate

`PASS_DECODE_Q8_CLOCK_AUTHORITY_CONTROLLED_FAST` requires a lane whose median-of-5 q8 lifecycle is `<=115.24us`, with
all sessions correct and lane setup successful.

If no lane clears, the result is `BLOCKED_DECODE_Q8_CLOCK_AUTHORITY_NO_CONTROLLED_FAST`, and the path should pivot away
from clock/warmup policy toward primitive-level rethink or explicit promotion-policy change.
