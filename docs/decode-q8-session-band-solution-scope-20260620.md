# Decode q8 Session-Band Solution Scope

Date: 2026-06-20

## What Happened

The q8 route is not failing because the owned producer is wrong or because the fused gate/up consumer body is obviously
bad. The failure is session-band authority.

The current q8 lifecycle has two real bands:

```text
fast: producer ~17us + consumer ~90us  = lifecycle ~108us  -> clears 115.24us
slow: producer ~20us + consumer ~101us = lifecycle ~122us  -> misses 115.24us
```

The fast band is real and repeatable in some sessions. The slow band is also real and dominates median-of-fresh-sessions.
Warmup policy did not force the fast band:

- 3-session matrix produced an apparent pass for `consumer_warm` and `lifecycle_warm`.
- 5-session confirmation failed: `consumer_warm` median `122.10us`, `lifecycle_warm` median `122.04us`.

So the 3-session pass was sampling luck, not authority.

Coarse `rocm-smi --showgpuclocks` boundary samples were not enough. They sample before/after blocks and miss the actual
GPU state during short decode kernels.

## Solution Direction

Build a decode-specific clock/DPM authority probe, borrowing the proven pieces from the prefill clock tooling:

- continuous sysfs telemetry sampler during the timing window;
- explicit DPM lane controls when allowed;
- classification by measured active-window telemetry, not boundary `rocm-smi` samples;
- restore GPU state after each lane.

Relevant existing tooling:

- `extra/qk_prefill_clock_dpm_authority.py`
- `extra/qk_prefill_clock_experiment.py`
- `extra/qk_prefill_clock_threeway.py`

## Proposed Phases

### P0 - Inventory

Record available controls:

- `power_dpm_force_performance_level`;
- `pp_dpm_sclk`;
- `pp_dpm_mclk`;
- `gpu_busy_percent`;
- `mem_busy_percent`;
- hwmon power/temp files;
- whether passwordless sudo can set DPM lanes.

Output:

```text
bench/qk-decode-primitive-transfer/decode_q8_clock_authority_inventory.json
```

### P1 - Continuous Telemetry Wrapper

Wrap the existing q8 lifecycle measurement with a background sampler at `20-60ms` cadence. Capture:

- sclk/mclk/fclk/socclk if present;
- gpu/mem busy;
- power/temp;
- perf level;
- timestamps covering the exact timing block.

Classify each child session by active-window telemetry, not pre/post samples.

### P2 - Lane Matrix

Run the q8 lifecycle under:

- `auto`;
- `high`;
- `profile_peak`;
- `manual_peak`;
- optional `rocm-smi --setperfdeterminism <mhz>` if supported.

Each lane must restore `auto` afterward.

Gate:

- if a lane holds stable telemetry and median-of-5 lifecycle clears `115.24us`, that lane becomes the q8 timing authority
  candidate;
- if lanes do not hold or do not clear target, q8 remains blocked.

### P3 - Policy Decision

If no controlled lane clears target, choose explicitly:

1. keep median-of-fresh-auto-sessions as authority: q8 remains blocked;
2. accept best-stable-session or controlled-lane authority: requires owner policy approval and clear labeling;
3. continue into lower-level perf-state/PMU work.

No route promotion can happen silently from best-row or 3-session luck.

## Non-Goals

- Do not rewrite q8 producer or consumer kernels in this scope.
- Do not change decode defaults.
- Do not use sparse boundary `rocm-smi` samples as promotion authority.
- Do not accept a 3-session pass without 5+ session confirmation.

## Success Criteria

A successful solution must produce one of:

```text
PASS_DECODE_Q8_CLOCK_AUTHORITY_CONTROLLED_FAST
```

with stable telemetry and median-of-5 lifecycle `<=115.24us`, or:

```text
BLOCKED_DECODE_Q8_CLOCK_AUTHORITY_NO_CONTROLLED_FAST
```

with enough telemetry to stop repeating warmup experiments.
