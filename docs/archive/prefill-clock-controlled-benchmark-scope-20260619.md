# SCOPE — clean clock-controlled tinygrad-vs-llama prefill benchmark (separate clock from efficiency)

## Why
The tinygrad-vs-llama prefill ratio swings 64%-82%+ across sessions; llama itself = +-26% stddev/run. Cause is GPU
CLOCK, not measured. We need to separate "tinygrad clocks lower" (a runtime/DPM issue) from "tinygrad is less
efficient at equal clock" (the real perf gap). The reliable number (concrete-KV = 1.24x over symbolic) is
clock-controlled; the cross-engine absolute is not.

## Grounded surface (verified)
- **tinygrad uses KFDIface** (amdgpu/KFD-managed) -> SAME clock domain as llama's HIP. Both controllable via the
  same amdgpu sysfs. (NOT the AM userspace driver -> no separate clock control needed.)
- card0 = 0x744c = RX 7900 XTX. DPM levels: sclk {500, 848, 2304} MHz; mclk {96, 456, 772, 1249} MHz.
- **At idle: SCLK=848 (level 1, NOT max 2304); MCLK=1249 (max).** `power_dpm_force_performance_level=high` did NOT
  pin SCLK to max. So SCLK is the confound (compute-bound prefill: 848 vs 2304 = 2.7x).
- Clock pin via sysfs (likely needs root): `echo manual > .../power_dpm_force_performance_level`,
  `echo 2 > .../pp_dpm_sclk` (2304), `echo 3 > .../pp_dpm_mclk` (1249). Reversible (`echo auto > ...level`).

## The KEY question this answers
**Does tinygrad's KFD path ramp SCLK to 2304 under load like llama's HIP does, or stay at 848?** If tinygrad runs
prefill at 848 while llama hits 2304, most of the "gap" is CLOCK (tinygrad runtime not ramping DPM) -> a separate,
fixable runtime issue, NOT an efficiency gap. If both hit 2304 and tinygrad is still slower, that's the real gap.

## Plan
- **P1 - measure clock UNDER LOAD (no pinning):** read `pp_dpm_sclk`/`pp_dpm_mclk` (the `*` level) WHILE a sustained
  tinygrad concrete-KV prefill loop runs, and again while `llama-bench -p 512` runs. Compare the steady-state SCLK
  each engine drives. (Cheap, no root, directly answers the key question.)
- **P2 - pin the clock (root):** force SCLK=2304, MCLK=1249 (sysfs manual DPM); verify `*` moves; verify it holds
  during a run. If root unavailable, user runs the `! echo ...` commands.
- **P3 - clean comparison at pinned clock:** tinygrad concrete-KV pp512 vs llama pp512, both verified at SCLK=2304
  MCLK=1249, best-of-N. Report the clock-controlled ratio = the TRUE efficiency gap.
- **P4 - if pinning impossible:** normalize tok/s by the measured SCLK each engine ran at (tok/s per GHz), report
  the clock-normalized ratio + flag the clock-ramp delta as a tinygrad runtime issue.

## Gates / output
A defensible single number: tinygrad concrete-KV pp512 as % of llama pp512 AT EQUAL VERIFIED CLOCK, + the clock
each ran at, + the clock-vs-efficiency split. Decode (Codex) clock applies too (decode is BW/MCLK-bound; MCLK is
already max, so decode less affected -- a useful corollary).

## Risks
- sysfs clock-pin needs root (user runs `!`); reversible.
- DPM may override the manual pin under thermal/power limits (verify it holds mid-run).
- tinygrad KFD may not honor amdgpu DPM the same way (P1 measures this directly).
