# RESULT - Prefill three-way clock authority (WMMA vs Tensile vs llama)

Date: 2026-06-20

## Scope

This closes the remaining clock question for pp512 prefill:

- tinygrad clean `PREFILL_V2` WMMA
- tinygrad `PREFILL_TENSILE_GEMM=1` research Tensile route
- llama.cpp `llama-bench` pp512

The key requirement was real clock telemetry for all three engines, not just WMMA. The probe uses a separate
process sampler for `rocm-smi --showgpuclocks` plus sysfs DPM/busy/power, so the timing loop is not perturbed by
in-process `rocm-smi` calls.

Driver: `extra/qk_prefill_clock_threeway.py`

Artifacts: `bench/qk-prefill-clock-threeway/*.json` (gitignored, reproducible)

## Method

Tinygrad runs:

- `DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_clock_threeway.py wmma --lane auto`
- `DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_clock_threeway.py tensile --lane auto`
- same commands with `--lane manual_peak`

Llama runs:

- `PYTHONPATH=. .venv/bin/python extra/qk_prefill_clock_threeway.py llama --lane auto --llama-reps 10`
- same command with `--lane manual_peak`

Tinygrad capture safety:

- Each TinyJit is traced and captured while the route flag is fixed.
- WMMA asserts no captured `tensile_*` programs.
- Tensile asserts captured `tensile_*` programs exist.

Clock lanes:

- `auto`: user-realistic DPM.
- `manual_peak`: sysfs `manual`, sclk level 2, mclk level 3. Restored to `auto` after each pinned run.

## Results

| lane | engine | tok/s | real SCLK median | DPM SCLK median | MCLK median | GPU busy median |
|---|---:|---:|---:|---:|---:|---:|
| auto | WMMA | 1436.0 | 2333 MHz | 2334 MHz | 1249 MHz | 100% |
| auto | Tensile | 2664.4 | 2313 MHz | 2314 MHz | 1249 MHz | 100% |
| auto | llama | 3136.2 avg / 3159.6 median | 2296 MHz | 2305.5 MHz | 1249 MHz | 100% |
| manual_peak | WMMA | 1438.2 | 2333 MHz | 2334 MHz | 1249 MHz | 100% |
| manual_peak | Tensile | 2661.9 | 2314.5 MHz | 2316.5 MHz | 1249 MHz | 100% |
| manual_peak | llama | 3139.5 avg / 3161.2 median | 2295 MHz | 2298 MHz | 1249 MHz | 100% |

Ratios:

| lane | Tensile / WMMA | WMMA / llama avg | Tensile / llama avg |
|---|---:|---:|---:|
| auto | 1.855x | 45.8% | 85.0% |
| manual_peak | 1.851x | 45.8% | 84.8% |

## Verdict

Clock is not the missing factor in the pp512 comparison.

All three engines already run at high real SCLK and max MCLK under `auto`. Forcing `manual_peak` does not materially
move throughput for WMMA, Tensile, or llama.

Therefore:

- tinygrad clean WMMA remains about 1436-1438 tok/s at high clock.
- Tensile remains about 2662-2664 tok/s at essentially the same clock.
- llama remains about 3136-3139 tok/s average at essentially the same clock.
- The Tensile/WMMA gap is not a clock artifact.
- The llama/Tensile gap is not a clock artifact.

This supersedes any remaining interpretation that prefill parity is blocked by DPM/SCLK behavior. The remaining gap is
kernel/runtime efficiency at equal clock.
