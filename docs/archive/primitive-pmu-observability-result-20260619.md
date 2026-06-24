# PMU observability result - PMU-1..PMU-3

Executed the first PMU observability scope from `primitive-pmu-observability-scope-20260619.md`.

## Verdict

**REDIRECT_HCQ_NATIVE_ADAPTER.**

ROCm profiling works as a PMU oracle for a HIP/rocBLAS control, but the tinygrad HCQ smoke workload does not produce
`rocprofv3` kernel/HSA trace files in this environment. That means PMU collection is available, but not directly
authoritative for tinygrad HCQ kernels until we add a tinygrad-side attribution/fallback adapter.

Artifact: `bench/qk-pmu-observability/result.json`

Probe: `extra/qk_pmu_observability.py`

## Results

| check | result |
|---|---:|
| ROCm tools found | yes (`/opt/rocm/bin`) |
| available counter sample count | 61 |
| requested PMCs | 9 |
| HIP control kernel trace dispatches | 341 |
| HIP control PMC rows | 2952 |
| HIP control nonzero PMC rows | 328 |
| tinygrad HCQ trace dispatches | 0 |

## Interpretation

This cleanly separates the two problems:

- **PMU collection itself works.** `rocprofv3` captures HIP/rocBLAS dispatches and counter rows on gfx1100.
- **HCQ attribution is the gap.** The same profiler wrapper around a tinygrad HCQ matmul runs successfully but emits
  no kernel trace rows, so the direct tinygrad path needs native queue/program attribution.

The result does not change any performance primitive verdict. It changes the observability plan:

1. Use ROCm PMU as the Level-4 oracle for HIP controls and separately launched extracted kernels.
2. Build a tinygrad-native Level-3 adapter for HCQ: program name, code object hash, launch geometry, queue submit,
   waits/signals, graph segment, kernarg size, and device timing.
3. Attach Level-4 PMU only when `rocprofv3` can actually see the dispatch.

## Next scope

PMU-4 should be a tinygrad HCQ attribution adapter, not a profiler clone. The required output is a primitive-local
row that can distinguish:

- `rocprof_hcq_visibility_gap`
- `graph_boundary`
- `host_sync`
- `program_cache_miss`
- `kernel_math_bound`

without claiming PMU-level root cause when the PMU counters are absent.
