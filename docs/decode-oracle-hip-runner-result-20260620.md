# Decode Oracle HIP Runner Result - 2026-06-20

Verdict: `PASS_DECODE_ORACLE_ROCPROF_VISIBLE_HIP_RUNNER`

The OES-5 runner-surface blocker is cleared. A minimal HIP host runner now launches the same `q8_mmvq_gateup` source under HIP, with the exact oracle launch geometry, and `rocprofv3 --kernel-trace` sees the dispatch.

## Runner

| field | value |
| --- | --- |
| source generator | `extra/qk_decode_oracle_hip_runner_probe.py` |
| generated source | `bench/qk-decode-primitive-transfer/oracle_hip_runner/q8_mmvq_gateup_runner.cpp` |
| generated executable | `bench/qk-decode-primitive-transfer/oracle_hip_runner/q8_mmvq_gateup_runner` |
| kernel | `q8_mmvq_gateup` |
| grid | `(12288, 2, 1)` |
| workgroup | `(32, 4, 1)` |
| direct HIP event time | `106.6us/kernel` |
| output sanity | finite, stable checksum |

The probe must use a consistent ROCm 7.2 stack: `/opt/rocm/bin/hipcc`, `/opt/rocm/lib` in `LD_LIBRARY_PATH`, and `/opt/rocm/bin/rocprofv3`. The first attempt mixed ROCm 7.2 `rocprofv3` with system HIP 5.7 libraries; under that mix, direct HIP ran but rocprof-wrapped HIP could not see a ROCm-capable device.

## Kernel Trace

`rocprofv3 --kernel-trace` emits `q8_mmvq_gateup` rows.

Observed resource/timing fields from the first gate/up dispatch:

| field | value |
| --- | ---: |
| `LDS_Block_Size` | 512 |
| `Scratch_Size` | 0 |
| `VGPR_Count` | 32 |
| `Accum_VGPR_Count` | 0 |
| `SGPR_Count` | 128 |
| `Workgroup_Size_X/Y/Z` | `32 / 4 / 1` |
| `Grid_Size_X/Y/Z` | `393216 / 8 / 1` |

The profiler grid fields are flattened relative to HIP's `(12288,2,1)` grid; the workgroup fields match exactly.

## OES-5 Status

This unlocks the coarse OES-5 path:

1. Kernel-trace resource/timing comparison is now possible.
2. ATT/thread-trace still needs a decoder library, tracked separately.
3. Native scheduling should still remain parked until stage-level attribution identifies whether the gap is S3 issue ordering, S4/S5 reduction, or runtime/launch.

Probe: `extra/qk_decode_oracle_hip_runner_probe.py`

