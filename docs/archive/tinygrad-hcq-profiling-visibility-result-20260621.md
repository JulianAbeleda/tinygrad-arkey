# Tinygrad HCQ Profiling Visibility — Result

Date: 2026-06-21

Tooling-only (no decode/model/default change). Fix/scope the low-level visibility blind spot from the decode
attribution: rocprofv3 is blind to tinygrad HCQ; rocprof-compute is broken. Artifacts:
`bench/qk-hcq-profiling-visibility/`.

## Decision: **`HCQ_VISIBILITY_USE_NATIVE_ATTRIBUTION_ONLY`**

rocprofv3 is **inherently** blind to tinygrad's HCQ (it bypasses the HSA queue path rocprof intercepts — not a
flag). Live HCQ PMU counters need a **deep** native-profiled-HCQ / ROCprofiler-integration project whose **bounded
paths the prior tooling work already KILLED**, and the rocprofv3 PMC backend **returns 0 for compute counters on
this gfx1100+ROCm-7.2.4 stack regardless**. Meanwhile **native attribution** (ISA + resources via llvm-objdump,
ProfileGraphEvent durations, and the existing ATT instruction-interval tracer) **already sufficed** for the
`FIXABLE_CODEGEN` decode attribution. So: **use native attribution; do not invest in HCQ live-counter visibility.**

## Why rocprofv3 is blind to HCQ (reproduced)

`rocprofv3 --kernel-trace` on a tinygrad matmul produces **0 trace files** (exit 0, no kernels). Root cause
(`tinygrad/runtime/ops_amd.py`): `AMDComputeQueue` / `AMDComputeAQLQueue` write PM4/AQL packets **directly into a
hardware compute ring** (`cq.ring[...]`, lines 431/475) and ring the **doorbell** themselves (`cq.signal_doorbell`,
lines 434/478) via the KFD/amdgpu ioctl path. tinygrad **never calls `hsa_queue_create`** or submits through the HSA
runtime queue. rocprof/rocprofv3 inject profiling by hooking `hsa_queue_create` + intercepting AQL packets/signals in
the HSA runtime — so tinygrad's direct-ring dispatches are **never seen**. This is inherent to HCQ, not a missing
option.

## rocprof-compute fix (attempted, contained venv)

`NOT_BOUNDED_AND_LOW_VALUE`. The install is missing a deep dependency chain — `astunparse==1.6.2` (exact pin vs the
system's 1.6.3), then `plotext`, `colorlover`, `plotly`, `dash>=3.0.0`, `dash-bootstrap-components`, `dash-svg`,
`matplotlib`, `pandas`, … — each install revealing more. **And even if fully satisfied, rocprof-compute wraps
rocprofv3's PMC counter backend, which returns 0 for compute counters here** → a working UI over a 0-counter backend,
and it would only ever help the llama/HIP side, never HCQ. Not worth fixing.

## Counter reliability on this stack

| source | result |
|---|---|
| rocprofv3 `--pmc` `SQ_WAVES` / `SQ_BUSY_CYCLES` (llama) | WORKS |
| rocprofv3 `--pmc` `SQ_INSTS_VALU/LDS`, `GRBM_GUI_ACTIVE` (llama) | **return 0** (gfx1100+ROCm-7.2.4 limit) |
| rocprofv3 anything on tinygrad HCQ | **0 output** (blind) |
| rocprof-compute | broken (deep deps) + wraps the 0-counter backend |

So **live compute counters are unavailable for both llama and tinygrad** on this stack (a ROCm/hardware-support
limit), independent of the HCQ blindness.

## What DOES work for HCQ (the recommended path)

| capability | tool | status |
|---|---|---|
| per-kernel GPU duration | tinygrad ProfileGraphEvent (`PROFILE=1`) | **WORKS** (used in every recent attribution) |
| program/graph attribution | `extra/qk_hcq_attribution.py` | **WORKS** (prior `PASS` PMU-4a..4c) |
| instruction-interval body trace | `extra/qk_att_primitive_atlas.py` (ATT) | **WORKS** (prior `PASS`; `NOT_DECISIVE_FOR_INMODEL_GAP`) |
| VGPR/SGPR/LDS/scratch/spill | `llvm-readelf` descriptors | **WORKS** |
| full ISA | `llvm-objdump` | **WORKS** |

This native stack produced the decode attribution's confident conclusion (un-tiled scalar `flash_partial` with
0 `v_dot2`/0 LDS, 100% occupancy, 0 spills) **without** live counters.

## Path ranking

| path | rank | verdict |
|---|---:|---|
| **F. native ISA/resource/ProfileGraphEvent + ATT intervals** | 1 | **chosen** — exists + sufficed |
| C. ATT/SQTT body-interval capture | 2 | works for intervals; use if needed |
| A. tinygrad-native HCQ PMC export | 3 | the real lever for live counters, but a **deep** runtime project + the PMC backend reads 0 here → low EV; defer |
| B. ROCProfiler/AQL HCQ integration | 4 | large; prior bounded attempts KILLED/BLOCKED; defer |
| E. fix rocprof-compute only | 5 | unbounded deps + 0-counter backend + llama-only |
| D. HIP-visible bridge runner | 6 | profiles a re-implemented HIP kernel, not the HCQ kernel → unrepresentative |

## Acceptance gates

| gate | result |
|---|---|
| G1 rocprof-compute failure reproduced + classified | PASS (deep dep chain; classified) |
| G2 rocprofv3 HCQ blindness reproduced + explained | PASS (0 traces; direct-ring/doorbell bypasses HSA interception) |
| G3 ≥1 bounded fix attempted if safe | PASS (rocprof-compute venv attempt → unbounded; not pursued) |
| G4 says whether HCQ visible now / needs integration / native | PASS (native suffices; live counters need a deep project + 0-backend) |
| G5 no decode/model/default change | PASS (`git diff tinygrad/` empty) |
| G6 policy guard passes | PASS |

## Next tooling step

**None required for decode attribution** — native attribution is sufficient and recommended. If live HCQ compute
counters ever become essential (they were not for the decode verdict), scope **native-profiled-HCQ** (tinygrad
programs PMC counters in its own AQL packets) as a separate large project — but note the gfx1100+ROCm-7.2.4 PMC
backend currently returns 0 for compute counters, so even that has low expected value on this stack.

## Boundary
Tooling only. No `tinygrad/`/model/default/decode-kernel change, no performance tuning. The reproduction runs were
offline profiling probes; no closed lane touched.
