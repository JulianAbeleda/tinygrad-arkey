# GPU Low-Level Control Tooling Reference

Date: 2026-06-21

This document is a future tooling map for the tinygrad AMD/Qwen project. It is not a current execution scope.

The purpose is to keep the low-level control surface clear in case decode work moves below ordinary tinygrad codegen
into profiler, disassembly, raw-kernel, or assembly-level tooling.

## Core Point

Modern GPUs do not expose complete control to ordinary software.

The practical target is:

```text
enough observability and control for the performance-determining primitive
```

For this project, the likely performance-determining primitive is decode attention:

```text
llama-style flash_attn_tile:
  q·k mapping
  KV split work decomposition
  GQA/query-head packing
  register/LDS dataflow
  online softmax + V accumulation
  launch/graph integration
```

## Tooling Layers

| layer | NVIDIA examples | AMD examples | what it exposes | project relevance |
|---|---|---|---|---|
| runtime timeline | Nsight Systems | ROCm Systems Profiler | CPU/GPU overlap, launch gaps, queues, waits | useful if W==D or graph routing regresses |
| kernel counters | Nsight Compute | `rocprof`, ROCProfiler | occupancy, memory counters, cache, instruction mix | useful for q·k tile diagnosis |
| hardware traces | Nsight/CUPTI/internal | RGP, SQTT/thread trace | low-level timing and wave behavior | useful if counters are not enough |
| disassembly | `cuobjdump`, `nvdisasm` | `llvm-objdump`, Radeon GPU Analyzer | final machine/ISA shape | required for codegen-quality comparison |
| assembly authoring | MaxAs, CuAssembler, TuringAs | AMDGCN/RDNA assembly, HSACO | direct kernel-body control | possible DeepSeek-style escape hatch |
| driver/runtime control | CUDA Driver API, NVBit, CUPTI | HSA/HCQ, ROCr, ROCm tools | launch contract, queue behavior, instrumentation | relevant to tinygrad HCQ/raw bridge |
| simulation | GPGPU-Sim, Accel-Sim | gem5/academic models | approximate cycle-level experiments | research only; not promotion authority |

## AMD-First Toolchain For This Repo

Use this stack first because the target hardware is AMD gfx1100 / RX 7900 XTX.

| tool | use | notes |
|---|---|---|
| `rocprof` / rocprofv3 | kernel traces, runtime/API traces, counters | promotion still comes from W==D; profiler timing is diagnostic |
| ROCProfiler / ROCm Systems Profiler | system-level tracing and profiling | useful for host/runtime and queue behavior |
| Radeon GPU Profiler (RGP) / SQTT | hardware thread tracing and low-level timing | useful when kernel counters do not explain stalls |
| AMDGCN/RDNA disassembly | inspect generated ISA | use to compare tinygrad codegen against llama/Tensile-style bodies |
| AMDGCN assembly + HSACO | hand-authored raw kernels | only justified as a measured escape hatch |
| tinygrad HCQ attribution | connect programs/kernels back to model roles | keeps raw observations tied to lifecycle rows |
| `decode_eval` + lifecycle-search | authority for candidate verdicts | required for all candidate promotion/refutation |

## NVIDIA / Cross-Vendor Reference Tooling

These tools are not directly usable for the AMD target, but they explain how other teams get closer to the hardware.

| tool | what it is | lesson |
|---|---|---|
| Nsight Compute | CUDA kernel profiler with detailed metrics and CLI/UI | mature per-kernel observability matters |
| SASS / `nvdisasm` | NVIDIA machine-level assembly/disassembly | high-performing libraries often live below source-level CUDA |
| CuAssembler | unofficial SASS assembler that writes cubins | non-vendor teams have pursued direct assembly control |
| TuringAs | open-source SASS assembler for Volta/Turing/Ampere | assembly-level escape hatches are possible but narrow |
| GPGPU-Sim / Accel-Sim | trace-driven GPU simulation frameworks | useful for research, but not real-hardware authority |

## Who Gets Farthest

| rank | group | control level | limit |
|---:|---|---|---|
| 1 | GPU vendors | hardware docs, firmware, compiler, profilers, libraries | still not all behavior is software-controlled |
| 2 | custom-silicon hyperscalers | hardware/compiler/runtime co-design | mostly inside their own stack |
| 3 | elite kernel/library teams | hand-tuned kernels, PTX/SASS/ISA, custom profilers | narrow primitives, high effort |
| 4 | reverse-engineering/open-driver communities | driver/ISA discovery | firmware and undocumented internals remain hard limits |
| 5 | academic simulator/compiler groups | high observability in models | simulator fidelity and speed limits |
| 6 | framework/compiler projects | productive abstractions and portability | backend quality determines the ceiling |

DeepSeek-style work sits around rank 3 for specific bottlenecks: custom kernels, PTX-level control, framework/data
format co-design, and measured lifecycle integration. It is not complete GPU control.

## When To Drop Lower

Do not use assembly/raw kernels because they are interesting.

Use a lower layer only when all are true:

1. The candidate names a specific missing control surface.
2. Higher-level tinygrad/codegen cannot express it efficiently.
3. There is a comparator and a correctness gate.
4. There is a local A/B gate and a W==D promotion path if local passes.
5. The path is default-off until it passes policy.
6. The result becomes a lifecycle-search row, not a one-off script.

For decode attention, a legitimate low-level escape hatch would look like:

```text
missing control:
  q·k mapping / register-LDS dataflow cannot match llama flash_attn_tile through current tinygrad lowering

lowest layer:
  AMDGCN/HSACO or a renderer/codegen change

gate:
  local A/B vs gqa_coop_vec and llama oracle, then W==D if local passes
```

## Useful References

- NVIDIA Nsight Compute:
  https://docs.nvidia.com/nsight-compute/NsightCompute/index.html
- AMD rocprof:
  https://rocm.docs.amd.com/projects/rocprofiler/en/docs-6.1.5/how-to/using-rocprof.html
- AMD Radeon GPU Profiler:
  https://gpuopen.com/manuals/rgp_manual/
- AMDGCN assembly / HSACO:
  https://gpuopen.com/learn/amdgcn-assembly/
- Reading AMDGCN ISA:
  https://rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html
- CuAssembler:
  https://github.com/cloudcores/CuAssembler
- TuringAs:
  https://github.com/daadaada/turingas
- GPGPU-Sim:
  https://github.com/gpgpu-sim/gpgpu-sim_distribution
- Accel-Sim:
  https://accel-sim.github.io/

## Project Rule

```text
Do not seek complete GPU control.
Seek the lowest layer that exposes the current bottleneck, then route it through the evaluator/lifecycle system.
```
