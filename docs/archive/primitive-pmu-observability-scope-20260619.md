# Scope - PMU observability for primitive analysis

Question: can we install/use ROCm profiling tools to understand PMU-level root causes, then build the tinygrad-side
equivalent we actually need?

## Local finding

The tools are already installed, but not on the default shell PATH:

| tool | local path / status |
|---|---|
| `rocprofv3` | `/opt/rocm/bin/rocprofv3`, version 1.1.0 / ROCm 7.2.4 |
| `rocprofv3-avail` | `/opt/rocm/bin/rocprofv3-avail` |
| `rocprof-compute` | `/opt/rocm/bin/rocprof-compute`, present but Python UI dependencies are incomplete |
| `rocm-smi` | `/usr/bin/rocm-smi` |

`rocprofv3-avail list` reports gfx1100 counters including occupancy, L2 hit/miss, VMEM/LDS wait, LDS bank conflict,
VALU/SALU/SMEM/VMEM instruction counts, wave counts, GPU busy, and memory unit busy.

So the blocker is not installation. The blocker is integration and attribution:

- ROCm can profile HIP/ROCm-launched kernels directly.
- tinygrad's AMD path uses HCQ/KFD directly, so `rocprofv3` may not emit the same trace/counter records for every
  tinygrad workload.
- even when counters are collected, they must be mapped back to a primitive row, role, candidate, shape, and gate.

## Smoke test

Command shape tested:

```bash
DEV=AMD PYTHONPATH=. /opt/rocm/bin/rocprofv3 --kernel-trace --hsa-trace --stats --summary \
  -f json csv --output-directory /tmp/qk-rocprof-smoke -- .venv/bin/python tinygrad_workload.py
```

The tinygrad HCQ workload ran successfully, but no output files were emitted in the requested directory. Treat this as
evidence that the direct HCQ path needs a dedicated adapter/probe before assuming ROCm traces are authoritative for
tinygrad kernels.

## What to copy from ROCm

Do not rebuild ROCm's PMU stack. Copy the data model:

| ROCm concept | tinygrad primitive equivalent |
|---|---|
| kernel dispatch trace | program name, launch dims, device time, queue id |
| counter collection | selected PMCs attached to one primitive observation |
| kernel include regex / iteration range | primitive candidate id + warmup/timed iteration selection |
| thread trace / PC sampling | optional deep diagnostic for one kernel after timing gate |
| summary/grouping | aggregate by primitive/role/shape, not by raw kernel symbol only |

## Build plan

### PMU-1 - path and capability inventory

Update the primitive ledger to find ROCm tools in `/opt/rocm/bin`, not only PATH. Record available counters and tool
version in `trace_plugins.json`.

Gate: ledger accurately reports `rocprofv3` as available on this machine.

### PMU-2 - HIP control capture

Run `rocprofv3` on a HIP-only control kernel that we know emits dispatches, preferably the existing rocBLAS/Tensile
ceiling binary. Parse `trace_kernel_trace.csv`, `trace_counter_collection.csv`, and `trace_results.json`.

Gate: one primitive row can attach Level-4 counters to a known HIP-launched Tensile kernel.

### PMU-3 - tinygrad HCQ capture attempt

Run the same trace/counter selection against a tinygrad HCQ kernel. If no records appear, classify that as
`rocprof_hcq_visibility_gap` rather than a performance result.

Gate: either tinygrad HCQ dispatches are visible to `rocprofv3`, or the visibility gap is documented with a minimal
reproducer.

### PMU-4 - tinygrad-native fallback

If ROCm cannot see HCQ dispatches reliably, build a tinygrad-side trace adapter:

- log program name, queue submit, signal wait, launch geometry, kernarg size, code object hash;
- attach static metadata already available from codegen/HSACO descriptors;
- optionally add SQTT/thread-trace integration using existing `extra/sqtt` assets;
- store all of this as Level 3 evidence, with PMU counters only when ROCm collection succeeds.

Gate: the primitive ledger can distinguish `kernel_math_bound`, `graph_boundary`, `host_sync`, `program_cache_miss`,
and `rocprof_hcq_visibility_gap` without overclaiming PMU evidence.

### PMU-5 - primitive PMU rows

Apply the adapter only to live rows:

- TPE-7 graph route / extracted Tensile kernels;
- pure tinygrad WMMA plateau if reopened by new codegen evidence;
- q8/MMVQ lifecycle only after a buildable candidate exists.

Do not run broad PMU collection over closed rows.

## Expected outcome

This should expand the evidence hierarchy, not change the current frontier by itself:

- Level 4 counters can explain why a candidate plateaued.
- They should not override correctness, device time, in-model throughput, or quality gates.
- The useful deliverable is a primitive-local PMU adapter, not a general profiler clone.
