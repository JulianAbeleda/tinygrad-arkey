# AMD ROCprofiler R1-P2 v2 AQLprofile exporter scope - 2026-06-19

Purpose: scope the remaining bounded reopen after R1-P1:

> Build a v2 AQLprofile packet exporter that binds command/output/control buffers to tinygrad-owned or
> tinygrad-mappable GPU VAs, then submits those exact vendor packets around one HCQ dispatch.

This is still a no-model tooling proof. It is not native profiled-HCQ and it does not change default runtime behavior.

## Starting Point

Already known:

- external ROCprofiler ATT is instruction-rich on HIP controls;
- tinygrad HCQ SQTT is lifecycle-only;
- `AMD_AQL=1` in tinygrad is stable but still lifecycle-only;
- old `hsa_ven_amd_aqlprofile_*` command buffers are useful for diffing but not a standalone replay blob;
- AQLprofile v2 exposes the needed hooks:
  - `aqlprofile_register_agent_info(...)`;
  - `aqlprofile_att_create_packets(...)`;
  - allocation callback with `host_access`, `device_access`, and `memory_hint`;
  - `aqlprofile_att_control_aql_packets_t` with start/stop vendor packets;
  - `aqlprofile_att_iterate_data(...)`.

The missing proof is whether those v2 packets can be generated with buffers we control and submitted through tinygrad's
HCQ/AQL queue.

## Scope

R1-P2 has four phases.

### P0 - v2 Packet Exporter Smoke

Goal: prove we can call the v2 ATT API without ROCprofiler's HSA queue service.

Build a small C helper or Python/ctypes wrapper that:

1. registers a gfx1100 agent using KFD/HSA-visible properties:
   - `agent_gfxip = gfx1100`;
   - `xcc_num = 1`;
   - `se_num = 6`;
   - `cu_num = 96`;
   - `shader_arrays_per_se = 2`;
   - domain/location when using `aqlprofile_agent_info_v1_t`;
2. calls `aqlprofile_att_create_packets(...)` with:
   - `TARGET_CU = 1`;
   - `SE_MASK = 1`;
   - `SIMD_SELECT = 1` or default;
   - `ATT_BUFFER_SIZE = 64MB` initially;
3. records every allocation callback request:
   - size;
   - `host_access`;
   - `device_access`;
   - memory hint;
   - returned pointer;
4. exports:
   - start packet bytes;
   - stop packet bytes;
   - allocation table;
   - handle value;
   - whether `aqlprofile_att_iterate_data` can be called before execution without crashing.

Pass gate:

- `aqlprofile_att_create_packets` returns success;
- start/stop packet bytes are nonzero;
- allocation callback table identifies trace output buffer and trace-control/command buffers.

Kill gate:

- v2 API cannot create packets from a registered KFD-style agent without a live HSA agent/runtime-owned memory pool.

### P1 - tinygrad-Mappable Allocation Strategy

Goal: make the v2 exporter return buffers compatible with tinygrad HCQ submission.

Two acceptable strategies:

1. **Exporter-owned HSA buffers, tinygrad maps/copies addresses.**
   Use HSA/AQLprofile allocation callbacks, then export GPU VAs and CPU pointers to Python. tinygrad submits the packet
   using those VAs and copies output back through the exported CPU mapping or callback.

2. **tinygrad-owned buffers, exporter receives fixed pointers.**
   Allocate trace output/control/command buffers in tinygrad first, pass their CPU/GPU addresses into the exporter, and
   return those addresses from the allocation callback when AQLprofile asks for matching sizes/hints.

Preferred first path: exporter-owned buffers. It is less invasive and proves packet submission before solving buffer
ownership purity.

Pass gate:

- all packet-referenced command/output/control buffers have stable GPU addresses;
- Python can see enough metadata to submit the start/stop packets and read/iterate output;
- no HIP runtime is initialized in the tinygrad process.

Kill gate:

- AQLprofile requires memory allocations that cannot be exposed to or submitted from tinygrad HCQ.

### P2 - One HCQ Dispatch Replay

Goal: run the minimal actual replay.

Use `AMD_AQL=1` and a tiny no-model kernel. Build an AQL queue sequence:

1. AQLprofile start vendor packet;
2. one tinygrad kernel dispatch packet;
3. AQLprofile stop vendor packet;
4. signal/synchronize;
5. read trace output through `aqlprofile_att_iterate_data(...)` or copied trace buffers.

Pass gate:

- process does not hang;
- stop packet completes;
- AQLprofile iteration or direct buffer copy returns nonzero SQTT bytes;
- tinygrad decoder sees body instruction packet classes, not only lifecycle packets.

Kill gate:

- start/stop packets execute but output remains lifecycle-only;
- packets cannot be submitted through tinygrad's KFD AQL queue;
- AQLprofile iteration needs hidden ROCprofiler queue state not present in HCQ.

### P3 - Decision

If P2 passes:

- promote to a reusable probe-local packet replay helper;
- use it for q8 body attribution;
- then decide whether native profiled-HCQ is worth starting.

If P2 fails:

- close HCQ body ATT;
- keep split tooling:
  - external ROCprofiler ATT for instruction oracle;
  - tinygrad-native PMCs for in-model HCQ attribution.

## Non-Goals

- no model route;
- no decode speed claims;
- no default runtime behavior change;
- no broad native profiled-HCQ implementation;
- no more `MASK/TOKEN/CTRL` sweeps.

## Expected Failure Modes

1. **Agent registration mismatch.**
   `aqlprofile_register_agent_info` may not be enough if AQLprofile needs a real HSA agent handle rather than registered
   static agent info.

2. **Memory ownership mismatch.**
   AQLprofile may allocate buffers through HSA pools that tinygrad's KFD/HCQ path cannot safely submit against.

3. **Vendor packet queue-state dependency.**
   Even if the packet bytes are valid, hardware may require the profiled-queue activation packet or HSA queue profiling
   state that ROCprofiler normally emits.

4. **Trace-control iteration dependency.**
   `aqlprofile_att_iterate_data` may require its internal memory manager handle to own the allocations and status layout.

These are valid kill results. The goal is to learn whether packet replay is genuinely bounded before starting native
profiled-HCQ.

## Deliverables

- `extra/amd_rocprofiler_r1p2_v2_exporter.py`
- optional generated C helper under `bench/amd-scheduler-tooling-backend/r1p2_v2_exporter_work/`
- `bench/amd-scheduler-tooling-backend/r1p2_v2_exporter.json`
- result doc with pass/kill verdict

## Recommendation

Do P0/P1 first. Do not proceed to P2 until the exporter can prove packet bytes plus allocation metadata. If P0/P1 cannot
produce controllable packet material, stop and keep split tooling.
