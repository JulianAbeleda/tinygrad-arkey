# AMD ROCprofiler R1-P2 v2 AQLprofile exporter result - 2026-06-19

Verdict: `P0_PASS_P1_NEEDS_HCQ_VA_BINDING`.

R1-P2 tested the remaining bounded reopen for HCQ body-level ATT:

> Build a v2 AQLprofile packet exporter that binds command/output/control buffers to tinygrad-owned or
> tinygrad-mappable GPU VAs, then submits those exact vendor packets around one HCQ dispatch.

P0 passes after correcting the local v2 ABI declaration. The v2 packet exporter can initialize HSA, find the real HSA GPU
agent, generate ATT start/stop packets, and expose the allocation callback table. P1 is now the next boundary: those
AQLprofile-requested buffers must be backed by tinygrad-owned or tinygrad-submittable GPU VAs before one HCQ dispatch can
be replayed.

The earlier failing result was caused by a local ABI mistake. The real `aql_profile_v2.h` declares
`aqlprofile_att_profile_t.agent` as `hsa_agent_t`, while the probe had declared it as `aqlprofile_agent_handle_t` and fed
the static registered-agent handle into `aqlprofile_att_create_packets(...)`. That made the ATT path receive an invalid
agent and reject every profile before allocation. Using the real `hsa_agent_t` from `hsa_iterate_agents(...)` fixes packet
creation.

## Artifact

- Probe: `extra/amd_rocprofiler_r1p2_v2_exporter.py`
- Result: `bench/amd-scheduler-tooling-backend/r1p2_v2_exporter.json`
- Generated helper: `bench/amd-scheduler-tooling-backend/r1p2_v2_exporter_work/v2_exporter_smoke.cpp`

The helper dynamically loads `libhsa-amd-aqlprofile64.so` because the installed public headers expose the legacy
`hsa_ven_amd_aqlprofile_*` API but not the v2 ATT packet-export symbols. The exact v2 declarations were checked against
`ROCm/aqlprofile/src/core/include/aqlprofile-sdk/aql_profile_v2.h`.

## What Passed

- `hsa_init()` succeeds.
- `hsa_iterate_agents(...)` finds the real GPU `hsa_agent_t`.
- `aqlprofile_register_agent_info(...)` succeeds for the static gfx1100 v1 agent record, but that handle is not the ATT
  profile agent type:
  - `gfx1100`;
  - `xcc_num=1`;
  - `se_num=6`;
  - `cu_num=96`;
  - `shader_arrays_per_se=2`.
- The v2 symbols resolve from `libhsa-amd-aqlprofile64.so`:
  - `aqlprofile_register_agent_info`;
  - `aqlprofile_att_create_packets`;
  - `aqlprofile_att_delete_packets`;
  - `aqlprofile_att_iterate_data`.

One implementation detail was important: without explicit `hsa_init()`, the v2 path fails with
`HSA_STATUS_ERROR_NOT_INITIALIZED`. With `hsa_init()` plus a real HSA GPU agent, packet creation succeeds.

## Why It Previously Rejected

The initial probe used this wrong local declaration:

```c
typedef struct {
  aqlprofile_agent_handle_t agent;
  const aqlprofile_att_parameter_t* parameters;
  uint32_t parameter_count;
} aqlprofile_att_profile_t;
```

The real v2 declaration is:

```c
typedef struct {
  hsa_agent_t agent;
  const aqlprofile_att_parameter_t* parameters;
  uint32_t parameter_count;
} aqlprofile_att_profile_t;
```

So the rejection was not a hidden ROCprofiler-service dependency at P0. It was a wrong handle type. The static
registered-agent handle is valid for APIs that accept `aqlprofile_agent_handle_t`, but ATT packet creation asks for the
runtime `hsa_agent_t`.

## What Passed After the Fix

All swept profile shapes returned success from `aqlprofile_att_create_packets(...)`:

| Attempt | Parameters | Result |
| --- | --- | --- |
| `cu_se_simd_buf` | `TARGET_CU`, `SE_MASK`, `SIMD_SELECTION`, `ATT_BUFFER_SIZE=64MB` | pass |
| `cu_se_simd` | `TARGET_CU`, `SE_MASK`, `SIMD_SELECTION` | pass |
| `cu_se_only` | `TARGET_CU`, `SE_MASK` | pass |
| `cu_only` | `TARGET_CU` | pass |
| `no_params` | none | pass |

Classification from the artifact:

```text
attempt_count: 5
working_attempts: [cu_se_simd_buf, cu_se_simd, cu_se_only, cu_only, no_params]
allocation_count: 15
has_device_access_alloc: true
has_host_access_alloc: true
start_nonzero_words: 6
stop_nonzero_words: 6
```

Each successful profile requests three buffers:

- small host/device-access trace-control state (`raw=7`, memory hint host);
- large device-only trace output (`raw=17`, memory hint device noncoherent);
- small host/device-access command buffer (`raw=19`, memory hint device noncoherent).

The start and stop AQL packets are nonzero and point at the generated command-buffer regions.

## Interpretation

This reopens the bounded route at the correct boundary.

The controls now separate cleanly:

- External `rocprofv3 --att` is instruction-rich on HIP controls, so the machine and decoder stack can produce body
  instruction records.
- tinygrad HCQ SQTT remains lifecycle-only.
- Legacy `hsa_ven_amd_aqlprofile_*` can generate gfx1100 ATT command material, but R1-P1 showed that material is not a
  direct HCQ replay blob.
- v2 `aqlprofile_att_create_packets(...)` can generate direct start/stop packet material when given a real HSA GPU agent.

The next unknown is no longer packet creation. It is whether the callback-requested buffers can be allocated through
tinygrad/KFD in a form that the vendor packets can use when submitted on tinygrad's HCQ queue.

## Decision

R1-P2 now lands here:

- P0: pass.
- P1: pending; bind AQLprofile allocation requests to tinygrad-owned or tinygrad-submittable GPU VAs.
- P2: pending; submit start packet, one HCQ dispatch, stop packet, then decode/iterate output.

The immediate next step is a P1/P2 replay helper:

1. allocate the trace-control, output, and command buffers through tinygrad/KFD or HSA memory pools with GPU-visible VAs;
2. return those exact pointers from the AQLprofile allocation callback;
3. submit the nonzero v2 start/stop packets around one tinygrad HCQ dispatch;
4. check whether `aqlprofile_att_iterate_data(...)` or direct output decode yields body instruction packets.

If that passes, the body-attribution route is alive without native profiled-HCQ. If it fails, the remaining blocker is
packet/buffer execution semantics, not API packet export.
