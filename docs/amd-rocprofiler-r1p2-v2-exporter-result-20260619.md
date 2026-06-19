# AMD ROCprofiler R1-P2 v2 AQLprofile exporter result - 2026-06-19

Verdict: `P0_FAIL_STATIC_AGENT_V2_CREATE_REJECTED`.

R1-P2 tested the remaining bounded reopen for HCQ body-level ATT:

> Build a v2 AQLprofile packet exporter that binds command/output/control buffers to tinygrad-owned or
> tinygrad-mappable GPU VAs, then submits those exact vendor packets around one HCQ dispatch.

The route stops at P0. The v2 packet exporter can initialize HSA and register the static gfx1100 agent info, but
`aqlprofile_att_create_packets(...)` rejects every attempted ATT profile before invoking the allocation callback. That
means there are no start/stop packet bytes and no buffer table to bind into tinygrad HCQ, so P1/P2 cannot honestly start.

## Artifact

- Probe: `extra/amd_rocprofiler_r1p2_v2_exporter.py`
- Result: `bench/amd-scheduler-tooling-backend/r1p2_v2_exporter.json`
- Generated helper: `bench/amd-scheduler-tooling-backend/r1p2_v2_exporter_work/v2_exporter_smoke.cpp`

The helper dynamically loads `libhsa-amd-aqlprofile64.so` because the installed public headers expose the legacy
`hsa_ven_amd_aqlprofile_*` API but not the v2 ATT packet-export symbols.

## What Passed

- `hsa_init()` succeeds.
- `aqlprofile_register_agent_info(...)` succeeds for the static gfx1100 v1 agent record:
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

One earlier implementation detail was important: without explicit `hsa_init()`, the v2 path fails with
`HSA_STATUS_ERROR_NOT_INITIALIZED`. With `hsa_init()`, registration succeeds, so the final failure is not just missing
HSA initialization.

## What Failed

All swept profile shapes returned status `4096` from `aqlprofile_att_create_packets(...)`:

| Attempt | Parameters | Result |
| --- | --- | --- |
| `cu_se_simd_buf` | `TARGET_CU`, `SE_MASK`, `SIMD_SELECTION`, `ATT_BUFFER_SIZE=64MB` | fail before allocation |
| `cu_se_simd` | `TARGET_CU`, `SE_MASK`, `SIMD_SELECTION` | fail before allocation |
| `cu_se_only` | `TARGET_CU`, `SE_MASK` | fail before allocation |
| `cu_only` | `TARGET_CU` | fail before allocation |
| `no_params` | none | fail before allocation |

Classification from the artifact:

```text
attempt_count: 5
working_attempts: []
allocation_count: 0
start_nonzero_words: 0
stop_nonzero_words: 0
```

Because the allocation callback is never called, this is upstream of buffer ownership. There is no trace output buffer,
trace-control buffer, command buffer, start packet, or stop packet to transplant into tinygrad's HCQ queue.

## Interpretation

This kills the scoped static-agent v2 exporter route, not all ROCprofiler ATT.

The controls now separate cleanly:

- Legacy `hsa_ven_amd_aqlprofile_*` can generate gfx1100 ATT command material, but R1-P1 showed that material is not a
  direct HCQ replay blob.
- External `rocprofv3 --att` is instruction-rich on HIP controls, so the machine and decoder stack can produce body
  instruction records.
- tinygrad HCQ SQTT remains lifecycle-only.
- The v2 static-agent packet exporter cannot produce packet material independently of ROCprofiler's normal queue/service
  setup.

The likely boundary is the same one identified in the source audit: ROCprofiler ATT is not just a few SQTT registers or a
standalone command-buffer blob. It depends on a profiled HSA queue path plus AQLprofile/ROCprofiler-owned setup state.

## Decision

R1-P2 stops here:

- P0: fail.
- P1: not started, because there are no v2 allocations to bind.
- P2: not started, because there are no vendor start/stop packets to submit.

The bounded reopen path is now closed unless one of these changes:

1. use ROCprofiler's real HSA agent/queue/service path instead of static-agent registration;
2. build native profiled-HCQ support;
3. keep split tooling and use external ROCprofiler ATT as the instruction oracle while tinygrad-native PMCs remain the
   in-model attribution tool.

Per the current principles, option 3 is the default. Option 2 is project-level and should start only if body-level HCQ
instruction attribution becomes a blocker for a funded scheduler/codegen project.
