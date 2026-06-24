# AMD ROCprofiler thread-trace audit result - 2026-06-19

Purpose: stop guessing why `rocprofv3 --att` produces body instruction records while tinygrad HCQ SQTT still captures
only lifecycle packets. This audits the installed ROCprofiler SDK sample/headers plus ROCprofiler/AQLprofile source
against `tinygrad/runtime/ops_amd.py`.

Artifacts:

- `extra/amd_rocprofiler_thread_trace_audit.py`
- `bench/amd-scheduler-tooling-backend/rocprofiler_thread_trace_audit.json`

## Verdict

**ROCPROFILER_ATT_MISSING_PROFILED_HSA_AQL_PATH_IN_HCQ.**

The missing piece is not a single SQTT register. ROCprofiler gets body ATT by running a full profiled-HSA-queue path:

1. configure ROCprofiler's dispatch thread-trace service;
2. intercept HSA queue writes;
3. mark the HSA queue/agent as profiling-active;
4. inject AQLprofile-generated vendor-specific ATT start/stop packets around the kernel dispatch;
5. use AQLprofile's trace-control buffer protocol to read status/WPTR and copy the payload;
6. emit ROC trace-decoder metadata packets for agent/code-object mapping.

tinygrad HCQ currently has direct SQTT PM4 register programming, but not that profiled-HSA-AQL lifecycle. That explains
why the prior bounded patches changed trace volume but never produced body instruction packets.

## What the audit checked

The audit script scans these local sources:

- `/opt/rocm-7.2.4/share/rocprofiler-sdk/samples/thread_trace/agent.cpp`
- `/opt/rocm-7.2.4/include/rocprofiler-sdk/experimental/thread-trace/core.h`
- `/tmp/rocm-systems-main-probe/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk-tool/tool.cpp`
- `/tmp/rocm-systems-main-probe/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/thread_trace/{core,service}.cpp`
- `/tmp/rocm-systems-main-probe/projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/{aql/packet_construct,hsa/aql_packet,hsa/queue}.cpp`
- `/tmp/rocm-systems-main-probe/projects/rocprofiler-sdk/source/lib/aqlprofile/core/threadtrace.cpp`
- `/tmp/rocm-systems-main-probe/projects/rocprofiler-sdk/source/lib/aqlprofile/pm4/sqtt_builder.h`
- `tinygrad/runtime/ops_amd.py`

All eight audited lifecycle components are present in the ROCprofiler/AQLprofile path and missing as a combined path in
tinygrad HCQ:

| component | ROCprofiler | tinygrad HCQ | meaning |
|---|---:|---:|---|
| dispatch thread-trace service | yes | no | service chooses which dispatch gets start/stop |
| HSA queue interposition | yes | no | profiler can inject packets around an HSA dispatch |
| profiled queue activation | yes | no | calls `hsa_amd_profiling_set_profiler_enabled` and emits profiler-active queue packet |
| AQLprofile packet factory | yes | no | start/stop are vendor-specific AQL PM4 packets, not only ad hoc register writes |
| trace-control buffer protocol | yes | partial/no | status/WPTR/counter copied to a known control buffer and iterated by AQLprofile |
| SQTT begin ordering | yes | partial/no | includes `BuildPrimeL2`, status zero, exact per-SE order, enable |
| decoder metadata markers | yes | partial/no | emits ROC decoder agent/code-object metadata via `SQ_THREAD_TRACE_USERDATA_2` |
| SQTT end ordering | yes | partial/no | disables, waits finish/busy, reads values, cache-flushes control buffer |

## Important source-level facts

`rocprofv3 --att` does not directly explain the whole mechanism. In the SDK source it configures parameters such as
`TARGET_CU`, `SIMD_SELECT`, `BUFFER_SIZE`, `SHADER_ENGINE_MASK`, `SERIALIZE_ALL`, and then normally chooses
`rocprofiler_configure_dispatch_thread_trace_service(...)`.

The dispatch service installs queue callbacks. The HSA write interceptor returns AQL instrumentation packets from
`DispatchThreadTracer::pre_kernel_call`, places them before the kernel, places stop/read packets after it, and runs
`DispatchThreadTracer::post_kernel_call` after completion.

The queue constructor explicitly calls `hsa_amd_profiling_set_profiler_enabled_fn(...)` and emits
`set_profiler_active_on_queue(...)` before profiling packets run. This is a major structural difference from tinygrad's
KFD/HCQ path: tinygrad does not submit through a normal HSA intercept queue.

AQLprofile manufactures the ATT packets through `aqlprofile_att_create_packets(...)`. On gfx11, its `Begin()` path does
more than the tinygrad SQTT code: broadcast GRBM, clear `SQ_THREAD_TRACE_STATUS`, prime L2 for the trace buffer, program
buffer size/base/mask/token/control in its chosen order, enable `COMPUTE_THREAD_TRACE_ENABLE`, and emit ROC decoder
agent metadata. Its `End()` path disables tracing, waits finish/busy, copies status/counter/WPTR into a trace-control
buffer, cache-flushes that buffer, and later `aqlprofile_att_iterate_data(...)` uses the control buffer to compute the
payload size.

## What is closed

The following have already been tried and are no longer plausible as the primary missing piece:

- `SQ_THREAD_TRACE_MASK`;
- `SQ_THREAD_TRACE_TOKEN_MASK`;
- `SQ_THREAD_TRACE_CTRL`;
- `SQTT_MODE`;
- `SQTT_TTRACE_EXEC`;
- `SQTT_ORACLE_TARGET_CU`;
- decoder availability;
- trace volume.

They can change packet volume or lifecycle records, but they did not produce body instruction packets.

## Reopen options

1. **AQLprofile packet import/replay.** Use AQLprofile to manufacture start/stop packets and adapt their command buffers
   into tinygrad HCQ submission, if the vendor AQL packet body can be decoded or directly replayed outside an HSA queue.
   This is medium-high risk but the most bounded real reopen.

2. **HCQ profiled-queue equivalent.** Implement the profiling-active queue state and ATT packet lifecycle natively for
   tinygrad's KFD/HCQ path. This is project-level work.

3. **Keep the split tooling model.** Use external ROCprofiler ATT on HIP controls/imported kernels for instruction-rich
   attribution, and use tinygrad-native PMCs for in-model HCQ attribution. This is the low-risk default.

## Decision

Do not continue Track T as a register-patch project. The audit identifies a lifecycle/queue-protocol gap, not a missing
small primitive knob.

If we reopen, reopen with an explicit AQLprofile packet import/replay scope. Otherwise, keep the external ROCprofiler ATT
oracle plus tinygrad PMCs as the observability stack.
