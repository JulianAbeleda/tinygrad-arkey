# 14B decode hardware-counter integration scope

Date: 2026-07-26

Repository: `/home/ubuntu/tinygrad-arkey`

Target: AMD RX 7900 XTX-class gfx1100, wave32, `xccs=1`.

## Objective

Make hardware-counter capture safe, truthful, and attributable for tinygrad's
child-process decode workload. Integrate the result with BoltBeam so it can
compare 14B G=5 against 8B G=4 and explain decode-depth decay without changing
the production path or risking another GPU reset.

This task is an instrumentation integration task. It is not a performance fix
by itself. The output must tell us whether occupancy, instruction pressure,
cache behavior, or another mechanism changes with depth.

## Current facts; do not re-derive

- Counter register programming exists in `tinygrad/runtime/ops_amd.py`.
- Graph counter insertion exists behind `PMC_GRAPH` in
  `tinygrad/runtime/graph/hcq.py`.
- Gfx1100 initialization calls `AMDKFD_IOC_PROFILER` to unlock perfmon.
- The ioctl returns `EPERM` even though `/dev/kfd` is readable by the current
  user through the `render` group.
- Before the fail-closed change, `PMC=1 PMC_GRAPH=1` on 14B decode timed out,
  raised the known SQC page fault at `0x0000ffffffbfe000`, and reset the GPU.
- Prefill PMC traces produced rows for non-GEMM kernels but did not attribute
  the hot packed GEMMs.
- Safe observer-only captures are valid. The last 500 14B dispatch binaries
  are identical at context 512 and 4096, so depth-dependent route/dispatch
  growth is not the cause.
- The live flash kernels are:
  - 14B: G=5, workgroup `[32,5,1]`, LDS 8192, VGPR 91, VGPR spill 0.
  - 8B: G=4, workgroup `[32,4,1]`, LDS 8192, VGPR 134, VGPR spill 0.
- HSACO dumps and the offline resource audit already exist in the campaign
  artifacts under `/home/ubuntu/boltbeam-runs/tinygrad-g5-decay-20260726/`.

## Required work packages

### 1. System profiler preflight

Record, in a machine-readable artifact:

- kernel version and amdgpu/KFD module state;
- ROCm version and exact `llvm-readobj`/profiler tool paths;
- `/dev/kfd` and render-node ownership/group/ACL;
- KFD ioctl version result;
- profiler ioctl operation, arguments, errno, and whether the failure is
  permission, unsupported operation, or driver state.

Do not change kernel parameters, reset the GPU, install a new driver, or use
`sudo` as part of an agent run. If the ioctl requires an operator/system
change, stop and produce the exact requested change rather than guessing.

### 2. Fail-closed tinygrad runtime

Promote the existing behavior in the isolated campaign worktree:

- `PMC=0` must remain byte/behavior identical to ordinary inference.
- `PMC=1` must abort before queue submission when perfmon setup fails.
- The error must name the ioctl and tell the operator to disable PMC or fix the
  profiler interface.
- No partial counter rows may be emitted after setup failure.
- Add CPU-only tests for setup success/failure handling; do not mock a clean
  result as measured hardware data.

### 3. Eager PMC positive control

Only after package 1 succeeds, test the non-graph path with a tiny matmul and a
single known kernel:

- collect raw counter words and scheduler metadata;
- require at least one known-good nonzero counter and a valid completion;
- verify the counter blob size matches the schedule;
- run twice in separate child processes;
- stop immediately on a wait timeout, page fault, reset, or malformed blob.

This package must prove the ioctl and eager read path independently of decode.

### 4. Graph attribution, separately gated

Do not re-enable `PMC_GRAPH` for decode by default. First make graph capture
safe on a tiny graph, then on prefill, then on one decode token.

The graph implementation must:

- associate each read buffer slot with exactly one program dispatch;
- prove ordering between kernel completion, counter read, and buffer copyout;
- handle multiple graph runtimes and repeated dispatches;
- reject unsupported counter blocks instead of converting zero to a metric;
- emit a positive-control record for the hot packed GEMM, not only norm or
  elementwise kernels.

If graph reads cannot be made safe, leave graph PMC disabled and use the
observer/HSACO path for identity only.

### 5. Observer and code-object integration

Keep the safe launch observer as the fallback identity authority:

- sidecar records must include candidate, source identity, binary hash,
  dispatch order, grid, workgroup, submit, and completion timestamps;
- optional content-addressed HSACO dumping must remain opt-in;
- batched sidecar writes must not change launch semantics;
- incomplete or timed-out sidecars must fail closed in BoltBeam.

The resource audit must join binary metadata to observed geometry and report
VGPR, SGPR, LDS, spill counts, wave size, and kernel name without claiming
hardware occupancy.

### 6. BoltBeam integration

Add or extend schemas for:

- profiler preflight result;
- raw PMC capture with counter-quality status;
- graph dispatch attribution;
- observer/resource audit evidence.

Joins must be candidate-scoped and atomic. A failed profiler preflight,
missing hot-GEMM row, stale source hash, incomplete sidecar, or reset must
produce `inconclusive`/`blocked` evidence, never a clean result.

### 7. Minimal live comparison

After all positive controls pass, run only these fixed workloads:

- 14B at context 512 and 4096;
- 8B at context 4096;
- one process per run, identical environment, isolated worktree, and GPU lock.

Capture both observer identity and validated counters. Compare the same binary
sequence at 14B depth, then compare the G=5 and G=4 flash kernels. Report raw
counters before derived occupancy/cache metrics.

## Acceptance criteria

The task is complete only if all are true:

1. PMC setup failure is fail-closed and cannot submit a kernel.
2. A tiny eager positive control produces valid raw hardware counters twice.
3. A graph positive control attributes at least one hot GEMM, or graph PMC is
   explicitly rejected and remains disabled.
4. No run in the campaign causes a GPU reset.
5. BoltBeam rejects malformed, incomplete, stale, or counterless evidence.
6. The final report distinguishes measured counters, static HSACO resources,
   observer timing, and inference.

## Stop conditions

Stop the GPU campaign on any wait timeout, SQC/page fault, GPU reset, KFD
quiesce failure, malformed counter blob, or silent zero-counter result without
a positive control. Preserve the artifact and kernel log; do not rerun the
same configuration.

## Non-goals

- Do not enable PMC in production inference.
- Do not treat `SQ_BUSY_CYCLES` alone as occupancy or bandwidth.
- Do not infer a spill from workgroup size alone.
- Do not replace the existing timing authority with profiler overhead.
- Do not delete the safe observer or resource artifacts until this scope is
  closed and its evidence is promoted.

## Existing evidence to consume

- `gpu-counter-probe/DECODE-PMC-FAILURE.md`
- `gpu-counter-probe/RESOURCE-AUDIT-RESULTS.md`
- `gpu-counter-probe/resource-audit.json`
- `extra/qk/decode/decode_launch_probe.py`
- `extra/qk/decode/decode_resource_audit.py`

## Promotion and cleanup

Work in an isolated worktree. Promote only the fail-closed runtime, observer
batching/binary dump, resource audit, and validated BoltBeam schemas. Keep the
native PMC graph changes quarantined until the graph acceptance criteria pass.
After promotion, remove only probes proven redundant by the final report; keep
the positive controls and failure artifact as regression fixtures.

## Preflight result from initial execution

The KFD profiler ioctl failure is capability-gated, not a device-node ACL
failure. The ordinary `ubuntu` process has no effective `CAP_PERFMON` or
`CAP_SYS_ADMIN`; a narrowly scoped root process succeeded and returned nonzero
raw PMC words on an 8x8 matmul. The durable fix must therefore be a dedicated
capability-bounded profiling wrapper or operator-approved root invocation. Do
not grant capabilities to the system-wide Python interpreter. This does not
clear the separate `PMC_GRAPH` decode safety gate.

The isolated campaign worktree provides
`extra/qk/decode/run_pmc_privileged.py`, a capability-bounded launcher for
eager PMC diagnostics. It uses `sudo -n`, an isolated root `HOME`, preserves
only approved observation variables, and refuses `PMC_GRAPH=1` unless an
operator explicitly overrides the guard.
