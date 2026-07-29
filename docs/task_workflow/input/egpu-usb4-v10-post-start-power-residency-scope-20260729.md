# TinyGPU v10 post-Start power-residency implementation scope

Date: 2026-07-29

Status: source implementation and host-side verification complete. The v10
candidate passes 76 focused tests, a clean full analyzer pass, and a signed
build-only verification; it is uncommitted and uninstalled. This scope
supersedes the uninstalled v9 diagnostic-start candidate. It does not authorize
installing or activating a DEXT, rebooting macOS, changing NVRAM, resetting or
replugging the eGPU, initializing AMD hardware, or running A0/A1.

Repository/worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, branch
`exp`, starting HEAD `7f4974d3b`. Existing uncommitted v9 work and evidence are
inputs to this scope and must not be discarded.

Primary audit:
`docs/task_workflow/output/egpu-usb4-v9-power-management-api-audit-20260729T014837Z.md`.

## 1. Problem statement

The installed v8 DEXT failed `Start(display)` before publishing `tinygpu`.
Matching xnu source establishes that `SetPowerOverride(true)` was called before
DriverKit joined the service to the power-management tree, so it deterministically
returned `kIOReturnError`. The same call is also the wrong mechanism: it ignores
the calling service's power-plane children rather than asserting an upstream
full-power requirement.

The uninstalled v9 candidate makes that error observable, but it still issues
both power calls during `Start_Impl`, interprets the non-informative
`ChangePowerState` return as acceptance, and requires an override that can never
become active. It can therefore publish diagnostics but cannot become healthy.
The v9 identity must never be installed.

## 2. Objective

Produce a source-only v10 candidate that:

- starts and publishes diagnostics without making pre-PM-join power requests;
- requests `kIOServicePowerCapabilityOn` from the first real timer callback,
  after `Start_Impl` has returned and automatic PM-tree join can complete;
- treats `SetPowerState(On)` plus a later successful PCI identity canary as the
  evidence required for power residency;
- keeps workload, reset, configuration, MMIO, BAR, and DMA admission fail-closed
  until that evidence is present;
- records enough lifecycle evidence to falsify the PM-tree timing diagnosis in
  one future activation;
- guarantees timer, mappings, power desire, PCI session, and superclass Stop
  teardown even when resources are unexpectedly outstanding or timer draining
  fails; and
- advances all native and DEXT identities atomically from v9 to v10.

## 3. Required implementation

1. Remove all `SetPowerOverride(true)` use and all override-required readiness
   conditions.
2. Use `kIOServicePowerCapabilityOff`, not Low, to withdraw the service's power
   desire.
3. In `Start_Impl`, make no `ChangePowerState` request. Record a side-effect-free
   pre-join `SetPowerOverride(false)` lifecycle probe, then retain the existing
   identity, timer, first-canary, naming, and registration flow.
4. On the first timer callback for which the action is non-null, request full
   power. Permit bounded reassertion while the request has not been confirmed.
5. Record a post-join `SetPowerOverride(false)` probe on the first request. The
   probes are diagnostic only and never participate in readiness.
6. Confirm the request only after `SetPowerState` observes On. Require a
   successful canary timestamp strictly later than the request timestamp before
   `PowerResidencyReady` can become true.
7. Publish `tinygpu.power-residency.v2` with request attempts/timestamp, both
   lifecycle-probe results, request/release results, observed transitions, and
   fail-closed `publishable` state. Preserve the frozen keepalive v1 payload.
8. Restructure `Stop_Impl` so no early path skips teardown or
   `Stop(provider, SUPERDISPATCH)`. Outstanding-resource counts and drain errors
   become diagnostics rather than vetoes on mandatory teardown.
9. Correct the PowerResidencyStatus user-client branch layout without changing
   selector behavior.
10. Update source guards, wire fixtures/validators, qualification expectations,
    install-state fixtures, build IDs, installer version, and DEXT project
    version for v10.

## 4. Explicit design decisions

- `ChangePowerState(On)` is implementation-confirmed on the running xnu but is
  outside the documented DriverKit contract. The v2 status must expose observed
  confirmation rather than claiming the return value proves residency.
- Automatic post-Start PM-tree join is the primary path. Do not call
  `JoinPMTree()` from `Start_Impl` in this candidate.
- Do not add `SetASPMState` or `EnablePCIPowerManagement` in v10. Those are
  inference-based link-policy experiments and would confound whether the
  post-Start power desire alone holds the USB4/ACIO tunnel.
- Do not add `CreatePMAssertion`; it does not address the observed awake-idle
  lifecycle and is unavailable before PM join.
- The first inline keepalive call from `Start_Impl` remains an identity canary
  only. It must not assert power because its action is null and Start has not
  returned.
- A successful API return is recorded as request acceptance only. It is never
  sufficient for readiness without the callback and post-request canary.

## 5. Safety and non-regression invariants

- Diagnostics remain available in `active_degraded`.
- All hardware-mutating and provider-resource RPCs remain unavailable until
  `PowerResidencyReady` is true.
- No hardware command is run while implementing or verifying this scope.
- Any future eGPU observation must use `/tmp/gpu-bench.lock` through
  `extra/usbgpu/tools/with_gpu_lock.py`; none is authorized here.
- Preserve unrelated worktree changes and the three existing v8/v9 evidence
  documents.
- Do not commit the candidate or install it without separate user direction.

## 6. Host-side acceptance

Before handoff:

1. Focused native, server, wire, remote-protocol, qualification, and installer
   tests pass.
2. `SetPowerOverride(true)` is absent; `ChangePowerState(On)` is absent from
   `Start_Impl` and present in the post-Start timer path.
3. `PowerResidencyReady` requires observed On and a post-request successful
   canary and contains no override predicate.
4. `Stop_Impl` has no pre-teardown return and always calls superclass Stop.
5. `xcodebuild ... clean analyze` succeeds with zero DriverKit analyzer
   diagnostics.
6. `install_nosip.sh --build` succeeds, the signed app verifies, and the DEXT
   reports version 10. This is a build-only action, not installation.
7. `git diff --check` passes and a source/build evidence document records exact
   commands, results, hashes, and remaining uncertainty.

## 7. Future activation gate

Installation and reboot require separate explicit approval. After a future
activation, the service must publish and the lifecycle probes, transition
callback, observed power flags, post-request canary, and `active_healthy` state
must match the audit's falsifiable predictions before A1 is attempted. A1 alone
can determine whether the DEXT child's full-power desire propagates far enough
to prevent removal of the upstream USB4/ACIO tunnel. No AMD initialization,
reset, replug, or workload is part of that first activation gate.

## 8. Source-only outcome

The implemented v10 candidate follows the required flow and intentionally omits
the speculative ASPM and PCI D0 controls. `tinygpu.power-residency.v2` preserves
degraded diagnostics while qualification and workload admission require the
expected lifecycle probes, an accepted request, a later On callback, and a
successful post-request identity canary. The build reports DEXT version 10 and
the native build IDs `tinygrad-arkey-native-v10` and `tinygrad-arkey-v10`.

Host-side verification and binary hashes are recorded in
`docs/task_workflow/output/egpu-usb4-v10-post-start-power-residency-candidate-20260729T020547Z.md`.
