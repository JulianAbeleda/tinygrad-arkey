# TinyGPU v11 synchronous power-confirmation reconciliation scope

Date: 2026-07-29

Status: implementation and initial host-side verification complete. The
candidate is not installed. The currently active v10 DEXT remains fail-closed
in `active_degraded`; no reset, replug, AMD initialization, socket server, A1,
or workload is authorized by this source scope.

Repository/worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, branch
`exp`, starting HEAD `eaeff7ee89092e906d109e6d2bdeb2442fd83e7e`.

Primary runtime evidence:
`docs/task_workflow/output/egpu-usb4-v10-post-reboot-R6.1-20260729T022601Z.md`.

## 1. Proven runtime defect

v10 fixed the earlier startup failure: version 10 activated cleanly, started,
published `tinygpu`, negotiated capability 11, and sustained successful PCI
identity canaries. R6.1 nevertheless stopped because keepalive state remained
`active_degraded`.

The separately authorized selector-10 status then showed:

- `full_power_requested=true` and `power_request_accepted=true`;
- zero lifecycle-probe and request errors;
- `desired_power_flags=2` and `last_observed_power_flags=2`;
- one observed transition and a later successful `0x744c1002` canary;
- three exhausted request attempts; but
- `power_request_confirmed=false` and `publishable=false`.

The v10 ordering is defective when `ChangePowerState(On)` synchronously
delivers `SetPowerState(On)`. The callback computes confirmation before the
request call returns, while `powerRequestAccepted` still contains false. After
the return records success, v10 never reconciles the already-observed ordered
transition. Retries cannot rely on another callback when the service is already
On and eventually leave the provider permanently degraded.

This is an implementation defect in v10's evidence bookkeeping. It does not
show that DriverKit rejected the power desire or that the PCI canary failed.

## 2. Objective

Produce a narrowly scoped v11 candidate that:

1. treats both synchronous and asynchronous `SetPowerState(On)` delivery as
   valid only when the transition timestamp is later than the current request;
2. records API acceptance before reconciling a synchronous callback;
3. rejects stale pre-request On observations and any request error or release;
4. preserves the v2 status schema, policy, power mechanism, fail-closed
   workload gate, timer, teardown, and qualification predicates; and
5. advances the installer version, both DEXT project configurations, fixture
   states, and native/server build IDs together from v10 to v11.

## 3. Required implementation

Add one shared `FullPowerTransitionConfirmsRequest` predicate requiring:

- a full-power request and accepted zero-error API result;
- no release attempt;
- at least one observed transition;
- observed flags equal to `kIOServicePowerCapabilityOn`; and
- nonzero request time with `lastPowerTransition > lastPowerRequestTick`.

Before each request, mark acceptance and confirmation false and set the desired
flags. After `ChangePowerState` returns, record its error and acceptance, then
recompute confirmation from the shared predicate. `SetPowerState_Impl` must
update the observed flags and timestamp and use the same predicate. Final
readiness must require both the confirmation latch, the shared ordered
predicate, and a later successful canary.

The timestamp requirement is essential: simply accepting any previously
observed On state would turn a stale PM notification into evidence for a new
request.

## 4. Non-goals and safety boundary

- Do not add ASPM, PCI D0, PM assertions, reset, or replug behavior.
- Do not change `tinygpu.keepalive.v1`, `tinygpu.power-residency.v2`, command
  numbers, capability bits, or the `driverkit_full_power_v1` policy.
- Do not loosen workload/BAR/DMA/reset admission.
- Do not initialize AMD hardware, start the TinyGPU socket server, or run A1
  while implementing or building the candidate.
- Continue to serialize every eGPU observation and any future install through
  `extra/usbgpu/tools/with_gpu_lock.py` and `/tmp/gpu-bench.lock`.

## 5. Host-side acceptance

Before installation handoff:

1. The focused native, installer, server, wire, remote-protocol, and
   qualification tests pass, including source guards for both post-return and
   callback reconciliation and an atomic v11 identity test.
2. `git diff --check` passes.
3. A clean Xcode analyzer pass succeeds and every analyzer plist has zero
   diagnostics.
4. `install_nosip.sh --build` succeeds, strict signature verification passes,
   the DEXT reports version 11, and the native build IDs contain v11 but not
   v10.
5. The source and verification evidence are committed before the audited
   installer is permitted to bind provenance to branch HEAD.

## 6. Future activation gate

An audited v11 replacement requires the installer's literal and interactive
approval under the GPU lock. If macOS reports a pending upgrade, do not run the
installer again; reboot once.

On the first v11 boot, repeat R6.1 in order. The power table must now show one
or at most three attempts, an accepted and confirmed request, an On transition
strictly after the retained request time, a later canary, and
`publishable=true`. Registration, handshake, keepalive health, resource-zero,
and downgrade-zero predicates remain unchanged.

Only after R6.1 is entirely green may the host-only provenance-finalization
checkpoint and A1 be considered. Reset/replug, AMD initialization, the socket
server, and workloads remain outside this first diagnostic boot.
