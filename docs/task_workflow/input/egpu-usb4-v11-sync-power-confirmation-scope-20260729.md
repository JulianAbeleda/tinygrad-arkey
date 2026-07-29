# TinyGPU v11 synchronous power-confirmation reconciliation scope

Date: 2026-07-29

Status: implementation commit `544bde177`, clean-commit host-side verification,
and the single audited v11 installation are complete. Two post-install boots
occurred. v11 launched, started, and published on both, but on the current boot
the ACIO/PCIe link failed 78.447 seconds after publication and the tunneled tree
was removed before R6.1 ran. R6.1 stopped at the unavailable handshake. Do not
run the installer again. No reset, replug, AMD initialization, socket server,
A1, or workload is authorized by this source scope.

Repository/worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, branch
`exp`, starting HEAD `eaeff7ee89092e906d109e6d2bdeb2442fd83e7e`.

Primary runtime evidence:
`docs/task_workflow/output/egpu-usb4-v10-post-reboot-R6.1-20260729T022601Z.md`.

Post-v11 runtime evidence:
`docs/task_workflow/output/egpu-usb4-v11-post-reboot-R6.1-20260729T025229Z.md`.

## Post-reboot disposition

This subsection supersedes the future activation gate below. Boot history
records post-install boots at `2026-07-29T02:45` and `02:48`, rather than the
single reboot requested by the handoff. The first boot launched and published
v11 and retained the DEXT until that boot was shut down 205.055 seconds later;
no preserved R6.1 payload from that boot is admitted.

On the second boot, kernelmanagerd selected installed v11 unique ID
`0459ffba1fd9685605db1efbfa432ae2efa18e9bd84a87b55b8edb55092750a5`.
The DEXT started and published at `2026-07-29T02:48:59.369Z`. At
`02:50:17.816Z`, repeated ACIO Gen2/3 errors on both lanes preceded a zero-link
PCI rescan, removal of the ASM2464 and every AMD function, force-close of
`tinygpu`, and `stopUsingTunnel`. The locked R6.1 audit began later: boot,
registration, and native registration gates passed, but selector 4 returned
`keepalive handshake unavailable` with exit 3. It stopped without selector 5,
selector 10, `ioreg`, endpoint query, provenance finalization, or A1.

This result does not provide a pre-failure v11 power payload, so the synchronous
confirmation fix is not runtime-admitted. It does prove that the active v11
child did not prevent the current boot's upstream ACIO/PCIe removal. Stop this
hardware path. Do not retry, recover, or advance to A1 under this scope. Any
next hardware transition requires a separate scope; source-only follow-up
belongs at the ACIO/Thunderbolt tunnel-policy and physical-link boundary.

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

The clean verification and exact operational handoff are recorded in
`docs/task_workflow/output/egpu-usb4-v11-sync-power-confirmation-candidate-20260729T024038Z.md`.
