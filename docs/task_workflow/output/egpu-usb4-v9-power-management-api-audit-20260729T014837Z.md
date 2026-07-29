# TinyGPU DriverKit power-management API audit and v10 implementation spec

Collected: 2026-07-29T01:48:37Z

Status: read-only source/API audit. No file in the repository was modified to
produce this document other than adding this document. The DEXT was not built,
installed, or activated; no reboot, reset, replug, AMD initialization, or A0/A1
run occurred. Existing uncommitted v9 changes are untouched.

Audit inputs:

- Branch `exp`, head `7f4974d3b`; installed v8 source commit
  `4e821c5d4a151024537a5cba5814e7ac7f35e528`.
- `docs/task_workflow/output/egpu-usb4-v8-post-reboot-R6.1-20260729T011847Z.md`
- `docs/task_workflow/output/egpu-usb4-v9-diagnostic-start-candidate-20260729T012600Z.md`
- `docs/task_workflow/input/egpu-usb4-tinygpu-runtime-initialization-scope-20260728.md`
- Local DriverKit 25.5 and macOS SDK headers.
- Apple open source `xnu` at tag `xnu-12377.121.6`, which matches the running
  kernel exactly (`uname -v` reports `xnu-12377.121.6~2/RELEASE_ARM64_T8132`,
  macOS 26.5 build 25F71). Kernel citations below are therefore the real
  implementation of the kernel this machine runs, not an approximation.

Throughout this document, **documented** means stated in an Apple header or
official documentation, and **implementation-confirmed** means read directly
from the matching xnu source tag. Anything else is marked **inference**.

---

## 1. Executive summary

v8 failed `Start` for a deterministic, hardware-independent reason:
`SetPowerOverride(true)` is unconditionally illegal inside `Start_Impl`, always
returns `kIOReturnError`, and v8 treated that as fatal.

v9 correctly stops treating it as fatal, but keeps three defects:

1. `SetPowerOverride` is the wrong API for keeping a provider powered. It
   suppresses the caller's **children's** power desires. TinyGPUDriver has no
   PM children. Applied successfully it *lowers* power rather than raising it.
2. `ChangePowerState` returns `kIOReturnSuccess` unconditionally for any valid
   argument, discarding the underlying error. v9's recorded
   `power_request_error: 0` is therefore a fabrication, not evidence.
3. `PowerResidencyReady` still requires `powerOverrideActive`, which can never
   be true, so v9 would publish a service permanently stuck in
   `active_degraded` with every workload path closed forever.

The corrective build (v10) removes `SetPowerOverride` entirely, moves
`ChangePowerState(kIOServicePowerCapabilityOn)` to the first post-`Start` timer
tick where it is legal, confirms residency only from the `SetPowerState`
callback corroborated by a live canary read, and makes `Stop` incapable of
skipping teardown. It also carries two zero-side-effect lifecycle probes so a
single install/reboot both applies the fix and proves the root-cause theory.

Whether this is *sufficient* to hold the USB4/ACIO tunnel awake-idle is
genuinely unknown and only A1 can answer it. The proposed build answers that
question in the same boot.

---

## 2. Root cause of the v8 `Start` failure

`TinyGPUDriver.cpp:160` calls `owner->SetPowerOverride(true)` from within
`Start_Impl`. That is a synchronous RPC into the kernel. The chain is:

1. `IOService::SetPowerOverride_Impl(true)` calls `powerOverrideOnPriv()`.
   *xnu `iokit/Kernel/IOUserServer.cpp:5307-5320`*
2. `powerOverrideOnPriv()` begins:
   `if (!initialized) { return IOPMNotYetInitialized; }`
   *xnu `iokit/Kernel/IOServicePM.cpp:2759-2766`*
3. `initialized` is set **only** by `IOService::PMinit()`.
   *xnu `IOServicePM.cpp:351, 530`*
4. `PMinit()` is called **only** from `IOUserServer::serviceJoinPMTree()`.
   *xnu `IOUserServer.cpp:4956`*
5. `serviceJoinPMTree()` is reached from `IOUserServer::serviceStarted()` only
   on the `result == true` branch, i.e. **after `Start` has already returned
   successfully**. *xnu `IOUserServer.cpp:5931-5937`*
6. `SetPowerOverride_Impl` flattens any non-`IOPMNoErr` result to
   `kIOReturnError`. *xnu `IOUserServer.cpp:5319`*

During `Start_Impl` the DEXT service has not joined the power-management tree,
so `initialized == false`, so `SetPowerOverride(true)` returns
`kIOReturnError` = `0xE00002BC` = `-536870212` as `int32_t`
(`DriverKit.sdk/.../IOReturn.h:83,124,129`), 100% of the time.

v8's `RequestPowerResidency` did `if (overrideError) return overrideError;` and
`Start_Impl` did `if (err) goto fail`, so `Start` returned that error. The
kernel then logged it with the exact format string observed:

```c
DKLOG(DKS "::start(" DKS ") %s\n", DKN(service), DKN(provider), result ? "ok" : "fail");
```
*xnu `IOUserServer.cpp:5916`*

which is `DK: TinyGPUDriver-0x100000c0a::start(display-0x100000aa2) fail`.

This explains every observed symptom: no crash, two idle threads, service never
published, diagnostics unreachable. v7 lacked these calls and published
successfully; v8 added them. The regression is exactly the power calls.

### Second, independent defect in the same function

Even with correct lifecycle, `SetPowerOverride` is the wrong primitive. Apple's
kernel header:

> `powerOverrideOnPriv` — Allows a driver to ignore its **children's** power
> management requests and only use `changePowerStateToPriv` to define its own
> power state. ... a driver may ensure a **lower** power state than otherwise
> required by itself and its children ... Turning on the override will initiate
> a power change if the driver's `changePowerStateToPriv` desired power state is
> different from the maximum of the `changePowerStateTo` desired power state and
> the children's desires.
>
> *`MacOSX.sdk/.../Kernel.framework/Headers/IOKit/IOService.h:2011-2016`*

Implementation-confirmed: with `fDeviceOverrideEnabled` set,
`computeDesiredState()` skips the `gIOPMPowerClientChildren` and
`gIOPMPowerClientDriver` clients and pins the service to the
`gIOPMPowerClientDevice` desire.
*xnu `IOServicePM.cpp:2874-2884`*

TinyGPUDriver has no PM children, so the override removes nothing it wants
removed. Called before any `ChangePowerState`, the Device desire is `0`, so a
*successful* `SetPowerOverride(true)` would pin the service to power state 0 —
the opposite of the goal. The call is wrong in kind, wrong in order, and wrong
in lifecycle position.

---

## 3. API reference table

| API | Correct receiver | Lifecycle precondition | Valid arguments | Synchronous? | Return behavior | Evidence |
|---|---|---|---|---|---|---|
| `ChangePowerState(uint32_t)` | `this` (the DEXT's own service). Never the provider. | Service must have joined the PM tree (`PMinit` done). Pre-join it is a **silent no-op**. | Documented: `kIOServicePowerCapabilityLow` only. Implementation-confirmed: `Off(0x0)→changePowerStateToPriv(0)`, `Low(0x10000)→(1)`, `On(0x2)→(3 = kUserServerMaxPowerState)`. Anything else → `kIOReturnBadArgument`. `LPW(0x20000)` is **not** accepted. | **No.** `requestPowerState` queues a PM request and returns. | **Always `kIOReturnSuccess` for a valid argument.** The return of `changePowerStateToPriv()` is discarded. Carries zero information about whether the request was applied. | Decl/doc `IOService.iig:263-272`, enum `:53-58`; impl xnu `IOUserServer.cpp:5287-5305`, `IOServicePM.cpp:2465-2477, 2609-2650` |
| `SetPowerOverride(bool)` | `this`, and only if the service has PM children whose desires must be ignored. Not applicable here. | Same PM-tree precondition. Pre-join → `kIOReturnError`. | `true` / `false` | Flag set in-gate or PM request queued; resulting power change is async. | `kIOReturnSuccess` iff `IOPMNoErr`, else `kIOReturnError`. All distinct failures flattened to one opaque code. | Decl/doc `IOService.iig:274-282`; impl xnu `IOUserServer.cpp:5307-5320`, `IOServicePM.cpp:2759-2792`; semantics `Kernel.framework/.../IOService.h:2011-2024` |
| `SetPowerState(uint32_t)` (driver override) | Implemented by the driver, **called by** DriverKit. | Delivered on the driver's **default dispatch queue**, only after PM-tree join. | Receives the target state's `capabilityFlags`: `0` / `kIOPMPowerOn(0x2)` / `kIOPMLowPower(0x10000)` / `kIOPMAOTPower(0x20000)`, numerically identical to `kIOServicePowerCapability*`. | Driver makes itself safe then calls super, which acknowledges. | Driver returns `kIOReturnSuccess`; super performs the ack. | Decl/doc `IOService.iig:249-261`; dispatch xnu `IOServicePM.cpp:4279-4281` → `IOUserServer.cpp:5021-5100`; flags `IOPM.h:102,104` |
| `JoinPMTree()` | `this` | Requires `uvars->userServer`, set at instantiation, so it **is** callable from `Start_Impl`. Idempotent; re-entry returns success once `userServerPM` is set. | none | Synchronous RPC; resulting `registerPowerDriver` work is async. | `kIOReturnNotReady` if uvars missing, else propagates `serviceJoinPMTree`. | Doc `IOService.iig:240-247`; impl xnu `IOUserServer.cpp:5250-5256, 4924-4977` |
| `RegisterService()` | `this` | Safe anywhere. If called before `started`, the kernel **defers** it and re-issues from `serviceStarted` after PM join with `kIOServiceDextRequirePowerForMatching`. | none | Async matching. | `kIOReturnSuccess` | Doc `IOService.iig:162-167`; impl xnu `IOUserServer.cpp:368-390, 5949` |
| `CreatePMAssertion(bits,&id,synced)` | `this` | **Explicitly rejected before PM join:** `if (!userServerPM) return kIOReturnNotReady;` | `kIOServicePMAssertionCPUBit`, `kIOServicePMAssertionForceFullWakeupBit`; CPU bit only when `synced`. | Async effect; `synced=true` returns `kIOReturnBusy` if sleep is irreversible. | `Success` / `NotReady` / `Busy` / `InternalError` | Doc `IOService.iig:566-580`; impl xnu `IOUserServer.cpp:5152-5180` |
| `IOPCIDevice::SetASPMState(bits)` | `ivars->pci` | Open session (already held). No PM-tree precondition. | `kIOPCILinkControlASPMBitsDisabled(0)`, `...L0s`, `...L1`, `...L0sL1` | Applies to device and its upstream bridge. | `kIOReturnSuccess` if no errors. | `IOPCIDevice.iig:553-564`; `IOPCIFamilyDefinitions.h:500-505` |
| `IOPCIDevice::EnablePCIPowerManagement(state)` | `ivars->pci` | Open session. | `kPCIPMCSPowerStateD0(0)` **disables** PCI bus PM; `0xffffffff` = device decides. | Synchronous config write. | `kIOReturnSuccess` if no errors. | `IOPCIDevice.iig:448-456`; `IOPCIFamilyDefinitions.h:363` |
| `RequireMaxBusStall(ns)` | `this` | none documented | `kIOMaxBusStall*` from `IOTypes.h` | — | `kIOReturnSuccess` | `IOService.iig:343-362` |

**The single most important row is `ChangePowerState`.** It cannot report
failure. Pre-join, `requestPowerState` returns `kIOPMNotYetInitialized`
(`IOPM.h:568`, value `8`) and nobody reads it; the DEXT still sees
`kIOReturnSuccess`.

---

## 4. Findings

### 4.1 Confirmed defects

**C1 — `SetPowerOverride(true)` in `Start_Impl` always fails; this is the v8
`start ... fail`.**
`TinyGPUDriver.cpp:160, 241`. Chain proven in section 2. Deterministic, not a
race, not hardware-dependent.

**C2 — `SetPowerOverride` is the wrong API for this goal at any lifecycle
point.** It suppresses children's desires on the calling service. TinyGPUDriver
has none. Successfully applied, it lowers rather than raises residency.
`Kernel.framework/.../IOService.h:2011-2016`; xnu `IOServicePM.cpp:2874-2884`.

**C3 — Ordering is inverted even on the theory the code is written to.**
`TinyGPUDriver.cpp:160` (override) precedes `:166` (state request). Enabling the
override immediately initiates a power change against the current
`changePowerStateToPriv` desire, which at that moment is `0`. The repository
currently **pins this ordering in a test**:
`test/unit/test_tinygpu_native_source.py:74` asserts `SetPowerOverride(true)`
appears before `ChangePowerState(...)`. The test guards the bug.

**C4 — v9's "record both errors" produces actively misleading evidence.**
`TinyGPUDriver.cpp:157-173` will record
`override_request_error = -536870212` and `power_request_error = 0`,
`power_request_accepted = true`. The second is a fabrication:
`ChangePowerState_Impl` returns success unconditionally while
`changePowerStateToPriv` returned `kIOPMNotYetInitialized` and did nothing. An
operator reading `tinygpu.power-residency.v1` would conclude the state request
worked. Preserving the *first* error is not diagnostically sound here, because
the second call's return value is structurally incapable of being an error.

**C5 — v9 as written can never leave `kActiveDegraded`; all workload paths stay
closed forever.**
`PowerResidencyReady` (`:142-146`) requires `powerOverrideActive`, which is only
set when `SetPowerOverride(true)` succeeds (`:163`), which cannot happen (C1).
It also requires `!powerOverrideRequestError`, which will always be non-zero.
Therefore `AcquireWorkloadLease` (`:572`), `ResetDevice` (`:455`), and every
`kActiveHealthy`-gated path (`CfgRead`, `CfgWrite`, `MMIORead`, `MMIOWrite`,
`MapBar`, `SetupDMA`) remain closed permanently. v9 buys observability at the
cost of a service that can never do anything. That may be acceptable as a
one-shot probe, but it should be a stated intent rather than an emergent
property.

**C6 — `Stop_Impl` has two paths that skip all teardown including
`Stop(provider, SUPERDISPATCH)`.**
`TinyGPUDriver.cpp:293` (`kIOReturnBusy` when leases/BARs/DMA are outstanding)
and `:294-298` (drain error). Neither releases power residency, closes the PCI
session, nor calls super. Termination proceeds regardless; returning an error
from `Stop` does not veto teardown. Power override release is therefore **not**
guaranteed on every applicable path. `free()` (`:216-225`) also does not and
cannot release it.

**C7 — Release uses `Low` rather than `Off`, leaving a residual desire.**
`kReleasedPowerFlags = kIOServicePowerCapabilityLow` (`:20`) maps to
`changePowerStateToPriv(1)`, leaving a permanent floor at `kIOPMLowPower` on
`gIOPMPowerClientDevice`. To actually withdraw the desire the argument must be
`kIOServicePowerCapabilityOff` → `changePowerStateToPriv(0)`. Low severity at
`Stop`; wrong in the `Start`-failure path where the object may linger.

### 4.2 Probable defects

**P1 — v9 can publish while holding a successful override with no state
desire.** If a later build fixes lifecycle but keeps the override, the branch
"`SetPowerOverride(true)` succeeds, `ChangePowerState` fails" leaves the service
pinned to Device desire 0 while published and while the 1 Hz keepalive keeps
issuing `ConfigurationRead32` against a device whose parent may have dropped.
v9's non-fatal change makes this branch reachable where v8 would have bailed.
Attempting both calls unconditionally is not merely uninformative; in that
branch it is unsafe.

**P2 — `desiredPowerFlags` is never updated** (`:61`, `:674`). After
`ReleasePowerResidency`, the JSON still reports `desired_power_flags: 2`.
Diagnostic inaccuracy in exactly the report an operator will be reading.

**P3 — Lock-ordering hazard: the driver holds its own gate across synchronous
PM RPCs.** `:241` runs `RequestPowerResidency` inside `gate->RunAction`, and
`SetPowerState_Impl:329` re-enters the same gate via `DispatchSync` from the
default queue. This is currently safe only because `ChangePowerState` returns
before any PM work runs. Once the calls take effect it becomes a real
cross-domain wait: the default queue blocked on a gate held by a thread doing a
kernel RPC. Not a deadlock today; one refactor away from being one.

**P4 — Brace/indent mis-nesting in the user client.**
`TinyGPUDriverUserClient.cpp:272-292`: the `PowerResidencyStatus` branch is
indented one level deeper and its closing `}` at `:292` disagrees with the
surrounding `else if` chain. It parses and dispatches correctly today, but it is
easy to break. Worth fixing in the same pass.

### 4.3 Unresolved uncertainty

**U1 — Whether `ChangePowerState(On)` on the DEXT service is *sufficient* to
hold a USB4/ACIO tunnel awake-idle.** Documented upward propagation says a power
child's desire raises the parent:

> Three things affect driver power state: `changePowerStateTo`,
> `changePowerStateToPriv`, and the desires of the driver's power plane
> children. Power management puts the device into the maximum state governed by
> those three entities.
>
> *`Kernel.framework/.../IOService.h:2001-2009`*

The DEXT joins under the nearest `gIOPowerPlane` ancestor, the `IOPCIDevice`
(xnu `IOUserServer.cpp:4962-4970`), so the mechanism exists. But the v7 symptom
was ACIO link loss and removal of the tunneled PCIe tree, which is a
Thunderbolt-controller-level event. Whether the PCI child's D0 desire is enough
to veto tunnel teardown is not documented and not verifiable from readable
source. **Only A1 can answer this.**

**U2 — Why v7 lost residency at all (inference).** After `registerPowerDriver`,
`adjustPowerState(tempDesire)` (xnu `IOServicePM.cpp:1379`) passes `tempDesire`
as a *local clamp* only; `computeDesiredState` applies it to `newPowerState` but
never to `gIOPMPowerClientDevice` / `fDeviceDesire`
(xnu `IOServicePM.cpp:2905-2919`). So a DEXT that never calls
`ChangePowerState` has a Device desire of `0`, and any later recomputation can
settle the service at state 0, withdrawing its demand on the `IOPCIDevice`. If
that is what happened, `ChangePowerState(kIOServicePowerCapabilityOn)` is
exactly the missing call; it just has to be made where it works.

**U3 — Documented-versus-implemented divergence for `ChangePowerState(On)`.**
The header documents only `kIOServicePowerCapabilityLow`
(`IOService.iig:263-272`). The kernel does accept `On` and maps it to
`kUserServerMaxPowerState`. This is real and current, but outside the documented
contract and could be narrowed without notice. Do not treat "it compiles" or
even "it works" as a contract. Corroborating evidence that Apple expects DEXT
services to sit at `kUserServerMaxPowerState`: on wake, `IOUserServer` restores
`changePowerStateWithOverrideTo(kUserServerMaxPowerState)` for `userServerPM`
services (xnu `IOUserServer.cpp:5781, 5804-5811`).

---

## 5. Recommended v10 control flow

Design principle: make every PM call at a point where it is legal; make the
driver's health depend on observed evidence rather than a return code that
cannot fail; never let teardown skip `super::Stop`.

```
Start_Impl(provider):
    Start(provider, SUPERDISPATCH)                 -> fail fast
    pci = OSDynamicCast(IOPCIDevice, provider)     -> kIOReturnNoDevice
    IODispatchQueue::Create(gate)                  -> fail fast
    gate->RunAction{ pci->Open(this, 0) }          -> fail fast

    state = kStarting; generation = NextGeneration()

    // Lifecycle probe A: proves the PM boundary, zero power side effects.
    // Expected: kIOReturnError (-536870212) because PM is not yet initialized.
    ivars->overrideProbePreJoinError = SetPowerOverride(false)

    // Optional, documented-safe link knobs (no PM-tree precondition):
    ivars->aspmError  = gate->RunAction{ pci->SetASPMState(kIOPCILinkControlASPMBitsDisabled) }
    ivars->pciPMError = gate->RunAction{ pci->EnablePCIPowerManagement(kPCIPMCSPowerStateD0) }

    // NO ChangePowerState here. NO SetPowerOverride(true). Ever.

    identity read / timer create / SetHandler / SetEnable / first canary   (unchanged)
    SetName("tinygpu"); RegisterService()          // kernel defers until post-join
    return kIOReturnSuccess

fail:                                              // unchanged, plus:
    ReleasePowerResidency(...)                     // now a no-op unless something was requested
```

```
KeepaliveTimer(action, time):        // action == nullptr means "called inline from Start"
    ... existing canary logic ...

    if (action != nullptr):          // guaranteed post-Start, therefore post-PM-join
        if (!fullPowerRequested) or
           (powerRequestAccepted and !powerRequestConfirmed and
            powerRequestAttempts < kMaxAssertAttempts):
              RequestPowerResidency(this, ivars)

    RefreshProviderHealth(ivars)
```

```
RequestPowerResidency(owner, state):               // SetPowerOverride(true) removed entirely
    ++state->powerRequestAttempts
    state->lastPowerRequestTick = UptimeNS()
    state->fullPowerRequested = true
    err = owner->ChangePowerState(kIOServicePowerCapabilityOn)
    state->powerRequestError    = err
    state->powerRequestAccepted = (err == kIOReturnSuccess)
    state->desiredPowerFlags    = kIOServicePowerCapabilityOn

    // Lifecycle probe B (first attempt only): expected kIOReturnSuccess now.
    if (state->powerRequestAttempts == 1)
        state->overrideProbePostJoinError = owner->SetPowerOverride(false)

    if (state->powerTransitions and state->lastObservedPowerFlags == kFullPowerFlags)
        state->powerRequestConfirmed = true
    return err
```

```
PowerResidencyReady(state):                        // override terms removed
    return  state->fullPowerRequested
        and state->powerRequestAccepted
        and !state->powerRequestError
        and !state->powerReleaseAttempted
        and state->lastObservedPowerFlags == kIOServicePowerCapabilityOn
        and state->powerRequestConfirmed
        and state->lastSuccess > state->lastPowerRequestTick     // canary corroboration
```

```
ReleasePowerResidency(owner, state):
    if (state->powerReleaseAttempted) return success
    state->powerReleaseAttempted = true
    state->powerRequestConfirmed = false
    if (state->powerOverrideActive)                              // dead code once override is gone
        state->powerOverrideReleaseError = owner->SetPowerOverride(false)   // FIRST
    if (state->powerRequestAccepted)
        state->powerReleaseError = owner->ChangePowerState(kIOServicePowerCapabilityOff)  // Off, not Low
    state->desiredPowerFlags = kIOServicePowerCapabilityOff
    return first non-zero
```

```
Stop_Impl(provider):
    // No path may return before super::Stop.
    gate->DispatchSync{ record outstanding leases/bars/dma into ivars->stopBusy*;
                        state = kQuiescing; timerEnabled = false }
    drainError = DrainTimer(gate, timer, true)
    gate->DispatchSync{ if (drainError) ivars->timerError = drainError }
    release timer; Cancel + release timerAction
    gate->DispatchSync{
        ReleaseMMIOMappings(ivars)
        releaseError = ReleasePowerResidency(this, ivars)
        if (pci) { pci->Close(this, 0); pci = nullptr }
        state = kStopped
    }
    stopError = Stop(provider, SUPERDISPATCH)       // ALWAYS reached
    return stopError ?: releaseError ?: drainError
```

### Admission gating

Unchanged in shape and already correct:

- Diagnostics (`GetKeepaliveStatus`, `GetPowerResidencyStatus`, handshake)
  require no residency and stay reachable in `active_degraded`.
- `AcquireWorkloadLease` and `ResetDevice` require `PowerResidencyReady`.
- Every MMIO / config / BAR / DMA path requires `kActiveHealthy`, which
  `RefreshProviderHealth` grants only when `PowerResidencyReady` holds.

The only change is that `PowerResidencyReady` becomes *achievable*.

### Rejected alternative and why

Calling `JoinPMTree()` from `Start_Impl` would make the PM calls legal
immediately. `JoinPMTree_Impl` needs only `uvars->userServer`, which exists
pre-`Start` (xnu `IOUserServer.cpp:5250-5256`), and re-entry from
`serviceStarted` is idempotent via the `userServerPM` early return
(`IOUserServer.cpp:4930`). It is tempting and probably safe.

It is rejected as the primary path because it is an undocumented use of an API
whose header states that matched services join automatically, and because it
introduces a synchronous RPC that triggers `registerPowerDriver` while the
default queue is occupied by `Start`. With a one-install budget, the
timer-deferred path achieves the same result using only documented lifecycle
guarantees. Keep `JoinPMTree()` in reserve if Phase 2 evidence shows the join is
not happening.

---

## 6. Patch plan

All paths relative to `extra/usbgpu/tbgpu/installer/` unless noted. Items 1-8,
10, and 11 are required. Items 9 and 12 are strongly recommended. The optional
`SetASPMState` / `EnablePCIPowerManagement` calls inside item 6 are
**inference-driven** mitigations for the ACIO symptom, not documented fixes for
it; they are cheap, reversible, and independently recorded, so a single install
can test them alongside the PM fix.

| # | File / symbol | Change |
|---|---|---|
| 1 | `TinyGPUDriverExtension/TinyGPUDriver.cpp:20` `kReleasedPowerFlags` | `kIOServicePowerCapabilityLow` → `kIOServicePowerCapabilityOff` |
| 2 | `:41-45, :70` ivars | Delete `powerOverrideRequested`, `powerOverrideActive`. Add `powerRequestAttempts`, `lastPowerRequestTick`, `overrideProbePreJoinError`, `overrideProbePostJoinError`, `aspmError`, `pciPMError`, `stopBusyLeases`, `stopBusyBars`, `stopBusyDMA` |
| 3 | `:142-146` `PowerResidencyReady` | Drop override conjuncts; add `lastSuccess > lastPowerRequestTick` |
| 4 | `:157-173` `RequestPowerResidency` | Remove `SetPowerOverride(true)`. Add attempt counter and tick. Keep the post-join `SetPowerOverride(false)` probe on attempt 1 only. Return `ChangePowerState`'s error. Set `desiredPowerFlags` |
| 5 | `:175-192` `ReleasePowerResidency` | Override-off **before** state change; `Off` instead of `Low`; update `desiredPowerFlags` |
| 6 | `:227-264` `Start_Impl` | Remove the `RequestPowerResidency` call and its `IOLog`. Add the pre-join `SetPowerOverride(false)` probe and the two optional `IOPCIDevice` link knobs. Leave identity/timer/canary/publish logic intact |
| 7 | `:335-373` `KeepaliveTimer` | Add the `action != nullptr` re-assert block from section 5, immediately before the final `RefreshProviderHealth` |
| 8 | `:285-314` `Stop_Impl` | Remove both early `return`s; record busy counts and drain error as diagnostics; guarantee `Stop(provider, SUPERDISPATCH)` on every path |
| 9 | `:652-705` `GetPowerResidencyStatus` | Replace `override_requested` / `override_active` with `override_probe_prejoin_error`, `override_probe_postjoin_error`, `power_request_attempts`, `last_power_request_monotonic_ns`, `aspm_error`, `pci_pm_error`. Bump schema to `tinygpu.power-residency.v2` |
| 10 | `test/unit/test_tinygpu_native_source.py:74` (repo root) | **Delete the assertion pinning `SetPowerOverride(true)` before `ChangePowerState`.** Replace with: `SetPowerOverride(true)` absent from the driver; `ChangePowerState(kFullPowerFlags)` absent from `Start_Impl` and present in `KeepaliveTimer`; `Stop_Impl` contains exactly one `return` and it is the super call |
| 11 | `Shared/server.c:264, :452`; `install_nosip.sh:11`; `TinyGPUDriverExtension.xcodeproj/project.pbxproj` `CURRENT_PROJECT_VERSION` (both configs) | `v9` → `v10` on all four, together, so the v9 identity is never installed |
| 12 | `TinyGPUDriverExtension/TinyGPUDriverUserClient.cpp:272-292` | Re-indent and re-brace the `PowerResidencyStatus` branch to match the rest of the `else if` chain |

Also update the `capabilities` comment and any wire-spec / protocol tests that
assert on the `tinygpu.power-residency.v1` schema string if item 9 is taken.

### Explicitly out of scope for this patch

- Do not install, activate, or reinstall any DEXT.
- Do not reboot, reset, replug, or power-cycle the endpoint.
- Do not initialize the AMD GPU or run any workload, A0, or A1.
- Do not modify the acceptance matrix or the keepalive architecture.
- Do not add `CreatePMAssertion`; it is rejected before PM join
  (xnu `IOUserServer.cpp:5162-5164`) and the v7 loss was awake-idle, not sleep.

---

## 7. Validation plan (at most one install/reboot cycle)

### Phase 0 — host-side, no hardware, no install

1. Run the six focused suites. The updated
   `test_tinygpu_native_source.py` must fail before the patch and pass after;
   that is the point of item 10.
2. `xcodebuild ... clean analyze` → `ANALYZE SUCCEEDED`, zero diagnostics in all
   four DriverKit analyzer plists.
3. `bash extra/usbgpu/tbgpu/installer/install_nosip.sh --build` →
   `BUILD SUCCEEDED`, ad-hoc sign, deep/strict bundle verification.
4. Grep gate: `SetPowerOverride(true)` must not appear anywhere in the source;
   `ChangePowerState` must not appear inside `Start_Impl`; `Stop_Impl` must
   contain exactly one `return` statement.
5. Commit the clean candidate and record source and build hashes so the
   installer's strict provenance gate passes.

### Phase 1 — the single install and reboot

Optionally set `sudo nvram boot-args="dk=0x9"` **before** the reboot. That is
`kIODKEnable | kIODKLogPM`
(`Kernel.framework/.../IOKit/IOKitDebug.h:135, 138`), which makes the kernel
emit `DKS::setPowerState(...)`, `DKS::serviceSetPowerState(...)`, and
`changePowerStateWithOverrideTo(...)` lines, giving an independent kernel-side
view of every transition. No sysctl equivalent is exposed on this host
(`sysctl -a | grep iodk` returns nothing), so this rides the reboot already
being spent. Remove the boot-arg afterwards.

### Phase 2 — falsifiable predictions from the published service

The service must publish. Then `selector-10` must return:

| Field | Predicted | What a mismatch means |
|---|---|---|
| `override_probe_prejoin_error` | `-536870212` (`kIOReturnError`) | The PM-boundary root cause (C1) is **wrong**; re-open the analysis |
| `override_probe_postjoin_error` | `0` | PM join is not happening post-`Start`; fall back to `JoinPMTree()` in `Start` |
| `power_request_attempts` | `1` | `>1` means the first assert did not confirm; read `transition_count` |
| `power_request_error` | `0` | Non-zero means the argument was rejected; `On` is not accepted on this build |
| `transition_count` | `>= 1` | `0` means no `SetPowerState` callback ever arrived; PM join failed silently |
| `last_observed_power_flags` | `2` | `0` or `65536` means the service settled Off/Low despite the request |
| `power_request_confirmed`, `publishable` | `true` | `false` with `transition_count >= 1` means the canary-corroboration clause is the blocker |
| keepalive `state` | `active_healthy` | anything else blocks A0 |

These four probe fields settle C1, C4, C5, and U2 in a single boot, whether or
not the fix works.

### Phase 3 — A1 idle continuity, the actual open question

Only if Phase 2 is green. Hold the service idle with no client for the A1
duration and watch `success_gap_over_leeway_count`, `max_success_gap_ms`,
`unexpected_downgrade_count`, and `last_identity_dword`.

This is the only test that can answer U1. If the tunnel still drops with
`last_observed_power_flags == 2` and `unexpected_downgrade_count == 0`
throughout, then DriverKit `IOService` power management is not the right layer
for this problem, and the investigation should move to ACIO / Thunderbolt tunnel
policy. The PM work will have been necessary but not sufficient, and that will
be known definitively rather than by inference.

No reset, replug, AMD initialization, or workload in any phase.

---

## 8. Answers to the audit questions

1. **Correct object?** Yes, `this` is the only sensible receiver, but the call
   itself is the wrong API (C2). Calling it on `ivars->pci` would be worse: it
   would make the `IOPCIDevice` ignore *our* desire.
2. **Who should request it?** The TinyGPU service, on itself, via
   `ChangePowerState`. There is no DriverKit API to request power *on* a
   provider, and none is needed: IOKit propagates a child's desire upward.
3. **Is `ChangePowerState(On)` valid?** Implementation-confirmed yes
   (`→ changePowerStateToPriv(3)`); documented no. Use it, record it, treat it
   as undocumented (U3).
4. **Exact declaration and semantics.**
   `virtual kern_return_t ChangePowerState(uint32_t powerFlags);` —
   `IOService.iig:270-272`. Doc: "Allow provider to enter a low power state ...
   `kIOServicePowerCapabilityLow`" (`:263-269`). Implementation accepts
   Off/Low/On, rejects everything else with `kIOReturnBadArgument`, and returns
   `kIOReturnSuccess` unconditionally otherwise
   (xnu `IOUserServer.cpp:5287-5305`). The header's "provider" wording is loose;
   the call operates on the calling service's own PM node.
5. **Legal during `Start_Impl`?** No. Both require `PMinit`, which happens in
   `serviceJoinPMTree`, which `serviceStarted` invokes only after `Start`
   returns success (xnu `IOUserServer.cpp:5931-5937`). `SetPowerOverride`
   reports this as `kIOReturnError`; `ChangePowerState` hides it.
   `RegisterService()` during `Start` **is** legal; the kernel defers and
   re-issues it post-join.
6. **Synchronous?** No. Both queue PM requests. The only reliable confirmation
   is the `SetPowerState(kIOServicePowerCapabilityOn)` callback, whose argument
   is the target state's `capabilityFlags`
   (xnu `IOServicePM.cpp:4279-4281`). Corroborate with the keepalive
   `ConfigurationRead32` succeeding *after* the request.
7. **Missing registration or callback?** DriverKit exposes no power-state
   registration; the kernel registers a fixed four-state table on the DEXT's
   behalf (`sPowerStates`, xnu `IOUserServer.cpp:4899-4920`). The required
   callback is `SetPowerState`, which the driver already overrides and correctly
   forwards with `SUPERDISPATCH` (`TinyGPUDriver.cpp:332`). Nothing is missing;
   the timing is wrong.
8. **Failing because not yet in the PM tree?** Yes. That is the whole failure
   (C1).
9. **Smallest correct sequence?** Section 5.
10. **Ownership / teardown / race problems in v9?** C5, C6, C7, P1, P2, P3. The
    `NewUserClient_Impl` change (`:384-388`) is genuinely equivalent to the old
    `IO_FOR_ANALYZER(service.get()->release())` — that macro expands to `x` under
    the analyzer and to nothing otherwise
    (`DriverKit.sdk/.../IOLib.h:49-51`) — so the explicit `detach()`/`release()`
    is a strict improvement.
11. **Is attempting both and keeping the first error sound?** No. See C4 and P1.
12. **Release in `Stop`, guaranteed?** No. See C6.

---

## 9. Citation index

**Local DriverKit SDK** —
`/Applications/Xcode.app/Contents/Developer/Platforms/DriverKit.platform/Developer/SDKs/DriverKit.sdk/System/DriverKit/System/Library/Frameworks/`

- `DriverKit.framework/Headers/IOService.iig:53-58, 113-134, 162-167, 240-247, 249-261, 263-272, 274-282, 343-362, 566-589`
- `DriverKit.framework/Headers/IOService.h:621-646, 693-730, 860-900`
- `DriverKit.framework/Headers/IOReturn.h:83, 124, 129`
- `DriverKit.framework/Headers/IOLib.h:49-51`
- `PCIDriverKit.framework/Headers/IOPCIDevice.iig:436-456, 495-518, 537-564`
- `PCIDriverKit.framework/Headers/IOPCIFamilyDefinitions.h:363, 500-505`

**Local macOS SDK (Kernel.framework)** —
`/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Kernel.framework/Headers/IOKit/`

- `IOService.h:1644-1650, 1716-1723, 1733-1743, 2001-2009, 2011-2016, 2018-2024`
- `pwr_mgt/IOPM.h:102, 104, 542, 568, 571, 578`
- `IOKitDebug.h:135-141, 153`

**Apple open source, tag `xnu-12377.121.6`** (byte-exact match to the running
kernel; `https://github.com/apple-oss-distributions/xnu`)

- `iokit/Kernel/IOUserServer.cpp:368-390, 4899-4920, 4924-4977, 5021-5100, 5152-5180, 5250-5256, 5278-5285, 5287-5305, 5307-5320, 5781, 5804-5811, 5910-5949`
- `iokit/Kernel/IOServicePM.cpp:351, 530, 1257-1385, 2440-2530, 2609-2650, 2759-2792, 2832-2930, 4262-4292`

**Repository (read-only)**

- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.cpp:18-20, 41-73, 142-146, 148-152, 157-173, 175-192, 227-283, 285-314, 316-333, 335-373, 375-394, 451-490, 568-579, 652-705`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.iig:24-28`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriverUserClient.cpp:272-303`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/Info.plist` (`IOPCITunnelCompatible`, `IOUserClass`)
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.NoSIP.entitlements`
- `test/unit/test_tinygpu_native_source.py:69-98`
- `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension.xcodeproj/project.pbxproj:445, 505, 530, 562`
