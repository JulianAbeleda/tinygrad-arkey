# USB4 eGPU persistent PCIe service scope

Date: 2026-07-27

Repository: `/Users/julianabeleda/env/tinygrad-arkey`

Status: input

Target host: Apple Silicon Mac mini M4

Target device: AMD Radeon RX 7900 XTX, PCI identity `1002:744c`, gfx1100

Target transport: Apple USB4 root complex -> UT4G/ASM2464-class PCIe tunnel -> PCIe riser -> GPU

## 1. Objective

Make the USB4-attached AMD GPU remain enumerated and usable while the Mac is awake, including when no tinygrad process is connected, without retaining workload DMA or BAR mappings between clients.

The implementation must move PCIe link-liveness ownership into the native TinyGPU stack, whose lifetime is tied to the PCI device, instead of relying on a short-lived Python inference process. It must behave like a narrow endpoint driver under an operating system PCI owner:

- own the endpoint for its device lifetime;
- perform the proven, harmless periodic PCI configuration read needed by this USB4 path;
- isolate persistent device/link ownership from temporary compute resources;
- serialize lifecycle transitions and destructive operations;
- expose enough status to prove that the keepalive is actually active;
- recover from ordinary client churn and hotplug without leaking resources;
- fail loudly, and never reset or power-cycle the GPU automatically, when the endpoint is gone.

This task is complete only after the native source is restored to an owned location, the installed application is proven to contain the implementation, awake-idle qualification passes beyond the historical failure window, and the canonical 8B tinygrad workload still runs after idle.

## 2. Why this is a driver-lifecycle problem

A motherboard does not keep a PCIe card alive by continuously running compute. The platform root complex and operating-system PCI stack retain ownership of the topology, manage link state, route transactions, and keep an endpoint driver bound while applications come and go.

This setup has the same layers, but the local TinyGPU implementation currently leaves a traffic gap:

```text
Mac M4 root complex
  -> Apple USB4 PCIe tunnel
    -> UT4G / ASM2464-class bridge
      -> PCIe riser
        -> RX 7900 XTX endpoint
          -> TinyGPU DriverKit extension
            -> TinyGPU.app Unix-socket server
              -> tinygrad APLRemotePCIDevice client
```

The root complex and macOS own enumeration and routing. The DriverKit extension binds to the endpoint and opens the `IOPCIDevice`. The application server creates a DriverKit user-client connection for a tinygrad client. Tinygrad maps BARs, prepares DMA, boots the AMD firmware/runtime, and submits work.

The missing behavior is not dummy GPU utilization. It is a small PCIe transaction during otherwise idle periods so this specific USB4 bridge chain does not enter a low-power state from which it has historically failed to retrain.

## 3. Evidence baseline

Treat the following as the starting evidence. Recheck identities at execution time, but do not reopen already-settled theories without contradictory data.

### 3.1 Hardware facts

- Replacing the old PCIe riser changed the GPU from non-enumerating to enumerating. The old riser was therefore a real signal-path blocker.
- With the new riser, macOS enumerated the RX 7900 XTX and TinyGPU opened it as `AMD gfx1100`.
- A minimal tinygrad compute test completed and returned `[2.0, 5.0, 10.0, 17.0]`.
- The UT4G connection has been observed at 40 Gb/s.
- Passing initialization and a small compute test does not prove long-idle stability, sustained-load power stability, or 8B model stability. Those are separate gates.

### 3.2 Proven idle failure

`docs/egpu-usb4-link-keepalive.md` records the prior failure and fix:

- while the Mac remained awake, the USB4/PCIe topology disappeared after an idle period;
- captures showed link errors around the ASPM/CLx idle transition and failed retraining;
- once the endpoint disappeared, software reinitialization was insufficient and a physical power cycle was required;
- a one-second PCI configuration-space read prevented the failure in the former Python remote bridge;
- commit `554800bef` is the reference implementation and evidence anchor.

The historical implementation read offset `0`, size `4`, once per second. That reads the vendor/device identity DWORD and creates a real PCIe configuration transaction without touching GPU workload state.

### 3.3 Current ownership gap

The current Python client is `APLRemotePCIDevice` in `tinygrad/runtime/support/system.py`.

- It starts `/Applications/TinyGPU.app/Contents/MacOS/TinyGPU server <socket>` on demand.
- It then connects over a Unix socket and creates workload resources through RPC.
- `REMOTE_KEEPALIVE_S=1` is not consumed by this Python class.
- The installed TinyGPU executable and the local Debug build executable have been checked for keepalive symbols and contain no `REMOTE_KEEPALIVE` or equivalent implementation marker.
- The installed executable matches the local build artifact by SHA-256, so the installed copy is not a separate newer keeper build.

The native source was pruned at commit `4c5e67cff`. The last local source immediately before that commit shows:

- `TinyGPUDriver::Start_Impl` opens and retains the `IOPCIDevice` until `Stop_Impl`;
- `run_server` persists across application clients;
- `handle_client` opens an `IOUserClient` connection only after a socket client connects;
- socket-client disconnect calls `cleanup()`;
- `cleanup()` unmaps BARs, unmaps and unlinks shared memory, closes DMA-related file descriptors, and calls `IOServiceClose(g_conn)`;
- while waiting in `accept()`, the app server has no DriverKit user-client connection and performs no configuration reads.

This lifecycle correctly frees workload resources, but it does not provide device traffic between tinygrad clients. Merely setting `REMOTE_KEEPALIVE_S=1` currently gives a false sense of protection.

### 3.4 Native API feasibility

Apple's DriverKit and PCIDriverKit APIs provide the required primitive at the layer that already owns the endpoint:

- `IOPCIDevice::ConfigurationRead32` synchronously reads PCI configuration space and returns all ones on error;
- `IOTimerDispatchSource` schedules a timer on an `IODispatchQueue`;
- `IOService::SetDispatchQueue` can assign a named serial queue to driver methods.

Primary references:

- <https://developer.apple.com/documentation/pcidriverkit/iopcidevice/configurationread32>
- <https://developer.apple.com/documentation/driverkit/iotimerdispatchsource/create>
- <https://developer.apple.com/documentation/driverkit/iotimerdispatchsource/wakeattime>
- <https://developer.apple.com/documentation/driverkit/ioservice/setdispatchqueue>

The implementation must still prove these APIs work in the project's signed DriverKit extension on the target macOS version. API availability alone is not runtime evidence.

## 4. Problem classification

Keep these failure classes separate throughout implementation and reporting.

| Class | Typical observation | Relevant owner | This task |
|---|---|---|---|
| Enumeration/signal failure | GPU never appears after plug-in | riser, cable, bridge, root complex | Observe only; new riser already changed this state |
| Awake-idle link loss | GPU works, then PCI identity disappears while idle | USB4/PCIe link power transition | Primary target |
| Load/power dropout | GPU disappears or resets under sustained compute | PSU, GPU power cabling, riser, thermal/load behavior | Qualify and classify; do not mislabel as idle failure |
| AMD firmware/runtime failure | PCI endpoint remains present but PSP/SMU/GFX init fails | AMD runtime and device state | Preserve evidence; separate from link presence |
| Application lifecycle failure | socket/server dies but PCI endpoint remains | TinyGPU.app transport | Repair without touching hardware |
| Actual system sleep loss | device fails across Mac sleep/wake | macOS/USB4 sleep policy | Secondary observation, not the initial acceptance gate |

A keeper cannot compensate for an undersized or unstable PSU, a bad power lead, an electrically marginal riser, or a bridge that disappears under load. Conversely, a successful load test does not prove the 40-minute idle transition is fixed.

## 5. Goals

1. Keep `1002:744c` enumerated for the entire awake-idle qualification window.
2. Tie periodic link traffic to the DriverKit device lifecycle, not a tinygrad process lifecycle.
3. Keep persistent state limited to the PCI provider, timer, policy, and health counters.
4. Give every workload connection fresh BAR mappings, DMA preparations, shared-memory objects, and socket state.
5. Ensure workload exit immediately releases all workload resources without stopping link management.
6. Serialize keepalive, stop, and reset transitions so no timer callback touches a closing provider.
7. Make installed-binary identity and keeper status queryable and recordable.
8. Preserve compatibility with ordinary tinygrad AMD-over-USB operation.
9. Validate minimal compute, client churn, idle-to-compute transition, and the canonical Qwen3 8B benchmark.
10. Produce durable evidence that distinguishes endpoint presence, keeper activity, AMD initialization, and workload completion.

## 6. Non-goals

- Do not emulate a PCI root complex or replace macOS PCI/USB4 power management.
- Do not run dummy kernels, allocate VRAM, or maintain GPU utilization merely to keep the link awake.
- Do not hold BAR mappings or prepared DMA memory between workload clients.
- Do not keep a Python process alive as the production solution.
- Do not poll `system_profiler` once per second as a keeper. It is an observer, not the transaction owner.
- Do not automatically issue FLR, hot reset, bus reset, USB4 reset, or smart-plug power cycle.
- Do not change ASPM, L1 substates, link speed, or USB4 CLx policy in the first implementation.
- Do not modify the AMD firmware boot sequence, small-BAR discovery policy, queue submission, or benchmark hot path as part of the keeper.
- Do not use `AM_REMOTE_SMALL_BAR_DISCOVERY=1` for normal execution. It is explicitly unsafe for this target.
- Do not claim system-sleep survival from an awake-idle test.
- Do not claim PSU sufficiency from keepalive success.
- Do not merge official upstream tinygrad. This work belongs to `tinygrad-arkey` and may use Git history only as source evidence.
- Do not commit ignored Xcode build products, signing material, provisioning profiles, downloaded applications, or local model files.

## 7. Required invariants

1. The DriverKit provider is either fully started and eligible to tick, or stopping/stopped and unable to tick.
2. A timer callback cannot execute after the provider has closed its `IOPCIDevice`.
3. A failed timer creation or enable operation is visible in status and logs; it cannot silently report the keeper as active.
4. A successful tick means the full identity DWORD equals the expected device/vendor value, not merely that a callback fired.
5. `0xffffffff`, `0x00000000`, an unexpected PCI identity, or a missed-provider state is a failed tick.
6. Keepalive failures never trigger an automatic reset or power cycle.
7. Keepalive state contains no BAR mapping, DMA command, IOMemoryDescriptor, shared-memory allocation, firmware state, or queue state.
8. Every workload user-client owns and releases its own DMA commands.
9. Every server-side workload cleanup is idempotent and safe after partial initialization.
10. A socket disconnect cannot disable the DriverKit keeper.
11. A server crash cannot disable the DriverKit keeper while the extension remains bound.
12. A tinygrad process crash cannot leave a workload lease or mapping indefinitely owned by the app server.
13. Reset and extension stop are mutually exclusive with a keepalive read.
14. The keeper does not write PCI configuration space.
15. The keeper does not change link policy in its normal tick.
16. Protocol commands and DriverKit selectors have one declared source of truth and are versioned.
17. A client that cannot prove keeper capability must say `unsupported` or `inactive`; it must not infer support from an environment variable.
18. The default interval for the admitted `1002:744c` USB4 target is one second until measured evidence supports another cadence.
19. Disabling the keeper is explicit, observable, and never a hidden consequence of workload cleanup.
20. All published qualification artifacts identify the native app, driver extension, Python commit, hardware, and exact test interval.

## 8. Architecture decision

### 8.1 Preferred placement: DriverKit provider timer

Implement the keeper in `TinyGPUDriver`, next to the `IOPCIDevice` that the driver opens in `Start_Impl` and closes in `Stop_Impl`.

This is the closest available equivalent to motherboard/OS endpoint ownership:

- its lifetime follows device binding rather than an application connection;
- it can issue `ConfigurationRead32(0, &identity)` directly;
- workload user clients may connect and disconnect without affecting it;
- the app server does not have to monopolize the only user-client connection;
- no LaunchAgent is required merely to keep the link alive;
- workload resources remain naturally scoped to `TinyGPUDriverUserClient`.

The driver extension must use a DriverKit-supported timer and serial execution context. The exact generated IIG declarations may depend on the Xcode/DriverKit SDK, but the behavioral contract is fixed:

```text
Start provider
  -> open IOPCIDevice
  -> read and validate 1002:744c
  -> enable required command-register bits
  -> create keeper serial queue and timer
  -> publish ACTIVE state
  -> arm first 1-second deadline

Timer callback
  -> confirm state is ACTIVE
  -> read config DWORD at offset 0
  -> validate 0x744c1002
  -> update counters and timestamp
  -> re-arm one-shot deadline

Stop provider
  -> publish STOPPING state
  -> cancel/disable timer and wait for callback drain
  -> release timer and queue
  -> close IOPCIDevice
  -> publish STOPPED state
```

Use a re-armed one-shot timer rather than assuming a repeating timer cannot overlap stop. The callback and lifecycle transitions must execute on, or synchronize through, one serial keeper queue.

### 8.2 Workload resource placement

Keep these resources per user client and per socket workload lease:

- mapped BAR address and size records;
- system-memory mappings and shared-memory file descriptors;
- `IODMACommand` objects and `PrepareForDMA` state;
- bulk transfer buffers whose contents belong to a client request;
- AMD runtime mappings, rings, queues, firmware allocations, and synchronization state;
- client RPC statistics and errors.

The DriverKit user client already owns an expandable list of DMA commands and completes them during `Stop_Impl`. Preserve that ownership. Add regression coverage for partial allocation and repeated stop.

Do not transfer any of these resources to the persistent driver object for convenience. The provider may expose device identity and keepalive telemetry to user clients, but it must not adopt their workload memory.

### 8.3 App-server role after the change

The native app server remains a transport and workload-lifetime manager:

- accept at most one exclusive workload client unless multi-client semantics are separately designed;
- open a fresh DriverKit user client for the socket client;
- map and allocate only on explicit workload requests;
- clean all workload resources on orderly disconnect, protocol error, timeout, or process termination;
- expose a read-only keeper status RPC forwarded from the driver;
- never represent socket-server liveness as proof of PCI keeper liveness.

The server may remain on-demand. A fixed socket and LaunchAgent can improve operational reliability and status access, but they are not required for link liveness when the preferred DriverKit timer is working.

### 8.4 Fallback placement: persistent app-side idle holder

Use this only if the preferred DriverKit timer cannot be made reliable or signable on the target host.

The fallback TinyGPU.app server must keep one DriverKit connection open while it waits for workload clients. That idle connection performs the same one-second configuration read. On workload arrival it must either:

- retain the idle connection and open a separate user-client connection for workload resources, if DriverKit supports both concurrently; or
- synchronously pause the idle timer, close the idle connection, open a fresh workload connection, then reacquire the idle connection immediately after workload cleanup.

The fallback requires process supervision, because link liveness would then depend on TinyGPU.app. Install it as a per-user LaunchAgent with a stable socket path, restart-on-failure policy, log destination, and no embedded signing secrets.

The fallback must satisfy the same resource isolation and status contracts. It is inferior because application death creates a keeper gap, so it must not be selected merely because it is easier to prototype.

### 8.5 Rejected placements

| Placement | Reason rejected |
|---|---|
| tinygrad Python client | Exits between workloads; cannot protect idle periods |
| separate Python keeper connected to the socket | Competes for the server's single client and still depends on Python/process supervision |
| periodic dummy AMD kernel | Boots and retains far more state than required; adds heat, power, and failure modes |
| `system_profiler` loop | Expensive observer with no ownership guarantee; not the proven config-read primitive |
| periodic automatic reset | Destructive, races workloads, and cannot recover a physically missing endpoint safely |
| disabling all system sleep | Addresses a different event and does not prove prevention of awake-idle link loss |

## 9. Driver state model

Use an explicit provider keeper state. Names may follow local C++ conventions, but transitions and meaning must remain visible.

| State | Meaning | Timer allowed | User client allowed |
|---|---|---:|---:|
| `DETACHED` | No provider is bound | No | No |
| `STARTING` | Provider exists; identity/timer setup incomplete | No | No |
| `ACTIVE_HEALTHY` | Timer armed; last identity read matched | Yes | Yes |
| `ACTIVE_DEGRADED` | Provider remains bound but one or more ticks failed | Yes, bounded retry only | Yes only if ordinary device checks pass |
| `QUIESCING` | Reset or stop owns the provider transition | No | No new client |
| `STOPPED` | Timer drained and provider closed | No | No |

Required transitions:

```text
DETACHED -> STARTING -> ACTIVE_HEALTHY
STARTING -> STOPPED                    setup failure
ACTIVE_HEALTHY -> ACTIVE_DEGRADED      invalid/missed identity read
ACTIVE_DEGRADED -> ACTIVE_HEALTHY      later matching identity read
ACTIVE_* -> QUIESCING -> STOPPED       extension stop / device termination
ACTIVE_* -> QUIESCING -> ACTIVE_*      explicit operator-requested reset, if retained
```

There is no unbounded `RECOVERING` loop. When a tick fails, record the failure and continue only bounded read retries at the normal cadence. Do not reset. If the system terminates the service, stop cleanly and allow normal hotplug rebinding.

Workload connection state is orthogonal:

```text
NO_CLIENT -> CONNECTING -> ACTIVE_CLIENT -> DRAINING -> NO_CLIENT
```

Do not encode workload presence into keeper enablement. The config read is sufficiently small to continue at one hertz during a workload; serialize it with reset/stop and measure its overhead. If evidence shows an active-workload interference, suppression may use a short `last_activity` window, but it must never create an unbounded gap and it must not depend on mapped-MMIO accesses being visible to the provider.

## 10. Keepalive policy

### 10.1 Default

- Default enabled for the admitted `1002:744c` TinyGPU target.
- Default interval: 1000 ms.
- Read: PCI configuration offset `0`, width `4`.
- Expected little-endian value: `0x744c1002`.
- Timer leeway must remain small enough that the maximum observed gap is below the proven safe cadence. Start with no more than 100 ms leeway.
- Arm only after the provider is open and initial identity validation succeeds.

Do not silently apply the target-specific policy to unrelated PCI devices. If the driver extension's match rules admit additional devices later, require an explicit supported-device policy table.

### 10.2 Configuration surface

The device-lifetime default must not depend on an app environment variable because DriverKit service launch does not inherit tinygrad's environment reliably.

Provide these read-only facts through a native status interface:

- compiled default interval;
- effective interval;
- enabled/disabled state;
- expected identity and last observed value;
- keeper state;
- total ticks, successful ticks, failed ticks, and consecutive failures;
- monotonic time of the last attempt and last success;
- maximum observed inter-success gap;
- timer setup/enable error, if any;
- driver start generation or boot UUID.

An operator override may be added through a native CLI/RPC, but it must be explicit and status-visible. `REMOTE_KEEPALIVE_S` may be translated by `APLRemotePCIDevice` into a native set-policy request for development compatibility only after the status/version handshake exists. It must not remain a no-op.

For production qualification, use the one-second default. A `0` override is for controlled negative testing and must print a warning that awake-idle protection is disabled.

### 10.3 Failure accounting

A callback fire and a successful link transaction are different metrics.

On every tick:

1. increment `tick_attempts`;
2. read the identity DWORD;
3. store the observed value and monotonic timestamp;
4. if it matches, increment `tick_successes`, clear consecutive failures, and update the last-success gap;
5. if it does not match, increment total and consecutive failures and enter `ACTIVE_DEGRADED`;
6. emit logs on failure counts `1`, `10`, and `100`, plus recovery after any failure;
7. re-arm only if state remains active.

Counters must use widths that cannot wrap during an overnight run. Status reads must be synchronized and must not race updates.

## 11. Reset, hotplug, and power behavior

### 11.1 Reset serialization

If the existing reset RPC remains supported:

- acquire the provider lifecycle gate;
- transition to `QUIESCING`;
- disable and drain the timer;
- reject new workload requests;
- require that the current workload lease is already drained, unless the reset is part of an explicit fatal-workload teardown contract;
- issue only the operator-requested reset;
- revalidate identity and required PCI command bits;
- re-arm the timer and publish the resulting state.

No timer callback may run during reset. No reset may be triggered by keeper failures.

### 11.2 Unplug

On physical unplug or service termination:

- block new operations;
- cancel and drain the timer;
- release per-client resources through normal user-client stop;
- close the provider exactly once;
- terminate or mark the app-server connection failed;
- leave an artifact/log event with the last successful tick and last observed identity;
- wait for normal macOS reprobe after replug.

Do not spin on `IOServiceGetMatchingService` at high frequency.

### 11.3 Replug

The extension's next start is a new generation:

- clear counters or preserve them under a new generation identifier;
- reread vendor/device identity;
- recreate the timer rather than reusing a canceled object;
- require a new workload user-client connection;
- do not reuse old BAR addresses, DMA addresses, shared-memory handles, or AMD runtime state.

### 11.4 Endpoint already missing

Once config reads return all ones and macOS removes the endpoint, the keeper has failed to prevent the transition. The safe response is evidence preservation and an operator-visible error. Physical power-cycle automation remains separate in `extra/remote/amd_power_cycle.py` and must never run implicitly from this driver task.

## 12. Protocol and status contract

The Python `RemoteCmd` enum currently contains commands through `SYSMEM_SYNC`, while the last retained native `server.c` predates some of those commands. Restoring old files without reconciling the protocol would preserve silent drift.

Required protocol work:

1. Define protocol version and capability bits in one small shared schema or generated header/module.
2. Preserve existing numeric command values.
3. Add a version/capability handshake before using new keeper commands.
4. Add read-only `KEEPALIVE_STATUS` support.
5. Add `KEEPALIVE_SET_POLICY` only if an override is implemented.
6. Validate request sizes, enum values, device identity, and response payload lengths.
7. Return structured errors for unsupported protocol, unsupported device, inactive driver, and malformed request.
8. Keep status queries side-effect free.
9. Add compatibility tests for an old server: it must report unsupported cleanly rather than hang or misparse a payload.
10. Add a native CLI command that prints status without starting tinygrad, for example `TinyGPU keepalive status`.

Do not make Python parse native log strings as the status API. Use a versioned binary or JSON payload with fixed field names and explicit units.

Suggested capability bits:

- `CAP_PROTOCOL_VERSION`;
- `CAP_KEEPALIVE_STATUS`;
- `CAP_KEEPALIVE_POLICY`;
- `CAP_HEALTH_STATUS`;
- `CAP_SYSMEM_SYNC`.

Suggested status schema:

```json
{
  "schema": "tinygpu.keepalive.v1",
  "driver_generation": 1,
  "state": "active_healthy",
  "enabled": true,
  "interval_ms": 1000,
  "expected_pci_identity": "1002:744c",
  "last_identity_dword": "0x744c1002",
  "tick_attempts": 3600,
  "tick_successes": 3600,
  "tick_failures": 0,
  "consecutive_failures": 0,
  "last_attempt_monotonic_ns": 0,
  "last_success_monotonic_ns": 0,
  "max_success_gap_ms": 0,
  "timer_error": 0
}
```

Monotonic timestamps are meaningful only with the artifact's collection timestamps and generation. Do not present them as wall-clock time.

## 13. Source restoration and repository ownership

The task must restore native source to Git. An ignored `extra/usbgpu/.../build` directory and an installed application are not maintainable source authorities.

Recover the last local source as a starting point from `4c5e67cff^`, including only the files needed to build, sign, install, and test TinyGPU:

- `extra/usbgpu/tbgpu/installer/Shared/server.c`;
- `extra/usbgpu/tbgpu/installer/Shared/TinyGPUCLIRunner.swift`;
- `extra/usbgpu/tbgpu/installer/Shared/TinyGPUApp.swift`;
- DriverKit provider and user-client C++/IIG sources;
- Xcode project metadata and non-secret entitlements;
- reproducible build/install scripts;
- source asset metadata required by Xcode.

Before changing behavior:

- reconcile restored native protocol values against `tinygrad/runtime/support/system.py`;
- retain the local bundle identifiers `org.tinygrad.arkey` and `org.tinygrad.arkey.tinygpu.driver2` unless signing requirements force a documented change;
- preserve the existing MMIO-write acknowledgement fix;
- inspect current build products only as evidence, never as source;
- exclude `build/`, provisioning profiles, signing identities, notarization credentials, and downloaded zips;
- document the exact Xcode and macOS SDK versions used.

If the native project is split into its own repository during implementation, this input scope remains the cross-repository contract. The final output must identify both commits and prove the installed app was built from the native-source commit. Do not leave the only modified native source outside version control.

## 14. Implementation work packages

### Package 0: freeze the pre-change state

Record without modifying the device:

- tinygrad commit and dirty state;
- installed TinyGPU app bundle identity, version, code-sign identity, SHA-256, and dext identifier;
- installed system-extension state;
- Mac model, macOS build, USB4 topology, negotiated link rate, and PCI identity;
- whether `1002:744c` is visible through System Information and I/O Registry;
- current keeper capability result, expected to be unsupported;
- last known successful minimal-compute command and result;
- current riser, cable, enclosure, PSU, and GPU power-lead topology in operator notes.

Deliverable: `docs/task_workflow/output/egpu-usb4-persistent-pcie-phase0-20260727.md` plus a machine-readable JSON artifact.

### Package 1: restore and build the native source

- Restore the bounded native source set from Git history.
- Reconcile protocol enums and current Python expectations.
- Build Debug without installing.
- Run static protocol/layout checks.
- Verify no signing material or build product becomes tracked.
- Record the unsigned Debug executable hash.

Gate: the restored project builds from a clean checkout and ordinary existing RPC tests pass before keeper code is added.

### Package 2: implement provider-lifetime keeper

- Add keeper state and counters to `TinyGPUDriver` private state.
- Create a serial dispatch queue and `IOTimerDispatchSource` during start.
- Validate `1002:744c` before arming.
- Implement the one-second identity read and one-shot rearm.
- Implement synchronous timer disable/drain during stop.
- Serialize reset with the lifecycle gate.
- Add bounded logs and no automatic recovery action.
- Keep all workload DMA ownership in `TinyGPUDriverUserClient`.

Gate: DriverKit unitable state logic passes, the extension builds, and review can prove there is no timer-after-close path.

### Package 3: expose status and protocol identity

- Add protocol handshake/capability query.
- Add driver user-client selector for keeper status.
- Forward status through `server.c`.
- Add native CLI status output.
- Add Python query support without making inference startup depend on string parsing.
- Make `REMOTE_KEEPALIVE_S` either a supported explicit override or a fail-loud deprecated setting.

Gate: status shows actual native counters advancing while no tinygrad workload is connected.

### Package 4: harden workload cleanup

- Split persistent provider cleanup from per-client cleanup in naming and code.
- Make partial-client cleanup idempotent.
- Ensure every mapped BAR is unmapped from the correct connection.
- Ensure every prepared DMA command is completed and released.
- Ensure shared-memory files are unlinked and descriptors closed.
- Close the DriverKit user-client connection after resources drain.
- Reconnect a fresh workload client without restarting the extension or keeper.
- Reject a second simultaneous workload client explicitly.

Gate: repeated socket and process churn leaves keeper counters advancing and resource counts at the pre-client baseline.

### Package 5: install with provenance

- Build the installable app and embedded dext from the recorded source commit.
- Sign using the existing operator-controlled identity without exposing credentials.
- Verify signatures, entitlements, bundle identifiers, and embedded extension.
- Install/replace the app and extension through the supported flow.
- Complete any required System Settings approval as an explicit operator action.
- Verify installed executable hashes match build outputs.
- Verify `TinyGPU keepalive status` reports enabled and counters advance.

Gate: do not begin an idle soak if installed provenance or keeper activity is unproven.

### Package 6: hardware qualification

Run the test sequence in Section 17 in increasing risk order. Stop on the first disappearance, reset, timeout, malformed status, timer stall, or workload cleanup leak. Preserve the complete artifact before any replug or power cycle.

### Package 7: production integration and cleanup

- Keep the keeper default limited to the admitted target.
- Add concise operator status and uninstall/rollback commands.
- Update `docs/egpu-usb4-link-keepalive.md` from `OPEN` to the measured result.
- Update `docs/README.md` so the native-source and status authorities are discoverable.
- Produce the final output report and machine-readable qualification artifact.
- Remove temporary probes after banking unique evidence.
- Keep focused protocol, lifecycle, and cleanup regression tests.
- Remove or rewrite any documentation that says `REMOTE_KEEPALIVE_S=1` alone proves protection.

## 15. File-level change surface

Expected files or ownership areas follow. Exact filenames may change to match the restored Xcode project, but every responsibility needs a named owner.

| Area | Required change |
|---|---|
| `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.cpp` | Provider keeper lifecycle, timer, config read, state, counters, reset serialization |
| Driver provider header/IIG | Keeper status/policy interfaces and serialized execution declarations |
| `TinyGPUDriverUserClient.cpp` and IIG | Read-only keeper status forwarding; preserve per-client DMA ownership |
| `Shared/server.c` | Current protocol reconciliation, status RPC, hardened cleanup, version handshake |
| `Shared/TinyGPUCLIRunner.swift` | Native keepalive status command and structured output |
| `Shared/TinyGPUApp.swift` | No link-liveness dependency; only UI/status changes if needed |
| Xcode project | Restored source membership, test target, build settings, deployment target |
| build/install scripts | Clean build, provenance hash, signature verification, secret exclusion |
| `tinygrad/runtime/support/system.py` | Capability/status query and explicit handling of legacy keepalive env setting |
| focused unit/integration tests | Protocol layouts, old-server behavior, state transitions, cleanup, status schema |
| `docs/egpu-usb4-link-keepalive.md` | Final implementation location and measured qualification result |
| `docs/task_workflow/output/...` | Phase 0 and final execution reports |

Do not edit AMD compute code unless a qualification failure independently proves a defect there.

## 16. Test strategy

### 16.1 Host-only and build tests

These must run without the eGPU:

- request/response struct size and byte-order tests;
- command numeric stability tests;
- protocol-version and capability negotiation tests;
- malformed, truncated, and oversized request rejection;
- old-server unsupported-command behavior;
- keepalive status schema validation;
- state-transition table tests;
- timer setup failure path;
- stop before first tick;
- stop concurrent with a pending tick;
- repeated start/stop generation handling;
- failure-counter and recovery-log thresholds;
- unexpected PCI identity handling;
- cleanup after each partial resource-allocation step;
- second-client rejection;
- Xcode Debug and Release builds;
- signature and entitlement inspection scripts in dry-run mode.

Use dependency-injected identity reads for state tests. A mocked success proves control flow only and must never be reported as hardware evidence.

### 16.2 Native integration tests on hardware

- Extension starts and initial identity is `1002:744c`.
- Status counters advance with TinyGPU.app server stopped, proving DriverKit placement.
- Starting and stopping the app server does not reset counters or generation.
- Connecting and disconnecting a raw protocol client does not stop the timer.
- A tinygrad process exit releases the workload connection while counters continue.
- An intentionally malformed request closes only that workload client.
- Physical unplug terminates cleanly without callback-after-free or app crash loop.
- Replug creates a new generation and a fresh successful identity sequence.

If counters stop when TinyGPU.app exits, the implementation is app-lifetime, not provider-lifetime, and the preferred architecture gate has failed.

### 16.3 Resource-lifetime tests

Capture resource counts before and after each workload:

- open user-client connections;
- mapped BAR records;
- shared-memory objects matching the TinyGPU prefix;
- open shared-memory file descriptors;
- prepared DMA command count;
- server socket/client descriptors;
- app and Python process count.

After cleanup, all workload counts must return to baseline. Keeper counters and provider generation must remain continuous.

### 16.4 Fault injection

Use software fault injection before physical tests:

- force the identity read adapter to return all ones;
- return zero or a different identity;
- fail timer creation;
- fail timer enable/rearm;
- close the socket during each RPC payload phase;
- fail each BAR/DMA/shared-memory allocation step;
- terminate the Python client without cleanup;
- terminate TinyGPU.app during a workload only after a non-destructive test fixture exists;
- request reset while a client lease is active and require rejection;
- request stop with a pending timer callback and prove drain ordering.

Do not fault-inject destructive PCI writes or resets on the physical GPU.

## 17. Hardware qualification sequence

Every phase records native keeper status before and after, endpoint visibility, app/dext identity, and any macOS/DriverKit logs. Run from a clean repository commit and do not combine phases after a failure.

### Q0: enumeration preflight

Require:

- target visible as `1002:744c`;
- UT4G topology visible at the expected negotiated rate;
- TinyGPU dext activated and enabled;
- provider state `ACTIVE_HEALTHY`;
- at least 10 consecutive successful ticks;
- no active tinygrad workload.

### Q1: no-client keeper proof

Stop TinyGPU.app if the extension remains loaded, wait 120 seconds, then query status through a fresh CLI/app connection.

Require:

- provider generation unchanged;
- success counter increased by approximately the expected number of ticks;
- maximum success gap remains within the declared cadence bound;
- endpoint stayed visible;
- no workload resources existed.

This is the decisive test of preferred placement.

### Q2: minimal compute

Run the established small tensor computation with:

```bash
DEV=AMD \
AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_REMOTE_SKIP_RESIZE_BAR=1 \
PYTHONPATH=. \
.venv/bin/python <minimal-compute-test>
```

Require exact expected output `[2.0, 5.0, 10.0, 17.0]`, clean process exit, workload resources back at baseline, and continued keeper success.

Do not include `AM_REMOTE_SMALL_BAR_DISCOVERY=1`.

### Q3: short client churn

Run 25 independent minimal-compute processes. Between processes, leave five seconds with no client.

Require:

- 25 correct results;
- no server/extension restart;
- no provider-generation change;
- no accumulated workload resource;
- no keeper failure;
- endpoint continuously visible.

### Q4: historical-window awake idle

With macOS awake and no tinygrad process connected, run a 90-minute idle soak. The previous failure appeared around 40 minutes, so a shorter run is not qualification.

Sample status and independent endpoint visibility every 60 seconds for evidence, but do not use the observer as the keeper.

Require:

- one continuous provider generation;
- endpoint visible for every sample;
- keeper success counter monotonically increasing;
- zero invalid identity reads;
- no success gap over the allowed bound, excluding a separately marked host scheduling event;
- no unexplained app or extension restart.

### Q5: idle-to-compute transition

Immediately after Q4, without replug, restart, reset, or power cycle, run the minimal compute test.

Require the exact expected result and normal cleanup. This proves the endpoint is not merely listed but usable after the protected idle interval.

### Q6: canonical 8B smoke

Use the existing official model file:

`/Users/julianabeleda/models/Qwen3-8B-Q4_K_M.gguf`

Run the repository authority:

```bash
DEV=AMD \
JIT=1 \
AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c \
AM_REMOTE_SKIP_RESIZE_BAR=1 \
PYTHONPATH=. \
.venv/bin/python extra/qk/bench.py \
  --model /Users/julianabeleda/models/Qwen3-8B-Q4_K_M.gguf \
  --prefill \
  --prefill-mode smoke
```

Require:

- model admission and AMD initialization succeed;
- the authority produces a parsable throughput result and its expected artifact;
- no timeout, endpoint disappearance, reset, or malformed keeper status;
- workload resources return to baseline;
- keeper resumes/continues with the same provider generation.

This gate answers whether the 8B model works. Throughput optimization is outside this task.

### Q7: canonical 8B prefill and decode

After smoke passes, run `extra/qk/bench.py` through its default authority path, or run explicit `--prefill` and `--decode` phases separately for failure attribution. Do not substitute a custom throughput script.

Record:

- exact command and environment;
- model SHA-256 and size;
- tinygrad commit and dirty state;
- native app/dext source commit and installed hashes;
- prefill and decode artifacts;
- keeper status before, during, and after;
- endpoint visibility and provider generation.

Do not publish performance comparisons unless the benchmark authority's normal identity and measurement gates pass.

### Q8: load-to-idle cycles

Run five cycles:

```text
minimal or 8B smoke workload -> clean disconnect -> 15 minutes awake idle
```

Require no provider restart, no resource growth, no invalid tick, and correct compute after the final idle.

### Q9: overnight awake-idle soak

Run at least eight continuous hours with the Mac awake and no workload. Then run minimal compute and the 8B smoke without replug.

This is the production-confidence gate. Preserve the full time series and summary rather than a screenshot alone.

### Q10: unplug/replug lifecycle

After the non-destructive gates pass:

- unplug only while no workload lease exists;
- verify clean stop and no callback-after-close report;
- replug once;
- verify a new provider generation and advancing keeper counters;
- rerun minimal compute.

This gate validates lifecycle correctness, not automatic recovery from a dead bridge.

### Q11: sleep/wake observation

Run only after awake-idle acceptance. Record one ordinary Mac sleep/wake cycle with no workload.

Classify the result separately:

- survives and revalidates;
- extension restarts and revalidates with a new generation;
- endpoint disappears and needs replug/power cycle.

Failure here does not falsify the awake-idle fix, but it blocks any claim of sleep/wake support.

### Q12: bounded sustained-load observation

Run a bounded 8B workload long enough to observe load stability and power behavior. Stop on endpoint disappearance, GPU reset, thermal alarm, PSU protection, cable heating, or repeated AMD wait timeout.

This phase may reveal a separate PSU/riser/power problem. Report it as load/power evidence, not keepalive failure, unless keeper/identity evidence shows the link failed first.

## 18. Observability and artifact contract

Create a machine-readable artifact for each hardware phase. Suggested schema: `tinygpu.egpu_qualification.v1`.

Required run-level fields:

- phase and run ID;
- start/end wall-clock timestamps with timezone;
- monotonic start/end times;
- host model and macOS build;
- tinygrad commit, branch, worktree, and dirty state;
- native source commit and dirty state;
- app/dext versions, bundle IDs, code-sign identities, and SHA-256 values;
- protocol version and capability bits;
- PCI identity, USB4 topology, and negotiated link rate;
- riser/enclosure/cable/PSU operator labels;
- keeper interval, state, provider generation, and counter snapshots;
- endpoint-observer samples;
- workload command, environment allowlist, return code, and artifact paths;
- resource counts before and after;
- reset, unplug, replug, sleep, and power-cycle events;
- final classification and first failure.

Required keeper sample fields:

- wall and monotonic timestamp;
- provider generation;
- state and enabled flag;
- attempts, successes, total failures, and consecutive failures;
- last identity value;
- last-success age and maximum gap;
- app-server PID if present;
- workload-client PID/lease ID if present.

Use unified logging for native events with stable subsystem/category names. Rate-limit ordinary success logs; counters provide success evidence. Never rate-limit the first failure, state transition, timer setup failure, unexpected identity, reset request, stop, or provider generation change.

## 19. Safety and stop conditions

Stop the current hardware phase immediately on any of these:

- PCI identity disappears or changes;
- keeper counter stops advancing beyond the allowed gap;
- identity read returns all ones, zero, or an unexpected device;
- provider generation changes unexpectedly;
- Timer setup, enable, rearm, or drain fails;
- DriverKit extension crashes or is disabled;
- TinyGPU.app enters a restart loop;
- socket RPC hangs or returns a malformed status payload;
- BAR/DMA/shared-memory resources do not return to baseline;
- AMD wait timeout, PSP/SMU/GFX boot failure, page fault, or GPU reset;
- macOS reports USB4/PCIe link errors correlated with disappearance;
- PSU protection, cable overheating, visible power instability, or unsafe hardware behavior.

After stopping:

1. preserve status, logs, process state, endpoint visibility, and artifact files;
2. classify the earliest failure before downstream AMD errors;
3. do not rerun the same configuration blindly;
4. do not issue an automatic reset;
5. ask the operator before unplug, replug, power cycle, system sleep, or signing/extension replacement.

## 20. Acceptance criteria

The task is complete only when all of the following are true:

1. Native TinyGPU source is tracked in an owned repository location and builds from a clean checkout.
2. The installed app and dext hashes are tied to the recorded native-source commit.
3. The preferred DriverKit provider timer is used, or the final report documents why it was impossible and proves the supervised fallback meets the same behavioral gates.
4. Native status proves one-second configuration reads continue with no tinygrad client and, for preferred placement, with TinyGPU.app stopped.
5. Every successful tick validates `0x744c1002`.
6. Timer stop/reset ordering prevents callback-after-provider-close.
7. Keepalive failure never triggers reset or power cycle.
8. Workload BAR, DMA, shared-memory, socket, and AMD runtime resources are absent between clients.
9. Twenty-five independent minimal-compute clients pass without provider restart or resource growth.
10. The GPU remains continuously enumerated through a 90-minute awake-idle test.
11. Minimal compute passes immediately after the 90-minute idle without replug or reset.
12. Canonical Qwen3 8B prefill smoke passes through `extra/qk/bench.py`.
13. Canonical 8B prefill and decode either pass, or a failure is conclusively classified outside the keeper with endpoint continuity proven.
14. Five load-to-idle cycles pass.
15. An eight-hour awake-idle soak passes and is followed by minimal compute and 8B smoke.
16. One no-workload unplug/replug cycle stops cleanly, creates a new generation, and returns to correct compute.
17. Protocol compatibility, malformed-input, state, timer, and cleanup tests pass.
18. `docs/egpu-usb4-link-keepalive.md` names the shipped implementation and measured result.
19. No documentation or operator command treats `REMOTE_KEEPALIVE_S=1` as proof unless native status confirms the effective policy.
20. The final report distinguishes enumeration, idle-link, load/power, AMD-runtime, application, and sleep/wake results.

## 21. Rollout and rollback

### Rollout

1. Land tracked native source and build reproducibility first.
2. Land keeper and protocol tests without replacing the installed extension.
3. Build and inspect the candidate app/dext.
4. Install only with operator participation and record the previous installed hashes.
5. Run Q0-Q3 before any long soak.
6. Run Q4-Q8 in order.
7. Run Q9 overnight.
8. Run destructive-lifecycle observations Q10-Q12 last.
9. Promote documentation and default status checks only after evidence is banked.

### Rollback

Rollback must be possible at both source and installed-bundle levels:

- retain the prior app/dext version identifier and hash;
- provide exact supported uninstall/reinstall commands;
- do not overwrite the only copy of a known-working signed bundle without a recorded recovery path;
- if the new extension fails to start, restore the prior bundle through the normal system-extension workflow;
- if the keeper causes workload interference, disable only through the explicit native policy/status interface for diagnosis, then revert the native commit;
- do not use a Python environment variable as the sole rollback mechanism.

Any rollback report must say whether link protection is absent after rollback.

## 22. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Timer callback races provider stop | Serial lifecycle queue, one-shot rearm, synchronous cancel/drain, targeted concurrency tests |
| Native source restored with stale protocol | Version handshake, shared schema, numeric stability tests, current Python reconciliation before behavior changes |
| Keepalive appears enabled but no PCI read occurs | Validate identity value, expose attempt/success counters and maximum gap, test without app/client |
| Workload cleanup disables keeper | Put keeper on provider, separate cleanup functions and state, churn test |
| Persistent driver accidentally owns DMA | Enforce user-client ownership, resource baseline tests, code review invariant |
| One-hertz read affects workload | Measure canonical workload with counters active; keep transaction read-only and serialized only with lifecycle operations |
| Link still drops despite reads | Preserve first-failure evidence; do not auto-reset; investigate cadence/topology/power as a new bounded scope |
| New riser fixes one class but power remains weak | Separate sustained-load gate and physical power topology record |
| Actual sleep suspends timer | Treat sleep/wake separately; do not claim support from awake-idle results |
| Signing or System Extension approval blocks install | Stop at operator action, provide exact state and command, do not weaken system security silently |
| Build products become the source authority | Track source, exclude build/signing products, verify installed hash against clean build |
| Device policy accidentally applies to other cards | Match explicit vendor/device policy table and fail unsupported identities |
| Failure logging floods storage | Log state transitions and bounded failure counts; sample counters into structured artifacts |

## 23. Open decisions requiring evidence

These decisions must be closed during execution and recorded in the output report:

1. Which DriverKit SDK and deployment target are required for `IOTimerDispatchSource` on this Mac's macOS build?
2. Does the extension remain active and its timer advance when TinyGPU.app is fully stopped?
3. What timer leeway produces a measured maximum gap safely below the prior failure threshold?
4. Should configuration reads continue unconditionally during workloads, or does measured interference justify a bounded recent-activity suppression window?
5. Can keepalive status be exposed through the existing user-client without changing signing entitlements?
6. Should the protocol source of truth be a generated C/Python schema or paired compile-time assertions?
7. Does the current native server need all `RemoteCmd` operations through `SYSMEM_SYNC`, or should unsupported commands be capability-gated?
8. Is a dynamic interval override worth the additional state and persistence surface, or is fixed target policy sufficient?
9. Does ordinary macOS sleep restart the extension cleanly, preserve it, or remove the endpoint?
10. Under sustained 8B load, does any failure begin with power/link evidence or with AMD runtime evidence while PCI identity remains valid?

No open decision permits silently weakening the invariants or acceptance gates.

## 24. Required deliverables

- Restored, tracked TinyGPU native source and clean-build instructions.
- Provider-lifetime keepalive implementation with tests.
- Protocol version/capability and keeper-status interfaces.
- Native CLI status command.
- Python integration that reports actual native capability.
- Resource-lifecycle and malformed-client regression tests.
- Phase 0 report and JSON artifact.
- Per-phase hardware qualification artifacts.
- Final report at `docs/task_workflow/output/egpu-usb4-persistent-pcie-service-report-20260727.md`.
- Updated `docs/egpu-usb4-link-keepalive.md` and `docs/README.md`.
- Rollback instructions tied to installed hashes.

## 25. Definition of done

Done means the eGPU can sit idle while the Mac remains awake, beyond the previously observed failure window and through an overnight soak, then run minimal tinygrad compute and the canonical Qwen3 8B workload without replugging. Native counters must prove that the DriverKit-owned PCI configuration read ran throughout the idle interval. At the same time, inspection and resource counts must prove that no workload BAR, DMA, shared-memory, or AMD runtime state remained alive between clients.

Anything less is one of:

- a prototype keeper;
- an app-lifetime workaround;
- a short smoke test;
- an unproven environment setting;
- or evidence for a different hardware failure class.

It is not completion of this scope.

## 26. Review remediation addendum

Review date: 2026-07-27

Review authority: `structure/Development/coding-principles.md` and
`structure/Development/tinygrad-coding-overrides.md`

This addendum converts the implementation review into required remediation
work. It is part of this input scope. The keeper is not ready for installation
or hardware qualification until the high-severity items below are closed.

### 26.1 Review verdict

The provider-lifetime placement is correct in principle, and the DriverKit
target builds with the current Xcode toolchain. The current implementation is
still a prototype because it does not yet prove the invariants that make a
persistent PCI owner safe:

- reset and configuration RPCs are not serialized with the timer callback;
- timer destruction does not use a completion contract that proves callback
  drain;
- Python and native protocol definitions are version-skewed and have no
  capability handshake;
- the restored native source is untracked and therefore is not yet a durable
  source authority;
- target policy is compiled into the driver without a central policy/status
  surface;
- initial identity-read failure silently disables the keeper;
- native socket framing accepts partial requests and payloads;
- no native lifecycle or protocol tests exercise the dangerous boundaries.

These are implementation defects or release blockers, not optional polish.

### 26.2 Principles-to-remediation mapping

| Principle | Current violation | Required remediation |
|---|---|---|
| Centralize authority | Python command enum and native command enum diverge | One versioned protocol schema or generated header/module |
| Centralize config | Identity and cadence are private C++ constants | Named policy table with one owner and status exposure |
| Encode invariants | Reset, config access, and timer access use different queues | One provider operation gate for all PCI operations |
| Contain dangerous power | Reset can run during a keepalive transaction | Reset must acquire the provider gate and quiesce the timer |
| Treat errors as system information | Invalid initial identity silently skips keeper | Typed inactive/unsupported/failed state and structured error |
| Test behavior at the boundary | Build-only verification | Native lifecycle, framing, protocol, and hardware integration gates |
| Design for replacement | Python cannot distinguish old server from new server | Capability negotiation before keeper/status commands |
| Keep artifacts portable | Native source exists only as untracked local recovery | Track source and record repo-relative build provenance |
| Keep public surfaces boring | Status is inferred from logs or environment | Explicit native status API and CLI |
| Orthogonalize | Provider, timer, workload, reset, and transport state overlap | Separate provider policy, operation gate, workload lease, and transport state |

### 26.3 Remediation R0: establish reviewable source ownership

#### Required outcome

The native source that implements the keeper must be a tracked, reviewable
source set. An ignored Xcode build product or an installed `/Applications`
bundle is not an implementation authority.

#### Required actions

1. Add the minimum recovered TinyGPU installer subtree to Git:

   - DriverKit provider and user-client C++/IIG files;
   - native server and CLI/app sources;
   - Xcode project metadata required to build those sources;
   - non-secret entitlements, plist files, and build scripts;
   - source assets required by the Xcode project.

2. Do not add:

   - `build/` products;
   - DerivedData;
   - embedded provisioning profiles;
   - signing identities, private keys, notarization credentials, or download
     tokens;
   - installed application bundles;
   - local model files or machine-specific logs.

3. Add a source manifest or build report that records:

   - repository-relative native source paths;
   - native source commit;
   - Xcode version and SDK;
   - DriverKit deployment target;
   - build command;
   - output bundle hashes;
   - signing state (`unsigned`, `ad hoc`, or release-signed).

4. Make the clean build command work from the repository root, not only from
   the Xcode project directory. The command must name the project explicitly.

5. Add a source-ownership check that fails if a native file included by the
   Xcode project is untracked or if a build source path resolves outside the
   repository without an explicit generated-source declaration.

#### R0 acceptance

- `git ls-files extra/usbgpu/tbgpu/installer` lists every source needed by the
  build;
- `git status --short` is clean in the validation worktree;
- a clean checkout builds with `xcodebuild -project ...`;
- no build product is required as an input to another clean build;
- the installed bundle, when one exists, can be traced to a recorded source
  commit.

### 26.4 Remediation R1: define one native/Python protocol authority

#### Required outcome

The native server, DriverKit user client, Python client, and tests must share
one declared wire contract. No new command may be assigned by editing only one
endpoint.

#### Current defect

Python currently reserves command values through `SYSMEM_SYNC`, while the
retained native server implements only the older subset. The server has no
version or capability handshake. A generic unknown-command error cannot tell a
client whether it contacted an old binary or a binary with a different command
assignment.

#### Required protocol design

1. Preserve the numeric values and wire layout of all commands already used by
   deployed tinygrad clients.

2. Define a versioned schema containing:

   - protocol major and minor version;
   - request and response struct sizes;
   - command numeric values;
   - capability bits;
   - maximum request and response sizes;
   - error status values and typed error payloads;
   - keeper-status payload schema.

3. Generate or mechanically validate both native and Python declarations from
   that schema. If generation is impractical, require compile-time/native
   assertions and Python numeric/layout tests against a checked-in manifest.

4. Add a side-effect-free handshake before any new status/policy request. The
   handshake must return:

   - protocol version;
   - supported capabilities;
   - native app version/source identity when available;
   - maximum supported payload size;
   - explicit unsupported status for legacy servers.

5. Add a read-only keeper status capability and command only after the
   handshake is defined. It must return a versioned structured payload, not
   log text.

6. Add a policy command only if a runtime override is genuinely required. A
   policy command must identify the caller, validate the target identity, and
   return the effective policy after the request.

7. Define legacy behavior:

   - a client first performs the handshake;
   - a legacy server returns an explicit unsupported result or closes cleanly;
   - the client does not send a status command to an unknown server;
   - a generic RPC failure is never interpreted as active keepalive.

8. Keep `REMOTE_KEEPALIVE_S` non-authoritative until a negotiated policy
   command exists. It must either be translated into a native request after
   capability confirmation or fail loudly as deprecated.

#### Required status payload

Use a fixed schema such as `tinygpu.keepalive.v1` with exact field units:

```json
{
  "schema": "tinygpu.keepalive.v1",
  "provider_generation": 1,
  "state": "active_healthy",
  "enabled": true,
  "policy_id": "usb4_1002_744c_v1",
  "interval_ms": 1000,
  "expected_identity": "1002:744c",
  "last_identity_dword": "0x744c1002",
  "attempts": 100,
  "successes": 100,
  "failures": 0,
  "consecutive_failures": 0,
  "last_attempt_monotonic_ns": 0,
  "last_success_monotonic_ns": 0,
  "max_success_gap_ms": 0,
  "timer_error": 0
}
```

The exact values above are illustrative. The field names, units, required
versus optional status, and maximum encoded size must be frozen in the schema
before implementation.

#### R1 acceptance

- native and Python numeric/layout tests pass;
- an old server returns unsupported without hanging;
- a new server exposes capabilities before status is requested;
- malformed and truncated status payloads fail closed;
- Python never reports active keeper solely because an environment variable is
  set;
- status distinguishes `unsupported`, `inactive`, `degraded`, and
  `active_healthy`.

### 26.5 Remediation R2: serialize every PCI operation

#### Required outcome

The provider must have one explicit operation gate. A timer read, client config
read, client config write, reset, stop, and provider close must not access the
same `IOPCIDevice` concurrently.

#### Required state and queue model

Use a provider-owned serial `IODispatchQueue` for all operations that touch
`ivars->pci`:

```text
provider queue:
  keeper tick -> ConfigurationRead32
  client CfgRead -> ConfigurationRead{8,16,32}
  client CfgWrite -> ConfigurationWrite{8,16,32}
  client Reset -> quiesce, reset, revalidate, rearm
  provider Stop -> disable, drain, close
```

Do not assume that creating the timer on a queue automatically moves ordinary
`ExternalMethod` calls or `LOCALONLY` provider methods onto that queue. The
IIG declarations or explicit `DispatchSync`/`DispatchAsync` calls must make
the ownership visible.

#### Reset contract

1. Reject reset while a workload lease is active unless the caller has an
   explicit destructive-reset authority.
2. Acquire the provider operation gate.
3. Mark the keeper `QUIESCING`.
4. Disable the timer and wait for pending callback completion.
5. Issue the requested reset exactly once, with no automatic fallback unless
   the existing reset contract explicitly requires it.
6. Reread and validate the expected identity.
7. Restore required PCI command bits if reset cleared them.
8. Reset counters or advance `provider_generation` according to the status
   contract.
9. Re-arm the timer and publish `ACTIVE_HEALTHY` or a typed failure.

#### Config RPC contract

- Validate offset, width, and value before entering the provider queue.
- Return `kIOReturnBadArgument` for unsupported widths instead of returning
  success with an unchanged output.
- Serialize config reads/writes with the keeper tick.
- Forbid arbitrary config writes from the keeper path; the keeper is read-only.
- Record reset and config-write operations in the native event log.

#### R2 acceptance

- a stress test interleaving timer ticks, config RPCs, and reset requests has no
  concurrent provider access;
- reset cannot overlap a configuration read;
- stop cannot close the PCI provider while any provider operation is active;
- invalid config widths fail explicitly;
- the operation queue and state transitions are visible in code and tests.

### 26.6 Remediation R3: make timer shutdown provably safe

#### Required outcome

The implementation must prove that no timer handler, rearm, or config read can
run after the timer/action/queue or PCI provider is released.

#### Required teardown sequence

Use one shared teardown helper from normal stop, failed start, provider free,
and hotplug termination:

```text
enter QUIESCING
disable timer with completion callback
wait for pending handler completion
cancel queued provider work
mark keeper inactive
release timer action
cancel/drain keeper queue
release keeper queue
close IOPCIDevice
clear provider pointer
enter STOPPED
```

The exact DriverKit calls may differ by SDK, but the observable guarantee must
be the same. `Cancel` followed immediately by `release` is not sufficient as
the only proof. `free()` must not contain a separate unsafe destruction path.

#### Start-failure sequence

If timer creation, action creation, handler registration, enable, or first
deadline setup fails:

- mark setup failure in status/logs;
- disable and drain any partially-created source;
- release action, timer, and queue in dependency order;
- close the PCI provider;
- return a non-success start error for the target device;
- do not register a service that appears usable without a keeper.

#### R3 acceptance

- repeated start/stop under a mocked timer has no callback after release;
- stop while a callback is in flight waits for completion;
- stop with a queued but not started callback does not execute it after release;
- failed setup leaves no timer, action, queue, or PCI connection;
- provider `free()` invokes the same safe teardown helper or proves that all
  lifecycle paths have already drained it.

### 26.7 Remediation R4: replace incidental hardcoding with named policy

#### Review judgment

`1002:744c` and one second are currently intentional for the proven RX 7900
XTX USB4 path. They are not arbitrary values. They are nevertheless hardcoded
inside execution logic, with no central policy identity, no effective-status
report, and no testable extension point.

#### Required policy shape

Create one policy table owned by the native provider, for example:

```text
policy_id: usb4_1002_744c_v1
vendor_id: 0x1002
device_id: 0x744c
config_offset: 0
config_width: 4
interval_ms: 1000
enabled_by_default: true
```

The table must be data, not a chain of device-specific `if` statements. The
provider selects exactly one policy after validating the PCI identity. An
unknown identity is `unsupported`, not silently treated as the target.

#### Configuration rules

- The default policy is compiled and reviewable for the current target.
- Runtime override is optional and must be explicit, bounded, and status-visible.
- An environment variable cannot directly mutate provider state.
- The effective policy, policy ID, and identity are returned by status.
- A malformed or out-of-range interval is rejected.
- `0` is allowed only for controlled negative testing and produces an explicit
  warning/status state.

#### R4 acceptance

- adding a future device means adding one policy row and tests;
- no execution function contains an unexplained identity or interval literal;
- status reports the policy ID and effective interval;
- the current target still defaults to 1,000 ms;
- the unknown-device path cannot claim active keepalive.

### 26.8 Remediation R5: fail closed on identity/setup failure

#### Current defect

The provider reads vendor/device IDs and only creates the keeper when they match.
If the read fails or returns all ones, the extension can continue and register a
service without a keeper. The broad PCI class match makes this distinction
important.

#### Required behavior

1. Read the full identity DWORD once through the provider operation gate.
2. Classify it as `supported`, `unsupported`, or `read_failed`.
3. For `supported`, create and arm the policy-selected keeper.
4. For `unsupported`, either decline the device cleanly or expose a clearly
   inactive provider; do not expose target-specific status.
5. For `read_failed` on an otherwise matched endpoint, fail provider start and
   log the exact error/value.
6. Do not publish `ACTIVE_HEALTHY` until the timer is armed and the first
   successful identity transaction has occurred.
7. Expose setup errors through status after a client can connect, or preserve a
   machine-readable start artifact if no user client can be created.

#### R5 acceptance

- all-ones, zero, and wrong-identity tests produce typed non-active states;
- timer creation/enable/rearm failure cannot result in a healthy service;
- the target cannot be used while the keeper is absent unless an explicit
  development override is recorded;
- logs identify whether failure occurred at identity read, timer creation,
  timer enable, first deadline, or first tick.

### 26.9 Remediation R6: harden native socket framing and errors

#### Required outcome

The Unix socket boundary must treat framing, truncation, oversized values, and
partial writes as protocol errors. It must never operate on stale payload data.

#### Required changes

- Replace one-shot request-header `recv` with an exact-length receive helper.
- Make exact receive return a typed EOF/error result, not `void`.
- Reject a partial header and close only that client connection.
- For payload commands, receive exactly the declared length before touching the
  BAR or shared buffer.
- Bound every declared size before allocation, receive, or pointer arithmetic.
- Return a structured error for unsupported commands instead of a generic
  status-only failure.
- Use a send-all helper for fixed responses, error bodies, MMIO responses, and
  file-descriptor responses.
- Validate request command, device ID, BAR index, offset, size, and reserved
  fields before dispatch.
- Reset per-client resource state before accepting the next client.
- Preserve the provider keeper when a client framing error closes the socket.

#### Required malformed-input tests

- header split across multiple writes;
- header truncated at every byte boundary;
- payload split across writes;
- payload truncated before declared length;
- declared size zero, one byte over the maximum, and integer-overflow values;
- invalid command and reserved fields;
- invalid BAR and offset/range combinations;
- disconnect during response and during FD passing;
- second client while one client owns the lease.

#### R6 acceptance

No malformed client request reaches MMIO, config, reset, BAR mapping, or DMA
code. Every malformed case returns a typed error or clean disconnect, and the
next valid client can connect with a fresh resource baseline.

### 26.10 Remediation R7: separate provider, workload, and transport state

Create explicit structures or modules for these independent concerns:

```text
ProviderState
  identity, policy, keeper state, counters, generation

ProviderOperationGate
  serialized PCI access, reset, stop, and timer drain

WorkloadLease
  user-client connection, BAR maps, DMA, shared memory, AMD workload state

TransportState
  socket framing, protocol version, capabilities, client lifecycle
```

Rules:

- provider state outlives socket clients;
- workload state never outlives its user-client lease;
- transport failure releases workload state but does not stop provider keeper;
- reset requires provider gate and explicit workload authority;
- transport status is not substituted for provider status;
- policy selection happens once at provider start and is exposed through status.

Do not add a second general-purpose framework. The goal is a small ownership
boundary around existing code, not a large abstraction layer.

### 26.11 Remediation R8: tests and evidence

#### Host-only tests

- protocol command numeric/layout stability;
- handshake capability negotiation;
- old-server unsupported response;
- malformed/truncated status payload rejection;
- policy table selection and unknown identity;
- interval bounds and explicit disabled mode;
- provider state transition table;
- reset gate admission/rejection;
- timer teardown ordering with injected callbacks;
- exact socket receive/send helpers;
- per-client cleanup after every partial allocation;
- source/build manifest coverage.

#### Native integration tests

- provider start on supported identity;
- provider start on unknown identity;
- provider start with failed identity read;
- timer setup failure;
- timer rearm failure;
- status counter advancement with no app server;
- app server start/stop without provider generation change;
- workload connect/disconnect without keeper interruption;
- reset during idle and reset during workload;
- provider stop with callback in flight;
- unplug and replug generation transition.

#### Hardware evidence

Do not call the implementation qualified from a build or unit test. Require,
in order:

1. installed binary/dext provenance;
2. status capability and counter advancement;
3. 120-second no-client keeper proof;
4. 90-minute awake-idle soak;
5. post-idle minimal compute;
6. canonical Qwen3 8B smoke;
7. five load-to-idle cycles;
8. eight-hour awake-idle soak;
9. bounded unplug/replug lifecycle;
10. separate sleep/wake and sustained-load classification.

Every run records source commits, bundle hashes, provider generation, keeper
counters, endpoint visibility, resource baselines, command, model identity,
and first failure.

### 26.12 Remediation execution order

The order is mandatory because later work depends on earlier authority:

```text
R0 tracked source
  -> R1 protocol schema/handshake
    -> R2 provider operation gate
      -> R3 timer drain/teardown
        -> R4 policy table/status
          -> R5 fail-closed startup
            -> R6 socket framing
              -> R7 ownership separation
                -> R8 tests and hardware evidence
```

Do not add Python status integration before R1. Do not install the new dext
before R2/R3. Do not run long hardware soaks before R5 and the native status
counter proof. Do not claim an 8B benchmark result as keeper evidence unless
the provider status artifact covers the complete idle-to-workload transition.

### 26.13 Remediation stop conditions

Stop implementation review and return to design if:

- any PCI access remains outside the provider gate;
- teardown cannot prove callback completion before release;
- protocol IDs are edited independently at either endpoint;
- the target can report healthy without a successful first tick;
- Python status relies on log parsing or environment inference;
- malformed socket input can reach MMIO, DMA, config, or reset;
- native source remains untracked;
- a test claims hardware behavior from a mocked identity read;
- a release install cannot be traced to a recorded native source commit.

Stop hardware qualification on any endpoint disappearance, timer stall, reset,
wait timeout, page fault, malformed status, resource leak, or power hazard.

### 26.14 Remediation definition of done

This review addendum is closed only when:

1. All native implementation files are tracked and have one source authority.
2. One versioned protocol schema governs native and Python command/status data.
3. All PCI operations, including reset and config RPCs, are serialized.
4. Timer disable and provider stop have a tested callback-drain guarantee.
5. Policy is named, centralized, bounded, and status-visible.
6. Startup fails closed when identity or timer setup is invalid.
7. Native framing rejects every tested partial/malformed request safely.
8. Provider, workload, and transport lifetimes are independently testable.
9. Native, protocol, cleanup, and concurrency tests pass.
10. Installed provenance is recorded and verified.
11. Awake-idle, post-idle compute, 8B smoke, churn, and soak gates pass.
12. Remaining signing, sleep/wake, and sustained-load limitations are recorded
    as explicit external gates rather than implied support.

Until these conditions hold, the implementation remains an unqualified
prototype even if the DriverKit project compiles.
