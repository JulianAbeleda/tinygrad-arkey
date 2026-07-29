# TinyGPU v5 runtime initialization and RPC-disconnect scope

Date: 2026-07-28

Status: open; post-reboot A0 passed on v7, but A1 stopped when the tunneled PCIe
tree disappeared during awake idle. No reset or AMD initialization followed.
This scope is a follow-on to
`egpu-usb4-persistent-pcie-service-scope-20260727.md`. It does not replace the
DriverKit-owned keepalive architecture or its A0-A11 acceptance matrix.

Repository/worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, branch `exp`.

GPU discipline: every command that can inspect, open, initialize, or exercise the
eGPU runs under `/tmp/gpu-bench.lock` through
`extra/usbgpu/tools/with_gpu_lock.py`. No broad benchmark, reset, power-cycle, or
sleep test is authorized by this scope.

## 1. Executive alignment

The installation and endpoint state at the latest handoff is:

- `org.tinygrad.arkey.tinygpu.driver2` version `1.0.0/7` is
  `[activated enabled]`, and it is the only arkey registration.
- The legacy `org.tinygrad.tinygpu.driver2` version `1.0.0/3` is
  `[activated disabled]`.
- The audited v7 app and activated provenance are bound to clean commit
  `c380ab4d0`.
- After the reboot, kernelmanagerd selected replacement unique ID
  `4e0fa54f6b09bc1e6274b7dc836eb373616790401964cf3334ebb63876099237`,
  and the live provider CDHash matched the installed v7 CDHash.
- A0 passed while `1002:744c` and its three companion functions were present at
  x16 and 16.0 GT/s and selector 5 reported a healthy keeper.
- Fifteen seconds after the A0 status sample, ACIO link errors preceded removal
  of the entire tunneled PCIe tree. The USB4 UT4G bridge remains connected, but
  there is now no `tinygpu` IOUserService and keepalive status is unavailable.

Those observations establish installation, activation, selector-5 marshalling,
and the single A0 snapshot only. They do not establish no-client continuity or
that the provider can safely service AMD runtime MMIO, DMA, or queue
initialization.

The original AMD runtime blocker, which has not been re-exercised after the v7
status fix, is:

```text
Device["AMD"]
  -> KFDIface: /dev/kfd absent (expected on macOS)
  -> PCIIface/APLRemotePCIDevice
  -> handshake and workload lease path
  -> AMD AMDev initialization
  -> init_sw performs earlier BAR reads and VRAM writes
  -> PSP sOS liveness read returns nonzero
  -> SMU liveness check starts by clearing C2PMSG90
  -> TinyGPU RPC connection receives EOF
  -> Python reports an aggregate "No interface for AMD:0" error
```

## 13. Handoff state — 2026-07-28

Implementation is committed on `exp`:

- `dda9ad2b4`: provider-gated MMIO, typed native/provider errors, Python error
  preservation, protocol updates, and runtime scope.
- `f4eb0b281`: prefer the active extension registration when stale terminating
  entries precede it.
- `289e9015f`: bump the DriverKit bundle version from 5 to 6.

CPU/source/protocol tests pass: `24 passed`. The v5 build and signed install
completed successfully under `extra/usbgpu/tools/with_gpu_lock.py`.

The v6 replacement was rejected by macOS with `OSSystemExtensionErrorDomain
error 4` because the old v5 registration remains active while another stale
registration is terminating. The installer rolled back to the known v5 app.
No GPU reset, unplug, or power-cycle was performed.

After the first reboot, the extra stale arkey registration was gone but
`org.tinygrad.arkey.tinygpu.driver2` version 5 remained `[activated enabled]`.
The installed app reported ready, the v1 handshake succeeded with capabilities
`3`, and keeper status remained unavailable. Headless app and
`systemextensionsctl` deactivation attempts were rejected because macOS could
not obtain interactive authorization.

The same app deactivation request was then run from the logged-in Terminal while
the GPU lock was held:

```sh
sudo /Applications/TinyGPU.app/Contents/MacOS/TinyGPU uninstall
```

It completed successfully with `Will complete after reboot.` The current system
entry is now version 5 in `[terminating for uninstall but still running]`; the
legacy `org.tinygrad.tinygpu.driver2` version 3 entry remains
`[activated disabled]`. No AMD initialization, reset, unplug, or power-cycle was
performed during this registration cleanup.

The required next action is another reboot so macOS can finish removing v5,
followed by these checks under the GPU lock:

```sh
systemextensionsctl list | grep -E 'tinygpu|TinyGPU'
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU status
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU keepalive status
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU keepalive handshake
```

If the v5 entry is gone, rerun the audited installer with the approval token
and provenance output, then run A0/A1 before any AMD initialization or M0-M7
qualification. If v5 is still terminating, do not retry v6 activation; retain
the exact system-extension state and continue registration diagnosis. This is a
registration cleanup, not an AMD firmware or PSP/GART operation.

The last locked minimal probe failed with `TinyGPU disconnect` while
`AMDev.is_smu_alive()` was issuing the first write in the standard SMU
`GetSmuVersion` mailbox transaction: a 32-bit zero to response register
`mmMP1_SMN_C2PMSG_90`. Cleanup then reported a broken pipe while releasing the
lease. After the probe, the app-level keepalive status command reported
unavailable even though the system extension remained enabled; this is a
provider/service-lifecycle failure, not a sudo or approval failure. It is the
first *observed failing* register write, not proof that it was the first runtime
write or the root cause.

### v7 inline status follow-up

The second reboot removed the v5 registration. The v6 activation initially
required System Settings approval, then completed as `[activated enabled]`; a
second audited installer pass aligned `/Applications/TinyGPU.app`, the live
v6 registration, and activated provenance at commit `b9b5fe06d`.

One bounded eGPU reset was required after the UT4G bridge returned without the
PCI endpoint. The reset restored `1002:744c`, all four PCI functions, and a
16.0 GT/s x16 link. The v6 DEXT then launched, published an active `tinygpu`
IOUserService, and successfully handled selector 4 handshakes. A0 nevertheless
failed consistently at selector 5 because keepalive status remained
unavailable; no AMD initialization ran.

The failure is an output-marshalling defect rather than another GPU-reset
boundary. `IOConnectCallStructMethod` supplies the protocol's 4096-byte status
buffer as inline DriverKit `OSData`, while the v6 user client accepts only
`structureOutputDescriptor`. The v7 candidate supports both inline `OSData` and
large descriptor outputs, retains the 4096-byte protocol cap, and bumps the
DEXT bundle version so macOS will replace v6. The six targeted CPU/source suites
pass (`61 passed`), and a clean Debug DriverKit build succeeds.

The next action is the audited v7 install followed by A0 and A1. Do not reset
the GPU again before those gates: the endpoint and v6 provider are both present,
and the remaining observed failure is selector-5 reply marshalling.

### v7 reboot handoff

This subsection supersedes the earlier Section 13 next actions.

Implementation commit `114d9c6d6` adds inline `OSData` handling for selector 5,
retains the descriptor path, bumps the DEXT version to 7, records the three v6
A0 failures, and passes `61` targeted tests plus a clean Debug DriverKit build.
The audited v7 install completed successfully and
`org.tinygrad.arkey.tinygpu.driver2` version 7 reached `[activated enabled]`.
The installed v7 DEXT CDHash is
`7b9d17f219d094d904d4758880b9bf1d3770e057`.

The first v7 A0 artifact is
`docs/task_workflow/output/egpu-usb4-persistent-pcie-A0-20260728T231606Z-6242.json`.
It failed at keepalive status because the still-bound provider was v6 (CDHash
`72240ba968978570b90e38ab47a21b331b1ec026`), not the newly registered v7.
No AMD initialization ran.

A second user-authorized bounded eGPU reset detached that v6 IOUserService. The
endpoint then returned successfully: UT4G is connected at 40 Gb/s, `1002:744c`
and its three companion functions are present, and every PCI function reports
x16 at 16.0 GT/s with the link up. The reset did not solve the DEXT handoff:
there is currently no `tinygpu` IOUserService, while the old v6 process remains
at PID 4492 and keepalive status is unavailable.

DriverKit logs establish the reason. During the v7 upgrade, sysextd delegated
v6 termination and explicitly scheduled its uninstallation for the next reboot.
When the endpoint returned, kernelmanagerd selected the v6 unique ID
`c2d486c17e62dc24af9469f97bcbb95896678b3f96a5b0826b7baf032f66279b`
instead of v7, then launchd rejected the v6 executable with `EACCES` because
sysextd had cleared its executable bit. v7 is registered as loaded under unique
ID `bdb12e12f1643474be8a566f3c6822ab570083e85e5b46b8f7f23f2848ffe6a3`,
but cannot bind while the stale v6 personality wins matching.

The required next action is a Mac reboot. Do not reset the GPU again and do not
run AMD initialization. After reboot, run these observations under the GPU
lock:

```sh
systemextensionsctl list | grep -E 'tinygpu|TinyGPU'
system_profiler SPThunderboltDataType SPPCIDataType
ioreg -r -n tinygpu -l
ps -ww -axo pid=,ppid=,command= | grep -Ei 'tinygpu|org.tinygrad.arkey'
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU status
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU keepalive handshake
/Applications/TinyGPU.app/Contents/MacOS/TinyGPU keepalive status
```

Proceed only if v6 is gone, v7 remains `[activated enabled]`, the endpoint is
present, the `tinygpu` IOUserService CDHash is the installed v7 CDHash above,
and the status payload validates as healthy. Because this handoff commit advances
`HEAD` beyond the installed provenance's `114d9c6d6` source commit, refresh the
audited install provenance from the clean handoff commit before running A0 and
A1. A0/A1 remain the only authorized gates; M0-M7 and AMD initialization stay
blocked until both pass.

### Single-reboot discipline and same-version v7 follow-up

Avoid repeated reboot requests. Before any future DEXT install, version bump,
deactivation, or reboot, perform one thorough read-only audit under the GPU lock
and record all of the following together:

- every current and historical registration for both TinyGPU bundle IDs,
  including version, state, SystemExtensions path, unique ID, executable mode,
  CDHash, and which registration kernelmanagerd selects for PCI matching;
- the installed, built, registered, and live-provider app/DEXT hashes and bundle
  versions, plus the install provenance source commit;
- USB4 and PCI enumeration, the `tinygpu` IOUserService properties, DEXT process
  census, selector-4 handshake, selector-5 status, and the relevant sysextd,
  kernelmanagerd, launchd, and DriverKit log interval;
- every pending source or activation-contract change that could require another
  DEXT version. Batch those changes into a single final candidate before asking
  the operator to reboot.

Do not re-register an identical DEXT at the same bundle version merely to refresh
provenance. Refreshing the v7 install from clean commit `c380ab4d0` created a
second version-7 registration: the original v7 became `[terminating for upgrade
via delegate]`, sysextd cleared its executable bit, and the replacement v7 became
`[activated enabled]`. After one operator reset returned `1002:744c` at x16 and
16.0 GT/s, kernelmanagerd still selected the terminating original v7 unique ID
`bdb12e12f1643474be8a566f3c6822ab570083e85e5b46b8f7f23f2848ffe6a3` rather
than replacement unique ID
`4e0fa54f6b09bc1e6274b7dc836eb373616790401964cf3334ebb63876099237`.
launchd then rejected the original executable with `EACCES`. The endpoint is
present, but no `tinygpu` IOUserService is published and A0 has not run.

The next action is one Mac reboot to complete this delegated v7-to-v7 upgrade.
Do not reset the GPU again before that reboot. After reboot, run the complete
read-only audit above and proceed directly to A0/A1 only if there is exactly one
active arkey v7 registration, the endpoint and expected live-provider CDHash are
present, and selector 4/5 are healthy. Do not run the installer again solely
because this documentation is later committed; the audited provenance already
binds the installed feature binaries to clean source commit `c380ab4d0`. If this
note is committed and pushed before reboot, park the qualification worktree at
detached commit `c380ab4d0` for A0/A1, then return to `exp` after preserving the
gate artifacts. That keeps the gate runner's strict current-commit provenance
check valid without another DEXT registration.

### Post-reboot A0 pass and A1 awake-idle failure

This subsection supersedes all earlier next actions.

The Mac rebooted at `2026-07-29T00:10:16Z`. The complete locked audit found one
active arkey v7 registration at
`/Library/SystemExtensions/443F48B3-355D-448A-9413-99CB8DBDE7AC/`, with no
historical arkey registration remaining. Kernelmanagerd explicitly selected
unique ID
`4e0fa54f6b09bc1e6274b7dc836eb373616790401964cf3334ebb63876099237`.
The registered executable was mode `0755`, its SHA-256 was
`2737afd33a8cbbe1c178602cd94e46b64f7349424f5e1ff4328f89a90dd563f6`,
and its CDHash was `7b9d17f219d094d904d4758880b9bf1d3770e057`. Those values matched the
installed and built v7 DEXT. The installed and built app executable SHA-256 was
`7b4fedf967a92ba57dc34e8f637438561ca5ada4e46b3e94050d4e6d6543ba07`,
and the install provenance named clean commit `c380ab4d0`.

The UT4G bridge was connected at 40 Gb/s, all four AMD PCI functions were
present at x16 and 16.0 GT/s, and the `tinygpu` IOUserService published the
expected unique ID and CDHash. Selector 4 returned protocol v1 capabilities
`3`. Selector 5 returned `active_healthy`, initially with `33/33` successful
ticks and then with `51/51` in the formal A0 sample; both had zero failures,
zero active workload resources, zero gaps over leeway, and a 1100 ms maximum
success gap.

Formal A0 passed in
`docs/task_workflow/output/egpu-usb4-persistent-pcie-A0-20260729T001505Z-1319.json`.
Formal A1 then stopped before its first status sample because its endpoint check
returned false. The preserved failure artifact is
`docs/task_workflow/output/egpu-usb4-persistent-pcie-A1-20260729T001521Z-1374.json`.
No 120-second continuity interval ran.

This was an actual link loss, not a transient `system_profiler` result. At
`2026-07-29T00:15:17.123Z`, the kernel recorded repeated ACIO Gen2/3 errors on
both lanes with codes `83`, `84`, `87`, and `88`. At `00:15:17.149Z`,
`AppleT8132PCIeC` disabled `pcic0-bridge`, found the ASM2464 child dead, and
marked `1002:744c`, `1002:ab30`, `1002:7446`, and `1002:7444` dead. It then
force-closed `tinygpu`, removed all four functions, and stopped using the
Thunderbolt tunnel. The last healthy A0 status transaction completed at
`00:15:02.960Z`, less than fifteen seconds earlier.

After the failure, the USB4 bridge remained visible but `SPPCIDataType` was
empty, `ioreg -r -n tinygpu -l` returned no service, and native keepalive status
was unavailable. Two v7 DriverKit processes from the two short-lived provider
instances remained at PIDs 308 and 1263 at capture time. No TinyGPU Unix-socket
server was running and no workload lease, BAR mapping, DMA allocation, AMD
initialization, reset, replug, power cycle, or sleep transition occurred.

The result reopens the keeper design boundary. It proves that the current
DriverKit config-read implementation can report a healthy one-Hz history until
immediately before the known ACIO/CLx-style failure, but it does not establish
whether later timer callbacks stalled or whether
`IOPCIDevice::ConfigurationRead32` failed to generate the same tunneled traffic
as the historical user-space config read. The current scope explicitly limits
the keeper to that config read and forbids changing ASPM/CLx or power policy, so
do not guess at a power override or broaden the keeper action in place.

Stop hardware qualification here. Do not retry A1, run M0-M7, reset/replug the
GPU, reinstall v7, or request another reboot. Before another DEXT version or
operator action, define and CPU/build-test one bounded diagnostic/design
candidate that can discriminate timer progress from a shadowed/non-tunneling
config read, and batch all required lifecycle changes into that single version.

## 2. Evidence boundary: facts versus hypotheses

### Established facts

1. Exactly one arkey v7 DEXT registration is enabled after reboot; the legacy
   signed v3 DEXT remains disabled.
2. A0 passed with matched v7 provenance, a healthy selector-5 payload, and the
   admitted `1002:744c` endpoint present at x16 and 16.0 GT/s.
3. A1 stopped at its first endpoint check after ACIO link errors caused macOS to
   remove the tunneled PCIe tree; the USB4 bridge remained present.
4. The AMD interface selector tries KFD first, then the macOS PCI interface.
5. The failing PCIIface path reaches `tinygrad/runtime/support/am/amdev.py` and
   fails in `SMU.is_smu_alive()` rather than at device discovery, handshake, or
   BAR mapping. The exact failing operation is the 32-bit write of zero to BAR5
   register `mmMP1_SMN_C2PMSG_90` at the start of `GetSmuVersion`.
6. Python evaluates `self.psp.is_sos_alive() and self.smu.is_smu_alive()` with
   short-circuit semantics. Reaching the SMU call proves that the immediately
   preceding PSP C2PMSG81 read returned nonzero: sOS was already alive.
7. This attempt had not entered `AMDev.init_hw`, `AM_GMC.init_hw`,
   `AM_PSP.init_hw`, a mode1 reset, PSP firmware loading, or
   `AM_GMC.setup_psp_gart`. The ordinary invocation does not set
   `AM_PSP_SYSMSG1_GART`; that experimental PSP-GART path is opt-in and was not
   selected.
8. `AMDev.init_sw` is not read-only. `AMMemoryManager` zeroes a 4 KiB root page
   table through BAR0, and `AM_PSP.init_sw` zeroes a 4 KiB fence allocation
   through BAR0 before the later SMU liveness check. The complete successful and
   failing operation order has not yet been captured.
9. The historical bridge at `554800bef` used the same AM constructor branch,
   `is_sos_alive`, `is_smu_alive`, and direct aligned 32-bit SMU store shape.
   The AMD algorithm at this boundary is therefore not a new v5 code path.
10. The historical bridge did not issue bulk BAR writes like v5. After observed
    4 KiB BAR0 failures, commits `8e1959f41` and `26387e206` split slice writes
    into 32-bit stores and read each stored dword back before the next store.
    Current `server.c::mmio_copy` instead emits a tight loop of volatile 32-bit
    stores without those completion fences.
11. The historical one-Hz keepalive ran only from the single-threaded server's
    idle `select`/`accept` timeout. It could not overlap a request handler. The v5
    provider timer is independent while app-side mapped BAR loads/stores bypass
    the provider queue. These concurrency models are not equivalent.
12. The Python remote transport classifies a clean socket EOF as
    `TinyGPU disconnect`; its atexit lease release can then see `EPIPE`.
13. `APLRemotePCIDevice.__init__` starts the audited TinyGPU app server and
    acquires one workload lease before AMD initialization continues.
14. `tinygrad/runtime/support/system.py` suppresses the app server's stdout and
    stderr when Python starts it.
15. The native server's I/O service termination callback currently calls
    `_exit(0)`, which terminates the server process instead of returning a
    structured provider-lifecycle error to the client.
16. Native DEXT RPC helpers currently collapse `IOConnectCall*` return values to
    success/failure, losing the `kern_return_t` needed to distinguish bad
    arguments, not-ready state, device loss, and provider termination.
17. Workload BAR mappings are created through the DriverKit user client, but
    `server.c` subsequently dereferences the mapped BAR directly. Those accesses
    are not executed by the provider's serial operation gate. Apple's DriverKit
    queue contract serializes submitted queue blocks; mapping an
    `IOMemoryDescriptor` into a client does not submit the client's later memory
    accesses to that queue.

### Hypotheses to discriminate; none is yet the root-cause claim

- The pre-existing PSP/SMU state requires the existing mode1-reset branch, but
  the transport/provider fails before SMU responsiveness can be established.
- One or more earlier tight-loop 4 KiB BAR0 writes destabilize the TinyGPU path,
  and the later C2PMSG90 write is only where provider termination becomes
  observable.
- The C2PMSG90 write itself is invalid for the current GPU/provider state and
  causes a device or provider termination.
- A DriverKit user-client, BAR-map, or `IOPCIDevice` lifetime error terminates the
  provider when the mapped region is touched.
- Direct app-side BAR access races the provider's one-second config-read timer or
  stop/termination path because it bypasses the provider gate.
- The provider returns a meaningful IOKit error, but the server's error collapse,
  termination callback, and suppressed logs hide it as EOF.
- A stale/rebound DEXT/provider instance is involved after repeated activation or
  failed probes; this must be ruled out with provider generation, IORegistry, and
  process identity evidence rather than inferred from process count.

The implementation must not choose among these hypotheses by adding an AMD
reset, unsafe small-BAR discovery, `AM_REMOTE_UNSAFE_INDIRECT_VRAM_WRITE`, or a
power-cycle workaround.

## 3. Objective and invariants

Make the v5 TinyGPU path fail diagnostically and safely at runtime, then make the
minimal AMD computation pass without weakening the DriverKit keeper contract.

Required invariants:

1. A provider/service termination is reported as a typed provider-unavailable
   or device-lost result, with the originating RPC selector and IOKit status
   retained in local diagnostic evidence where the protocol permits it.
2. The server never dereferences a BAR mapping after provider termination, and it
   never calls `_exit(0)` from an IOKit interest callback.
3. Client disconnect, provider termination, and failed lease release are
   idempotent and cannot produce a second unsafe RPC on a dead connection.
4. Keepalive state is independent of workload lease state. A failed workload
   operation cannot silently claim that the keeper is healthy.
5. Provider timer reads, config operations, reset, BAR/DMA admission, stop, and
   provider close have a documented serialization boundary. Direct mapped BAR
   access may not be called “gate-serialized” without an implementation and test
   proving that property.
6. No workload BAR, DMA, shared-memory, or AMD runtime state survives a failed
   client or provider termination.
7. Python preserves the first meaningful runtime failure. It must not reduce a
   reachable endpoint plus a failed runtime operation to a discovery-only
   `No interface` diagnosis.
8. The minimal probe remains the exact correctness test:
   `[1, 2, 3, 4] -> x*x + 1 -> [2.0, 5.0, 10.0, 17.0]`.
9. A passing status/handshake, PCI enumeration, or app liveness check is not a
   compute pass.
10. No root-cause claim may name C2PMSG90, PSP, GART, the keeper timer, or BAR0
    bulk ordering until R0 records every operation from lease acquisition to
    termination and the discriminating microgates isolate the first causal
    boundary.
11. Runtime qualification explicitly rejects inherited behavior-changing AMD
    experiment variables. In particular, `AM_PSP_SYSMSG1_GART`, all PSP/GART
    setup/audit controls, `AM_PSP_GMC_INIT_TRACE`,
    `AM_PRE_PSP_MODE1_RESET`, `AM_RESET`, unsafe indirect VRAM, and unsafe
    small-BAR discovery must be unset. The harness records and enforces the
    effective allowlist rather than trusting the caller's shell.
12. No PSP/GART setup experiment is part of this task. Reopening one requires a
    new evidence-backed scope after M7, not an ad hoc response to this transport
    failure.

## 4. Workstream R0: reproduce and instrument the boundary

Before changing runtime behavior, create a lock-held diagnostic path that records
one attempt end-to-end:

- repository commit, dirty state, worktree, lock owner, macOS/Xcode/SDK;
- installed app/DEXT bundle IDs, versions, hashes, signing/provenance;
- `systemextensionsctl list`, PCI identity/link, IORegistry provider identity,
  provider generation, and keepalive status before the attempt;
- exact effective environment, enforcing the five admitted runtime settings
  (`DEV`, `JIT`, `PYTHONPATH`, `AM_REMOTE_DISCOVERY_PROFILE`, and
  `AM_REMOTE_SKIP_RESIZE_BAR`) and rejecting inherited `AM_PSP_*`, `AM_RESET`,
  `AM_PRE_PSP_MODE1_RESET`, `AM_REMOTE_SMALL_BAR_DISCOVERY`,
  `AM_REMOTE_UNSAFE_INDIRECT_VRAM_WRITE`, and `REMOTE_KEEPALIVE_S`;
- handshake, lease acquire, BAR map, config read, and every MMIO operation in
  order, including BAR, offset, width/byte count, read/write, completion status,
  and whether a write used a completion fence;
- the first failing command, BAR, offset, width, byte count, and operation type;
- provider/service state and endpoint visibility after the failure; and
- client/server/DEXT process identity and exit status.

The Python-launched server must offer a development-only bounded diagnostic log
or inherited log descriptor. It must not write credentials, private environment
values, or uncontrolled raw logs into Git. Suppressed `stdout`/`stderr` is not
acceptable for a runtime blocker because it prevents distinguishing an IOKit
error from a provider termination.

R0 must add no recovery action. If the endpoint disappears, preserve evidence and
stop the attempt.

### R0.1 Historical parity audit

Treat the old bridge as behavioral evidence, not code to copy blindly. Before a
hardware write, record an explicit old-versus-v5 parity table for:

- exact aligned 32-bit MMIO reads/writes;
- bulk BAR0 read/write chunking and per-dword readback fences;
- request-handler serialization relative to keepalive traffic;
- connection/provider termination behavior;
- dirty-state admission and cleanup; and
- AMD constructor, PSP/SMU liveness, and default msg1 allocation selection.

Initial source audit:

| Boundary | Historical working path | v5 path | Disposition |
|---|---|---|---|
| Aligned 32-bit MMIO | one direct 32-bit access | one direct 32-bit access in `mmio_copy` | shape matches; lifetime/result still unproven |
| Multi-dword BAR write | 32-bit chunks with same-dword readback after every store | tight volatile 32-bit store loop | correctness regression candidate; restore/prove before hardware |
| Keepalive versus request | idle timeout in the same server loop | independent provider timer; mapped access outside provider gate | concurrency differs; serialize bootstrap MMIO with timer |
| AM liveness branch | `is_sos_alive()` then `is_smu_alive()` | same branch and register protocol | not an AMD algorithm migration |
| PSP msg1/GART | default VRAM msg1 unless experiment enabled | same default; `setup_psp_gart` remains opt-in | keep all PSP/GART experiments disabled |
| Provider termination | lower app could exit; outer bridge retained dirty/error context when possible | app callback `_exit(0)` becomes socket EOF and cleanup `EPIPE` | preserve native error and first failure |

Each difference must be either restored for correctness, proven equivalent by a
CPU/native test, or isolated by one bounded hardware microgate. "The old bridge
worked" and "the new path uses the same register" are not parity proofs.

### R0.2 Truly read-only firmware-state gate

After lifecycle/error instrumentation is installed, but before any BAR write,
capture a lock-held read-only snapshot of PCI command/state, AMD scratch state,
PSP C2PMSG33/35/36/81, SMU C2PMSG90/92, and the relevant MMHUB invalidate
SEM/REQ/ACK/CID2 registers. Classify without mutating:

- `preboot_ready`: sOS is zero and the PSP bootloader mailboxes match the known
  pre-KDB baseline;
- `existing_sos`: sOS is nonzero and all sampled registers remain readable and
  non-all-ones; this is a normal input to AMDev's existing mode1-reset decision,
  not automatically a clean pre-KDB state;
- `inaccessible`: endpoint/provider loss or all-ones MMIO; stop; or
- `ambiguous`: unexpected but readable mailbox state; stop and preserve it.

Do not call `remote_psp_setup_clean_gate` or enable
`AM_PSP_GMC_INIT_TRACE` for this gate: both can release/write the MMHUB invalidate
semaphore. The historical `psp-clean-gate` label "DIRTY" for sOS-alive applied to
fresh KDB/GART experiments; it must not be reused as a universal runtime-state
classifier. This gate performs no SMU command, reset, mailbox clear, semaphore
release, firmware load, or GART programming.

## 5. Workstream R1: native server/provider lifecycle errors

### R1.1 IOKit error preservation

Refactor `dext_rpc` and each native call site so the original `kern_return_t` is
retained and classified. At minimum distinguish:

- invalid request or unsupported selector;
- not-ready/no active lease;
- busy/resource lifetime conflict;
- no device/provider termination;
- map/DMA failure; and
- transport/socket failure.

The wire response must remain conformant with the v1 specification. If a new
typed provider-unavailable code is needed, allocate it in the protocol document
and fixtures first; do not repurpose a frozen selector or make Python infer the
meaning of an empty legacy error.

### R1.2 Service-termination callback

Replace the current `_exit(0)` behavior with a bounded lifecycle transition:

- mark the connection unusable exactly once;
- prevent new RPCs and new leases;
- let the active operation finish or return a provider-unavailable error;
- close/unmap/clear workload resources without using stale IOKit handles; and
- close the client socket with an explicit, machine-readable classification.

The callback must not perform unsafe IOKit calls or race cleanup. The server must
remain able to report the first failure to the diagnostic harness, even if the
provider cannot be reopened in that process.

### R1.3 Server admission and cleanup

Audit handshake, status, lease acquire/release, map, DMA, MMIO, reset, and
disconnect paths for:

- exact request/response framing;
- response on every non-fatal native error;
- no double release of leases, BARs, DMA commands, or file descriptors;
- no lease-release RPC after provider termination;
- no use of a stale `io_connect_t`; and
- deterministic behavior when a second client connects.

Add a server build identity and diagnostic operation sequence to the evidence,
not to a workload-visible success claim.

## 6. Workstream R2: BAR/MMIO access contract

This is the principal design decision and must be resolved before claiming the
operation-gate invariant.

### Preferred design: provider-mediated bounded MMIO

Add explicit, validated user-client methods for the narrow reads/writes required
by the runtime bootstrap. The provider executes them on its serial gate, checks
lease/state/range/alignment, invokes the `IOPCIDevice` memory operation, and
returns a typed result. The server does not retain or directly dereference an
unbounded BAR mapping for those operations.

Correctness must preserve the historical ordering that made the old path work:

- exact aligned 32-bit reads and writes remain one transaction;
- multi-dword BAR reads are split into ordered 32-bit reads;
- multi-dword BAR0 writes are split into ordered 32-bit writes with a readback
  of the stored dword before issuing the next write;
- a single BAR5 register write is not given an invented readback, because AMD
  register semantics supply their own command/response polling; and
- keeper config reads and every provider-mediated MMIO operation execute on the
  same serial provider queue.

This is the safe baseline, not a permanent throughput claim. A later batching
optimization requires evidence that it preserves completion ordering and does
not recreate the tight-store failure from `26387e206`.

### Alternative design: explicit direct-map semantics

If direct BAR mappings are required for workload throughput, document and prove:

- mapping ownership and lifetime;
- what happens when the provider terminates;
- how timer/config/stop operations synchronize with user-space loads/stores;
- how stale mappings become unusable;
- how invalid offsets and widths are blocked; and
- how all mapping counters reach zero before provider close.

The current provider gate alone does not prove this: `server.c`'s `mmio_copy`
executes in the app process after mapping. Direct mapping may remain for a later
performance phase, but it cannot be used for R0-R6 bootstrap acceptance unless
these semantics are implemented and tested.

### Required access matrix

Use the smallest lock-held microprobes, in order, stopping at first failure:

| Probe | Operation | Purpose |
|---|---|---|
| M0 | handshake/status | provider and protocol readiness |
| M1 | lease acquire/release | workload admission and cleanup |
| M2 | lease plus BAR metadata/map, no dereference | mapping lifetime |
| M3 | config read offset 0/width 4 | safe provider-gated PCI access |
| M4 | truly read-only R0.2 firmware/MMHUB snapshot | state classification and bounded read path |
| M5a | controlled `AMDev.init_sw`-only path, stopping before liveness/reset/init_hw | normal 4 KiB BAR0 zero/write-readback ordering |
| M5b | complete SMU `GetSmuVersion` mailbox transaction when R0.2 reports `existing_sos`; not applicable for `preboot_ready` | exact failing boundary without leaving an isolated mailbox clear |
| M6 | AMD `Device["AMD"]` initialization with the admitted environment only | SMU/bootstrap integration without PSP/GART experiments |
| M7 | tracked four-value minimal compute | end-to-end correctness |

M5a must use the normal root-page-table and PSP-fence allocation behavior, not a
guessed VRAM scratch address. M5b is the complete existing three-write/poll SMU
protocol only; do not issue C2PMSG90=0 as a standalone probe. If R0.2 returns
`preboot_ready`, the ordinary AMDev short-circuit does not query SMU before PSP
initialization, so M5b is recorded as not applicable rather than forced. No PSP
mailbox write, firmware load, GART setup, queue submission, reset, or benchmark
is authorized before M6.

## 7. Workstream R3: DriverKit state and gate audit

Audit `TinyGPUDriver.cpp` and `TinyGPUDriverUserClient.cpp` for the following:

- provider start validates identity before publishing `tinygpu`;
- all provider-owned PCI calls use the provider gate;
- timer disable and callback drain complete before timer/action/PCI release;
- provider stop cannot return busy while a client cleanup path still holds a
  lease, BAR, or DMA reference;
- user-client gate recursion cannot deadlock with provider gate calls;
- user-client close and provider stop cannot race resource counters;
- `CfgRead`/`CfgWrite`/map/DMA paths return the actual IOKit result;
- a provider termination invalidates all client-visible mappings and handles; and
- status reports `active_workload_leases`, `active_bar_mappings`, and
  `active_dma_allocations` consistently before and after every microprobe.

Do not broaden the keeper policy. The one-Hz config read remains the only
keepalive action; workload bootstrap must not be folded into the keeper timer.

## 8. Workstream R4: Python error and lifecycle handling

Update `tinygrad/runtime/support/system.py` only after the native wire behavior
and fixtures are fixed:

- preserve `TinyGPUWireError` kinds for provider-unavailable, device-lost,
  partial-read, partial-write, timeout, typed native error, and legacy error;
- attach the operation context (`handshake`, `lease`, `map`, `cfg`, `mmio`, or
  `release`) without exposing secrets;
- do not retry a failed MMIO or lease operation automatically;
- make close/lease-release idempotent and suppress only a known dead-transport
  cleanup error after recording the original failure;
- make interface selection distinguish “no device discovered” from “device
  discovered but runtime initialization failed”; and
- keep `REMOTE_KEEPALIVE_S` unsupported and reject unsafe small-BAR overrides.

The final Python exception should identify the first runtime failure and preserve
the nested cause. It must not turn a provider crash into a false zero-device
diagnosis.

## 9. Workstream R5: tests and fixtures

CPU-only tests must cover:

- the effective-environment allowlist and rejection of every behavior-changing
  PSP/GART/reset/unsafe override named in invariant 11;
- pure R0.2 classification for `preboot_ready`, `existing_sos`, `inaccessible`,
  and `ambiguous`, with no write-capable helper call;
- split/partial headers and payloads;
- typed native errors and legacy generic errors;
- EOF during handshake, lease, response, and cleanup;
- provider termination during each RPC phase;
- repeated close and release after dead transport;
- second-client busy behavior;
- invalid BAR/offset/width/size and reserved fields;
- map lifetime and counter cleanup;
- exact error preservation from every native `IOConnectCall*` path;
- Python interface-selection error classification; and
- server diagnostic logging without credential/environment leakage.

Native tests or reviewable harness assertions must cover:

- operation-gate serialization;
- exact 32-bit versus multi-dword MMIO dispatch;
- per-dword BAR0 write/readback ordering and stop-on-first-error behavior;
- timer callback versus MMIO/config/stop;
- provider termination while a BAR is mapped;
- lease disconnect with BAR/DMA state;
- no `_exit` from an interest callback; and
- one and only one provider close.

Protocol fixtures remain independently declared in C and Python and are checked
against `extra/usbgpu/protocol/tinygpu-wire-v1.md`.

## 10. Workstream R6: locked hardware gates

Hardware work resumes only after R0-R5 CPU/build checks pass and a fresh audited
v5 install provenance check succeeds.

Required order:

1. R6.0: v5 enabled, legacy disabled, endpoint/link up, provider IORegistry
   identity present, keepalive handshake/status valid.
2. R6.1: M0-M3 pass with no resource growth.
3. R6.2: M4, M5a, and conditionally applicable M5b produce classified results
   without provider disappearance, stale-handle use, standalone mailbox clears,
   or resource growth.
4. R6.3: M6 reaches AMD initialization or returns a precise supported runtime
   error; it may not report discovery failure.
5. R6.4: M7 returns `[2.0, 5.0, 10.0, 17.0]` and status/counters remain valid.
6. Repeat the existing A0 and A1 evidence gates.
7. Only then rerun A2 and proceed to A3-A9. A4/A8 idle claims remain gated on
   the authoritative keeper matrix; A10/A11 remain separate classifications.

Every failed hardware gate stops immediately and records endpoint, provider,
status, process, and resource evidence before any operator action. No automatic
reset, hotplug, power cycle, or sleep transition is part of R6.

## 11. Ownership and files

| Area | Primary files | Responsibility |
|---|---|---|
| DriverKit lifecycle/gate | `extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.cpp` and headers | provider state, termination, timer, serialized PCI operations |
| User client | `TinyGPUDriverUserClient.cpp` and IIG | validated RPC selectors, leases, maps, DMA, typed IOKit results |
| Unix RPC | `extra/usbgpu/tbgpu/installer/Shared/server.c` | framing, error preservation, provider-dead transition, cleanup, diagnostics |
| Python client | `tinygrad/runtime/support/system.py` | protocol decoding, operation context, interface error classification, cleanup |
| AMD boundary | `tinygrad/runtime/support/am/amdev.py`, `tinygrad/runtime/support/am/ip/smu.py` | diagnostic context only unless a separate runtime defect is proven |
| Qualification | `extra/usbgpu/tests/qualify.py`, `minimal_amd_compute.py` | microprobes, stop conditions, evidence artifacts |
| Wire contract | `extra/usbgpu/protocol/tinygpu-wire-v1.md` and fixtures | IDs, framing, errors, bounds |
| Workflow evidence | `docs/task_workflow/output/` | locked diagnostic and gate artifacts |

Do not modify AMD algorithms, queue scheduling, model code, power policy, or
keepalive cadence under this task unless R0 proves the failure occurs there.

## 12. Definition of done

This blocker is resolved only when all of the following are true:

- a locked diagnostic artifact identifies the original failing operation and its
  native result;
- native server/provider errors are typed, observable, and lifecycle-safe;
- provider termination cannot call `_exit(0)` or leave stale client mappings;
- the BAR/MMIO access contract is implemented and its gate semantics are tested;
- Python preserves runtime initialization failures distinctly from discovery
  failures and cleans up without a secondary broken-pipe diagnosis;
- CPU/native/protocol tests pass;
- the old-versus-v5 parity audit is complete and the bulk-write ordering and
  keeper/MMIO serialization differences are resolved;
- a truly read-only preflight classifies firmware state without running the
  semaphore-writing PSP setup gate;
- M0-M6 pass under the GPU lock without endpoint/provider disappearance; and
- M7 passes with the exact four-value result, followed by fresh A0/A1 and A2
  acceptance evidence.

Until then, the correct project status is: **v5 installed and activated;
runtime compute qualification blocked by a native TinyGPU provider/RPC
disconnect during AMD initialization**.
