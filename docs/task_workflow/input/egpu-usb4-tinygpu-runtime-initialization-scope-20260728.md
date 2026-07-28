# TinyGPU v5 runtime initialization and RPC-disconnect scope

Date: 2026-07-28

Status: open; implementation and hardware qualification are blocked at the first
AMD runtime probe. This scope is a follow-on to
`egpu-usb4-persistent-pcie-service-scope-20260727.md`. It does not replace the
DriverKit-owned keepalive architecture or its A0-A11 acceptance matrix.

Repository/worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, branch `exp`.

GPU discipline: every command that can inspect, open, initialize, or exercise the
eGPU runs under `/tmp/gpu-bench.lock` through
`extra/usbgpu/tools/with_gpu_lock.py`. No broad benchmark, reset, power-cycle, or
sleep test is authorized by this scope.

## 1. Executive alignment

The v5 installation problem is resolved sufficiently to proceed to runtime
diagnosis:

- `org.tinygrad.arkey.tinygpu.driver2` version `1.0.0/5` is
  `[activated enabled]`.
- The legacy `org.tinygrad.tinygpu.driver2` version `1.0.0/3` is
  `[activated disabled]`.
- The audited app reports the extension as ready.
- macOS has observed the AMD endpoint `1002:744c` with the PCI link up at
  16.0 GT/s, and tinygrad has observed `['1002:744c']` in its macOS PCI scan.

Those observations establish installation, activation, and enumeration only.
They do not establish that the provider can safely service AMD runtime MMIO,
DMA, or queue initialization.

The current blocker is:

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

The last locked minimal probe failed with `TinyGPU disconnect` while
`AMDev.is_smu_alive()` was issuing the first write in the standard SMU
`GetSmuVersion` mailbox transaction: a 32-bit zero to response register
`mmMP1_SMN_C2PMSG_90`. Cleanup then reported a broken pipe while releasing the
lease. After the probe, the app-level keepalive status command reported
unavailable even though the system extension remained enabled; this is a
provider/service-lifecycle failure, not a sudo or approval failure. It is the
first *observed failing* register write, not proof that it was the first runtime
write or the root cause.

## 2. Evidence boundary: facts versus hypotheses

### Established facts

1. The v5 DEXT is enabled and the legacy DEXT is disabled.
2. The USB4 bridge can be present while the PCI endpoint is absent, and the PCI
   endpoint can later re-enumerate without reinstalling the DEXT.
3. When the endpoint is present, the macOS PCI scan returns `1002:744c` and the
   AMD runtime device table admits `0x744c`.
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
