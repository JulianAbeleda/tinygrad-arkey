# USB4 eGPU persistent PCIe service scope

Date: 2026-07-27

Status: authoritative input

Repository and native-source owner: `tinygrad-arkey`, under `extra/usbgpu/`

Task owner: assigned implementation owner; no implementation begins without one named in the phase-0 artifact.

Branch and worktree: one dedicated `feature/egpu-usb4-keeper` branch and worktree. Do not implement, build, install, or qualify from the production worktree.

GPU exclusion: every command that can open, initialize, exercise, install against, or inspect the eGPU while it is active must hold `/tmp/gpu-bench.lock`. Record lock owner and worktree in every artifact.

Target host: Apple Silicon Mac mini M4

Target device: AMD Radeon RX 7900 XTX, PCI identity `1002:744c`, gfx1100

Target transport: Apple USB4 root complex -> UT4G/ASM2464-class PCIe tunnel -> PCIe riser -> GPU

This document supersedes all earlier work-package plans and remediation addenda for this task. It is the only implementation order, frozen status-semantics authority, and acceptance authority. P1 creates `extra/usbgpu/protocol/tinygpu-wire-v1.md` as the sole wire-encoding authority. Execution output belongs under `docs/task_workflow/output/`.

## 1. Objective

Keep the USB4-attached AMD GPU enumerated and usable while the Mac is awake, including periods with no tinygrad process, without retaining workload DMA or BAR mappings between clients.

The native TinyGPU DriverKit provider owns link liveness for its bound device. TinyGPU.app remains transport and workload-lifetime management. Python is a client, never the production keeper. The admitted policy performs a harmless, periodic PCI configuration read; it does not keep the GPU busy, change PCIe power policy, or reset hardware.

Completion requires tracked native source, a versioned and independently implemented native/Python wire contract, an audited development-installed build with recorded provenance, awake-idle evidence beyond the historical failure window, and post-idle tinygrad compute including the canonical Qwen3 8B smoke workload.

## 2. Evidence and problem boundary

- Replacing the old PCIe riser changed the GPU from non-enumerating to enumerating. The former riser was a signal-path blocker.
- With the new riser, macOS enumerated the RX 7900 XTX, TinyGPU opened `AMD gfx1100`, and a minimal tinygrad computation returned `[2.0, 5.0, 10.0, 17.0]`.
- `docs/egpu-usb4-link-keepalive.md` records a prior awake-idle link loss, with USB4/PCIe errors around ASPM/CLx and failed retraining. A PCI config-space read at offset `0`, width `4`, every second prevented that failure in the former Python remote bridge. Commit `554800bef` is evidence only, not an implementation authority.
- The installed TinyGPU executable did not contain a keeper implementation, and `REMOTE_KEEPALIVE_S=1` is not consumed by the current Python client. It is not evidence of protection.
- Native source was pruned at `4c5e67cff`. The bounded historical source immediately before that commit is the restoration starting point, but all restored protocol values must be reconciled with current Python before behavior changes.
- That historical source is currently recovered under untracked `extra/usbgpu/`, and an unsigned Debug DriverKit prototype build succeeded. The recovered source and successful build are starting evidence only: the source is not yet an owned, review-qualified implementation and the build is not installed provenance.

Keep these failure classes separate in artifacts and conclusions:

| Class | Observation | Task handling |
|---|---|---|
| Signal/enumeration | Device never appears | Observe only; not solved by a keeper |
| Awake-idle link loss | Endpoint disappears while Mac is awake and idle | Primary target |
| Load/power dropout | Endpoint resets/disappears during load | Separate PSU, cabling, riser, and thermal classification |
| AMD runtime failure | Endpoint remains but AMD boot/runtime fails | Preserve and classify separately |
| App lifecycle failure | Socket/app fails while endpoint remains | Transport fix, not link proof |
| Sleep/wake failure | Failure across macOS sleep | Separate, post-initial qualification gate |

## 3. Goals, non-goals, and invariants

Goals:

1. Keep the admitted `1002:744c` endpoint present through awake-idle qualification.
2. Tie periodic traffic to the DriverKit provider lifetime rather than tinygrad or TinyGPU.app lifetime.
3. Give each workload lease fresh BAR maps, DMA state, shared memory, AMD runtime state, and socket state.
4. Expose read-only status proving actual native keeper activity.
5. Serialize keepalive reads, configuration RPCs, reset, provider stop, and provider termination.
6. Fail visibly without automatic reset, hotplug, USB4 reset, or power cycle.

Non-goals:

- Do not emulate a root complex, run dummy kernels, retain VRAM allocations, or use `system_profiler` polling as a keeper.
- Do not modify ASPM/L1/L1SS/CLx policy, AMD firmware boot, queue submission, small-BAR discovery, or benchmark hot paths in this task.
- Do not use `AM_REMOTE_SMALL_BAR_DISCOVERY=1` for ordinary execution.
- Do not infer awake-idle, load-power, or sleep/wake results from each other.
- Do not commit build products, signing assets, provisioning profiles, downloaded applications, model files, or raw private logs.

Required invariants:

1. The provider is fully active and eligible to tick, or stopping/stopped and unable to tick.
2. All PCI operations use one provider operation gate. No timer read races reset, config RPC, stop, or provider close.
3. Timer disable waits for callback drain before timer, action, queue, or provider release.
4. A successful tick is a matching config DWORD `0x744c1002`, not merely a callback fire.
5. `0xffffffff`, `0x00000000`, a wrong value, or unavailable provider is a failed tick.
6. Keeper failures only update state, counters, logs, and evidence. They never trigger recovery reset or power action.
7. Persistent provider state contains identity, policy, state, counters, and lifecycle synchronization only. It owns no workload DMA, BAR, shared-memory, firmware, ring, or queue state.
8. Socket disconnect releases its workload lease but cannot disable the provider keeper.
9. At most one workload lease is active until explicit multi-client semantics are designed.
10. A client cannot claim keepalive support from an environment variable or app liveness; it needs a successful capability/status exchange.

## 4. Architecture and state model

```text
Mac PCI/USB4 owner
  -> TinyGPU DriverKit provider: provider state + policy + operation gate + timer
    -> TinyGPU user client: read-only status plus one workload lease
      -> TinyGPU.app: Unix-socket framing and lease cleanup
        -> tinygrad APLRemotePCIDevice: independent protocol endpoint
```

The selected architecture is DriverKit-owned only. An app-owned idle holder is rejected for this release because it makes link liveness depend on an app/LaunchAgent, conflicts with the no-app-server counter gate, and creates a competing acceptance branch. If DriverKit cannot be made signable or reliable, record the blocker and return to design; do not silently substitute a fallback.

Provider states are `DETACHED`, `STARTING`, `ACTIVE_HEALTHY`, `ACTIVE_DEGRADED`, `QUIESCING`, and `STOPPED`. Workload states are independently `NO_CLIENT`, `CONNECTING`, `ACTIVE_CLIENT`, and `DRAINING`.

Required transitions:

```text
DETACHED -> STARTING -> ACTIVE_HEALTHY
STARTING -> STOPPED                         identity or timer setup failure
ACTIVE_HEALTHY -> ACTIVE_DEGRADED           invalid/missed identity read
ACTIVE_DEGRADED -> ACTIVE_HEALTHY           subsequent matching read
ACTIVE_* -> QUIESCING -> STOPPED            stop, termination, unplug
ACTIVE_* -> QUIESCING -> ACTIVE_*           explicit reset after drained lease
```

`ACTIVE_HEALTHY` is published only after provider open, supported identity validation, timer creation and enable, and the first successful identity transaction. There is no unbounded recovery loop. Provider termination stops cleanly; normal macOS reprobe owns replug handling.

Implementation shape:

- Create one serial provider operation gate used by timer callback, config reads/writes, reset, start-finalization, and stop.
- A new DriverKit user-client connection begins diagnostic-only. Status is allowed in that role. An explicit, serialized workload-lease acquisition is required before BAR mapping, DMA preparation, config mutation, reset, or other workload selectors; release/disconnect decrements every provider-tracked resource count exactly once.
- Use a re-armed one-shot `IOTimerDispatchSource`; stop/reset disables it with a completion callback, drains the operation gate, then releases it before closing `IOPCIDevice`.
- Timer tick reads config offset `0`, width `4`, under the gate, validates the full identity DWORD, updates state/counters, and rearms only while active.
- Reset requires an explicit operator request, no active workload lease, timer drain, one reset only, identity revalidation, and fresh timer arm. Keeper failure never calls reset.
- On unplug/termination: reject new work, drain timer, release each lease normally, close once, record last identity/tick, and wait for normal rebind.

## 5. Named policy

The initial named policy is `usb4_amd_744c_v1`:

| Field | Value |
|---|---|
| Identity | `1002:744c` / DWORD `0x744c1002` |
| Read | config offset `0`, width `4` |
| Default enabled | yes |
| Interval | 1000 ms |
| Maximum timer leeway | 100 ms |
| Disabled mode | explicit development-only native policy, with warning/status |

Implement policy as a named table owned by the DriverKit provider, not unexplained literals inside execution functions. Unknown devices are unsupported and cannot report active keeper. Runtime policy override is optional; any override must be bounded, explicit, native, and status-visible. `REMOTE_KEEPALIVE_S` remains unsupported until a negotiated native policy command exists; Python must warn or fail clearly rather than imply it took effect.

Cadence acceptance method: on every success after the first success in a provider generation, the provider computes the monotonic gap from the preceding success. It updates `max_success_gap_ms` and increments `success_gap_over_leeway_count` when the gap exceeds `interval_ms + maximum_timer_leeway_ms` (1100 ms for the admitted policy). Qualification records start/end status with the same `provider_generation` and nondecreasing, unsaturated counters. `observed_gaps` is `successes_end - successes_start`; `over_leeway_gaps` is the corresponding counter delta. Passing requires `observed_gaps > 0`, `over_leeway_gaps / observed_gaps <= 0.01`, cumulative per-generation `max_success_gap_ms <= 2000`, and `failures_end == failures_start`. A generation change, counter decrease/saturation, or timer stall fails the interval even if the endpoint remains present. The 2000 ms ceiling detects a missed one-shot rearm.

## 6. Protocol requirements and frozen status semantics

P1 must add `extra/usbgpu/protocol/tinygpu-wire-v1.md` and machine-readable fixtures under `extra/usbgpu/protocol/fixtures/`. That document becomes the sole wire-encoding authority and must allocate every command/selector ID; define request, response, error, and handshake framing; fix little-endian field widths and reserved values; define capability bits and version negotiation; define UTF-8 JSON encoding and maximum payload bytes; specify every enum, type, range, unit, and malformed-input result; and record existing command IDs/layouts unchanged. No handshake/status endpoint code may land before this specification and its fixtures.

Native C and Python each retain independently declared wire identifiers and independently implemented encoders/decoders. Do not generate shared declarations across endpoints: the fork's coding override intentionally requires duplicate protocol identifiers at independent client/server boundaries. Prevent drift with conformance tests that compare each endpoint to the wire specification and test real encoded fixtures.

Rules:

1. Preserve the common native/Python command IDs `0..11` and their layouts. Python-reserved IDs `12..14` (`PING`, `HEALTH`, `SYSMEM_SYNC`) cannot be reused; P1 either implements their current Python semantics natively or marks them reserved/unsupported. Handshake, lease, and status IDs are allocated at `>=15` only in the wire specification.
2. On a new server, the first new request is a side-effect-free handshake. A client must not send status/policy commands to an unknown server.
3. A new-server handshake returns protocol major/minor, capabilities, server build identity, and typed unsupported-version/capability errors.
4. For the recovered legacy server, the client sends only the bounded handshake probe selected by the wire specification. A complete legacy 17-byte response with `status=RESP_ERR`, `resp0=0`, and `resp1=0`, or a clean EOF before any response byte, maps locally to `unsupported_protocol`; the client closes immediately and never sends status. Timeout, partial response, nonzero legacy error length, or any other bytes are protocol errors, not capability absence. The probe uses a strict documented timeout and cannot claim keeper activity.
5. `KEEPALIVE_STATUS` is read-only. `KEEPALIVE_SET_POLICY` exists only when a native override is implemented.
6. Header and payload reads/writes use exact-length helpers. Partial header/payload, oversized length, invalid enum, invalid reserved field, bad range, and disconnect are typed protocol errors or clean client disconnects; none may reach MMIO, DMA, config, reset, or stale buffer contents.
7. Status is side-effect free, uses fixed units, and is never inferred from logs or environment variables.

Frozen `tinygpu.keepalive.v1` status payload:

```json
{
  "schema": "tinygpu.keepalive.v1",
  "provider_generation": 1,
  "state": "active_healthy",
  "enabled": true,
  "policy_id": "usb4_amd_744c_v1",
  "interval_ms": 1000,
  "maximum_timer_leeway_ms": 100,
  "expected_identity": "1002:744c",
  "last_identity_dword": "0x744c1002",
  "attempts": 3600,
  "successes": 3600,
  "failures": 0,
  "consecutive_failures": 0,
  "last_attempt_monotonic_ns": 3600000000000,
  "last_success_monotonic_ns": 3600000000000,
  "success_gap_over_leeway_count": 0,
  "max_success_gap_ms": 1000,
  "timer_error": 0,
  "counter_saturated": false,
  "active_workload_leases": 0,
  "active_bar_mappings": 0,
  "active_dma_allocations": 0
}
```

`provider_generation` increments for every bind/rebind. Tick/gap counters and `max_success_gap_ms` are cumulative within that generation and reset only on a new generation; `attempts == successes + failures`. All counters/timestamps are unsigned 64-bit values that saturate instead of wrapping; saturation sets `counter_saturated=true`, degrades state, and fails qualification. Active resource counts are unsigned 32-bit values. `timer_error` is a signed 32-bit native `IOReturn`, with zero meaning no timer error. Timestamps are monotonic and meaningful only alongside artifact collection time. States are exactly `unsupported`, `inactive`, `active_healthy`, `active_degraded`, `quiescing`, and `stopped`. The provider compares gaps in nanoseconds before conversion; `max_success_gap_ms` is the ceiling of the largest gap to an unsigned integer millisecond. The P1 wire specification fixes JSON string forms, integer ranges, ordering/whitespace policy, and payload bounds without changing these semantic fields.

## 7. Source, build, and signing authority

Restore only the bounded TinyGPU source needed to build, sign, install, and test under `extra/usbgpu/` from `4c5e67cff^`: provider/user-client C++ and IIG, `Shared/server.c`, `TinyGPUCLIRunner.swift`, `TinyGPUApp.swift`, required Xcode metadata, non-secret entitlements, and reproducible build/install helpers. `tinygrad-arkey` is the sole native source authority for this task; no separate native repository is permitted.

Before behavior changes, reconcile native command values against `tinygrad/runtime/support/system.py`, retain local bundle IDs unless a documented signing requirement requires a change, preserve the existing MMIO-write acknowledgement fix, and ensure restored source is tracked. Installed apps and `build/` directories are evidence, never source authority.

Known signing state: the currently observed app is ad-hoc signed, SIP is disabled on this host, and historical release provisioning profiles are absent. The task must not represent release signing or an Apple-supported production deployment as available. Hardware qualification may use only an audited development install on this named host: `csrutil status` must report disabled; the exact NoSIP entitlements, ad-hoc signatures, bundle IDs, source/build hashes, previous installed-app hash, and system-extension state must be recorded; the install helper must not download an app or delete/replace one without an immediately preceding operator approval; and post-install `codesign` plus system-extension identity checks must pass. This establishes development-test provenance, not production trust. Phase 0 records Xcode/macOS/SDK versions and any remaining install blocker. No credentials, private keys, profiles, or downloaded bundles enter Git.

## 8. Authoritative implementation order

### P0: baseline and ownership

- Create the dedicated feature worktree/branch and name the implementation owner.
- Add and CPU-test the `extra/usbgpu/tools/with_gpu_lock.py` runner before the first hardware observation. This phase-0 tooling change is allowed before native behavior work; it uses the macOS-supported `fcntl.flock` API, retains the lock for the child lifetime, records PID/worktree/command metadata, forwards termination, and returns the child's exit status.
- Record repository commit/dirty state, the current untracked recovery and intended tracked owner under `extra/usbgpu/`, installed app/dext provenance, system-extension state, topology, negotiated link rate, PCI visibility, power topology, and current capability result.
- Acquire `/tmp/gpu-bench.lock` for all eGPU observations and record its owner.
- Write `docs/task_workflow/output/egpu-usb4-persistent-pcie-phase0-<UTC>-<pid>.md` plus the same-stem machine-readable JSON; published filenames contain resolved values.

Gate: the lock runner is CPU-tested, and the untracked recovery state, planned tracked owner, worktree, lock discipline, and known signing state are recorded before native behavior changes. P1, not P0, establishes tracked native source.

### P1: restore source and freeze the wire specification

- Restore bounded native source under `extra/usbgpu/`; exclude build/signing/model artifacts.
- Add `extra/usbgpu/protocol/tinygpu-wire-v1.md` plus its fixtures with every encoding decision required by section 6, then retain independent C/Python constants and codecs.
- Implement handshake and conformance fixtures before status or policy requests.
- Reconcile old server behavior exactly as section 6 defines: the client maps only the documented legacy generic-error/clean-EOF signatures to local `unsupported_protocol`, closes, and sends no status request.

Gate: clean checkout builds the restored project; the named wire specification and fixtures are tracked; fixtures prove existing numeric stability, new framing, and legacy behavior; and no new command is endpoint-only.

### P2: provider operation gate and safe lifecycle

- Introduce `ProviderState`, `ProviderOperationGate`, `WorkloadLease`, and `TransportState` boundaries without a second framework.
- Make diagnostic-only the default user-client role; add explicit workload-lease acquire/release, require that lease for workload selectors, and maintain provider-visible active lease/BAR/DMA counts.
- Route every PCI access, including existing config RPC and reset, through the provider gate.
- Implement timer disable-with-completion and callback drain before release/close.
- Fail provider start closed on identity read, unsupported identity, timer creation, enable, first deadline, or first successful tick failure.

Gate: injected lifecycle tests prove no callback-after-close and no reset/config/tick overlap.

### P3: policy, keeper, and status

- Add the named policy table and initial `usb4_amd_744c_v1` policy.
- Implement the DriverKit one-shot config-read keeper and bounded accounting.
- Add user-client status selector, server forwarding, native CLI `TinyGPU keepalive status`, and negotiated Python query support. The one-shot CLI status path opens only a diagnostic DriverKit user client; it must not enter Unix-socket server mode, acquire a workload lease, map a BAR, or create DMA/shared-memory resources.
- Make legacy `REMOTE_KEEPALIVE_S` explicit unsupported/deprecated unless P3 includes a negotiated native override.

Gate: with no TinyGPU Unix-socket server or workload client running, two one-shot native CLI status queries at least 120 seconds apart prove provider counters advance and match the frozen schema. Both samples have zero active workload leases/BAR mappings/DMA allocations; the short-lived diagnostic processes do not invalidate the no-workload/no-server condition. CLI dispatch tests prove `keepalive status` cannot enter the `server` command path.

### P4: transport and workload hardening

- Replace one-shot socket `recv`/send paths with exact-length helpers and bounds validation.
- Make all lease cleanup idempotent after partial initialization; release BAR, DMA, shared-memory, fd, and user-client state on every disconnect/error path.
- Reject concurrent workload clients until designed otherwise.
- Extend the canonical `extra/qk/bench.py` harness with positive `--decode-duration-s` and `--decode-cycle-timeout-s` options; the timeout defaults to 900 seconds. Duration mode requires explicit `--decode` and rejects prefill/both mode. A cycle runs all selected checkpoint contexts and `--decode-reps` through the existing decode authority. Start at least one cycle, start no new cycle after the duration is reached, and terminate/kill a cycle that exceeds its timeout with a five-second grace. Total wall time is bounded by requested duration plus cycle timeout plus grace. On child failure, timeout, SIGINT, or SIGTERM, stop immediately, forward termination, atomically write a non-passing aggregate artifact, and return nonzero.
- Duration mode writes `tinygrad.qk.decode.duration.v1` at `bench/qk-decode-duration/run-<time>-<pid>.json` or an explicit `--decode-duration-out`. It rejects simultaneous `--decode-out`. The aggregate records resolved argv/environment controls, model path/hash/size, requested/actual duration, timeout/grace, start/end monotonic and wall times, final status, and an ordered cycle list with start/end, exit status, parsed throughput, relative child-artifact path, and child SHA-256. Child artifacts remain the existing decode-authority schema. Add CPU-only argument, loop-bound, failure, signal, and artifact-schema tests; do not create a second benchmark wrapper or alter benchmark hot paths.
- Add a CPU-tested `extra/usbgpu/tests/qualify.py` orchestrator for A0-A11. Every eGPU subprocess and periodic sampler is its descendant, it requires lock-runner metadata, emits `docs/task_workflow/output/egpu-usb4-persistent-pcie-<gate>-<UTC>-<pid>.json` atomically, preserves the first failure, and never automates unplug, power, extension replacement, or sleep. Published filenames contain resolved values.
- For A1, match only the installed executable's `server` argv, send `TERM`, wait at most ten seconds, and fail if it remains. Record process census and configured socket reachability before the first direct status query and after the second. Both status samples must show zero active workload leases/BAR mappings/DMA allocations. Unit tests prove the CLI `keepalive status` dispatch cannot call `run_server`.

Gate: malformed-input and client-churn tests leave a fresh baseline and an uninterrupted provider keeper.

### P5: build, install, and provenance

- Build from the recorded feature commit; verify source membership, hashes, bundle IDs, entitlements, and signing state.
- Install only through the audited development path defined in section 7, with immediate operator approval and before/after provenance. Disable automatic app download/replacement during qualification.
- Do not begin hardware qualification without installed provenance and native counter proof.

Gate: installed app/dext are traceable to the exact source commit and expose the expected handshake/status.

### P6: hardware qualification and integration

- Run the acceptance matrix in order under the GPU lock.
- Stop on unplanned endpoint disappearance, timer stall, reset, malformed status, resource leak, page fault, timeout, or power hazard; preserve evidence before replug/power action. A9's single recorded operator replug is the only planned endpoint-disappearance exception.
- Update durable implementation/result documentation only after evidence is banked.

## 9. File-level ownership

| Area | Responsibility |
|---|---|
| `extra/usbgpu/.../TinyGPUDriver.cpp` and headers/IIG | Policy table, operation gate, state, timer lifecycle, config read, reset serialization |
| `TinyGPUDriverUserClient.*` | Read-only status forwarding; per-client DMA ownership only |
| `extra/usbgpu/.../Shared/server.c` | Wire-conformant framing, handshake/status forwarding, transport/lease cleanup |
| `TinyGPUCLIRunner.swift` | Structured status command without starting a workload |
| Xcode/build/install files | Reproducible provenance, non-secret build configuration, verification |
| `extra/usbgpu/tools/with_gpu_lock.py` | Portable exclusive eGPU command runner and lock-owner metadata |
| `extra/qk/bench.py` | Canonical duration-bounded decode orchestration and artifact data |
| `extra/usbgpu/tests/qualify.py` | Lock-aware A0-A11 orchestration, sampling, first-failure capture, gate artifacts |
| `tinygrad/runtime/support/system.py` | Independent protocol implementation, handshake/status use, explicit env handling |
| focused tests | Wire fixtures, lifecycle, timer, framing, cleanup, integration evidence |
| `docs/egpu-usb4-link-keepalive.md` | Final measured result only |
| `docs/task_workflow/output/` | Phase and final evidence |

Do not modify AMD compute code unless a separately demonstrated failure requires it.

## 10. Test plan and gated commands

These commands become executable only after their P0/P4 runner and harness deliverables pass CPU-only tests. They then run from the dedicated feature worktree through `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- ...`, which acquires `/tmp/gpu-bench.lock` before executing the command. For an acceptance row, the single lock-held child is `extra/usbgpu/tests/qualify.py`; it owns every command and sampler for that row, so no raw status, endpoint inspection, or benchmark command runs outside the lock. Set `MODEL_GGUF` to an existing local Qwen3 8B Q4_K_M GGUF path; the model file is not committed. Before a model command, the orchestrator requires `test -n "${MODEL_GGUF:-}" && test -f "$MODEL_GGUF"`, then records `shasum -a 256 "$MODEL_GGUF"` and `stat -f '%z' "$MODEL_GGUF"`. Published artifacts contain the resolved command and model SHA-256/size, not placeholders.

Common environment:

```sh
export DEV=AMD
export JIT=1
export PYTHONPATH=.
export AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c
export AM_REMOTE_SKIP_RESIZE_BAR=1
```

Minimal compute harness is a tracked test script added by P4 at `extra/usbgpu/tests/minimal_amd_compute.py`. It must allocate `[1, 2, 3, 4]`, evaluate `x*x + 1` on `Device["AMD"]`, transfer the four float results to host, require exact `[2.0, 5.0, 10.0, 17.0]`, and exit nonzero on device, transfer, or value failure. The exact command is:

```sh
.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- \
  .venv/bin/python extra/usbgpu/tests/minimal_amd_compute.py
```

Sustained-load classification uses the canonical benchmark route, not a new benchmark wrapper. Run a 30-minute decode loop with an existing model:

```sh
.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- \
  .venv/bin/python extra/qk/bench.py --model "$MODEL_GGUF" --decode --decode-duration-s 1800
```

The duration flag is a required P4 deliverable and must pass its CPU-only tests before P6. Sustained load is a classification gate, not keeper acceptance.

Host/native tests:

- Wire command numeric/layout fixtures, handshake negotiation, legacy-server unsupported mapping, status decoding, maximum payload bounds, and independent endpoint conformance.
- Socket tests for split/truncated headers/payloads, oversize/overflow lengths, bad enum/reserved fields, invalid BAR/ranges, partial write, disconnect during response/FD passing, and second client.
- Provider state transitions, supported/unknown/read-failed identity, timer setup/rearm failure, drain ordering, reset admission, config/reset/tick serialization, and partial lease cleanup.
- Native integration: no-app-server counter advancement, app start/stop without provider generation change, connect/disconnect without keeper gap, stop with callback in flight, and replug generation transition.

## 11. Single acceptance matrix

Every row requires provenance/status artifacts, endpoint visibility, source commit, worktree, lock owner, and first failure if any. A required-gate failure stops later keeper claims until resolved. Classification rows record a result without a required pass; they invalidate keeper conclusions only when evidence attributes the failure to keeper lifecycle behavior.

| Gate | Role | Exact lock-held procedure | Required result |
|---|---|---|---|
| A0 provenance | Required | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A0` inspects installed app/dext, handshake, and status schema | Exact source commit/hash and audited development-signing state recorded |
| A1 no-client proof | Required | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A1`; terminate/verify server absence, run direct native status twice at least 120 seconds apart | No server/workload resources; counters advance in one generation; cadence passes |
| A2 minimal baseline | Required | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A2` runs tracked minimal harness | Exact four-value result; no failure/status regression |
| A3 churn | Required | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A3` runs 25 minimal processes with five seconds idle between | All pass; no lease/resource growth; keeper uninterrupted |
| A4 awake-idle | Required | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A4 --include-post-idle`; 90 minutes no workload, status/endpoint sample every 60 seconds | Endpoint present, no failed tick, cadence passes |
| A5 post-idle compute | Required | Immediate final step of the same lock-held A4 invocation; no restart/replug/reset | Tracked minimal harness passes |
| A6 8B smoke | Required | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A6 --model "$MODEL_GGUF"`; inner command uses `bench.py --prefill --prefill-mode smoke` | Model admission/AMD init/workload completes; keeper spans idle-to-workload |
| A7 load-to-idle | Required | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A7 --model "$MODEL_GGUF"`; five 600-second canonical decode durations, each followed by 15-minute awake idle | Every decode exits zero; no endpoint/resource loss; every idle cadence passes |
| A8 long awake idle | Required | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A8 --model "$MODEL_GGUF"`; eight hours no workload, then internal A2 and A6 procedures | Cadence and both post-idle workloads pass without releasing the lock |
| A9 bounded replug | Required | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A9`; harness preserves baseline and prompts for one manual unplug/replug | New generation; no stale resources; fresh minimal harness passes |
| A10 load classification | Classification | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A10 --model "$MODEL_GGUF"`; inner command uses `bench.py --decode --decode-duration-s 1800` | Endpoint/runtime/power outcome recorded; no claim this proves idle behavior |
| A11 sleep/wake classification | Classification | `.venv/bin/python extra/usbgpu/tools/with_gpu_lock.py -- .venv/bin/python extra/usbgpu/tests/qualify.py --gate A11`; harness prompts for explicit manual sleep/wake | Result recorded separately; no claim this is required awake-idle behavior |

Full 8B prefill/decode performance is not a keeper completion requirement. It is a separate performance and power investigation after A6. Its result cannot invalidate a passing keeper unless endpoint/status evidence attributes the failure to this task's lifecycle.

## 12. Evidence, stop conditions, and definition of done

Each artifact records host/macOS/Xcode/SDK, hardware and power topology, topology/link rate, device identity, branch/worktree/commit/dirty state, lock ownership, app/dext bundle IDs and hashes, signing state, protocol/capabilities, policy/status before-during-after, endpoint visibility, commands, model identity/hash/size, resource baseline, and first failure.

Stop implementation review and return to design if any PCI access bypasses the operation gate; callback drain cannot be proven; protocol IDs change at one endpoint only; target reports healthy before first success; native framing permits malformed input to reach a hardware path; source is untracked; installed provenance is untraceable; or an acceptance claim comes from mocked identity only.

Stop hardware qualification on unplanned endpoint disappearance, timer stall, reset, wait timeout, malformed status, resource leak, page fault, or power hazard. Preserve artifacts before asking the operator to unplug, replug, power cycle, sleep, or replace an extension; A9's already-recorded single manual replug is the only endpoint-disappearance exception.

Done means all native source is tracked under `extra/usbgpu/`; P0-P5 gates pass; wire-spec and frozen status conformance pass; no-client native status proves DriverKit-owned one-second reads; required A0-A9 pass; classification A10-A11 are recorded; no workload state survives clients; audited development-install provenance and its production limitations are recorded; and durable documentation reports measured evidence without treating an environment variable, app liveness, build success, load success, or sleep result as a substitute.
