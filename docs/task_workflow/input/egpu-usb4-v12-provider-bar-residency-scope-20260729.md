# TinyGPU v12 provider BAR-residency recovery scope

Date: 2026-07-29

Status: implementation authorized; source/build validation in progress. Hardware admission
requires an audited v12 install, reboot if requested by macOS, and fresh R6.1/A0/A1 evidence.

## Decision

Restore the smallest persistent device-memory state that the historically working stack held.
The v12 DriverKit provider will retain BAR5 from provider start through provider stop while the
existing one-Hz config-space keepalive continues. The persistent BAR is provider infrastructure,
not a workload lease resource.

This supersedes the v11 observation-only next step. The history changes the prior uncertainty:

- `554800bef` kept opened devices, mapped BARs, and system-memory allocations alive across
  sessions while issuing the one-Hz read.
- `4c5e67cff` deleted that bridge during the native migration.
- `29be4c9fa` and `5bff0135c` recorded completed Qwen inference through the eGPU.
- AMD initialization requests BAR5 first (`tinygrad/runtime/support/am/amdev.py`).
- Apple's reference `_CopyDeviceMemoryWithIndex` path establishes the separate tunneled-PCIe
  L1 veto before returning device memory.

The cable is therefore not the lead hypothesis. The missing persistent device-memory request is.

## Exact implementation

1. After PCI open and `1002:744c` identity validation, request BAR5 information, copy its device
   memory descriptor, create a mapping, and retain both on the provider lifecycle gate.
2. Fail provider start if BAR5 lookup, descriptor creation, mapping, or nonzero-size validation
   fails. Do not publish partial residency.
3. Create and start the existing timer only after BAR5 residency succeeds.
4. Require both full-power confirmation and active BAR5 residency for healthy status and workload
   lease admission.
5. Keep the BAR5 descriptor/mapping separate from `active_bar_mappings`; workload disconnect must
   still report zero workload leases, BARs, and DMA allocations.
6. Drain the timer, release workload MMIO caches, release BAR5 residency, release the power request,
   then close PCI during Stop. An explicit function reset releases and reacquires BAR5 around reset.
7. Report the state in `tinygpu.power-residency.v3` under policy
   `driverkit_bar5_mapping_v1`, including requested/active, BAR number/type/bytes, and error.
8. Advance the app/DEXT/native identities atomically from v11 to v12.

## Non-goals

- No MMIO read or write from the keeper.
- No BAR0/VRAM or BAR2/doorbell retention in v12.
- No AMD reset or initialization at provider start.
- No DMA allocation, shared memory, model load, synthetic GPU work, ASPM write, link-speed change,
  tunnel-property write, private symbol, or automatic recovery after link loss.
- No claim that device-memory bytes themselves maintain the link. This experiment tests the
  provider-owned descriptor/mapping lifecycle and its IOPCIFamily tunnel state.

## Source and build gates

1. Protocol/qualification/native source tests pass, including BAR separation, start/stop/reset
   ordering, v3 fixture negatives, and atomic v12 identity.
2. All focused eGPU unit tests pass.
3. Python compilation, shell syntax, JSON parsing, and `git diff --check` pass.
4. A clean DriverKit analyzer pass reports zero diagnostics.
5. `install_nosip.sh --build` produces a strictly verified ad-hoc v12 app and DEXT.

## Hardware admission

Do not load a model first. Under `/tmp/gpu-bench.lock`, after the audited v12 activation/reboot:

1. R6.1: require exactly one active v12 registration, native-v12 handshake, healthy advancing
   keepalive, `tinygpu.power-residency.v3`, active BAR5 residency with nonzero bytes and zero error,
   and zero workload resource counts.
2. Record `IOPCITunnelL1Enable` and the PCI/USB4 topology immediately.
3. A0: repeat installed-byte/provenance binding and the same status requirements.
4. A1: with no socket server and no workload, observe at least 120 seconds in one provider
   generation, then continue the five-minute one-Hz early-tunnel capture.
5. Stop on the first ACIO both-lane burst, endpoint disappearance, provider generation change,
   failed canary, BAR-residency loss, or status failure. Preserve logs and do not reset/replug in
   the same evidence run.

If BAR5-only residency fails with the historical ACIO signature, the next candidate may add BAR0
residency. A minimally initialized AMD session/tiny allocation follows only if BAR0 is insufficient.
A fully loaded model is the final known-good control.
