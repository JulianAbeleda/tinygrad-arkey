# TinyGPU v13 PCI command residency scope

Date: 2026-07-29

Status: source-only candidate authorized by the operator's request to recreate
the last historically confirmed working TinyGPU startup path. This scope does
not authorize installation, reset, replug, reboot, enclosure power, AMD
initialization, DMA, or model loading.

Repository/worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, branch
`exp`, starting HEAD `f1efd2c2c447d80ec644922da96cbc9a699c3f5b`.

## Historical finding

The PSP/GART investigation did not establish a physical GART defect. The
2026-06-10 control matrix isolated a bad forced discovery profile, corrected
its MP0/MP1/NBIO versions, and then completed the first full Mac AM boot and
GPU tensor computation. The old DriverKit provider used by that run, retained
through `a0250a41d` immediately before the USBGPU prune, explicitly read PCI
command register offset `0x04`, ORed
`kIOPCICommandIOSpace | kIOPCICommandMemorySpace |
kIOPCICommandBusMaster`, and wrote the value before publishing `tinygpu`.

Commit `f23c05c57` restored the native provider on 2026-07-27 but omitted that
startup operation. Versions 7-12 added keepalive, power-request, and BAR
residency mechanisms without restoring the concrete PCI command enables. v12
therefore reports DriverKit capability-On while never proving that PCI I/O,
memory decode, and bus mastering are enabled. It also remains permanently
`active_degraded` when `ChangePowerState(On)` is accepted for an already-On
service and macOS correctly emits no redundant later callback.

## Objective

Produce a v13 candidate that restores the historical startup contract and
makes it auditable:

1. after opening and identifying `1002:744c`, read PCI command, set required
   mask `0x0007`, and read it back before acquiring provider-owned BAR5;
2. fail provider start if all required bits do not latch;
3. reapply and verify the same command mask after a function reset and before
   reacquiring BAR5;
4. sample the command register during each keepalive tick and degrade, without
   silently rewriting, if required bits disappear;
5. require a confirmed command readback and a later successful identity canary
   for workload admission; and
6. expose the requested mask, before/after values, timestamp, error, and
   confirmation in a versioned power-residency payload.

## Power evidence

Retain the existing DriverKit `ChangePowerState(On)` request, observed On
state, downgrade detection, and balanced release. A request against a service
that is already On is allowed to reuse the observed On notification; it is not
alone sufficient for readiness. Fresh readiness additionally requires:

- a successful PCI command write/readback after provider open;
- the required `0x0007` mask still present;
- retained BAR5 residency; and
- a successful `0x744c1002` canary after both the command operation and the
  accepted full-power request.

This removes v12's impossible dependency on a redundant post-request On
callback while replacing it with concrete PCI state plus later live-device
evidence.

## Safety and non-goals

- No ASPM/CLx register writes or undocumented power overrides.
- No automatic reset, replug, or recovery.
- No retry loop for a command mask that fails to latch.
- No relaxation of lease, BAR, DMA, or provider-lifecycle serialization.
- No claim that this repairs the independent ACIO lane-error population until
  an separately authorized hardware qualification proves continuity.

## Source acceptance

Before an installation handoff:

1. focused native-source, installer, wire, protocol, and qualification tests
   pass;
2. `git diff --check` passes;
3. Xcode analyze completes with zero analyzer diagnostics;
4. the unsigned/ad-hoc development build and strict signature checks pass;
5. DEXT and native server identities advance atomically from v12 to v13; and
6. the candidate and evidence are committed before any later installer binds
   provenance to branch HEAD.
