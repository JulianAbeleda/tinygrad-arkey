# TinyGPU v13 PCI command residency candidate

Collected: 2026-07-29T03:48Z through 2026-07-29T04:00Z

Status: source candidate complete and build-verified. Not installed. No reset,
replug, reboot, enclosure power action, AMD initialization, DMA, workload, or
model load was performed.

Authority:
`docs/task_workflow/input/egpu-usb4-v13-pci-command-residency-scope-20260729.md`.

Starting source commit:
`f1efd2c2c447d80ec644922da96cbc9a699c3f5b`.

## Historical reconstruction

The 2026-06-10 PSP/GART closeout at `bb0a10846` records the first full Mac AM
boot (`boot done`, 11.4 seconds, 26,263 round trips) and GPU tensor result
`[2, 3, 4]`. Its root cause was the wrong MP0/MP1/NBIO versions in the forced
remote discovery profile, not GART behavior. The next commit, `554800bef`,
added the one-Hz idle config-read keepalive after a successful session later
dropped during a 40-minute idle interval.

The concrete provider startup used by those runs remained present at
`a0250a41d`, the parent of the USBGPU prune. Before publishing `tinygpu`, its
`Start_Impl` read PCI command offset `0x04`, ORed
`kIOPCICommandIOSpace | kIOPCICommandBusMaster |
kIOPCICommandMemorySpace`, and wrote it back. Commit `4c5e67cff` deleted that
provider. Commit `f23c05c57` restored a new native provider on 2026-07-27 but
did not restore the PCI command operation. The v7-v12 recovery arc therefore
measured keepalive, abstract DriverKit power, and BAR residency around a
provider that never performed the old concrete PCI enable step.

## Candidate behavior

v13 restores the exact required mask (`0x0007`) with a stricter audit boundary:

- after provider open and `1002:744c` identity validation, read PCI command,
  OR the I/O-space, memory-space, and bus-master bits, write if needed, and
  read back;
- fail provider start if the readback does not contain the complete mask;
- repeat the operation after a function reset before reacquiring BAR5;
- observe, but do not silently rewrite, command state on every keepalive tick;
- include policy, request/confirmation, before/after mask, timestamp, and error
  in `tinygpu.power-residency.v4`; and
- require command confirmation plus an identity canary later than both the
  command operation and accepted DriverKit On request before workload lease
  admission.

The power predicate now admits the real already-On lifecycle observed by v12.
An accepted `ChangePowerState(On)` does not need to cause a redundant later On
callback. A pre-existing observed On state is only supplemental evidence; the
fresh PCI command readback and subsequent identity canary are mandatory.

## Verification

Focused suite:

```text
86 passed, 3 pytest-configuration warnings in 0.24s
```

The warnings are the pre-existing environment's unavailable pytest-timeout
configuration keys; there were no test failures. Python compilation, installer
and classifier `bash -n`, both changed JSON fixtures, and `git diff --check`
passed.

The clean Xcode analyzer command completed with `ANALYZE SUCCEEDED`. All five
static-analyzer plists (provider source/IIG, user-client source/IIG, and native
server) contain zero diagnostics.

The audited build-only path completed with `BUILD SUCCEEDED`, ad-hoc signed the
app and universal DEXT, and passed strict `codesign` verification. Built
identity checks report:

```text
CFBundleVersion=13
tinygrad-arkey-native-v13
tinygrad-arkey-v13
tinygpu.power-residency.v4
pci_command_enable_v1
```

No v12 server or v3 power-schema string is present in the built app/server.

## Remaining boundary

This evidence proves source consistency and buildability, not link continuity
or successful AMD execution. The earlier literal install and one-recovery
tokens were used by the v12 cycle and are not reused here. Installation of the
committed v13 candidate requires a new explicit development-install approval;
any reset, replug, reboot, or enclosure power action requires separately valid
operator authority.

After installation and provider start, the first discriminating check is the
read-only v4 status: required mask `7`, confirmed readback containing `7`, zero
command error, retained BAR5, accepted/confirmed On state, and a later canary.
Only after that passes should the exact June working runtime environment be
recreated: `AM_REMOTE_SKIP_RESIZE_BAR=1` and
`AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c`, followed by full AM boot, minimal
GPU compute, then persistent VRAM/model residency.
