# TinyGPU ACIO / tunneled-PCIe boundary audit

Collected: 2026-07-29T02:55:19Z through 2026-07-29T03:12:24Z

Status: source and preserved-evidence audit complete. The repeated failure is an
upstream USB4/ACIO link collapse, not a TinyGPU launch or registration failure.
There is no public PCIDriverKit API that directly holds the USB4 tunnel out of
L1. No reset, replug, reboot, install, AMD initialization, socket server, BAR
access, or workload ran.

Starting HEAD: `3900b2c96446c9bfee46f9e1a8aac56a4a1e8e72` on clean `exp`.

## Repeated failure signature

The v7 and v11 failures are the same ordered event at the useful level:

| Build / boot | Last known healthy state | Failure | Removal sequence |
|---|---|---|---|
| v7 | keepalive healthy at `2026-07-29T00:15:02.960Z` | both ACIO lanes reported Gen2/3 codes `83`, `84`, `87`, and `88` at `00:15:17.123Z` | zero-link rescan, ASM2464 dead, four AMD functions dead, TinyGPU force-close, tunnel stop |
| v11, second post-install boot | DEXT started and published at `02:48:59.369Z` | the same both-lane code set began at `02:50:17.816Z`, 78.447 seconds later | the same zero-link, downstream-tree removal, force-close, and `stopUsingTunnel` sequence |

The first v11 post-install boot retained the DEXT for 205.055 seconds until an
operator shutdown. That makes the elapsed time nondeterministic and individual
ACIO messages non-dispositive. The fatal signature is the both-lane burst plus
zero-link PCI rescan and full downstream removal.

The DEXT can therefore launch, publish, and return healthy one-Hz PCI config
reads immediately before the path disappears. A child-service On desire and
config-read traffic are not sufficient to prevent this failure.

## Public DriverKit boundary

The installed DriverKit 25.5 SDK documents and exports:

- `EnablePCIPowerManagement(D0)`, which disables PCI bus power management for
  the device's **system sleep** state. The Apple reference implementation clears
  `sleepControlBits` and `pmSleepEnabled`; it is not an awake-idle tunnel hold.
- `SetASPMState(Disabled)`, which disables ASPM/L1 substates on the endpoint and
  its immediate upstream PCIe bridge. It does not document control of the USB4
  tunnel's shared-root policy.
- `SetLinkSpeed`, which changes the upstream bridge target and can retrain the
  PCIe link. Retraining an already unstable path is not a first diagnostic.

The PCIDriverKit framework has no exported `setTunnelL1Enable` method. The
public `IOPCITunnelL1Enable` string is a registry property key, not a documented
DriverKit setter or Info.plist policy input.

## Apple reference implementation finding

Apple's published `IOPCIFamily-726.100.6` source is older than the running
macOS 26.5 binary and is used here only as a reference implementation. It has a
separate internal `setTunnelL1Enable` counter on the shared tunnel root. A false
request disables shared-root tunnel ASPM; a true request allows it when every
client agrees.

The same source defaults that tunnel request to false when a driver first asks
for device memory. Its DriverKit `_CopyDeviceMemoryWithIndex` implementation
contains the explicit comment `Make sure L1 is not set` before making that
request. TinyGPU v11 opens the PCI provider and performs configuration reads at
idle, but does not request a BAR memory descriptor until a workload asks for a
BAR or MMIO operation. Thus the reference implementation predicts that v11 has
not established this separate tunnel-L1 veto during A0/R6.1 idle.

This is a useful hypothesis, not yet a safe v12 design:

1. the method that controls the counter is kernel-internal, not a public
   DriverKit call;
2. requesting a BAR for its undocumented side effect would rely on a particular
   IOPCIFamily implementation;
3. the memory-descriptor path does not pair its false request with descriptor
   release; the ordinary `IOPCIDevice` detach path re-enables it, so TinyGPU
   cannot make this experiment cleanly reversible through the same API; and
4. no pre-loss v11 `IOPCITunnelL1Enable` registry value was captured.

## Decisions

- Do not add `EnablePCIPowerManagement(D0)`; it addresses system-sleep policy.
- Do not change link speed or request a retrain.
- Do not write the `IOPCITunnelL1Enable` registry property or call a private
  kernel symbol from DriverKit.
- Do not add a blind ASPM disable: it changes a different PCIe link boundary
  and cannot by itself identify the USB4-tunnel cause.
- Do not request/map/read a BAR merely to obtain the reference implementation's
  implicit tunnel veto until runtime registry evidence is captured.

The smallest discriminating next experiment keeps installed v11 and all
hardware unchanged, captures selector 4/5/10 immediately after one separately
approved clean reboot, records the PCI registry including
`IOPCITunnelL1Enable`, and samples idle state at one Hz for five minutes. The
host-only recorder for that scope is
`extra/usbgpu/tests/capture_tunnel_idle.py`.

## Host-side verification

The recorder and existing qualification/protocol surface passed 81 focused
unit tests. Direct Python compilation and `git diff --check` also passed. The
repository-wide documentation-link checker still reports 74 missing targets in
unchanged historical documents; none is referenced by the files in this
change.

## Sources audited

- Xcode 26.5 / DriverKit 25.5
  `PCIDriverKit.framework/Headers/IOPCIDevice.iig:425-456,497-518,548-564`
- Xcode 26.5 `PCIDriverKit.tbd` exported-symbol list
- Apple `IOPCIFamily-726.100.6`, commit
  `503b7da7c8d67bc2dedecd83dd07b72f4644ec4b`:
  `IOPCIBridge.cpp:1530-1648,1665-1714` and
  `IOPCIDevice.cpp:344-350,1401-1414,1522-1539,3397-3444,3631-3639`
- `docs/task_workflow/input/egpu-usb4-tinygpu-runtime-initialization-scope-20260728.md`
- `docs/task_workflow/output/egpu-usb4-v11-post-reboot-R6.1-20260729T025229Z.md`

Apple's support page also states that macOS eGPU support requires an Intel Mac.
This Apple-silicon path is experimental, so a supported OS-level tunnel pinning
contract cannot be assumed.
