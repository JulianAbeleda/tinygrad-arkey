# TinyGPU v11 post-reboot R6.1 audit

Collected: 2026-07-29T02:52:29Z through 2026-07-29T02:55:19Z

Status: R6.1 stopped at the selector-4 handshake after the tunneled PCIe tree
had already disappeared. Registration gates passed, but the native handshake
returned `unavailable` with exit 3. No selector-5 keepalive query, selector-10
power query, `ioreg`, endpoint query, provenance finalization, A1, reset,
replug, install, activation, AMD initialization, TinyGPU socket server, or
workload ran.

Scope:
`docs/task_workflow/input/egpu-usb4-v11-sync-power-confirmation-scope-20260729.md`.

Control worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, clean `exp` at
`ddc24db13a0fffabb15ecfb3306ea9c84274aaa1` when the audit began.

Installed source/provenance commit:
`6435cc0dd0fb3b453fec6ea304bd38e949052e3d`.

Every R6.1 gate, process observation, and boot-log query ran through
`extra/usbgpu/tools/with_gpu_lock.py` with `/tmp/gpu-bench.lock`.

## Boot boundary

The audited installation completed at approximately `2026-07-29T02:42:16Z`.
Boot history showed two later reboot transitions, not the single transition in
the handoff:

```text
reboot time                                Tue Jul 28 22:48
shutdown time                              Tue Jul 28 22:48
reboot time                                Tue Jul 28 22:45
shutdown time                              Tue Jul 28 22:44
```

The first post-install boot launched v11 at `2026-07-29T02:45:14.685Z`, spawned
PID 310, completed `tinygpu::start` at `02:45:14.721Z`, and published the child.
The service remained active until that boot was shut down at
`02:48:39.776Z`, 205.055 seconds after start. No preserved R6.1 payload from
that boot was used for admission.

The R6.1 audit ran on the second post-install boot. Its gate-1 evidence was:

```text
2026-07-29T02:52:29Z
{ sec = 1785293333, usec = 873352 } Tue Jul 28 22:48:53 2026
22:52  up 4 mins, 3 users, load averages: 12.85 15.15 7.16
```

This boot was still after the installation, but it was not the literal first
post-install boot requested by the handoff.

## Ordered R6.1 results

Registration passed. Exactly one arkey registration was present, at v11 and
`activated enabled`; v10 and v8 were absent, and legacy v3 was disabled:

```text
4 extension(s)
--- com.apple.system_extension.driver_extension (Go to 'System Settings > General > Login Items & Extensions > Driver Extensions' to modify these system extension(s))
enabled active teamID bundleID (version) name [state]
* * - org.tinygrad.arkey.tinygpu.driver2 (1.0.0/11) org.tinygrad.arkey.tinygpu.driver2 [activated enabled]
  * 9YG3G8543N org.tinygrad.tinygpu.driver2 (1.0.0/3) org.tinygrad.tinygpu.driver2 [activated disabled]
```

The native registration-only check also passed:

```text
Extension registration is clean and active. Verify keepalive and power status before using the GPU.
```

The first live-provider gate then failed:

```text
keepalive handshake unavailable
handshake_exit=3
```

The locked arkey DEXT process census was empty. Per the first-failure rule, no
later R6.1 selector or service/endpoint query ran.

## Current-boot lifecycle and link loss

The current boot did initially bind v11 correctly. kernelmanagerd selected the
installed unique ID
`0459ffba1fd9685605db1efbfa432ae2efa18e9bd84a87b55b8edb55092750a5`
from
`/Library/SystemExtensions/D3D6778A-F6F2-4B84-AB6B-94806323260A/`, launched
v11, and spawned PID 304. The kernel then recorded:

```text
2026-07-28 22:48:59.368670-0400 DK: TinyGPUDriver-0x100000c01 server launched, validating
2026-07-28 22:48:59.369175-0400 DK: tinygpu-0x100000c01::start(display-0x100000aa3) ok
2026-07-28 22:48:59.369370-0400 (IOPCIFamily) [childPublished()] child tinygpu(0x100000c01) published
```

At `22:50:17.816`, 78.447 seconds after start, ACIO began reporting repeated
Gen2/3 link errors on both lanes with codes `83`, `84`, `87`, and `88`. The PCI
rescan then found the ASM2464 child dead and marked the complete AMD function
set dead, including `1002:744c`, `1002:ab30`, `1002:7446`, and `1002:7444`:

```text
2026-07-28 22:50:17.842599-0400 bridge [i3]0:0:0(0x106b:0x1017) linkStatus 0x0000
2026-07-28 22:50:17.842715-0400 bridge <private> dead child at [i4]1:0:0(0x1b21:0x2461)
2026-07-28 22:50:17.842718-0400 bridge [i7]4:0:0(5:128) marking child <private> [i8]5:0:0(0x1002:0x744c) dead
2026-07-28 22:50:17.844094-0400 DK: tinygpu-0x100000c01:force close (display-0x100000aa3)
2026-07-28 22:50:17.844874-0400 bridge [i7]4:0:0(5:128) removing child <private> [i8]5:0:0(0x1002:0x744c)
2026-07-28 22:50:18.265186-0400 AppleTunneledPCIE::setPowerState (apciec0) 0
2026-07-28 22:50:18.266171-0400 IOTBTTunnelClientInterface(0@0:0x3)::stopUsingTunnel
```

This sequence establishes a physical/tunneled PCIe-tree loss before R6.1
reached its first live-provider query. It is not explained by stale
registration, DEXT launch failure, a DEXT crash, a workload, or an explicit
recovery action.

## Disposition

R6.1 is failed and A1 remains unauthorized. The v11 change may repair the
synchronous DriverKit confirmation bookkeeping, but this boot provides no
selector-10 payload from before the link loss and therefore does not admit that
predicate as runtime-proven. More importantly, the child service's power desire
did not prevent the observed upstream ACIO/PCIe failure on the current boot.

Do not retry R6.1 against the missing provider, reinstall v11, reset/replug the
enclosure, initialize AMD, start the socket server, or run a workload. Any next
hardware transition needs a new explicit scope. Source-only follow-up should
move to the ACIO/Thunderbolt tunnel-policy and physical-link boundary rather
than adding another unproven TinyGPU child power-state change.
