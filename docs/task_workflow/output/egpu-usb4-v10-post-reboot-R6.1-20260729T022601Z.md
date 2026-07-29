# TinyGPU v10 post-reboot R6.1 audit

Collected: 2026-07-29T02:26:01Z through 2026-07-29T02:27:16Z

Status: R6.1 failed at keepalive health before the power-residency query and
before A1. No provenance finalization, A1 command, reset, replug, install,
activation, power-cycle, sleep transition, AMD initialization, TinyGPU socket
server, or workload ran.

## Source, installed payload, and lock

- Control worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, clean `exp`
  at `2dc9f2c03682db96a69ebde2b9e1954b93d32f8f`.
- Installed source/provenance commit:
  `0d9f3a1bf6ff0111fdb405b3959000b741603205`, as recorded by the audited
  pre-reboot installation handoff.
- Every R6.1 gate and boot-log observation ran through
  `extra/usbgpu/tools/with_gpu_lock.py` with `/tmp/gpu-bench.lock`.
- The diagnostic stopped on the first failed predicate. In particular,
  selector-10 power status and `ioreg` were not queried.

## R6.1 raw results

The boot began after the installation, at `2026-07-29T02:19:41Z`:

```text
2026-07-29T02:26:01Z
{ sec = 1785291581, usec = 961181 } Tue Jul 28 22:19:41 2026
22:26  up 6 mins, 3 users, load averages: 30.81 17.89 7.67
```

Registration passed. Exactly one arkey registration was present, at v10 and
`activated enabled`; v8 was gone, and legacy v3 remained disabled:

```text
4 extension(s)
--- com.apple.system_extension.driver_extension (Go to 'System Settings > General > Login Items & Extensions > Driver Extensions' to modify these system extension(s))
enabled active teamID bundleID (version) name [state]
* * - org.tinygrad.arkey.tinygpu.driver2 (1.0.0/10) org.tinygrad.arkey.tinygpu.driver2 [activated enabled]
  * 9YG3G8543N org.tinygrad.tinygpu.driver2 (1.0.0/3) org.tinygrad.tinygpu.driver2 [activated disabled]
```

The native registration check passed:

```text
Extension registration is clean and active. Verify keepalive and power status before using the GPU.
```

The selector-4 handshake passed with the v10 identity and capability 11:

```json
{"schema":"tinygpu.handshake.v1","protocol_major":1,"protocol_minor":0,"capabilities":11,"server_build_id":"tinygrad-arkey-native-v10"}
```

The selector-5 keepalive gate failed because `state` was `active_degraded`, not
the required `active_healthy`:

```json
{"schema":"tinygpu.keepalive.v1","provider_generation":1,"state":"active_degraded","enabled":true,"policy_id":"usb4_amd_744c_v1","interval_ms":1000,"maximum_timer_leeway_ms":100,"expected_identity":"1002:744c","last_identity_dword":"0x744c1002","attempts":360,"successes":360,"failures":0,"consecutive_failures":0,"last_attempt_monotonic_ns":402096622041,"last_success_monotonic_ns":402096622041,"success_gap_over_leeway_count":4,"max_success_gap_ms":1101,"timer_error":0,"counter_saturated":false,"active_workload_leases":0,"active_bar_mappings":0,"active_dma_allocations":0}
```

The PCI identity canary itself was live: all 360 attempts succeeded, the last
identity was `0x744c1002`, there were no failures or timer error, counters were
not saturated, and workload/BAR/DMA resource counts were zero. Four cumulative
gaps exceeded the 1100 ms leeway threshold by at most 1 ms. That gap counter is
not a direct input to `RefreshProviderHealth`, so it does not explain the
`active_degraded` state.

In the installed source, a successful current canary makes `canaryHealthy`
true, and `RefreshProviderHealth` also requires `PowerResidencyReady`. With the
reported canary, timer, and saturation fields green, the degraded state narrows
the failed runtime admission to the power-residency readiness predicate. The
exact failed power field is intentionally not inferred: the required stop at
selector 5 prohibited the later selector-10 query.

## Relevant boot logs

The filtered post-boot DriverKit and kernelmanagerd log showed:

- v10 at
  `/Library/SystemExtensions/06021A7B-3D08-4838-8010-71B1BABAD6C2/org.tinygrad.arkey.tinygpu.driver2.dext`
  replaced the v8 record;
- kernelmanagerd selected unique ID
  `7b24a6984d94c5997f742ced6e4f6df0d4280b26efa8cbf8289a987c83380cbf`;
- the DEXT launched as PID 314 at `2026-07-29T02:19:47.667Z`;
- the kernel reported zero recorded crashes;
- `DK: tinygpu-0x100000c27::start(display-0x100000a9e) ok` appeared at
  `2026-07-29T02:19:47.702Z`; and
- IOPCIFamily immediately recorded `child tinygpu(0x100000c27) published`.

The narrow log did not contain a TinyGPU startup failure or crash. The source
does not log individual power-request or power-transition fields, so the boot
log cannot safely distinguish which `PowerResidencyReady` term failed.

## Disposition and stop boundary

The v10 lifecycle change fixed the earlier v8 pre-publication failure: the DEXT
starts, publishes diagnostics, and maintains a successful PCI identity canary.
It did not satisfy the first post-reboot health gate, so R6.1 is not admitted
and A1 must not run from this boot under the current audit procedure.

No recovery action was attempted. A separately reviewed next step is required
before collecting any additional eGPU state or altering the gate; this audit
does not authorize selector-10 power status, reset/replug, another install or
reboot, AMD initialization, the socket server, or a workload.
