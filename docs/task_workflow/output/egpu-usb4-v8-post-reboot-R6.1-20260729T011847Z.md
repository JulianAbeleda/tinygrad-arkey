# TinyGPU v8 post-reboot R6.1 audit

Collected: 2026-07-29T01:12:01Z through 2026-07-29T01:15:09Z

Status: R6.1 failed before A0. No A0/A1 command, reset, replug, install,
power-cycle, sleep transition, AMD initialization, or workload ran.

## Source and lock

- Control worktree: `/Users/julianabeleda/env/tinygrad-arkey-exp`, clean `exp`
  at `7f4974d3b7d1386828e63ccab44db3ab6c921c14`.
- Qualification worktree: detached clean installed-source commit
  `4e821c5d4a151024537a5cba5814e7ac7f35e528` at
  `/Users/julianabeleda/worktrees/tinygrad-v8-a0a1`.
- Every eGPU observation ran through `extra/usbgpu/tools/with_gpu_lock.py` with
  `/tmp/gpu-bench.lock` and the acceptance environment. Dangerous AMD/reset
  experiment variables were unset.
- Install provenance SHA-256:
  `cc0ea67bc8738b8bd34609ba2efb5a0980866795820af2d9b98a2821f9c0b8af`.

## Registration and binary identity

- Boot time: `2026-07-29T01:09:24Z`.
- One arkey registration exists:
  `org.tinygrad.arkey.tinygpu.driver2 (1.0.0/8) [activated enabled]`.
- The legacy `org.tinygrad.tinygpu.driver2 (1.0.0/3)` is activated but disabled.
- Registered path:
  `/Library/SystemExtensions/8078E517-B18A-4ED5-9528-12679932709E/org.tinygrad.arkey.tinygpu.driver2.dext`.
- kernelmanagerd selected unique ID
  `38e827a984d9416d1ec10fce547693e7a3ba0af9a45adb2413bd16850ec63c48`.
- Registered executable mode: `0755`, owner `root:wheel`, size `336320`.
- Registered and installed DEXT SHA-256:
  `e8b967357030773cc49eee05bd5653e3e5fbcd6f64d9eb868ae8f5b79abafbb3`.
- Registered and installed DEXT CDHash:
  `384a80d6144b2c10b6c9466123ae0730f8df6b5f`.
- Installed app SHA-256:
  `b39c4939cd590656a4705279e6dba40597ddf65e20f87e894e37b10c6b2c87c7`.

## Topology and provider result

- ADTLINK UT4G is connected in USB4 mode at 40 Gb/s, firmware `5c.9`.
- `1002:744c`, `1002:ab30`, `1002:7446`, and `1002:7444` are all present at
  x16 and 16.0 GT/s with link up.
- kernelmanagerd launched v8 as PID 312 at `2026-07-29T01:09:30.266Z`.
- The kernel recorded
  `DK: TinyGPUDriver-0x100000c0a::start(display-0x100000aa2) fail` at
  `2026-07-29T01:09:30.281Z`.
- The DEXT recorded zero crashes. A two-second privileged sample found its two
  threads idle in the workqueue/signal wait paths, consistent with returned
  startup failure rather than a hang.
- The `display@0` PCI provider remains registered, matched, active, and has no
  TinyGPU child. The DEXT IOUserServer remains present for PID 312.
- `ioreg -r -n tinygpu -l` returned no service. Consequently selector-4
  handshake, selector-5 keepalive status, and selector-10 power status all
  returned `unavailable`.

## First causal boundary and disposition

Activation, version selection, executable permissions, registered/installed
binary identity, USB4 transport, and PCI enumeration all passed. The first
failure is v8 `TinyGPUDriver::Start_Impl`, before `RegisterService`.

The installed source calls `RequestPowerResidency` and immediately jumps to the
partial-start unwind on its first native error. Because `tinygpu` is never
published, the new diagnostic payload cannot expose which request failed. The
bounded source disposition is a version-9 candidate that attempts and records
both requests, logs the native errors, continues only through successful PCI
identity and first-canary checks, and publishes a degraded diagnostic service
while keeping workload/reset/BAR/DMA/configuration admission fail-closed behind
`PowerResidencyReady`. It requires CPU tests and a clean DriverKit build before
separate operator review. This audit does not authorize installing it.
