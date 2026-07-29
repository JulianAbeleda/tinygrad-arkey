# TinyGPU v13 historical AMD run recreation

Collected: 2026-07-29T04:11:21Z through 2026-07-29T04:17:47Z

Status: the June working runtime path was recreated twice. v13 enabled and
confirmed PCI command mask `0x0007`, the full AM device opened, and the minimal
GPU computation returned `[2.0, 5.0, 10.0, 17.0]` on two consecutive A2 runs.
The independent USB4 ACIO lane-error population continues and caused one
uncommanded tunnel/provider rebind before the successful runs.

Installed DEXT source commit:
`8f7afc45f274f8c2a4ffbeee286684a2a1013c42`.

Authority: the operator requested recreation of the last historically
confirmed run and supplied `APPROVE_ONE_EGPU_RECOVERY`. No agent reset, replug,
reboot, enclosure-power action, smart-plug action, or provider termination was
performed. The recovery approval remains unspent because the endpoint and
provider returned through macOS's own USB4 link recovery.

## v13 admission

The initial and post-rebind A0 gates passed:

- `egpu-usb4-persistent-pcie-A0-20260729T041121Z-9190.json`
- `egpu-usb4-persistent-pcie-A0-20260729T041654Z-9607.json`

The post-rebind provider reported `tinygpu.power-residency.v4`, command
`before=0`, `after=7`, required mask `7`, confirmed readback, retained BAR5,
confirmed full power, and a later successful `0x744c1002` canary. It was
`active_healthy` and `publishable=true` before workload admission.

## Two client defects found during recreation

The first A2 artifact,
`egpu-usb4-persistent-pcie-A2-20260729T041140Z-9229.json`, exposed a test-harness
API error. `Device["AMD"]` opened and returned an `AMDDevice`, but the harness
then passed that object to `Tensor(device=...)`; current tinygrad requires the
canonical string `"AMD"`. The resulting error was
`'AMDDevice' object has no attribute 'split'`. The harness and its unit test now
use `device="AMD"`, matching the June probe.

That process also exposed a lifecycle-ordering bug. `APLRemotePCIDevice`
registered its own `atexit` close after tinygrad's global device finalizer, so
Python's LIFO exit order released the TinyGPU lease and closed the socket before
`AMDDevice.finalize()` could run `AMDev.fini()`. Finalization then hit a broken
socket and left the AM scratch state dirty.

The next corrected-harness attempt,
`egpu-usb4-persistent-pcie-A2-20260729T041305Z-9351.json`, detected that dirty
state and entered the existing mode-1-reset branch. The preserved DriverKit log
shows the client intentionally changing PCI command `7 -> 3`, followed by the
reset reading command `0`. A v13 keepalive tick observed the missing bus-master
bit before the client restored it, degraded the provider, and the interface
selection failed closed. This was not an endpoint disappearance.

The lifecycle fix removes the transport-owned `atexit` close. `PCIIface` now
runs `AMDev.fini()` first and closes a close-capable PCI transport in a `finally`
block. This both preserves GPU finalization ordering and guarantees lease/socket
cleanup if finalization itself raises. A behavioral regression test proves the
`fini -> close` order and the failure-path close.

## Confirmed execution

At 00:15:49 local time, continuing both-lane ACIO errors caused macOS to request
a USB4 port power-down and retrain the link. macOS published a new TinyGPU
provider at 00:15:53. No agent action initiated this transition. The fresh
provider again enabled command mask `7` before publishing.

The exact historical environment was used under `/tmp/gpu-bench.lock`:

```text
DEV=AMD
JIT=1
PYTHONPATH=.
AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c
AM_REMOTE_SKIP_RESIZE_BAR=1
```

`REMOTE_KEEPALIVE_S` and `AM_REMOTE_SMALL_BAR_DISCOVERY` were unset.

The first passing A2 artifact is
`egpu-usb4-persistent-pcie-A2-20260729T041710Z-9646.json`. The minimal command
completed in about 10.1 seconds and printed exactly:

```text
[2.0, 5.0, 10.0, 17.0]
```

The before/after endpoint checks stayed visible, provider generation remained
1, keepalive advanced from 60 to 70 successes with zero failures, PCI command
remained `7`, and all workload lease/BAR/DMA counters returned to zero.

The immediate second A2 artifact is
`egpu-usb4-persistent-pcie-A2-20260729T041747Z-9705.json`. It completed in about
3.7 seconds with the same exact tensor result. Keepalive advanced from 101 to
104 successes in the same generation; the provider remained healthy and
publishable with command mask `7` and zero leaked workload resources. This
second pass is the hardware proof that the repaired exit path leaves the AM
state reusable.

## Verification and remaining boundary

The complete focused TinyGPU/eGPU set passed: 90 tests. Python compilation and
`git diff --check` also passed.

The compute result proves the old functional path is back. It does not prove
long-duration USB4 continuity. Both-lane ACIO errors `83/87` continued before,
during, and after the successful computation. The missing PCI command flag and
the physical-link instability are therefore separate faults: v13 repaired the
former, while cable/connector/port/UT4G signal integrity remains a live
explanation for the spontaneous tunnel power-downs. A persistent VRAM/model
residency run is the next workload-level continuity test; a known-good cable or
port A/B remains the clean physical-path discriminator.
