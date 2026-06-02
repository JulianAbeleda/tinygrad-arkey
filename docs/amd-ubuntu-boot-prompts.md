# AMD Ubuntu Boot Prompts

Reusable prompts for the RX 7900 GRE / Navi31 Ubuntu host during PSP/KDB testing.

The important rule is to keep every shutdown sequenced: queue the GRUB one-shot
entry, print the state, `sync`, pause, then power off. After KDB attempts or
failed gates, always queue the next boot back to normal.

## Current State

Run this before deciding what to do next.

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_state.sh
```

To update the checkout first:

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_state.sh --pull
```

Important interpreted states:

- `NORMAL_HEALTHY`: safe to queue a blacklisted test boot.
- `BLACKLISTED_READY`: safe to run a PSP/KDB attempt.
- `GPU_MISSING_FROM_PCI`: stop software testing; use normal shutdown plus full
  hardware power cycle.
- `MIXED_OR_DIRTY`: do not run KDB until the state is understood.

## Timed Normal Shutdown

Use this whenever the host must return to a normal boot, including after a failed
blacklisted gate or when the GPU is missing from PCI.

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_poweroff_normal.sh
```

After poweroff, use a full hardware power cycle when PCI enumeration is wedged:
PSU off or AC unplug, hold the power button for 10-15 seconds, wait 1-2 minutes,
then boot.

## Timed Blacklisted Shutdown

Use this from `NORMAL_HEALTHY` to queue the next boot for a clean blacklisted
PSP/KDB attempt.

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_poweroff_blacklist.sh
```

After poweroff, use a full hardware power cycle, then boot Ubuntu.

## Blacklisted Preflight

Run this after a blacklisted boot before starting the remote bridge.

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_blacklisted_preflight.sh
```

Required state:

- `modprobe.blacklist=amdgpu` is present in `/proc/cmdline`.
- `amdgpu` is not loaded.
- `0000:08:00.0` is present.
- `0000:08:00.0` has no `Kernel driver in use` line.

If this fails, do not run the bridge or KDB attempt. Use timed normal shutdown.

## sOS Delay Attempt

Run this only after blacklisted preflight passes. It starts the bridge, runs one
KDB attempt, stops the bridge, queues normal boot, and reports the latest log.
It does not power off by default.

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_run_kdb_once.sh --variant sos-delay20
```

After reviewing the report, run the separate timed normal shutdown:

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_poweroff_normal.sh
```

To intentionally combine the attempt and poweroff in one command:

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_run_kdb_once.sh --variant sos-delay20 --poweroff
```

If sudo cannot prompt inside the helper, start the bridge separately:

```bash
cd ~/tinygrad-arkey || exit 1
sudo .venv/bin/python extra/remote/serve.py 6667
```

Then run the attempt in another shell:

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_run_kdb_once.sh --variant sos-delay20 --use-existing-bridge
```

Report the `rc`, log path, SHA256, pipeline lines, component writes, wait-BL
lines, `sOS wait delay`, `C2PMSG81`, and whether `AMDDevice ready` appeared.

## Normal Recovery Check

Run current state after a normal boot. The healthy state is `NORMAL_HEALTHY`.

```bash
cd ~/tinygrad-arkey || exit 1
extra/amdpci/linux_amd_state.sh
```

If the GPU is missing from PCI, stop software testing and use timed normal
shutdown plus a full hardware power cycle.
