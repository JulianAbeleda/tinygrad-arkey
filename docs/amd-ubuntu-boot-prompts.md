# AMD Ubuntu Boot Prompts

Reusable prompts for the RX 7900 GRE / Navi31 Ubuntu host during PSP/KDB testing.

The important rule is to keep every shutdown sequenced: queue the GRUB one-shot entry,
print the state, `sync`, pause, then power off. After KDB attempts or failed gates,
always queue the next boot back to normal.

## Timed Normal Shutdown

Use this whenever the host must return to a normal boot, including after a failed
blacklisted gate or when the GPU is missing from PCI.

```bash
cd ~/tinygrad-arkey || exit 1

echo "STEP 1: queue normal next boot"
sudo extra/amdpci/linux_amdgpu_grub_switch.sh next-normal || true
sleep 1

echo
echo "STEP 2: verify grub one-shot state"
sudo grub-editenv list || true
sleep 3

echo
echo "STEP 3: sync disks"
sync
sleep 3

echo
echo "STEP 4: poweroff in 10 seconds"
sleep 10

sudo poweroff
```

After poweroff, use a full hardware power cycle when PCI enumeration is wedged:
PSU off or AC unplug, hold the power button for 10-15 seconds, wait 1-2 minutes,
then boot.

## Timed Blacklisted Shutdown

Use this to queue the next boot for a clean blacklisted PSP/KDB attempt.

```bash
cd ~/tinygrad-arkey || exit 1

echo "STEP 1: pull latest"
git pull --ff-only || exit 1
sleep 1

echo
echo "STEP 2: stop old bridge if any"
sudo pkill -f 'extra/remote/serve.py 6667' || true
sleep 1

echo
echo "STEP 3: queue blacklisted next boot"
sudo extra/amdpci/linux_amdgpu_grub_switch.sh next-blacklist || exit 1
sleep 1

echo
echo "STEP 4: verify grub one-shot state"
sudo grub-editenv list || true
sleep 3

echo
echo "STEP 5: sync disks"
sync
sleep 3

echo
echo "STEP 6: poweroff in 10 seconds"
sleep 10

sudo poweroff
```

After poweroff, use a full hardware power cycle, then boot Ubuntu.

## Blacklisted Preflight

Run this after a blacklisted boot before starting the remote bridge. Continue only if
all required conditions are met.

```bash
cd ~/tinygrad-arkey || exit 1

echo "STEP 1: pull latest"
git pull --ff-only || exit 1

echo
echo "STEP 2: boot command line"
cat /proc/cmdline

echo
echo "STEP 3: amdgpu module"
lsmod | grep '^amdgpu' || true

echo
echo "STEP 4: target GPU"
lspci -Dnnk -s 0000:08:00.0 || true
```

Required state:

- `modprobe.blacklist=amdgpu` is present in `/proc/cmdline`.
- `amdgpu` is not loaded.
- `0000:08:00.0` is present.
- `0000:08:00.0` has no `Kernel driver in use` line.

If any condition fails, do not run the bridge or KDB attempt. Use the timed normal
shutdown prompt.

## KDB Pipeline Attempt

Run this only after the blacklisted preflight passes.

```bash
cd ~/tinygrad-arkey || exit 1

echo "STEP 1: start bridge"
sudo .venv/bin/python extra/remote/serve.py 6667 > /tmp/kdb-bridge.log 2>&1 &
bridge_pid=$!
echo "bridge_pid=$bridge_pid"
sleep 2

echo
echo "STEP 2: run kdb-pipeline-seq"
extra/amdpci/run_remote_kdb_attempt.sh --variant kdb-pipeline-seq
rc=$?

echo
echo "STEP 3: stop bridge"
sudo kill "$bridge_pid" || true
sudo pkill -f 'extra/remote/serve.py 6667' || true
pgrep -af 'extra/remote/serve.py|extra/remote/amd_repro.py' || true

echo
echo "STEP 4: queue normal next boot"
sudo extra/amdpci/linux_amdgpu_grub_switch.sh next-normal || true
sudo grub-editenv list || true

echo
echo "STEP 5: report latest log"
log=$(ls -t extra/amdpci/captures/kdb-kdb-pipeline-seq-*.log | head -1)
echo "rc=$rc"
echo "log=$log"
sha256sum "$log"

grep -n "KDB pipeline continue\|KDB pipeline skip prewait\|pre-KDB invalidate burst\|write msg1\|write compid\|wait BL\|C2PMSG35\|C2PMSG36\|C2PMSG81\|AMDDevice ready\|Traceback\|RuntimeError\|TimeoutError" "$log" | tail -240 || true

echo
echo "STEP 6: sync and poweroff to normal boot in 10 seconds"
sync
sleep 10
sudo poweroff
```

Report the `rc`, log path, SHA256, KDB pipeline lines, component writes, wait-BL
lines, and whether `AMDDevice ready` appeared.

## Normal Recovery Check

Run this after a normal boot. The healthy state is `Kernel driver in use: amdgpu`.

```bash
cd ~/tinygrad-arkey || exit 1
git pull --ff-only

echo "=== boot command line ==="
cat /proc/cmdline

echo
echo "=== amdgpu module ==="
lsmod | grep '^amdgpu' || true

echo
echo "=== target GPU ==="
lspci -Dnnk -s 0000:08:00.0 || true

echo
echo "=== AMDGPU init markers ==="
sudo dmesg | grep -E 'Detected VRAM|PCIE GART|ring .* ib test pass|Initialized amdgpu' | tail -40
```

If the GPU is missing from PCI, stop software testing and use the timed normal
shutdown plus a full hardware power cycle.
