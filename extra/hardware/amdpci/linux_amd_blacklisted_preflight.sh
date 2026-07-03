#!/usr/bin/env bash
set -euo pipefail

TARGET_BDF="${TARGET_BDF:-0000:08:00.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "STEP 1: pull latest"
git pull --ff-only origin master

echo
echo "STEP 2: status"
extra/hardware/amdpci/linux_amdgpu_grub_switch.sh status

cmdline="$(cat /proc/cmdline)"
if ! grep -q 'modprobe.blacklist=amdgpu' <<<"$cmdline"; then
  echo "FAIL: modprobe.blacklist=amdgpu is not present"
  exit 2
fi

if lsmod | awk '{print $1}' | grep -qx 'amdgpu'; then
  echo "FAIL: amdgpu is loaded"
  exit 2
fi

gpu_info="$(lspci -Dnnk -s "$TARGET_BDF" 2>/dev/null || true)"
if [ -z "$gpu_info" ]; then
  echo "FAIL: GPU is missing at $TARGET_BDF"
  exit 2
fi

if grep -q 'Kernel driver in use:' <<<"$gpu_info"; then
  echo "FAIL: GPU is already bound"
  exit 2
fi

echo
echo "PASS: BLACKLISTED_READY"
