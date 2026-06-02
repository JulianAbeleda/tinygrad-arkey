#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "=== repo ==="
git pull --ff-only

echo
echo "=== boot/grub/gpu status ==="
extra/amdpci/linux_amdgpu_grub_switch.sh status

echo
echo "=== bridge/repro processes ==="
pgrep -af 'extra/remote/serve.py|extra/remote/amd_repro.py' || true

echo
echo "=== AMDGPU init markers ==="
if command -v sudo >/dev/null 2>&1; then
  sudo dmesg | grep -E 'Detected VRAM|PCIE GART|ring .* ib test pass|Initialized amdgpu|SMU is initialized successfully|added device 1002:744c' | tail -80 || true
else
  dmesg | grep -E 'Detected VRAM|PCIE GART|ring .* ib test pass|Initialized amdgpu|SMU is initialized successfully|added device 1002:744c' | tail -80 || true
fi
