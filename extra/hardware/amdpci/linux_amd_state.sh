#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DO_PULL=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --pull)
      DO_PULL=1
      shift
      ;;
    -h|--help)
      echo "usage: linux_amd_state.sh [--pull]"
      exit 0
      ;;
    *)
      echo "error: unknown argument $1" >&2
      exit 2
      ;;
  esac
done

echo "=== repo ==="
if [ "$DO_PULL" -eq 1 ]; then
  git pull --ff-only origin master || true
else
  git status --short --branch || true
  echo "note: state query did not pull; use --pull to update first"
fi

echo
echo "=== boot/grub/gpu status ==="
extra/hardware/amdpci/linux_amdgpu_grub_switch.sh status

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
