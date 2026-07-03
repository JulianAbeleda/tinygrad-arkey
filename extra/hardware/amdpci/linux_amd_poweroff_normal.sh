#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "STEP 1: queue normal next boot"
sudo extra/hardware/amdpci/linux_amdgpu_grub_switch.sh next-normal || true
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
