#!/usr/bin/env bash
set -euo pipefail

BRIDGE_PORT="${BRIDGE_PORT:-6667}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "STEP 1: pull latest"
git pull --ff-only origin master
sleep 1

echo
echo "STEP 2: stop old bridge if any"
old_bridge_pids=$(pgrep -af 'extra/remote/serve[.]py' | awk -v port="$BRIDGE_PORT" '$0 ~ (" " port "$") {print $1}' || true)
if [ -n "$old_bridge_pids" ]; then
  echo "$old_bridge_pids" | xargs sudo kill
fi
sleep 1

echo
echo "STEP 3: queue blacklisted next boot"
sudo extra/hardware/amdpci/linux_amdgpu_grub_switch.sh next-blacklist
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
