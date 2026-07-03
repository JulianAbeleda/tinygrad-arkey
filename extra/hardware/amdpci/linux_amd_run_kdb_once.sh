#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: linux_amd_run_kdb_once.sh [--variant NAME] [--remote HOST:PORT] [--out DIR] [--use-existing-bridge] [--stop-existing-bridge] [--poweroff]

Runs one blacklisted-boot KDB attempt:
  1. validates BLACKLISTED_READY,
  2. starts extra/remote/serve.py, unless --use-existing-bridge is passed,
  3. runs run_remote_kdb_attempt.sh,
  4. stops the bridge,
  5. queues next-normal,
  6. prints the latest log report.

Default variant: sos-pipeline-slow
Default remote: 127.0.0.1:6667

By default this does not power off. Pass --poweroff to run the normal timed
poweroff sequence after the report, or run linux_amd_poweroff_normal.sh later.

Use --use-existing-bridge when serve.py is already running as root in another
terminal. In that mode this script does not require sudo to start or stop the
bridge.

Use --stop-existing-bridge with --use-existing-bridge to kill the existing bridge
at the end of this run.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VARIANT="sos-pipeline-slow"
REMOTE="127.0.0.1:6667"
OUT_DIR="extra/hardware/amdpci/captures"
POWEROFF=0
USE_EXISTING_BRIDGE=0
STOP_EXISTING_BRIDGE=0
BRIDGE_PID=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --variant)
      [ "$#" -ge 2 ] || { echo "error: --variant needs NAME" >&2; exit 2; }
      VARIANT="$2"
      shift 2
      ;;
    --remote)
      [ "$#" -ge 2 ] || { echo "error: --remote needs HOST:PORT" >&2; exit 2; }
      REMOTE="$2"
      shift 2
      ;;
    --out)
      [ "$#" -ge 2 ] || { echo "error: --out needs DIR" >&2; exit 2; }
      OUT_DIR="$2"
      shift 2
      ;;
    --poweroff)
      POWEROFF=1
      shift
      ;;
    --use-existing-bridge)
      USE_EXISTING_BRIDGE=1
      shift
      ;;
    --stop-existing-bridge)
      STOP_EXISTING_BRIDGE=1
      shift
      ;;
    --no-poweroff)
      # Backward-compatible no-op. No poweroff is now the default.
      POWEROFF=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      echo "error: unknown argument $1" >&2
      exit 2
      ;;
  esac
done

cleanup_bridge() {
  if [ -n "$BRIDGE_PID" ] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
    sudo kill "$BRIDGE_PID" 2>/dev/null || true
    sleep 1
  fi
  if [ "$USE_EXISTING_BRIDGE" -eq 1 ] && [ "$STOP_EXISTING_BRIDGE" -eq 0 ]; then
    return 0
  fi
  old_bridge_pids=$(pgrep -af 'extra/remote/serve[.]py' | awk -v port="${REMOTE##*:}" '$0 ~ (" " port "$") {print $1}' || true)
  if [ -n "$old_bridge_pids" ]; then
    echo "$old_bridge_pids" | xargs sudo kill || true
  fi
}
trap cleanup_bridge EXIT

echo "STEP 1: validate blacklisted preflight"
extra/hardware/amdpci/linux_amd_blacklisted_preflight.sh

echo
echo "STEP 2: start bridge"
mkdir -p "$OUT_DIR"
if [ "$USE_EXISTING_BRIDGE" -eq 1 ]; then
  echo "using existing bridge at $REMOTE"
  if ! pgrep -af 'extra/remote/serve[.]py' | awk -v port="${REMOTE##*:}" '$0 ~ (" " port "$") {found=1} END {exit !found}'; then
    echo "FAIL: no existing serve.py bridge found on port ${REMOTE##*:}"
    exit 3
  fi
else
  bridge_log="$OUT_DIR/bridge-${VARIANT}-$(date +%Y%m%d-%H%M%S).log"
  if ! sudo -n true 2>/dev/null; then
    echo "FAIL: sudo is required to start serve.py and cannot prompt here."
    echo "Run 'sudo -v' in an interactive terminal first, or start the bridge manually and rerun with --use-existing-bridge."
    exit 3
  fi
  sudo .venv/bin/python extra/remote/serve.py "${REMOTE##*:}" > "$bridge_log" 2>&1 &
  BRIDGE_PID=$!
  echo "bridge_pid=$BRIDGE_PID"
  echo "bridge_log=$bridge_log"
  sleep 2
  if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "FAIL: bridge exited early"
    tail -80 "$bridge_log" || true
    exit 3
  fi
fi

echo
echo "STEP 3: run $VARIANT"
set +e
extra/hardware/amdpci/run_remote_kdb_attempt.sh --variant "$VARIANT" --remote "$REMOTE" --out "$OUT_DIR"
rc=$?
set -e

echo
echo "STEP 4: stop bridge"
cleanup_bridge
BRIDGE_PID=""
if [ "$USE_EXISTING_BRIDGE" -eq 1 ]; then
  if [ "$STOP_EXISTING_BRIDGE" -eq 1 ]; then
    echo "stopped existing bridge (requested)"
  else
    echo "left existing bridge running"
    echo "to clean it in this run, add --stop-existing-bridge"
  fi
else
  pgrep -af 'extra/remote/serve.py|extra/remote/amd_repro.py' || true
fi

echo
echo "STEP 5: queue normal next boot"
sudo extra/hardware/amdpci/linux_amdgpu_grub_switch.sh next-normal || true
sudo grub-editenv list || true

echo
echo "STEP 6: report latest log"
log=$(ls -t "$OUT_DIR"/kdb-"$VARIANT"-*.log 2>/dev/null | head -1 || true)
echo "rc=$rc"
echo "log=${log:-not found}"
if [ -n "$log" ]; then
  sha256sum "$log"
  grep -n "bootloader payload audit\\|bootloader pipeline continue\\|bootloader pipeline skip prewait\\|KDB pipeline continue\\|KDB pipeline skip prewait\\|pre-KDB invalidate burst\\|pre-KDB GART audit\\|pre-KDB CID2 audit\\|pre-KDB linux final invalidate\\|pre-KDB linux MMHUB window\\|msg1 primary sync\\|regMM\\|write msg1\\|write compid\\|wait BL\\|sOS final state audit\\|sOS wait delay\\|sOS\\|C2PMSG35\\|C2PMSG36\\|C2PMSG81\\|AMDDevice ready\\|Traceback\\|RuntimeError\\|TimeoutError" "$log" | tail -320 || true
fi

if [ "$POWEROFF" -eq 1 ]; then
  echo
  echo "STEP 7: timed normal poweroff"
  extra/hardware/amdpci/linux_amd_poweroff_normal.sh
else
  echo
  echo "STEP 7: poweroff skipped"
  echo "Run this when ready:"
  echo "  extra/hardware/amdpci/linux_amd_poweroff_normal.sh"
fi

exit "$rc"
