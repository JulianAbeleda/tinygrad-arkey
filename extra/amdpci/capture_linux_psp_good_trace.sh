#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: capture_linux_psp_good_trace.sh [--out DIR] [--bind-bdf BDF | --rebind-bdf BDF]

Capture a Linux-good AMD PSP boot trace for Navi31/RX 7900 XTX.

Modes:
  --bind-bdf BDF   Recommended. Hold this PCI device unbound, load amdgpu so
                   PSP symbols are visible, attach bpftrace, then bind BDF to
                   amdgpu. Boot with amdgpu blacklisted for the cleanest run.
  --rebind-bdf BDF Unbind an already-bound amdgpu device, attach bpftrace, then
                   bind it again. Do not use if this GPU drives the display.

If no mode is provided, the script records baseline files and starts bpftrace;
bind the GPU to amdgpu from another shell while it is running. Manual mode
requires the amdgpu PSP symbols to already be visible in /proc/kallsyms.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

OUT=""
MODE="manual"
BIND_BDF=""
DEV=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --out)
      [ "$#" -ge 2 ] || die "--out needs a directory"
      OUT="$2"
      shift 2
      ;;
    --bind-bdf)
      [ "$#" -ge 2 ] || die "--bind-bdf needs a PCI BDF like 0000:03:00.0"
      MODE="bind"
      BIND_BDF="$2"
      shift 2
      ;;
    --rebind-bdf)
      [ "$#" -ge 2 ] || die "--rebind-bdf needs a PCI BDF like 0000:03:00.0"
      MODE="rebind"
      BIND_BDF="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

[ "$(id -u)" -eq 0 ] || die "run with sudo/root so bpftrace and modprobe can attach"
need_cmd bpftrace
need_cmd lspci
need_cmd modinfo
need_cmd sha256sum
need_cmd modprobe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_SCRIPT="$SCRIPT_DIR/trace_amdgpu_psp.bt"
[ -f "$TRACE_SCRIPT" ] || die "missing trace script: $TRACE_SCRIPT"

if [ -z "$OUT" ]; then
  OUT="psp-linux-good-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$OUT"

echo "writing capture to $OUT"

{
  date -Is
  uname -a
  echo
  lspci -nn | grep -Ei 'amd|vga|display|3d' || true
  echo
  modinfo amdgpu | sed -n '1,80p' || true
  echo
  if [ -e /sys/kernel/btf/vmlinux ]; then
    echo "btf: /sys/kernel/btf/vmlinux present"
  else
    echo "btf: /sys/kernel/btf/vmlinux missing"
  fi
} > "$OUT/baseline.txt"

dmesg -T > "$OUT/dmesg-before.txt" || true

if [ -e /sys/kernel/btf/vmlinux ]; then
  cp /sys/kernel/btf/vmlinux "$OUT/vmlinux.btf" 2>/dev/null || true
fi

grep -E ' psp_hw_start$| psp_v13_0_bootloader_load_component$| psp_v13_0_wait_for_bootloader$| psp_v13_0_wait_for_vmbx_ready$| psp_v13_0_memory_training_send_msg$' \
  /proc/kallsyms > "$OUT/psp-symbols-before-setup.txt" || true

find /lib/firmware/amdgpu -maxdepth 1 -type f \( \
  -name 'psp_13_0_10_sos.bin' -o \
  -name 'gc_11_0_3*.bin' -o \
  -name 'smu_13_0_10*.bin' -o \
  -name 'sdma_6_0_3*.bin' -o \
  -name 'vcn_3_1_2*.bin' \
\) -print0 2>/dev/null | sort -z | xargs -0r sha256sum > "$OUT/firmware-sha256.txt"

lspci -Dnn | grep -Ei '1002:744c|amd.*(vga|display|3d)' > "$OUT/gpu-candidates.txt" || true
while read -r bdf _; do
  [ -n "$bdf" ] || continue
  lspci -vvnn -s "$bdf" > "$OUT/lspci-${bdf//[:.]/_}.txt" || true
done < "$OUT/gpu-candidates.txt"

if [ "$MODE" = "bind" ] || [ "$MODE" = "rebind" ]; then
  DEV="/sys/bus/pci/devices/$BIND_BDF"
  [ -d "$DEV" ] || die "PCI device not found: $BIND_BDF"
fi

if [ "$MODE" = "bind" ]; then
  if [ -e "$DEV/driver" ]; then
    die "$BIND_BDF is already bound to $(basename "$(readlink "$DEV/driver")"); boot with amdgpu blacklisted or unbind it first"
  fi
  echo "__tinygrad_hold__" > "$DEV/driver_override"
  modprobe amdgpu
elif [ "$MODE" = "rebind" ]; then
  [ -e "$DEV/driver" ] || die "$BIND_BDF is not currently bound to a driver"
  DRIVER="$(basename "$(readlink "$DEV/driver")")"
  [ "$DRIVER" = "amdgpu" ] || die "$BIND_BDF is bound to $DRIVER, not amdgpu"
  echo "$BIND_BDF" > "$DEV/driver/unbind"
fi

if ! grep -q ' psp_v13_0_bootloader_load_component$' /proc/kallsyms; then
  die "PSP symbols are still not visible after setup; install/load amdgpu with kallsyms available"
fi

grep -E ' psp_hw_start$| psp_v13_0_bootloader_load_component$| psp_v13_0_wait_for_bootloader$| psp_v13_0_wait_for_vmbx_ready$| psp_v13_0_memory_training_send_msg$' \
  /proc/kallsyms > "$OUT/psp-symbols-after-setup.txt" || true

if [ -e /sys/kernel/btf/amdgpu ]; then
  cp /sys/kernel/btf/amdgpu "$OUT/amdgpu.btf" 2>/dev/null || true
  echo "btf: /sys/kernel/btf/amdgpu present" >> "$OUT/baseline.txt"
else
  echo "btf: /sys/kernel/btf/amdgpu missing" >> "$OUT/baseline.txt"
fi

TRACE_OUT="$OUT/psp-linux-good.trace"
echo "starting bpftrace: $TRACE_OUT"
bpftrace "$TRACE_SCRIPT" | tee "$TRACE_OUT" &
TRACE_PID=$!

cleanup() {
  if kill -0 "$TRACE_PID" >/dev/null 2>&1; then
    kill -INT "$TRACE_PID" >/dev/null 2>&1 || true
    wait "$TRACE_PID" >/dev/null 2>&1 || true
  fi
  if [ "$MODE" = "bind" ] && [ -n "$DEV" ] && [ -e "$DEV/driver_override" ]; then
    echo "" > "$DEV/driver_override" || true
  fi
  dmesg -T > "$OUT/dmesg-after.txt" || true
}
trap cleanup EXIT INT TERM

sleep 2
if [ "$MODE" = "bind" ] || [ "$MODE" = "rebind" ]; then
  echo "amdgpu" > "$DEV/driver_override"
  echo "$BIND_BDF" > /sys/bus/pci/drivers/amdgpu/bind
  sleep 20
else
  echo "manual mode: load or reload amdgpu now, then press Ctrl-C after PSP init completes"
  wait "$TRACE_PID"
fi
