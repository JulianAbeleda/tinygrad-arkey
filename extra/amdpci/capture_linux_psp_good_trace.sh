#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: capture_linux_psp_good_trace.sh [--deep] [--out DIR] [--bind-bdf BDF | --rebind-bdf BDF]

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

Options:
  --deep           Generate a wider PSP trace from visible kallsyms and include
                   optional pre-KDB PSP/MMHUB/register probes when available.
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
DEEP=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --deep)
      DEEP=1
      shift
      ;;
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
if [ "$DEEP" -eq 1 ]; then
  need_cmd python3
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRACE_SCRIPT="$SCRIPT_DIR/trace_amdgpu_psp.bt"
DEEP_GENERATOR="$SCRIPT_DIR/generate_deep_psp_trace.py"
SNAPSHOT_SCRIPT="$SCRIPT_DIR/linux_mmhub_gart_snapshot.py"
[ -f "$TRACE_SCRIPT" ] || die "missing trace script: $TRACE_SCRIPT"
[ "$DEEP" -eq 0 ] || [ -f "$DEEP_GENERATOR" ] || die "missing deep trace generator: $DEEP_GENERATOR"
TRACE_SYMBOL_RE=' psp_hw_start([[:space:]]|$)| psp_v13_0_bootloader_load_component([[:space:]]|$)| psp_v13_0_wait_for_bootloader([[:space:]]|$)| psp_v13_0_wait_for_vmbx_ready([[:space:]]|$)| psp_v13_0_memory_training_send_msg([[:space:]]|$)| amdgpu_gart_map([[:space:]]|$)'
DEEP_SYMBOL_RE="$TRACE_SYMBOL_RE| amdgpu_device_[rw]reg([[:space:]]|$)| psp_v13_0_.*([[:space:]]|$)| psp_.*tmr.*([[:space:]]|$)| psp_ring.*([[:space:]]|$)"

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

grep -E "$TRACE_SYMBOL_RE" /proc/kallsyms > "$OUT/psp-symbols-before-setup.txt" || true
[ "$DEEP" -eq 0 ] || grep -E "$DEEP_SYMBOL_RE" /proc/kallsyms > "$OUT/psp-deep-symbols-before-setup.txt" || true

find /lib/firmware/amdgpu -maxdepth 1 -type f \( \
  -name 'psp_13_0_10_sos.bin' -o -name 'psp_13_0_10_sos.bin.zst' -o \
  -name 'gc_11_0_3*.bin' -o -name 'gc_11_0_3*.bin.zst' -o \
  -name 'smu_13_0_10*.bin' -o -name 'smu_13_0_10*.bin.zst' -o \
  -name 'sdma_6_0_3*.bin' -o -name 'sdma_6_0_3*.bin.zst' -o \
  -name 'vcn_3_1_2*.bin' -o -name 'vcn_3_1_2*.bin.zst' \
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

if ! grep -Eq ' psp_v13_0_bootloader_load_component([[:space:]]|$)' /proc/kallsyms; then
  die "PSP symbols are still not visible after setup; install/load amdgpu with kallsyms available"
fi
if ! grep -Eq ' amdgpu_gart_map([[:space:]]|$)' /proc/kallsyms; then
  die "amdgpu_gart_map is not visible after setup; install/load amdgpu with kallsyms available"
fi

grep -E "$TRACE_SYMBOL_RE" /proc/kallsyms > "$OUT/psp-symbols-after-setup.txt" || true
[ "$DEEP" -eq 0 ] || grep -E "$DEEP_SYMBOL_RE" /proc/kallsyms > "$OUT/psp-deep-symbols-after-setup.txt" || true

if [ -e /sys/kernel/btf/amdgpu ]; then
  cp /sys/kernel/btf/amdgpu "$OUT/amdgpu.btf" 2>/dev/null || true
  echo "btf: /sys/kernel/btf/amdgpu present" >> "$OUT/baseline.txt"
else
  echo "btf: /sys/kernel/btf/amdgpu missing" >> "$OUT/baseline.txt"
fi

if [ "$DEEP" -eq 1 ]; then
  TRACE_SCRIPT="$OUT/trace_amdgpu_psp_deep.generated.bt"
  python3 "$DEEP_GENERATOR" --out "$TRACE_SCRIPT" --symbols-out "$OUT/psp-deep-generated-symbols.txt"
fi

TRACE_OUT="$OUT/psp-linux-good.trace"
[ "$DEEP" -eq 0 ] || TRACE_OUT="$OUT/psp-linux-good-deep.trace"
TRACE_ERR="$OUT/bpftrace.stderr"
echo "starting bpftrace: $TRACE_OUT"
bpftrace "$TRACE_SCRIPT" 2>"$TRACE_ERR" | tee "$TRACE_OUT" &
TRACE_PID=$!

stop_trace() {
  if kill -0 "$TRACE_PID" >/dev/null 2>&1; then
    kill -INT "$TRACE_PID" >/dev/null 2>&1 || true
    wait "$TRACE_PID" >/dev/null 2>&1 || true
  fi
}

postprocess_trace() {
  grep -Ei 'psp_hw_start|bl_load|wait_bl|mem_train|gart_map|wreg|rreg|C2PMSG|ring|tmr|toc|LOAD|SETUP|0x1606[13478]|0x1609[01c]|0x160b3|0x1a7|0x1a8|0x80000|0x07fff007' \
    "$TRACE_OUT" > "$OUT/linux-pre-kdb-key-events.txt" || true
  grep -Ei ' [rw]reg .*reg=0x160[4-9a-b][0-9a-f]' \
    "$TRACE_OUT" > "$OUT/linux-c2pmsg-events.txt" || true
}

cleanup() {
  stop_trace
  if [ "$MODE" = "bind" ] && [ -n "$DEV" ] && [ -e "$DEV/driver_override" ]; then
    echo "" > "$DEV/driver_override" || true
  fi
  dmesg -T > "$OUT/dmesg-after.txt" || true
}
trap cleanup EXIT INT TERM

sleep 2
if ! kill -0 "$TRACE_PID" >/dev/null 2>&1; then
  wait "$TRACE_PID" >/dev/null 2>&1 || true
  die "bpftrace exited before GPU bind; see $TRACE_ERR"
fi
if [ "$MODE" = "bind" ] || [ "$MODE" = "rebind" ]; then
  echo "amdgpu" > "$DEV/driver_override"
  echo "$BIND_BDF" > /sys/bus/pci/drivers/amdgpu/bind
  sleep 20
  if [ -f "$SNAPSHOT_SCRIPT" ]; then
    PYTHON_BIN="${PYTHON:-python3}"
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
      echo "capturing MMHUB/GART snapshot"
      REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
      PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" "$SNAPSHOT_SCRIPT" \
        --bdf "$BIND_BDF" --out "$OUT" > "$OUT/mmhub-gart-snapshot.stdout" 2>"$OUT/mmhub-gart-snapshot.err" || \
        echo "warning: MMHUB/GART snapshot failed; see $OUT/mmhub-gart-snapshot.err" >&2
    else
      echo "warning: $PYTHON_BIN not found; skipping MMHUB/GART snapshot" >&2
    fi
  fi
else
  echo "manual mode: load or reload amdgpu now, then press Ctrl-C after PSP init completes"
  wait "$TRACE_PID"
fi

cleanup
trap - EXIT INT TERM
postprocess_trace
if [ "$DEEP" -eq 1 ]; then
  tar -czf "$OUT.tar.gz" "$OUT"
  sha256sum "$OUT.tar.gz" | tee "$OUT.tar.gz.sha256"
fi
