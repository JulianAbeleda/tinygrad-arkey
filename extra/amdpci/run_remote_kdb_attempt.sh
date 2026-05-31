#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
usage: run_remote_kdb_attempt.sh [--remote HOST:PORT] [--variant NAME] [--out DIR]

Run the second-shell remote PSP KDB repro command after the operator has:
  1. booted Ubuntu with amdgpu blacklisted,
  2. confirmed the GPU is present and unbound,
  3. started extra/remote/serve.py in another shell.

Variants:
  real-sync-order   Real KDB attempt with msg1 sysmem sync and mailbox ordering.
  sync-invalidate   Real KDB attempt plus AM_PSP_MSG1_SYSMEM_SYNC_INVALIDATE=1.
  contig-msg1-gart Real KDB attempt plus AM_PSP_SYSMSG1_GART_CONTIG=1.
  audit             Stop before KDB mailbox writes at AM_PSP_AUDIT_PRE_KDB=1.

Options:
  --remote HOST:PORT  Remote bridge address. Default: 127.0.0.1:6667
  --variant NAME      Variant to run. Default: contig-msg1-gart
  --out DIR           Capture directory. Default: extra/amdpci/captures
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

REMOTE="127.0.0.1:6667"
VARIANT="contig-msg1-gart"
OUT_DIR="extra/amdpci/captures"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --remote)
      [ "$#" -ge 2 ] || die "--remote needs HOST:PORT"
      REMOTE="$2"
      shift 2
      ;;
    --variant)
      [ "$#" -ge 2 ] || die "--variant needs a name"
      VARIANT="$2"
      shift 2
      ;;
    --out)
      [ "$#" -ge 2 ] || die "--out needs a directory"
      OUT_DIR="$2"
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

case "$VARIANT" in
  real-sync-order|sync-invalidate|contig-msg1-gart|audit) ;;
  *) die "unknown variant: $VARIANT" ;;
esac

mkdir -p "$OUT_DIR"
ts="$(date +%Y%m%d-%H%M%S)"
out="$OUT_DIR/kdb-${VARIANT}-${ts}.log"

echo "variant=$VARIANT"
echo "remote=$REMOTE"
echo "out=$out"

envs=(
  AM_REMOTE_DISCOVERY_PROFILE=gfx1100_744c
  AM_REMOTE_SMALL_BAR_DISCOVERY=1
  AM_REMOTE_SKIP_RESIZE_BAR=1
  AM_PSP_TRACE_MAP_BAR0_LAST=1
  AM_PSP_TLB_TRACE=1
  AM_PSP_GART_SETUP_TRACE=1
  AM_PSP_GMC_INIT_TRACE=1
  AM_PSP_TRACE=1
  AM_PSP_TRACE_REGS=1
  AM_PSP_PARITY_TRACE=1
  AM_PSP_TRACE_C2PMSG_DENSE=1
  AM_PSP_KDB_FAIL_CAPTURE=1
  AM_PSP_KDB_FAIL_CAPTURE_MS=50
  AM_PSP_KDB_FAIL_CAPTURE_READS=512
  AM_PSP_MSG1_SYSMEM_SYNC=1
  AM_PSP_KDB_ORDER_BARRIER=1
  AM_PSP_MAILBOX_STRONG_ORDER=1
  AM_PSP_SYSMSG1_GART=1
  AM_PSP_GART_LINUX_CONTEXT=1
  AM_PSP_GART_LINUX_FULL_CONTEXT=1
  AM_PSP_GART_STRONG_INVALIDATE=1
  AM_PSP_GART_MSG1_OFFSET=0x700000
  AM_PSP_GART_APERTURE_HIGH=0x217fbf
  AM_PSP_GART_DEFAULT_ADDR=0x5feaff
  AM_PSP_GART_FAULT_DEFAULT_ADDR=0xbdfff
  AM_PSP_KDB_SKIP_PREFIX=0x640
  AM_PSP_ZERO_MSG1=1
  AM_PSP_MSG1_READBACK=1
  REMOTE_TIMEOUT=3
)

if [ "$VARIANT" = "contig-msg1-gart" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1)
fi

if [ "$VARIANT" = "sync-invalidate" ]; then
  envs+=(AM_PSP_MSG1_SYSMEM_SYNC_INVALIDATE=1)
fi

if [ "$VARIANT" = "audit" ]; then
  envs+=(AM_PSP_AUDIT_PRE_KDB=1)
fi

set +e
env "${envs[@]}" .venv/bin/python extra/remote/amd_repro.py "$REMOTE" \
  --stage psp-setup-clean-gate-amd-boot --sizes 4096 --repeat 0 > "$out" 2>&1
rc=$?
set -e

echo "rc=$rc"
echo "out=$out"
sha256sum "$out"

grep -n "setup-gate\\|released invalidate17_sem\\|_released=1\\|msg1 sysmem gart\\|msg1 sysmem sync\\|KDB order barrier\\|mailbox before-reg36\\|mailbox post-compid\\|KDB\\|load component\\|write msg1\\|write compid\\|wait BL\\|C2PMSG35\\|C2PMSG36\\|C2PMSG81\\|kdb fail capture\\|AMDDevice ready\\|Traceback\\|RuntimeError\\|TimeoutError" "$out" | tail -560 || true

tail -240 "$out"
exit "$rc"
