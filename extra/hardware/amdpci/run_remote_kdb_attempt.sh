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
  contig-top-table Real KDB attempt with contiguous msg1 and top sparse GART table.
  contig-top-quiet Real KDB attempt with contiguous/top table and minimal observers.
  linux-pre-kdb-seq Real KDB attempt with Linux-like pre-KDB invalidate cadence.
  kdb-pipeline-seq Real KDB attempt that pipelines the post-KDB component.
  bl-pipeline-seq Real KDB attempt that pipelines early bootloader components.
  sos-pipeline-seq Real KDB attempt that pipelines the full sOS bootloader sequence.
  sos-pipeline-slow Real KDB attempt that pipelines full sOS with a 2500us gap.
  sos-payload-audit Real slow sOS pipeline plus all-component payload audit.
  sos-delay20    Real slow sOS pipeline plus Linux-like post-SOS 20ms delay.
  sos-final-state-audit Real sOS delay attempt plus final mailbox state audit.
  bl-boundary-1..8 Pipeline to a bootloader boundary, then wait and audit state.
  tos-spl-audit Boundary-1 run plus bootloader payload audit for the 0x10000000 load.
  tos-spl-normal-wait Normal first-KDB wait, then audit/wait the 0x10000000 load.
  kdb-first-audit Normal first-KDB wait plus payload, boundary, and fail-window audit.
  kdb-first-linux-invalidate First-KDB audit with Linux-count pre-KDB invalidate burst.
  kdb-first-minimal-gart First-KDB audit with simpler msg1 GART placement.
  kdb-metadata-audit Audit bootloader component metadata and stop before mailbox writes.
  kdb-metadata-pair-audit Audit first two KDB-family component metadata without mailbox writes.
  tos-source-metadata-audit Inventory available PSP firmware blobs before mailbox writes.
  kdb-header-audit Audit raw KDB header dwords and candidate offsets before mailbox writes.
  kdb-record-audit Audit repeated raw KDB records before mailbox writes.
  kdb-first-wait-trace-dense First-KDB minimal GART with dense post-command C2PMSG sample.
  kdb-slice-000/100/400/500/600/640 First-KDB dense trace with fixed 0x1700 KDB slice.
  kdb-rec-150/540/690/7e0/a80 First-KDB dense trace with record-boundary KDB slice.
  kdb-rec-690-tail/7e0-tail/a80-tail First-KDB dense trace from record boundary to KDB end.
  kdb-full-raw First-KDB dense trace with full raw KDB blob, matching Linux v13 memcpy path.
  kdb-full-raw-linux-wait Full raw KDB with minimal post-command observers before BL wait.
  kdb-full-raw-primary-sync Full raw KDB plus full 1MB msg1 sync/readback before BL wait.
  kdb-full-raw-gart-audit Audit full raw KDB msg1/MMHUB/GART state before mailbox writes.
  kdb-full-raw-cid2-audit Audit whether Linux-observed CID2 can stick before mailbox writes.
  kdb-full-raw-linux-final-invalidate Full raw KDB with Linux-observed final CID2/invalidate before mailbox writes.
  kdb-full-raw-linux-mmhub-window Full raw KDB with Linux-observed final MMHUB write window before mailbox writes.
  kdb-linux-mmhub-window Linux-sized KDB payload with Linux-observed final MMHUB write window before mailbox writes.
  kdb-linux-msg1-full-audit Linux-sized KDB/MMHUB path with full 1MB msg1 zero-tail audit.
  kdb-linux-fw-pri-equivalence Linux-sized KDB/MMHUB path with fw_pri/GART equivalence audit.
  kdb-linux-mmhub-sync-invalidate Linux-sized KDB/MMHUB path with msg1 sync invalidate before mailbox writes.
  kdb-linux-mmhub-nosnoop Linux-sized KDB/MMHUB path with non-snooped sysmem GART PTEs.
  kdb-linux-mmhub-map-unlocked Linux-sized KDB/MMHUB path with unlocked contiguous remote sysmem.
  kdb-linux-mmhub-map-nopopulate Linux-sized KDB/MMHUB path with no-MAP_POPULATE contiguous remote sysmem.
  kdb-pair-linux-timing Send 0x80000 then 0x10000000 with Linux-like timing.
  vram-msg1-quiet Real KDB attempt with VRAM msg1 and minimal observers.
  sorted-msg1-gart Real KDB attempt with msg1 GART pages sorted by paddr.
  top-table-sparse Real KDB attempt with sparse GART table near Linux top-of-VRAM.
  payload-audit   Real KDB attempt with exact msg1 payload byte audit.
  audit             Stop before KDB mailbox writes at AM_PSP_AUDIT_PRE_KDB=1.

Options:
  --remote HOST:PORT  Remote bridge address. Default: 127.0.0.1:6667
  --variant NAME      Variant to run. Default: contig-msg1-gart
  --out DIR           Capture directory. Default: extra/hardware/amdpci/captures
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

drop_env() {
  local name="$1"
  local kept=()
  local item
  for item in "${envs[@]}"; do
    case "$item" in
      "$name"=*) ;;
      *) kept+=("$item") ;;
    esac
  done
  envs=("${kept[@]}")
}

REMOTE="127.0.0.1:6667"
VARIANT="contig-msg1-gart"
OUT_DIR="extra/hardware/amdpci/captures"

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
  real-sync-order|sync-invalidate|contig-msg1-gart|contig-top-table|contig-top-quiet|linux-pre-kdb-seq|kdb-pipeline-seq|bl-pipeline-seq|sos-pipeline-seq|sos-pipeline-slow|sos-payload-audit|sos-delay20|sos-final-state-audit|tos-spl-audit|tos-spl-normal-wait|kdb-first-audit|kdb-first-linux-invalidate|kdb-first-minimal-gart|kdb-metadata-audit|kdb-metadata-pair-audit|tos-source-metadata-audit|kdb-header-audit|kdb-record-audit|kdb-first-wait-trace-dense|kdb-slice-000|kdb-slice-100|kdb-slice-400|kdb-slice-500|kdb-slice-600|kdb-slice-640|kdb-rec-150|kdb-rec-540|kdb-rec-690|kdb-rec-7e0|kdb-rec-a80|kdb-rec-690-tail|kdb-rec-7e0-tail|kdb-rec-a80-tail|kdb-full-raw|kdb-full-raw-linux-wait|kdb-full-raw-primary-sync|kdb-full-raw-gart-audit|kdb-full-raw-cid2-audit|kdb-full-raw-linux-final-invalidate|kdb-full-raw-linux-mmhub-window|kdb-linux-mmhub-window|kdb-linux-msg1-full-audit|kdb-linux-fw-pri-equivalence|kdb-linux-mmhub-sync-invalidate|kdb-linux-mmhub-nosnoop|kdb-linux-mmhub-map-unlocked|kdb-linux-mmhub-map-nopopulate|kdb-pair-linux-timing|bl-boundary-1|bl-boundary-2|bl-boundary-3|bl-boundary-4|bl-boundary-5|bl-boundary-6|bl-boundary-7|bl-boundary-8|vram-msg1-quiet|sorted-msg1-gart|top-table-sparse|payload-audit|audit) ;;
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

if [ "$VARIANT" = "contig-top-table" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000)
fi

if [ "$VARIANT" = "contig-top-quiet" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "linux-pre-kdb-seq" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 AM_PSP_PRE_KDB_INVALIDATE_BURST=16)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-pipeline-seq" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_KDB_PIPELINE_SEQ=1 AM_PSP_KDB_PIPELINE_COUNT=1 AM_PSP_KDB_PIPELINE_DELAY_US=900)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "bl-pipeline-seq" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_PIPELINE_COUNT=3 AM_PSP_BL_PIPELINE_DELAY_US=900)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "sos-pipeline-seq" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_PIPELINE_COUNT=8 AM_PSP_BL_PIPELINE_DELAY_US=900)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "sos-pipeline-slow" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_PIPELINE_COUNT=8 AM_PSP_BL_PIPELINE_DELAY_US=2500)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "sos-payload-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_PIPELINE_COUNT=8 AM_PSP_BL_PIPELINE_DELAY_US=2500 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=128)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "sos-delay20" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_PIPELINE_COUNT=8 AM_PSP_BL_PIPELINE_DELAY_US=2500 \
         AM_PSP_SOS_WAIT_DELAY_MS=20)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "sos-final-state-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_PIPELINE_COUNT=8 AM_PSP_BL_PIPELINE_DELAY_US=2500 \
         AM_PSP_SOS_WAIT_DELAY_MS=20 AM_PSP_SOS_FINAL_STATE_AUDIT=1)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [[ "$VARIANT" =~ ^bl-boundary-([1-8])$ ]]; then
  boundary_count="${BASH_REMATCH[1]}"
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_PIPELINE_COUNT="$boundary_count" AM_PSP_BL_PIPELINE_DELAY_US=2500 \
         AM_PSP_BL_BOUNDARY_AUDIT=1)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "tos-spl-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_PIPELINE_COUNT=1 AM_PSP_BL_PIPELINE_DELAY_US=2500 \
         AM_PSP_BL_BOUNDARY_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "tos-spl-normal-wait" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_BOUNDARY_AUDIT=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-first-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=16 AM_PSP_BL_BOUNDARY_AUDIT=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256 \
         AM_PSP_KDB_FAIL_CAPTURE=1 AM_PSP_KDB_FAIL_CAPTURE_MS=50 AM_PSP_KDB_FAIL_CAPTURE_READS=512)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-first-linux-invalidate" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000 \
         AM_PSP_PRE_KDB_INVALIDATE_BURST=30 AM_PSP_BL_BOUNDARY_AUDIT=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256 \
         AM_PSP_KDB_FAIL_CAPTURE=1 AM_PSP_KDB_FAIL_CAPTURE_PRE_COMMAND=0 AM_PSP_KDB_FAIL_CAPTURE_MS=50 AM_PSP_KDB_FAIL_CAPTURE_READS=512)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-first-minimal-gart" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_BL_BOUNDARY_AUDIT=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256 \
         AM_PSP_KDB_FAIL_CAPTURE=1 AM_PSP_KDB_FAIL_CAPTURE_PRE_COMMAND=0 AM_PSP_KDB_FAIL_CAPTURE_MS=50 AM_PSP_KDB_FAIL_CAPTURE_READS=512)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-metadata-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_BL_METADATA_AUDIT=1 AM_PSP_BL_METADATA_AUDIT_BYTES=128 AM_PSP_BL_METADATA_AUDIT_STOP=1)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-metadata-pair-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_BL_METADATA_AUDIT=1 AM_PSP_BL_METADATA_AUDIT_BYTES=128 \
         AM_PSP_BL_METADATA_AUDIT_STOP=1 AM_PSP_BL_METADATA_AUDIT_STOP_AFTER=2)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "tos-source-metadata-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_SOS_FW_INVENTORY_AUDIT=1 AM_PSP_SOS_FW_INVENTORY_AUDIT_BYTES=128 AM_PSP_SOS_FW_INVENTORY_AUDIT_STOP=1)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-header-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_KDB_HEADER_AUDIT=1 AM_PSP_KDB_HEADER_AUDIT_BYTES=0x200 AM_PSP_KDB_HEADER_AUDIT_STOP=1)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-record-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_KDB_RECORD_AUDIT=1 AM_PSP_KDB_RECORD_AUDIT_START=0x150 \
         AM_PSP_KDB_RECORD_AUDIT_STRIDE=0x150 AM_PSP_KDB_RECORD_AUDIT_BYTES=64 AM_PSP_KDB_RECORD_AUDIT_STOP=1)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-first-wait-trace-dense" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256 \
         AM_PSP_KDB_FAIL_CAPTURE=1 AM_PSP_KDB_FAIL_CAPTURE_PRE_COMMAND=0 AM_PSP_KDB_FAIL_CAPTURE_MS=2 AM_PSP_KDB_FAIL_CAPTURE_READS=256)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [[ "$VARIANT" =~ ^kdb-slice-(000|100|400|500|600|640)$ ]]; then
  slice_off="0x${BASH_REMATCH[1]}"
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_KDB_SLICE_OFFSET="$slice_off" AM_PSP_KDB_SLICE_SIZE=0x1700 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256 \
         AM_PSP_KDB_FAIL_CAPTURE=1 AM_PSP_KDB_FAIL_CAPTURE_PRE_COMMAND=0 AM_PSP_KDB_FAIL_CAPTURE_MS=2 AM_PSP_KDB_FAIL_CAPTURE_READS=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [[ "$VARIANT" =~ ^kdb-rec-(150|540|690|7e0|a80)$ ]]; then
  slice_off="0x${BASH_REMATCH[1]}"
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_KDB_SLICE_OFFSET="$slice_off" AM_PSP_KDB_SLICE_SIZE=0x1700 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256 \
         AM_PSP_KDB_FAIL_CAPTURE=1 AM_PSP_KDB_FAIL_CAPTURE_PRE_COMMAND=0 AM_PSP_KDB_FAIL_CAPTURE_MS=2 AM_PSP_KDB_FAIL_CAPTURE_READS=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [[ "$VARIANT" =~ ^kdb-rec-(690|7e0|a80)-tail$ ]]; then
  slice_off="0x${BASH_REMATCH[1]}"
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_KDB_SLICE_OFFSET="$slice_off" \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256 \
         AM_PSP_KDB_FAIL_CAPTURE=1 AM_PSP_KDB_FAIL_CAPTURE_PRE_COMMAND=0 AM_PSP_KDB_FAIL_CAPTURE_MS=2 AM_PSP_KDB_FAIL_CAPTURE_READS=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_KDB_SLICE_SIZE AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-full-raw" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256 \
         AM_PSP_KDB_FAIL_CAPTURE=1 AM_PSP_KDB_FAIL_CAPTURE_PRE_COMMAND=0 AM_PSP_KDB_FAIL_CAPTURE_MS=2 AM_PSP_KDB_FAIL_CAPTURE_READS=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE \
              AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-full-raw-linux-wait" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-full-raw-primary-sync" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-full-raw-gart-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 \
         AM_PSP_PRE_KDB_GART_AUDIT=1 AM_PSP_PRE_KDB_GART_AUDIT_STOP=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-full-raw-cid2-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 \
         AM_PSP_PRE_KDB_CID2_AUDIT=1 AM_PSP_PRE_KDB_CID2_AUDIT_STOP=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-full-raw-linux-final-invalidate" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 \
         AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE=1 AM_PSP_PRE_KDB_LINUX_FINAL_CID2=0x12104010 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-full-raw-linux-mmhub-window" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 \
         AM_PSP_PRE_KDB_LINUX_MMHUB_WINDOW=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SKIP_PREFIX AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-linux-mmhub-window" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 AM_PSP_KDB_SKIP_PREFIX=0x640 \
         AM_PSP_PRE_KDB_LINUX_MMHUB_WINDOW=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-linux-msg1-full-audit" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 AM_PSP_MSG1_FULL_AUDIT=1 AM_PSP_KDB_SKIP_PREFIX=0x640 \
         AM_PSP_PRE_KDB_LINUX_MMHUB_WINDOW=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-linux-fw-pri-equivalence" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 AM_PSP_MSG1_FULL_AUDIT=1 \
         AM_PSP_FW_PRI_EQUIV_AUDIT=1 AM_PSP_KDB_SKIP_PREFIX=0x640 \
         AM_PSP_PRE_KDB_LINUX_MMHUB_WINDOW=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-linux-mmhub-sync-invalidate" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 AM_PSP_MSG1_SYSMEM_SYNC_INVALIDATE=1 \
         AM_PSP_KDB_SKIP_PREFIX=0x640 AM_PSP_PRE_KDB_LINUX_MMHUB_WINDOW=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-linux-mmhub-nosnoop" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 AM_PSP_KDB_SKIP_PREFIX=0x640 \
         AM_PSP_GART_SNOOPED=0 AM_PSP_PRE_KDB_LINUX_MMHUB_WINDOW=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-linux-mmhub-map-unlocked" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 AM_PSP_KDB_SKIP_PREFIX=0x640 \
         AM_REMOTE_SYSMEM_MODE=3 AM_PSP_PRE_KDB_LINUX_MMHUB_WINDOW=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-linux-mmhub-map-nopopulate" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_MSG1_PRIMARY_SYNC=1 AM_PSP_KDB_SKIP_PREFIX=0x640 \
         AM_REMOTE_SYSMEM_MODE=4 AM_PSP_PRE_KDB_LINUX_MMHUB_WINDOW=1 \
         AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_KDB_SLICE_OFFSET AM_PSP_KDB_SLICE_SIZE \
              AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE \
              AM_PSP_KDB_FAIL_CAPTURE AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS \
              AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER AM_PSP_PRE_KDB_LINUX_FINAL_INVALIDATE; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "kdb-pair-linux-timing" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_CONTIG=1 AM_PSP_KDB_PIPELINE_SEQ=1 AM_PSP_KDB_PIPELINE_COUNT=1 AM_PSP_KDB_PIPELINE_DELAY_US=700 \
         AM_PSP_BL_BOUNDARY_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT=1 AM_PSP_BL_PAYLOAD_AUDIT_BYTES=256)
  for name in AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "vram-msg1-quiet" ]; then
  envs+=(AM_PSP_SYSMSG1_VRAM=1)
  for name in AM_PSP_SYSMSG1_GART AM_PSP_TRACE_REGS AM_PSP_PARITY_TRACE AM_PSP_TRACE_C2PMSG_DENSE AM_PSP_KDB_FAIL_CAPTURE \
              AM_PSP_KDB_FAIL_CAPTURE_MS AM_PSP_KDB_FAIL_CAPTURE_READS AM_PSP_KDB_ORDER_BARRIER AM_PSP_MAILBOX_STRONG_ORDER; do
    drop_env "$name"
  done
fi

if [ "$VARIANT" = "sorted-msg1-gart" ]; then
  envs+=(AM_PSP_SYSMSG1_GART_SORT_PADDRS=1 AM_PSP_KDB_PAYLOAD_AUDIT=1 AM_PSP_KDB_PAYLOAD_AUDIT_BYTES=128)
fi

if [ "$VARIANT" = "sync-invalidate" ]; then
  envs+=(AM_PSP_MSG1_SYSMEM_SYNC_INVALIDATE=1)
fi

if [ "$VARIANT" = "top-table-sparse" ]; then
  envs+=(AM_PSP_GART_TABLE_TOP=1 AM_PSP_GART_TABLE_SPARSE=1 AM_PSP_GART_TABLE_ADDR=0x5feb00000)
fi

if [ "$VARIANT" = "payload-audit" ]; then
  envs+=(AM_PSP_KDB_PAYLOAD_AUDIT=1 AM_PSP_KDB_PAYLOAD_AUDIT_BYTES=128)
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

grep -n "setup-gate\\|released invalidate17_sem\\|_released=1\\|msg1 sysmem gart\\|msg1 sysmem sync\\|msg1 primary sync\\|msg1 full audit\\|fw_pri equivalence\\|pre-KDB GART audit\\|pre-KDB CID2 audit\\|pre-KDB linux final invalidate\\|pre-KDB linux MMHUB window\\|regMM\\|KDB order barrier\\|KDB payload audit\\|bootloader payload audit\\|bootloader metadata audit\\|sos fw inventory\\|KDB header audit\\|KDB record audit\\|gart pte\\|mailbox before-reg36\\|mailbox post-compid\\|bootloader pipeline\\|sOS final state audit\\|sOS wait delay\\|KDB\\|load component\\|write msg1\\|write compid\\|wait BL\\|sOS\\|C2PMSG35\\|C2PMSG36\\|C2PMSG81\\|kdb fail capture\\|AMDDevice ready\\|Traceback\\|RuntimeError\\|TimeoutError" "$out" | tail -560 || true

tail -240 "$out"
exit "$rc"
