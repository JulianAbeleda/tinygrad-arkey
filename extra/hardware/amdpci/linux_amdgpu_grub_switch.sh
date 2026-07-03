#!/usr/bin/env bash
set -euo pipefail

ENTRY_TITLE="Ubuntu blacklisted amdgpu for tinygrad PSP audit"
ENTRY_ID="tinygrad-amdgpu-blacklist"
CUSTOM_FILE="/etc/grub.d/40_custom"
MARK_BEGIN="# tinygrad-amdgpu-blacklist begin"
MARK_END="# tinygrad-amdgpu-blacklist end"
TARGET_BDF="${TARGET_BDF:-0000:08:00.0}"

usage() {
  cat <<EOF
usage: linux_amdgpu_grub_switch.sh COMMAND

Manage a persistent GRUB entry that boots Ubuntu with amdgpu blacklisted.
The normal GRUB default entry is left untouched.

Commands:
  status           Show whether the entry exists and the current boot state.
  install          Add or refresh the blacklisted GRUB menu entry.
  remove           Remove the blacklisted GRUB menu entry.
  next-blacklist   Use the blacklisted entry for the next reboot only.
  next-normal      Use the normal first GRUB entry for the next reboot only.

Typical use:
  sudo extra/hardware/amdpci/linux_amdgpu_grub_switch.sh install
  sudo extra/hardware/amdpci/linux_amdgpu_grub_switch.sh next-blacklist
  sudo reboot

Then, to return to normal boot:
  sudo extra/hardware/amdpci/linux_amdgpu_grub_switch.sh next-normal
  sudo reboot
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

need_linux() {
  [ "$(uname -s)" = "Linux" ] || die "this tool only runs on Linux"
}

need_root() {
  [ "$(id -u)" -eq 0 ] || die "run with sudo/root"
}

update_grub_cmd() {
  if command -v update-grub >/dev/null 2>&1; then
    echo update-grub
  elif command -v grub-mkconfig >/dev/null 2>&1; then
    echo "grub-mkconfig -o /boot/grub/grub.cfg"
  else
    die "missing update-grub or grub-mkconfig"
  fi
}

root_arg() {
  tr ' ' '\n' < /proc/cmdline | awk '/^root=/{print; exit}'
}

root_uuid() {
  findmnt -no UUID / 2>/dev/null | head -n1
}

entry_exists() {
  [ -f "$CUSTOM_FILE" ] && grep -qF "$MARK_BEGIN" "$CUSTOM_FILE"
}

grub_cfg_has_entry() {
  [ -f /boot/grub/grub.cfg ] && grep -qF "$ENTRY_TITLE" /boot/grub/grub.cfg
}

grub_cfg_state() {
  if [ ! -f /boot/grub/grub.cfg ]; then
    echo "missing"
  elif [ ! -r /boot/grub/grub.cfg ]; then
    echo "unreadable"
  elif grep -qF "$ENTRY_TITLE" /boot/grub/grub.cfg; then
    echo "present"
  else
    echo "missing"
  fi
}

write_entry() {
  local kver root rootuuid search_line tmp
  kver="$(uname -r)"
  root="$(root_arg)"
  [ -n "$root" ] || die "could not find root= in /proc/cmdline"
  rootuuid="$(root_uuid || true)"
  search_line=""
  if [ -n "$rootuuid" ]; then
    search_line="    search --no-floppy --fs-uuid --set=root $rootuuid"
  fi

  tmp="$(mktemp)"
  if [ -f "$CUSTOM_FILE" ]; then
    awk -v begin="$MARK_BEGIN" -v end="$MARK_END" '
      $0 == begin {skip=1; next}
      $0 == end {skip=0; next}
      !skip {print}
    ' "$CUSTOM_FILE" > "$tmp"
  else
    cat > "$tmp" <<'EOF'
#!/bin/sh
exec tail -n +3 $0
EOF
  fi

  cat >> "$tmp" <<EOF

$MARK_BEGIN
menuentry '$ENTRY_TITLE' --id '$ENTRY_ID' {
    recordfail
    load_video
    gfxmode \$linux_gfx_mode
    insmod gzio
    if [ x\$grub_platform = xxen ]; then insmod xzio; insmod lzopio; fi
    insmod part_gpt
    insmod ext2
$search_line
    linux /boot/vmlinuz-$kver $root ro quiet splash modprobe.blacklist=amdgpu
    initrd /boot/initrd.img-$kver
}
$MARK_END
EOF

  install -m 0755 "$tmp" "$CUSTOM_FILE"
  rm -f "$tmp"
}

cmd_status() {
  need_linux
  local cmdline amdgpu_loaded gpu_info blacklist_present gpu_present gpu_bound next_state
  cmdline="$(cat /proc/cmdline)"
  echo "current kernel: $(uname -r)"
  echo "root arg: $(root_arg || true)"
  echo "cmdline: $cmdline"
  if grep -q 'modprobe.blacklist=amdgpu' <<<"$cmdline"; then
    blacklist_present=1
    echo "current boot mode: blacklisted"
  else
    blacklist_present=0
    echo "current boot mode: normal"
  fi

  if command -v grub-editenv >/dev/null 2>&1; then
    next_state="$(grub-editenv list 2>/dev/null || true)"
    echo "grub one-shot state: ${next_state:-<empty>}"
  else
    echo "grub one-shot state: unavailable; missing grub-editenv"
  fi

  if lsmod | awk '{print $1}' | grep -qx 'amdgpu'; then
    amdgpu_loaded=1
    echo "amdgpu module: loaded"
  else
    amdgpu_loaded=0
    echo "amdgpu module: not loaded"
  fi
  if entry_exists; then
    echo "custom entry: installed in $CUSTOM_FILE"
  else
    echo "custom entry: not installed in $CUSTOM_FILE"
  fi
  case "$(grub_cfg_state)" in
    present) echo "grub.cfg entry: present" ;;
    unreadable) echo "grub.cfg entry: unknown; /boot/grub/grub.cfg is not readable by this user" ;;
    *) echo "grub.cfg entry: missing" ;;
  esac

  echo "target BDF: $TARGET_BDF"
  if command -v lspci >/dev/null 2>&1; then
    gpu_info="$(lspci -Dnnk -s "$TARGET_BDF" 2>/dev/null || true)"
    if [ -n "$gpu_info" ]; then
      gpu_present=1
      echo "target GPU: present"
      echo "$gpu_info"
    else
      gpu_present=0
      echo "target GPU: missing at $TARGET_BDF"
    fi
    echo "Navi31 functions:"
    lspci -Dnn | grep -Ei '1002:744c|1002:ab30|1002:7446|1002:7444|vga|3d|display' || true
  else
    gpu_present=0
    echo "target GPU: unknown; missing lspci"
  fi

  if [ "${gpu_present:-0}" -eq 1 ] && grep -q 'Kernel driver in use:' <<<"$gpu_info"; then
    gpu_bound=1
  else
    gpu_bound=0
  fi

  if [ "$blacklist_present" -eq 0 ] && [ "$amdgpu_loaded" -eq 1 ] && [ "${gpu_present:-0}" -eq 1 ] && [ "$gpu_bound" -eq 1 ]; then
    echo "interpreted state: NORMAL_HEALTHY"
  elif [ "$blacklist_present" -eq 1 ] && [ "$amdgpu_loaded" -eq 0 ] && [ "${gpu_present:-0}" -eq 1 ] && [ "$gpu_bound" -eq 0 ]; then
    echo "interpreted state: BLACKLISTED_READY"
  elif [ "${gpu_present:-0}" -eq 0 ]; then
    echo "interpreted state: GPU_MISSING_FROM_PCI"
  else
    echo "interpreted state: MIXED_OR_DIRTY"
  fi
}

cmd_install() {
  need_linux
  need_root
  need_cmd findmnt
  mkdir -p "$(dirname "$CUSTOM_FILE")"
  if [ -f "$CUSTOM_FILE" ]; then
    cp "$CUSTOM_FILE" "$CUSTOM_FILE.bak.$(date +%Y%m%d-%H%M%S)"
  fi
  write_entry
  sh -c "$(update_grub_cmd)"
  grep -nF "$ENTRY_TITLE" /boot/grub/grub.cfg || die "updated grub.cfg does not contain the blacklisted entry"
  echo "installed: $ENTRY_TITLE"
}

cmd_remove() {
  need_linux
  need_root
  [ -f "$CUSTOM_FILE" ] || die "$CUSTOM_FILE does not exist"
  cp "$CUSTOM_FILE" "$CUSTOM_FILE.bak.$(date +%Y%m%d-%H%M%S)"
  local tmp
  tmp="$(mktemp)"
  awk -v begin="$MARK_BEGIN" -v end="$MARK_END" '
    $0 == begin {skip=1; next}
    $0 == end {skip=0; next}
    !skip {print}
  ' "$CUSTOM_FILE" > "$tmp"
  install -m 0755 "$tmp" "$CUSTOM_FILE"
  rm -f "$tmp"
  sh -c "$(update_grub_cmd)"
  echo "removed: $ENTRY_TITLE"
}

cmd_next_blacklist() {
  need_linux
  need_root
  need_cmd grub-reboot
  if ! grub_cfg_has_entry; then
    echo "blacklisted entry is not present in grub.cfg; installing it first"
    cmd_install
  fi
  grub-reboot "$ENTRY_ID"
  echo "next boot: $ENTRY_TITLE"
}

cmd_next_normal() {
  need_linux
  need_root
  need_cmd grub-reboot
  grub-reboot 0
  echo "next boot: normal GRUB entry 0"
}

case "${1:-}" in
  status) cmd_status ;;
  install) cmd_install ;;
  remove) cmd_remove ;;
  next-blacklist) cmd_next_blacklist ;;
  next-normal) cmd_next_normal ;;
  -h|--help|"") usage ;;
  *) usage >&2; die "unknown command: $1" ;;
esac
