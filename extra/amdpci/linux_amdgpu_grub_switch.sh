#!/usr/bin/env bash
set -euo pipefail

ENTRY_TITLE="Ubuntu blacklisted amdgpu for tinygrad PSP audit"
ENTRY_ID="tinygrad-amdgpu-blacklist"
CUSTOM_FILE="/etc/grub.d/40_custom"
MARK_BEGIN="# tinygrad-amdgpu-blacklist begin"
MARK_END="# tinygrad-amdgpu-blacklist end"

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
  sudo extra/amdpci/linux_amdgpu_grub_switch.sh install
  sudo extra/amdpci/linux_amdgpu_grub_switch.sh next-blacklist
  sudo reboot

Then, to return to normal boot:
  sudo extra/amdpci/linux_amdgpu_grub_switch.sh next-normal
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
  echo "current kernel: $(uname -r)"
  echo "root arg: $(root_arg || true)"
  echo "cmdline: $(cat /proc/cmdline)"
  if lsmod | grep -q '^amdgpu '; then
    echo "amdgpu module: loaded"
  else
    echo "amdgpu module: not loaded"
  fi
  if entry_exists; then
    echo "custom entry: installed in $CUSTOM_FILE"
  else
    echo "custom entry: not installed in $CUSTOM_FILE"
  fi
  if grub_cfg_has_entry; then
    echo "grub.cfg entry: present"
  else
    echo "grub.cfg entry: missing"
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
