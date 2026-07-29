#!/bin/bash
# Pure classifier for `systemextensionsctl list` output. Keep this separate from
# the installer so captured multi-registration states can exercise the same
# parser without building or changing a system extension.
set -euo pipefail

tinygpu_classify_extension_rows() {
  local dext_id="$1" legacy_dext_id="$2" expected_version="$3"
  local line arkey_count=0 current_active=0 other_active=0 waiting=0 transitional=0 legacy_active=0

  while IFS= read -r line; do
    if [[ "$line" == *"$dext_id ("* ]]; then
      ((arkey_count += 1))
      [[ "$line" == *"terminating"* || "$line" == *"waiting to uninstall"* || "$line" == *"being replaced"* ]] && \
        ((transitional += 1))
      [[ "$line" == *"[activated waiting for user]"* ]] && ((waiting += 1))
      if [[ "$line" == *"[activated enabled]"* ]]; then
        if [[ "$line" == *"/$expected_version)"* ]]; then ((current_active += 1)); else ((other_active += 1)); fi
      fi
    elif [[ "$line" == *"$legacy_dext_id ("* && "$line" == *"[activated enabled]"* ]]; then
      ((legacy_active += 1))
    fi
  done

  if (( arkey_count == 0 )); then
    (( legacy_active == 0 )) && printf '%s\n' inactive || printf '%s\n' pending_reboot
  elif (( current_active == 1 && arkey_count == 1 && transitional == 0 && legacy_active == 0 )); then
    printf '%s\n' active_current
  elif (( other_active == 1 && arkey_count == 1 && transitional == 0 && legacy_active == 0 )); then
    printf '%s\n' active_other_version
  elif (( waiting == 1 && arkey_count == 1 && transitional == 0 && legacy_active == 0 )); then
    printf '%s\n' needs_approval
  elif (( arkey_count > 1 || transitional > 0 || legacy_active > 0 )); then
    printf '%s\n' pending_reboot
  else
    printf '%s\n' activating
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  [[ $# == 3 ]] || { echo "usage: $(basename "$0") DEXT_ID LEGACY_DEXT_ID EXPECTED_VERSION" >&2; exit 2; }
  tinygpu_classify_extension_rows "$1" "$2" "$3"
fi
