#!/bin/bash
# Audited development-only TinyGPU build/install path for this SIP-disabled host.
set -euo pipefail

APP_NAME="TinyGPU.app"
APP_ID="org.tinygrad.arkey.tinygpu.installer"
DEXT_ID="org.tinygrad.arkey.tinygpu.driver2"
# Increment this whenever the DriverKit binary or its activation contract changes;
# macOS will not replace an already-active extension at the same bundle version.
DEXT_VERSION="5"
FEATURE_BRANCH="exp"
APPROVAL_TOKEN="APPROVE_TINYGPU_DEVELOPMENT_INSTALL"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EXPECTED_PROVENANCE="$REPO_ROOT/docs/task_workflow/output/tinygpu-development-install-provenance.txt"
DERIVED_DATA="$SCRIPT_DIR/build/DerivedData"
BUILD_APP="$DERIVED_DATA/Build/Products/Debug/$APP_NAME"
BUILD_DEXT="$BUILD_APP/Contents/Library/SystemExtensions/$DEXT_ID.dext"
APPLICATIONS_DIR="/Applications"
INSTALL_APP="$APPLICATIONS_DIR/$APP_NAME"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"

mode="build"
build_seen=0
install_seen=0
provenance_out=""
provenance_tmp=""
stage_dir=""
backup_app=""
replacement_moved=0
previous_extension_active=0
source_commit=""

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [--build]
  $(basename "$0") --install $APPROVAL_TOKEN --provenance-out "$EXPECTED_PROVENANCE"

Builds a Debug, ad-hoc development app by default. Installation is accepted
only from the clean linked $FEATURE_BRANCH worktree under /tmp/gpu-bench.lock,
then requires the approval token again interactively immediately before the
/Applications replacement. This is development provenance, not production trust.
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }

record_command() {
  local label="$1"
  shift
  {
    printf '\n[%s]\n' "$label"
    printf 'argv='; printf '%q ' "$@"; printf '\n'
    "$@" 2>&1 || printf 'command_exit=%s\n' "$?"
  } >> "$provenance_tmp"
}

record_file() {
  local label="$1" file="$2"
  {
    printf '\n[%s]\n' "$label"
    if [[ -f "$file" ]]; then
      stat -f 'path=%N size=%z modified=%Sm' "$file"
      shasum -a 256 "$file"
    else
      printf 'absent=%s\n' "$file"
    fi
  } >> "$provenance_tmp"
}

record_tree() {
  local label="$1" root="$2"
  {
    printf '\n[%s]\n' "$label"
    if [[ -d "$root" ]]; then
      find "$root" -type f -print | LC_ALL=C sort | while IFS= read -r file; do shasum -a 256 "$file"; done
    else
      printf 'absent=%s\n' "$root"
    fi
  } >> "$provenance_tmp"
}

record_source_manifest() {
  {
    printf '\n[source_manifest]\n'
    git -C "$REPO_ROOT" ls-files -- \
      extra/usbgpu/protocol extra/usbgpu/tests extra/usbgpu/tools extra/usbgpu/tbgpu/installer \
      extra/llm_research/bench.py tinygrad/runtime/support/system.py test/unit | \
      LC_ALL=C sort | while IFS= read -r path; do shasum -a 256 "$REPO_ROOT/$path"; done
  } >> "$provenance_tmp"
}

record_bundle() {
  local label="$1" app="$2"
  local dext="$app/Contents/Library/SystemExtensions/$DEXT_ID.dext"
  record_tree "${label}_app_tree" "$app"
  record_command "${label}_app_bundle_id" plutil -extract CFBundleIdentifier raw -o - "$app/Contents/Info.plist"
  record_command "${label}_app_codesign" codesign -dvvv "$app"
  record_command "${label}_app_entitlements" codesign -d --entitlements :- "$app"
  record_tree "${label}_dext_tree" "$dext"
  record_command "${label}_dext_bundle_id" plutil -extract CFBundleIdentifier raw -o - "$dext/Info.plist"
  record_command "${label}_dext_codesign" codesign -dvvv "$dext"
  record_command "${label}_dext_entitlements" codesign -d --entitlements :- "$dext"
}

record_provenance() {
  local phase="$1"
  {
    printf '\n=== %s ===\n' "$phase"
    printf 'schema=tinygpu.development-install.provenance.v1\n'
    printf 'recorded_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'repository=%s\nrun_id=%s\nsource_commit=%s\n' "$REPO_ROOT" "$RUN_ID" "$source_commit"
  } >> "$provenance_tmp"
  record_command git_head git -C "$REPO_ROOT" rev-parse HEAD
  record_command git_branch git -C "$REPO_ROOT" branch --show-current
  record_command git_status git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all
  record_command xcode xcodebuild -version
  record_command macos_sdk xcrun --sdk macosx --show-sdk-version
  record_command driverkit_sdk xcrun --sdk driverkit --show-sdk-version
  record_command csrutil_status csrutil status
  record_command source_nosip_entitlements plutil -p "$SCRIPT_DIR/TinyGPUDriverExtension/TinyGPUDriver.NoSIP.entitlements"
  record_command source_app_entitlements plutil -p "$SCRIPT_DIR/macOS/macOS.entitlements"
  record_file source_nosip_entitlements_hash "$SCRIPT_DIR/TinyGPUDriverExtension/TinyGPUDriver.NoSIP.entitlements"
  record_file source_app_entitlements_hash "$SCRIPT_DIR/macOS/macOS.entitlements"
  record_source_manifest
  record_command system_extensions systemextensionsctl list
  record_bundle installed "$INSTALL_APP"
  record_bundle built "$BUILD_APP"
}

publish_provenance() {
  [[ -n "$provenance_out" && -n "$provenance_tmp" ]] || return 0
  mv -f "$provenance_tmp" "$provenance_out"
  provenance_tmp=""
}

bundle_id() { plutil -extract CFBundleIdentifier raw -o - "$1" 2>/dev/null; }

verify_bundle() {
  local app="$1"
  local dext="$app/Contents/Library/SystemExtensions/$DEXT_ID.dext"
  [[ -d "$app" && -f "$app/Contents/MacOS/TinyGPU" && -f "$dext/$DEXT_ID" ]] || return 1
  [[ "$(bundle_id "$app/Contents/Info.plist")" == "$APP_ID" ]] || return 1
  [[ "$(bundle_id "$dext/Info.plist")" == "$DEXT_ID" ]] || return 1
  [[ "$(plutil -extract CFBundleVersion raw -o - "$dext/Info.plist" 2>/dev/null)" == "$DEXT_VERSION" ]] || return 1
  codesign --verify --strict --verbose=4 "$dext"
  codesign --verify --strict --verbose=4 "$app"
  codesign -dvvv "$app" 2>&1 | grep -F "Identifier=$APP_ID" >/dev/null
  codesign -dvvv "$app" 2>&1 | grep -F 'Signature=adhoc' >/dev/null
  codesign -dvvv "$dext" 2>&1 | grep -F "Identifier=$DEXT_ID" >/dev/null
  codesign -dvvv "$dext" 2>&1 | grep -F 'Signature=adhoc' >/dev/null
}

extension_active() {
  systemextensionsctl list 2>/dev/null | grep -F "$DEXT_ID" | grep -Fq '[activated enabled]'
}

wait_extension_state() {
  local expected="$1" deadline=$((SECONDS + 30))
  while (( SECONDS < deadline )); do
    if [[ "$expected" == active ]] && extension_active; then return 0; fi
    if [[ "$expected" == inactive ]] && ! extension_active; then return 0; fi
    sleep 1
  done
  return 1
}

validate_feature_source() {
  local git_dir common_dir status
  git_dir="$(git -C "$REPO_ROOT" rev-parse --absolute-git-dir)"
  common_dir="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
  [[ "$git_dir" != "$common_dir" ]] || die "installation from the production worktree is forbidden"
  [[ "$(git -C "$REPO_ROOT" branch --show-current)" == "$FEATURE_BRANCH" ]] || die "installation requires branch $FEATURE_BRANCH"
  status="$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)"
  [[ -z "$status" ]] || die "installation requires a clean tracked feature commit"
  for path in \
    extra/usbgpu/protocol/tinygpu-wire-v1.md \
    extra/usbgpu/tests/qualify.py \
    extra/usbgpu/tbgpu/installer/Shared/server.c \
    extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriver.cpp \
    extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension/TinyGPUDriverUserClient.cpp; do
    git -C "$REPO_ROOT" ls-files --error-unmatch "$path" >/dev/null || die "required source is untracked: $path"
  done
  source_commit="$(git -C "$REPO_ROOT" rev-parse HEAD)"
}

validate_gpu_lock() {
  local fd="${TINYGRAD_GPU_LOCK_FD:-}" path="${TINYGRAD_GPU_LOCK_PATH:-}" nonce="${TINYGRAD_GPU_LOCK_NONCE:-}"
  [[ "$fd" =~ ^[0-9]+$ && -n "$path" && -n "$nonce" ]] || die "installation requires inherited GPU lock metadata"
  [[ "$(cd "$(dirname "$path")" && pwd -P)/$(basename "$path")" == "$(cd /tmp && pwd -P)/gpu-bench.lock" ]] || die "installation requires /tmp/gpu-bench.lock"
  [[ -r "/dev/fd/$fd" && "$(stat -f '%i' "/dev/fd/$fd")" == "$(stat -f '%i' "$path")" ]] || die "GPU lock descriptor mismatch"
  [[ "$(plutil -extract schema raw -o - "$path")" == "tinygrad.gpu.lock.v1" ]] || die "GPU lock schema mismatch"
  [[ "$(plutil -extract nonce raw -o - "$path")" == "$nonce" ]] || die "GPU lock nonce mismatch"
  [[ "$(plutil -extract pid raw -o - "$path")" == "$PPID" ]] || die "GPU lock was not inherited from the lock runner"
}

validate_developer_mode() {
  [[ "$(systemextensionsctl developer 2>&1)" == *"Developer mode is on"* ]] || \
    die "DriverKit development mode is off; run 'systemextensionsctl developer on' in an administrator Terminal and retry"
}

rollback_replacement() {
  local rollback_failed=0
  if [[ "$replacement_moved" == 1 && -d "$INSTALL_APP" ]]; then
    if extension_active; then
      "$INSTALL_APP/Contents/MacOS/TinyGPU" uninstall || rollback_failed=1
      wait_extension_state inactive || rollback_failed=1
    fi
    mv "$INSTALL_APP" "$stage_dir/failed-$APP_NAME" || rollback_failed=1
  fi
  if [[ -n "$backup_app" && -d "$backup_app" && ! -e "$INSTALL_APP" ]]; then
    mv "$backup_app" "$INSTALL_APP" || rollback_failed=1
    verify_bundle "$INSTALL_APP" || rollback_failed=1
    if [[ "$previous_extension_active" == 1 ]]; then
      "$INSTALL_APP/Contents/MacOS/TinyGPU" install || rollback_failed=1
      wait_extension_state active || rollback_failed=1
    fi
  fi
  replacement_moved=0
  return "$rollback_failed"
}

finish() {
  local rc="$?" rollback_rc=0
  trap - EXIT
  if [[ "$rc" != 0 && "$mode" == install ]]; then
    rollback_replacement || rollback_rc=$?
    record_provenance rollback || true
    if [[ "$rollback_rc" != 0 ]]; then echo "ERROR: rollback verification failed; inspect $stage_dir and provenance" >&2; rc=1; fi
  fi
  if [[ "$mode" == install ]]; then publish_provenance || rc=1; fi
  exit "$rc"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build)
      [[ "$build_seen" == 0 && "$install_seen" == 0 ]] || die "--build cannot be repeated or combined with --install"
      build_seen=1
      ;;
    --install)
      [[ "$build_seen" == 0 && "$install_seen" == 0 ]] || die "--install cannot be repeated or combined with --build"
      [[ $# -ge 2 ]] || die "--install requires the literal approval token"
      [[ "$2" == "$APPROVAL_TOKEN" ]] || die "approval token rejected; no build or installation was attempted"
      mode="install"; install_seen=1; shift
      ;;
    --provenance-out)
      [[ $# -ge 2 && -z "$provenance_out" ]] || die "--provenance-out requires one path"
      provenance_out="$2"; shift
      ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
  shift
done

if [[ "$mode" == build ]]; then
  [[ -z "$provenance_out" ]] || die "--provenance-out is valid only with --install"
else
  [[ -n "$provenance_out" ]] || die "--install requires --provenance-out PATH"
  [[ -t 0 ]] || die "installation requires an interactive terminal for immediate approval"
  [[ "$provenance_out" = /* ]] || provenance_out="$PWD/$provenance_out"
  [[ "$provenance_out" == "$EXPECTED_PROVENANCE" ]] || die "installation provenance must use $EXPECTED_PROVENANCE"
  validate_gpu_lock
  validate_feature_source
  [[ "$(csrutil status 2>&1)" == *"disabled"* ]] || die "SIP must be disabled for this audited development install"
  validate_developer_mode
  mkdir -p "$(dirname "$provenance_out")"
  provenance_tmp="$(mktemp "/tmp/.${APP_NAME}.provenance.XXXXXX")"
  trap finish EXIT
fi

cd "$SCRIPT_DIR"
xcodebuild -project TinyGPUDriverExtension.xcodeproj -scheme TinyGPU -configuration Debug -derivedDataPath "$DERIVED_DATA" clean build \
  CODE_SIGN_IDENTITY="" CODE_SIGNING_REQUIRED=NO CODE_SIGNING_ALLOWED=NO

[[ -d "$BUILD_APP" && -f "$BUILD_DEXT/$DEXT_ID" ]] || die "Debug build did not produce the expected app and dext"
codesign --sign - --entitlements ./TinyGPUDriverExtension/TinyGPUDriver.NoSIP.entitlements --force --timestamp=none --verbose "$BUILD_DEXT"
codesign --sign - --entitlements ./macOS/macOS.entitlements --force --timestamp=none --verbose "$BUILD_APP"
verify_bundle "$BUILD_APP" || die "signed build identity verification failed"

if [[ "$mode" == build ]]; then
  echo "Build complete (not installed): $BUILD_APP"
  exit 0
fi

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$source_commit" ]] || die "source commit changed during build"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=all)" ]] || die "source worktree changed during build"
record_provenance before
stage_dir="$(mktemp -d "$APPLICATIONS_DIR/.TinyGPU.stage.$RUN_ID.XXXXXX")"
[[ "$(stat -f %d "$stage_dir")" == "$(stat -f %d "$APPLICATIONS_DIR")" ]] || die "staging directory is not on the /Applications volume"
ditto "$BUILD_APP" "$stage_dir/$APP_NAME"
verify_bundle "$stage_dir/$APP_NAME" || die "staged build verification failed"
if extension_active; then previous_extension_active=1; fi

record_provenance ready_for_approval
printf 'Type %s to replace %s: ' "$APPROVAL_TOKEN" "$INSTALL_APP" >&2
IFS= read -r immediate_approval
[[ "$immediate_approval" == "$APPROVAL_TOKEN" ]] || die "immediate approval rejected; installed app was not changed"
printf '\n=== operator_approval ===\nrecorded_utc=%s\ntoken_matched=true\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$provenance_tmp"

if [[ -e "$INSTALL_APP" ]]; then
  backup_app="$APPLICATIONS_DIR/.TinyGPU.previous.$RUN_ID.app"
  mv "$INSTALL_APP" "$backup_app"
fi
mv "$stage_dir/$APP_NAME" "$INSTALL_APP"
replacement_moved=1
verify_bundle "$INSTALL_APP" || die "installed bundle verification failed"
record_provenance replaced

"$INSTALL_APP/Contents/MacOS/TinyGPU" install
wait_extension_state active || die "system extension did not reach [activated enabled]"
verify_bundle "$INSTALL_APP" || die "post-activation bundle verification failed"
record_provenance activated
publish_provenance
trap - EXIT
echo "Installed audited development build: $INSTALL_APP"
[[ -n "$backup_app" ]] && echo "Previous app retained for rollback: $backup_app"
