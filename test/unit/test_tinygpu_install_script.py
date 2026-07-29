import json, pathlib
import subprocess

ROOT = pathlib.Path(__file__).parents[2]
SCRIPT = ROOT / "extra/usbgpu/tbgpu/installer/install_nosip.sh"
APP = ROOT / "extra/usbgpu/tbgpu/installer/Shared/TinyGPUApp.swift"
CLI = ROOT / "extra/usbgpu/tbgpu/installer/Shared/TinyGPUCLIRunner.swift"
CLASSIFIER = ROOT / "extra/usbgpu/tbgpu/installer/classify_system_extension_state.sh"
STATE_FIXTURES = json.loads((ROOT / "test/unit/fixtures/tinygpu_system_extension_states.json").read_text())
TOKEN = "APPROVE_TINYGPU_DEVELOPMENT_INSTALL"


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
  return subprocess.run(["bash", str(SCRIPT), *args], text=True, capture_output=True, check=False)


def test_install_helper_has_valid_bash_syntax_and_non_operational_help():
  assert subprocess.run(["bash", "-n", str(SCRIPT)], check=False).returncode == 0
  assert subprocess.run(["bash", "-n", str(CLASSIFIER)], check=False).returncode == 0
  result = run_script("--help")
  assert result.returncode == 0 and "--install " + TOKEN in result.stdout
  assert "clean linked exp worktree" in result.stdout


def test_install_rejects_bad_modes_and_approval_before_build():
  cases = [
    ("--install",),
    ("--install", "not-approved", "--provenance-out", "/tmp/unused"),
    ("--build", "--install", TOKEN, "--provenance-out", "/tmp/unused"),
    ("--build", "--provenance-out", "/tmp/unused"),
  ]
  for args in cases:
    result = run_script(*args)
    assert result.returncode == 2
    assert "xcodebuild" not in result.stdout + result.stderr


def test_install_requires_clean_linked_feature_source_and_inherited_lock():
  source = SCRIPT.read_text()
  for token in ("--absolute-git-dir", "--git-common-dir", 'FEATURE_BRANCH="exp"', "status --porcelain=v1 --untracked-files=all",
                "ls-files --error-unmatch", "TINYGRAD_GPU_LOCK_FD", "/tmp/gpu-bench.lock", "GPU lock nonce mismatch"):
    assert token in source
  assert "systemextensionsctl developer" in source and "DriverKit development mode is off" in source
  assert source.index("validate_gpu_lock") < source.index("xcodebuild -project")
  assert source.index("validate_feature_source") < source.index("xcodebuild -project")


def test_build_output_signing_and_provenance_are_explicit():
  source = SCRIPT.read_text()
  for token in ("-derivedDataPath \"$DERIVED_DATA\"", "--timestamp=none", "Signature=adhoc", "Identifier=$APP_ID", "Identifier=$DEXT_ID",
                "source_manifest", "source_nosip_entitlements_hash", "source_app_entitlements_hash", "record_tree", "systemextensionsctl list",
                    "tinygpu-development-install-provenance.txt", "DEXT_VERSION=\"8\"", "CFBundleVersion"):
    assert token in source
  assert "curl " not in source and "download" not in source.lower() and "rm -rf" not in source
  assert "provisionprofile" not in source.lower()


def test_immediate_interactive_approval_precedes_only_replacement_path():
  source = SCRIPT.read_text()
  prompt = source.index("Type %s to replace %s")
  approval = source.index("token_matched=true")
  replacement = source.index('mv "$stage_dir/$APP_NAME" "$INSTALL_APP"')
  assert "[[ -t 0 ]]" in source
  assert prompt < approval < replacement


def test_rollback_deactivates_new_extension_and_restores_prior_state():
  source = SCRIPT.read_text()
  rollback = source[source.index("rollback_replacement()") : source.index("finish()")]
  assert '[[ "$previous_extension_active" != 1 ]] && extension_any_active' in rollback
  assert '"$INSTALL_APP/Contents/MacOS/TinyGPU" uninstall' in rollback
  assert "wait_extension_state inactive" in rollback
  assert 'mv "$backup_app" "$INSTALL_APP"' in rollback
  assert '"$INSTALL_APP/Contents/MacOS/TinyGPU" install' in rollback
  assert "wait_extension_state any_active" in rollback


def test_registration_classifier_rejects_reproduced_upgrade_state():
  expected = {
    "active_current":"active_current", "active_other_version":"active_other_version", "pending_reboot":"pending_reboot",
    "needs_approval":"needs_approval", "legacy_conflict":"pending_reboot", "inactive":"inactive",
  }
  for name, output in STATE_FIXTURES.items():
    result = subprocess.run(["bash", str(CLASSIFIER), "org.tinygrad.arkey.tinygpu.driver2", "org.tinygrad.tinygpu.driver2", "8"],
                            input=output, text=True, capture_output=True, check=False)
    assert result.returncode == 0 and result.stdout.strip() == expected[name], (name, result.stdout, result.stderr)


def test_install_distinguishes_clean_activation_from_pending_reboot():
  source = SCRIPT.read_text()
  assert "inactive|active_other_version" in source and "same-version reinstall is forbidden" in source
  assert "pending_reboot)" in source and "restart before another install" in source
  assert 'printf \'%s\\n\' unavailable' in source and "registration is incomplete" in source
  pending = source[source.index('if ! wait_extension_state active'):source.index('verify_bundle "$INSTALL_APP"', source.index('if ! wait_extension_state active'))]
  assert '[[ "$postinstall_state" == pending_reboot ]]' in pending
  assert "record_provenance pending_reboot" in pending and "trap - EXIT" in pending and 'exit "$REBOOT_REQUIRED_EXIT"' in pending


def test_gui_cannot_move_or_activate_itself_outside_audited_installer():
  source = APP.read_text()
  assert "NSAppleScript" not in source
  assert 'run(args: ["", "install"])' not in source
  assert "audited tinygrad-arkey development installer" in source


def test_cli_reports_system_extension_error_codes_accurately():
  source = CLI.read_text()
  assert "if code == 2" in source and "Missing entitlements" in source
  assert "else if code == 4" in source and "not found" in source
  assert "else if code == 13" in source and "Authorization is required" in source


def test_cli_does_not_treat_any_active_registration_as_ready():
  source = CLI.read_text()
  assert "case unloaded, activating, needsApproval, rebootRequired, activated" in source
  assert "currentActive.count == 1 && lines.count == 1" in source
  assert "lines.count > 1 || transitional || legacyActive" in source
  assert "Extension upgrade is pending. Restart macOS" in source
  assert "Verify keepalive and power status before using the GPU" in source
  assert "registration request completed; verify the complete registration set" in source
