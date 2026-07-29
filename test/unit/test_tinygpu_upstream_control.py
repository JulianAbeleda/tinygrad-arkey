import importlib.util, pathlib, subprocess, sys


ROOT = pathlib.Path(__file__).parents[2]
SCRIPT = ROOT / "extra/usbgpu/tools/run_upstream_control.py"
SPEC = importlib.util.spec_from_file_location("run_upstream_control", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CONTROL = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPT.parent))
try: SPEC.loader.exec_module(CONTROL)
finally: sys.path.pop(0)


def test_control_runner_has_valid_syntax_and_non_operational_help():
  assert subprocess.run([sys.executable, "-m", "py_compile", str(SCRIPT)], check=False).returncode == 0
  result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, check=False)
  assert result.returncode == 0 and "--check" in result.stdout and "a command is required" in result.stdout


def test_control_requires_upstream_enabled_and_arkey_not_enabled():
  upstream_enabled = "* * 9YG3G8543N org.tinygrad.tinygpu.driver2 (1.0.0/3) name [activated enabled]"
  upstream_disabled = "  * 9YG3G8543N org.tinygrad.tinygpu.driver2 (1.0.0/3) name [activated disabled]"
  arkey_enabled = "* * - org.tinygrad.arkey.tinygpu.driver2 (1.0.0/13) name [activated enabled]"
  assert CONTROL.registration_ready(upstream_enabled)
  assert not CONTROL.registration_ready(upstream_disabled)
  assert not CONTROL.registration_ready(upstream_enabled + "\n" + arkey_enabled)


def test_control_pins_matched_upstream_runtime_and_release_hashes():
  assert CONTROL.UPSTREAM_HEAD == "6ea7d366fa92842c0bc8b7b080e26e83a7406252"
  assert CONTROL.UPSTREAM_RELEASE == "c0d024f9ff0e1dc8fdf217f255da7101d91e8323"
  assert len(CONTROL.UPSTREAM_ZIP_SHA256) == len(CONTROL.UPSTREAM_APP_SHA256) == len(CONTROL.UPSTREAM_DEXT_SHA256) == 64
  source = SCRIPT.read_text()
  assert "never installs, activates, deactivates, or replaces" in source
  assert "standard app remains arkey" in source
  assert 'os.environ.get("TINYGRAD_GPU_LOCK_FD")' in source
