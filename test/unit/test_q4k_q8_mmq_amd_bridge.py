import pytest
from pathlib import Path

from extra.qk.q4k_q8_mmq_amd_bridge import AMDExecutionConfig, ROOT


def test_config_is_reproducible_and_does_not_change_route_defaults():
  config = AMDExecutionConfig()
  env = config.environment({"PREFILL_ROUTE": "direct_packed", "ALLOW_DEVICE_USAGE": "0", "PYTHONPATH": "/child"})
  assert env["DEV"] == "AMD" and env["ROCM_PATH"] == "/opt/rocm"
  assert env["HIP_PATH"] == "/opt/rocm" and env["HIPCC"] == "/opt/rocm/bin/hipcc"
  assert env["PYTHONPATH"].split(":") == [str(ROOT), "/child"]
  assert env["PREFILL_ROUTE"] == "direct_packed"


def test_dispatch_requires_explicit_opt_in():
  with pytest.raises(ValueError, match="dispatch is opt-in"):
    AMDExecutionConfig().command(bootstrap=__file__)


def test_command_passes_file_backed_bootstrap(tmp_path: Path):
  bootstrap = tmp_path / "bootstrap.json"
  bootstrap.write_text("{}")
  command = AMDExecutionConfig().command(bootstrap=bootstrap, dispatch=True)
  assert command[command.index("--bootstrap") + 1] == str(bootstrap)


def test_command_rejects_missing_bootstrap(tmp_path: Path):
  with pytest.raises(FileNotFoundError):
    AMDExecutionConfig().command(bootstrap=tmp_path / "missing.json", dispatch=True)


def test_preflight_reports_missing_gpu_without_tinygrad_device(monkeypatch):
  class Result:
    stdout = "Agent 1\\nDevice Type: CPU\\n"
    stderr = ""
  result = AMDExecutionConfig().preflight(runner=lambda *args, **kwargs: Result())
  assert result["gpu_agent_visible"] is False
  assert result["blocker"] == "rocminfo exposes no AMD GPU agent"
