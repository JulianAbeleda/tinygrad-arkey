import subprocess

from extra.qk import clock_pin
from extra.qk import timing_harness


def test_clock_pin_commands_are_centralized_and_noninteractive():
  assert "power_dpm_force_performance_level" in clock_pin.PIN_PEAK_CMD
  assert "pp_dpm_sclk" in clock_pin.PIN_PEAK_CMD
  assert "pp_dpm_mclk" in clock_pin.PIN_PEAK_CMD
  assert clock_pin.RESET_PERF_DETERMINISM[:3] == ["sudo", "-n", "rocm-smi"]


def test_pinned_peak_restores_auto_when_enabled(monkeypatch):
  calls = []
  monkeypatch.setattr(clock_pin, "pin_peak", lambda: calls.append("pin") or {"ok": True})
  monkeypatch.setattr(clock_pin, "restore_auto", lambda: calls.append("restore") or [{"ok": True}])

  with clock_pin.pinned_peak(enabled=True) as prov:
    assert prov == {"ok": True}
    calls.append("body")

  assert calls == ["pin", "body", "restore"]


def test_pinned_peak_disabled_does_not_mutate(monkeypatch):
  monkeypatch.setattr(clock_pin, "pin_peak", lambda: (_ for _ in ()).throw(AssertionError("pin called")))
  monkeypatch.setattr(clock_pin, "restore_auto", lambda: (_ for _ in ()).throw(AssertionError("restore called")))

  with clock_pin.pinned_peak(enabled=False) as prov:
    assert prov is None


def test_perflevel_wraps_rocm_smi(monkeypatch):
  seen = {}

  def fake_run(cmd, **kwargs):
    seen["cmd"] = cmd
    seen["kwargs"] = kwargs
    return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

  monkeypatch.setattr(clock_pin.subprocess, "run", fake_run)
  ret = clock_pin.perflevel("high")
  assert ret.returncode == 0
  assert seen["cmd"] == ["rocm-smi", "--setperflevel", "high"]
  assert seen["kwargs"] == {"capture_output": True, "text": True}


def test_timing_harness_clock_pin_env_is_canonical():
  env = {"OTHER": "1"}
  assert timing_harness.env_wants_clock_pin(env) is False
  assert timing_harness.set_clock_pin_env(env, True) is env
  assert env[timing_harness.PIN_CLOCK_ENV] == "1"
  assert timing_harness.env_wants_clock_pin(env) is True
  timing_harness.set_clock_pin_env(env, False)
  assert timing_harness.PIN_CLOCK_ENV not in env
