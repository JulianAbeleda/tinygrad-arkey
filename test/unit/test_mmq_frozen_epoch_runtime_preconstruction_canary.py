"""CPU-only contracts for frozen-v2 family runtime preconstruction."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from tinygrad.runtime.process_isolated import IsolatedResult

from extra.qk import mmq_frozen_epoch_runtime_preconstruction_canary as canary


def _binding(epochs: int = 20):
  programs = tuple(SimpleNamespace(arg=SimpleNamespace(function_name="mmq")) for _ in range(epochs))
  binaries = tuple(f"binary-{epoch}".encode() for epoch in range(epochs))
  keys = tuple(f"{epoch:064x}" for epoch in range(epochs))
  return SimpleNamespace(
    artifact=SimpleNamespace(programs=programs, binaries=binaries),
    program_keys=keys, family_identity="family")


def _child_pass(binding, role_spec, prefix_epochs: int) -> dict:
  keys = tuple(binding.program_keys[:prefix_epochs])
  identities = canary._target_identities(binding, prefix_epochs)
  return {
    "schema": canary.CHILD_SCHEMA, "status": "PASS", "passed": True,
    "role": role_spec.role, "shape": list(role_spec.shape),
    "family_identity": binding.family_identity,
    "program_keys": list(keys),
    "target_program_identities": [dict(row) for row in identities],
    "prefix_epochs": prefix_epochs, "complete_family": prefix_epochs == role_spec.epochs,
    "compile_performed": False, "requires_recompile": False,
    "hip_used": False, "no_fallback": True,
    "no_target_dispatch": True,
    "target_dispatch_count": 0, "target_runtime_called": False,
    "target_tensor_call_constructed": False,
    "tiny_health_passed": True,
    "runtime_preconstruction": {
      "enabled": True, "status": "PASS", "device": "AMD", "count": prefix_epochs,
      "ordered_program_keys": list(keys),
      "no_compute_dispatch_during_preconstruction": True,
      "runtime_cache_retains_code_allocations": True,
      "all_checks_pass": True,
      "runtimes": [{
        "epoch": epoch, "program_key": key,
        "program_identity": identities[epoch],
        "expected_program_identity": identities[epoch],
        "all_checks_pass": True,
      } for epoch, key in enumerate(keys)],
    },
  }


def _runner_pass(binding, role_spec):
  def runner(callback, *, args=(), timeout_seconds=0, start_method=None, **kwargs):
    assert callback is canary._run_frozen_epoch_runtime_preconstruction_worker
    assert Path(args[0]).name == "bundle"
    assert start_method == "spawn" and timeout_seconds > 0
    return IsolatedResult("passed", _child_pass(binding, role_spec, args[3]))
  return runner


def test_parent_pass_is_diagnostic_and_target_free(monkeypatch, tmp_path):
  monkeypatch.setattr(canary, "admit_exact_role_spec", lambda role_spec: role_spec)
  binding = _binding()
  monkeypatch.setattr(canary, "_load_binding", lambda role_spec, bundle: binding)
  health_calls: list[int] = []
  result = canary.run_frozen_epoch_runtime_preconstruction_canary(
    tmp_path / "bundle", prefix_epochs=None, timeout_seconds=1,
    runner=_runner_pass(binding, canary.DEFAULT_EXACT_ROLE_SPEC), fault_reader=lambda _: "",
    health_probe=lambda: health_calls.append(1) or True)
  assert result["status"] == "PASS" and result["passed"] is True
  assert result["prefix_epochs"] == 20 and result["complete_family"] is True
  assert result["research_only"] is True and result["diagnostic_only"] is True
  assert result["promotion_eligible"] is False
  assert result["target_tensor_call_constructed"] is False
  assert result["target_runtime_called"] is False and result["target_dispatch_count"] == 0
  assert result["no_target_dispatch"] is True and result["hip_used"] is False
  assert result["compile_performed"] is False and result["no_fallback"] is True
  assert health_calls == [1, 1]


def test_parent_rejects_non_admitted_prefix_before_health(monkeypatch, tmp_path):
  monkeypatch.setattr(canary, "admit_exact_role_spec", lambda role_spec: role_spec)
  calls: list[int] = []
  result = canary.run_frozen_epoch_runtime_preconstruction_canary(
    tmp_path / "bundle", prefix_epochs=4,
    runner=lambda *args, **kwargs: calls.append(1),
    health_probe=lambda: calls.append(2) or True)
  assert result["status"] == "BLOCKED" and "must be one of" in result["exact_blocker"]
  assert calls == []


def test_parent_rejects_minimal_child_pass_record(monkeypatch, tmp_path):
  monkeypatch.setattr(canary, "admit_exact_role_spec", lambda role_spec: role_spec)
  monkeypatch.setattr(canary, "_load_binding", lambda role_spec, bundle: _binding())
  minimal = {
    "schema": canary.CHILD_SCHEMA, "status": "PASS", "passed": True,
    "prefix_epochs": 1, "no_target_dispatch": True,
    "target_dispatch_count": 0, "target_runtime_called": False,
    "target_tensor_call_constructed": False,
  }
  result = canary.run_frozen_epoch_runtime_preconstruction_canary(
    tmp_path / "bundle", prefix_epochs=1,
    runner=lambda *args, **kwargs: IsolatedResult("passed", minimal),
    fault_reader=lambda _: "", health_probe=lambda: True)
  assert result["status"] == "BLOCKED"
  assert "did not close its exact no-target contract" in result["exact_blocker"]


def test_parent_fault_and_timeout_fail_closed_but_run_postflight(monkeypatch, tmp_path):
  monkeypatch.setattr(canary, "admit_exact_role_spec", lambda role_spec: role_spec)
  binding = _binding()
  monkeypatch.setattr(canary, "_load_binding", lambda role_spec, bundle: binding)
  health_calls: list[int] = []
  fault = canary.run_frozen_epoch_runtime_preconstruction_canary(
    tmp_path / "bundle", prefix_epochs=1,
    runner=_runner_pass(binding, canary.DEFAULT_EXACT_ROLE_SPEC),
    fault_reader=lambda _: "amdgpu: GPU reset begin",
    health_probe=lambda: health_calls.append(1) or True)
  assert fault["status"] == "BLOCKED" and fault["kernel_faults"]
  assert health_calls == [1, 1]

  health_calls.clear()
  timeout = canary.run_frozen_epoch_runtime_preconstruction_canary(
    tmp_path / "bundle", prefix_epochs=1,
    runner=lambda *args, **kwargs: IsolatedResult("timed_out", error="deadline", timed_out=True),
    fault_reader=lambda _: "", health_probe=lambda: health_calls.append(1) or True)
  assert timeout["status"] == "BLOCKED" and "deadline" in timeout["exact_blocker"]
  assert health_calls == [1, 1]


def test_runner_exception_still_scans_faults_and_runs_postflight(monkeypatch, tmp_path):
  monkeypatch.setattr(canary, "admit_exact_role_spec", lambda role_spec: role_spec)
  monkeypatch.setattr(canary, "_load_binding", lambda role_spec, bundle: _binding())
  calls: list[str] = []
  def raise_runner(*args, **kwargs):
    calls.append("runner")
    raise RuntimeError("spawn broke")
  result = canary.run_frozen_epoch_runtime_preconstruction_canary(
    tmp_path / "bundle", prefix_epochs=1, runner=raise_runner,
    fault_reader=lambda _: calls.append("fault_scan") or "",
    health_probe=lambda: calls.append("health") or True)
  assert result["status"] == "BLOCKED" and "spawn broke" in result["exact_blocker"]
  assert calls == ["health", "runner", "fault_scan", "health"]


def test_public_device_is_restricted_to_amd_before_health(monkeypatch, tmp_path):
  monkeypatch.setattr(canary, "admit_exact_role_spec", lambda role_spec: role_spec)
  calls: list[int] = []
  result = canary.run_frozen_epoch_runtime_preconstruction_canary(
    tmp_path / "bundle", prefix_epochs=1, device="CPU",
    health_probe=lambda: calls.append(1) or True)
  assert result["status"] == "BLOCKED" and "only admits device='AMD'" in result["exact_blocker"]
  assert calls == []


def test_worker_uses_family_helper_without_target_call(monkeypatch):
  programs = tuple(SimpleNamespace(arg=SimpleNamespace(function_name="mmq")) for _ in range(3))
  binaries = (b"zero", b"one", b"two")
  keys = tuple(f"{index:064x}" for index in range(3))
  binding = SimpleNamespace(
    artifact=SimpleNamespace(programs=programs, binaries=binaries),
    program_keys=keys, family_identity="family")
  monkeypatch.setattr(
    "extra.qk.mmq_frozen_epoch_program_set.load_frozen_epoch_program_set_binding",
    lambda role_spec, bundle: binding)
  captured = {}
  def preconstruct(selected_programs, selected_keys, identities, *, device):
    captured.update({
      "programs": selected_programs, "keys": selected_keys,
      "identities": identities, "device": device})
    return {
      "status": "PASS", "count": 3, "ordered_program_keys": list(keys),
      "no_compute_dispatch_during_preconstruction": True, "all_checks_pass": True,
    }
  monkeypatch.setattr(
    "extra.qk.mmq_llama_five_buffer_gpu_harness._preconstruct_frozen_program_runtimes",
    preconstruct)
  monkeypatch.setattr(canary, "exact_role_spec", lambda role, shape: SimpleNamespace(
    role=role, shape=shape, epochs=3))
  monkeypatch.setattr(canary, "_tiny_health", lambda device: True)

  result = canary._run_frozen_epoch_runtime_preconstruction_worker(
    "/tmp/bundle", "attn_kv", (512, 1024, 5120), 3)
  assert result["status"] == "PASS" and result["target_dispatch_count"] == 0
  assert captured["programs"] == programs and captured["keys"] == keys
  assert captured["device"] == "AMD"
  assert [row["binary_sha256"] for row in captured["identities"]] == [
    hashlib.sha256(binary).hexdigest() for binary in binaries]


def test_worker_retains_partial_preconstruction_error(monkeypatch):
  program = SimpleNamespace(arg=SimpleNamespace(function_name="mmq"))
  key = "0" * 64
  binding = SimpleNamespace(
    artifact=SimpleNamespace(programs=(program,), binaries=(b"binary",)),
    program_keys=(key,), family_identity="family")
  monkeypatch.setattr(
    "extra.qk.mmq_frozen_epoch_program_set.load_frozen_epoch_program_set_binding",
    lambda role_spec, bundle: binding)
  class Failed(RuntimeError):
    def __init__(self):
      super().__init__("upload failed")
      self.runtime_preconstruction = {"status": "PRECONSTRUCTION_ERROR", "count": 0}
  monkeypatch.setattr(
    "extra.qk.mmq_llama_five_buffer_gpu_harness._preconstruct_frozen_program_runtimes",
    lambda *args, **kwargs: (_ for _ in ()).throw(Failed()))
  monkeypatch.setattr(canary, "exact_role_spec", lambda role, shape: SimpleNamespace(
    role=role, shape=shape, epochs=1))

  result = canary._run_frozen_epoch_runtime_preconstruction_worker(
    "/tmp/bundle", "attn_kv", (512, 1024, 5120), 1)
  assert result["status"] == "BLOCKED" and result["tiny_health_passed"] is None
  assert result["runtime_preconstruction"]["status"] == "PRECONSTRUCTION_ERROR"
