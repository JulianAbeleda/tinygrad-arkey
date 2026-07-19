from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinygrad.runtime.process_isolated import IsolatedResult

from extra.qk.mmq_frozen_staged_family_execution import (
  ISOLATED_C4_SCHEMA, RUNTIME_CANARY_SCHEMA, SCHEMA, main,
  run_frozen_staged_family_prefix_from_canary_file,
  run_frozen_staged_runtime_canary_isolated,
  validate_frozen_staged_runtime_canary,
  validate_frozen_staged_runtime_canary_isolation,
)
from test.unit.test_mmq_frozen_staged_family_execution import (
  _canary, _family, _probe_result,
)


def _isolation(family, base=None, queue_mode="PM4"):
  base = _canary(family, queue_mode) if base is None else base
  return {
    "schema": ISOLATED_C4_SCHEMA, "status": "PASS", "exact_blocker": None,
    "containment_authority": "outer_parent_fresh_process_guards",
    "role": family.binding.role_spec.role,
    "shape": list(family.binding.role_spec.shape),
    "queue_mode": queue_mode,
    "family_identity": family.family_identity,
    "program_key": family.binding.program_key,
    "binary_sha256": family.binding.binary_sha256,
    "launched": True, "child_status": "passed",
    "timed_out": False, "error": None, "elapsed_seconds": 0.1,
    "stdout_tail": "", "stderr_tail": "", "timeout_seconds": 7,
    "target_dispatch_count": 0,
    "health_before": True, "health_after": True,
    "kernel_faults": [], "kernel_fault_evidence": {},
    "runtime_canary": base,
    "compile_performed": False, "requires_recompile": False,
  }


def _isolated_c4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, child: dict,
    *, health=(True, True), faults=(),
    isolated: IsolatedResult | None = None,
    ):
  tmp_path = tmp_path / "isolated-c4"
  tmp_path.mkdir()
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  health_values = iter(health)
  calls = []

  def runner(callback, **kwargs):
    calls.append((callback, kwargs))
    return isolated or IsolatedResult("passed", result=child)

  result = run_frozen_staged_runtime_canary_isolated(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    timeout_seconds=7, family_loader=loader, isolated_runner=runner,
    health_probe=lambda overrides: next(health_values),
    fault_collector=lambda started: (list(faults), {"since": started}))
  return result, calls, family


def test_c4_cli_boundary_isolated_success_is_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  callbacks = []

  def runner(callback, **kwargs):
    callbacks.append((callback, kwargs))
    return IsolatedResult("passed", result=_canary(family))

  health = iter((True, True))
  result = run_frozen_staged_runtime_canary_isolated(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    timeout_seconds=7, family_loader=loader, isolated_runner=runner,
    health_probe=lambda overrides: next(health),
    fault_collector=lambda started: ([], {"since": started}))
  assert result["schema"] == ISOLATED_C4_SCHEMA
  assert result["status"] == "PASS" and result["all_checks_pass"] is True
  assert result["target_dispatch_count"] == 0
  assert result["child_status"] == "passed"
  assert result["runtime_canary"]["schema"] == RUNTIME_CANARY_SCHEMA
  assert result["runtime_canary"]["all_checks_pass"] is True
  assert len(callbacks) == 1
  assert callbacks[0][1]["start_method"] == "spawn"
  assert callbacks[0][1]["timeout_seconds"] == 7.0


def test_c4_cli_boundary_timeout_is_fail_closed_and_health_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  health_calls = []
  result = run_frozen_staged_runtime_canary_isolated(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    timeout_seconds=1, family_loader=loader,
    isolated_runner=lambda *args, **kwargs: IsolatedResult(
      "timed_out", error="deadline", timed_out=True),
    health_probe=lambda overrides: health_calls.append(dict(overrides)) or True,
    fault_collector=lambda started: ([], {}))
  assert result["status"] == "BLOCKED"
  assert result["timed_out"] is True and "timed out" in result["exact_blocker"]
  assert result["target_dispatch_count"] == 0
  assert len(health_calls) == 2


def test_c4_cli_boundary_runner_error_still_collects_faults_and_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  role_spec, _, manifest_path, _, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  health_calls = []

  def raises(*args, **kwargs):
    raise RuntimeError("spawn refused")

  result = run_frozen_staged_runtime_canary_isolated(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    timeout_seconds=1, family_loader=loader, isolated_runner=raises,
    health_probe=lambda overrides: health_calls.append(dict(overrides)) or True,
    fault_collector=lambda started: ([], {"since": started}))
  assert result["status"] == "BLOCKED"
  assert result["child_status"] == "runner_error"
  assert "spawn refused" in result["exact_blocker"]
  assert len(health_calls) == 2


def test_c4_cli_boundary_fault_marker_overrides_passing_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  _, _, _, family, _ = _family(tmp_path)
  result, calls, _ = _isolated_c4(
    tmp_path, monkeypatch, _canary(family), faults=("amdgpu: GPU reset",))
  assert len(calls) == 1
  assert result["status"] == "BLOCKED"
  assert result["kernel_faults"] == ["amdgpu: GPU reset"]
  assert "fault/reset" in result["exact_blocker"]


def test_c4_parent_fault_window_covers_both_health_probes_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  events = []
  monkeypatch.setattr(
    "extra.qk.mmq_frozen_staged_family_execution.time.time",
    lambda: events.append("started") or 123.0)

  def health(overrides):
    events.append("health")
    return True

  def runner(*args, **kwargs):
    events.append("child")
    return IsolatedResult("passed", result=_canary(family))

  def faults(started):
    events.append("faults")
    assert started == 123.0
    return [], {"since": started}

  result = run_frozen_staged_runtime_canary_isolated(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    family_loader=loader, isolated_runner=runner,
    health_probe=health, fault_collector=faults)
  assert result["status"] == "PASS"
  assert events == ["started", "health", "child", "health", "faults"]


@pytest.mark.parametrize("health", [(False, True), (True, False)])
def test_c4_cli_boundary_unhealthy_gpu_never_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, health):
  _, _, _, family, _ = _family(tmp_path)
  result, calls, _ = _isolated_c4(
    tmp_path, monkeypatch, _canary(family), health=health)
  assert result["status"] == "BLOCKED"
  if health == (False, True):
    assert calls == []
    assert result["launched"] is False
    assert result["health_after"] is True
  else:
    assert len(calls) == 1
    assert result["health_after"] is False


def test_c4_cli_boundary_rejects_parent_queue_environment_before_health_or_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  role_spec, _, manifest_path, _, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "1")
  result = run_frozen_staged_runtime_canary_isolated(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, queue_mode="PM4",
    family_loader=loader,
    isolated_runner=lambda *args, **kwargs: pytest.fail("child must not start"),
    health_probe=lambda overrides: pytest.fail("health must not run"),
    fault_collector=lambda started: pytest.fail("fault scan must not run"))
  assert result["status"] == "BLOCKED"
  assert result["launched"] is False
  assert "AMD_AQL must be '0'" in result["error"]


@pytest.mark.parametrize("mutation", ["malformed", "identity", "queue", "target_dispatch"])
def test_c4_cli_boundary_rejects_malformed_or_drifted_child_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str):
  _, _, _, family, _ = _family(tmp_path)
  child = _canary(family)
  if mutation == "malformed":
    child = {"schema": RUNTIME_CANARY_SCHEMA, "status": "PASS",
             "health_before": True, "health_after": True, "kernel_faults": []}
  elif mutation == "identity":
    child["family_identity"] = "sha256:" + "0" * 64
  elif mutation == "queue":
    child["queue_mode"] = "AQL"
    child["amd_aql_effective"] = True
  else:
    child["target_dispatch_count"] = 1
  result, calls, _ = _isolated_c4(tmp_path, monkeypatch, child)
  assert len(calls) == 1
  assert result["status"] == "BLOCKED"
  if mutation == "target_dispatch":
    assert "invalid target dispatch count" in result["exact_blocker"]
  else:
    assert "evidence failed closed" in result["exact_blocker"]


@pytest.mark.parametrize(("field", "value"), [
  ("containment_authority", "child_claim"),
  ("launched", False),
  ("child_status", "failed"),
  ("timed_out", True),
  ("error", "hidden"),
  ("role", "ffn_down"),
  ("timeout_seconds", True),
  ("elapsed_seconds", False),
  ("target_dispatch_count", False),
  ("health_before", False),
  ("health_after", False),
  ("kernel_faults", ["amdgpu reset"]),
  ("kernel_fault_evidence", []),
])
def test_isolated_c4_validator_rejects_envelope_adversaries(
    tmp_path: Path, field: str, value):
  _, _, _, family, _ = _family(tmp_path)
  evidence = _isolation(family)
  evidence[field] = value
  with pytest.raises(ValueError, match="isolated staged runtime canary failed checks"):
    validate_frozen_staged_runtime_canary_isolation(
      evidence, family, queue_mode="PM4")


@pytest.mark.parametrize(("field", "value"), [
  ("runtime_count", True),
  ("target_dispatch_count", False),
])
def test_base_c4_validator_rejects_bool_counters(
    tmp_path: Path, field: str, value):
  _, _, _, family, _ = _family(tmp_path)
  evidence = _canary(family)
  evidence[field] = value
  with pytest.raises(ValueError, match=field):
    validate_frozen_staged_runtime_canary(
      evidence, family, queue_mode="PM4")


@pytest.mark.parametrize("nested_mutation", ["missing", "legacy_bool_count", "identity"])
def test_prefix_cli_never_calls_target_for_invalid_nested_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nested_mutation: str):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  evidence = _isolation(family)
  if nested_mutation == "missing":
    evidence["runtime_canary"] = None
  elif nested_mutation == "legacy_bool_count":
    evidence["runtime_canary"]["target_dispatch_count"] = False
  else:
    evidence["runtime_canary"]["program_key"] = "0" * 64
  canary_path = tmp_path / "c4-nested.json"
  canary_path.write_text(json.dumps(evidence))
  called = False

  def target(**kwargs):
    nonlocal called
    called = True
    return {}

  result = run_frozen_staged_family_prefix_from_canary_file(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, runtime_canary_path=canary_path,
    prefix_epochs=1, queue_mode="PM4", family_loader=loader,
    probe_runner=target)
  assert result["status"] == "BLOCKED"
  assert result["target_dispatch_attempted"] is False
  assert called is False


@pytest.mark.parametrize("mutation", ["malformed_json", "legacy_base", "identity", "queue"])
def test_prefix_cli_never_calls_target_for_invalid_persisted_c4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  canary_path = tmp_path / "c4.json"
  if mutation == "malformed_json":
    canary_path.write_text("{")
  else:
    canary = _canary(family) if mutation == "legacy_base" else _isolation(family)
    if mutation == "identity":
      canary["binary_sha256"] = "0" * 64
    elif mutation == "queue":
      canary["queue_mode"] = "AQL"
    canary_path.write_text(json.dumps(canary))
  called = False

  def target(**kwargs):
    nonlocal called
    called = True
    return {}

  result = run_frozen_staged_family_prefix_from_canary_file(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, runtime_canary_path=canary_path,
    prefix_epochs=1, queue_mode="PM4", family_loader=loader,
    probe_runner=target)
  assert result["schema"] == SCHEMA and result["status"] == "BLOCKED"
  assert result["target_dispatch_attempted"] is False
  assert called is False


def test_prefix_cli_calls_approved_wrapper_after_exact_c4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  canary_path = tmp_path / "c4.json"
  canary_path.write_text(json.dumps(_isolation(family)))
  calls = []

  def target(**kwargs):
    calls.append(kwargs)
    return _probe_result(
      family, kwargs["frozen_bundle"], kwargs["epoch_limit"], "PM4")

  result = run_frozen_staged_family_prefix_from_canary_file(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, runtime_canary_path=canary_path,
    prefix_epochs=1, queue_mode="PM4", timeout_seconds=9,
    family_loader=loader, probe_runner=target)
  assert result["status"] == "PASS" and result["gate"] == "C5"
  assert result["c4_runtime_canary_isolation"]["all_checks_pass"] is True
  assert len(calls) == 1
  assert calls[0]["timeout_seconds"] == 9
  assert calls[0]["child_env_overrides"] == {"AMD_AQL": "0"}


@pytest.mark.parametrize("mutation", [
  "preparation_count", "phase_epoch", "launch_count", "launch_epoch",
  "global_size", "argument_slot", "mismatch_count",
])
def test_prefix_evidence_rejects_bool_integer_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str):
  role_spec, _, manifest_path, family, loader = _family(tmp_path)
  monkeypatch.setenv("AMD_AQL", "0")
  canary_path = tmp_path / "c4.json"
  canary_path.write_text(json.dumps(_isolation(family)))

  def target(**kwargs):
    result = _probe_result(
      family, kwargs["frozen_bundle"], kwargs["epoch_limit"], "PM4")
    if mutation == "preparation_count":
      result["phase_isolation"]["preparation"]["target_dispatch_count"] = False
    elif mutation == "phase_epoch":
      result["phase_isolation"]["epochs"][0]["epoch"] = False
    elif mutation == "launch_count":
      result["runtime_evidence"]["launch_count"] = True
    elif mutation == "launch_epoch":
      result["runtime_evidence"]["launches"][0]["epoch"] = False
    elif mutation == "global_size":
      result["runtime_evidence"]["launches"][0]["global_size"][-1] = True
    elif mutation == "argument_slot":
      result["runtime_evidence"]["launches"][0]["arguments"][0]["slot"] = False
    else:
      result["correctness"]["comparison"]["mismatch_count"] = False
    return result

  result = run_frozen_staged_family_prefix_from_canary_file(
    role_spec=role_spec, frozen_bundle="/frozen/bundle",
    staged_family_manifest=manifest_path, runtime_canary_path=canary_path,
    prefix_epochs=1, queue_mode="PM4", family_loader=loader,
    probe_runner=target)
  assert result["status"] == "BLOCKED"
  assert result["exact_blocker"] == "guarded staged prefix evidence failed closed"


def test_cli_main_writes_one_atomic_structured_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
  role_spec, _, _, family, _ = _family(tmp_path)
  output = tmp_path / "evidence" / "c4.json"
  expected = _isolation(family)
  monkeypatch.setattr(
    "extra.qk.mmq_frozen_staged_family_execution.exact_role_spec",
    lambda *args, **kwargs: role_spec)
  monkeypatch.setattr(
    "extra.qk.mmq_frozen_staged_family_execution."
    "run_frozen_staged_runtime_canary_isolated",
    lambda **kwargs: expected)
  rc = main([
    "c4", "--role", "attn_qo", "--frozen-bundle", "/frozen/bundle",
    "--staged-family-manifest", "/frozen/family.json",
    "--queue-mode", "PM4", "--timeout-seconds", "7",
    "--output", str(output),
  ])
  assert rc == 0
  assert json.loads(output.read_text()) == expected
  assert not list(output.parent.glob(f".{output.name}.*.tmp"))
