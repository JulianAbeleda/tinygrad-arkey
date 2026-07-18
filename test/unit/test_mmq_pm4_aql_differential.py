from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from extra.qk import mmq_pm4_aql_differential as differential
from extra.qk.mmq_frozen_target_artifact import ACCUMULATION


def _artifact():
  fixture = {"schema": "fixture.v1", "q4": {"sha256": "q4"}, "q8": {"sha256": "q8"}}
  manifest = {
    "schema": "frozen.v1", "state": "FROZEN", "compile_calls": 1,
    "accumulate": True, "accumulation": ACCUMULATION,
    "consumer": {"requires_recompile": False},
    "program": {"key": "program-key", "function": "target"},
    "files": {"fixture.json": {"sha256": "fixture-file-sha", "nbytes": 1}},
    "artifacts": {
      "source_sha256": "source-sha", "binary_sha256": "binary-sha",
      "serialized_program_sha256": "program-sha",
    },
  }
  return SimpleNamespace(manifest=manifest, fixture=fixture)


def _passing_result(artifact, kwargs):
  expected = differential._frozen_identity(artifact)
  prefix = kwargs["epoch_limit"]
  amd_aql = kwargs["child_env_overrides"]["AMD_AQL"]
  return {
    "status": "PASS", "accumulation": ACCUMULATION, "no_fallback": True,
    "health_before": True, "health_after": True,
    "health_mode": {"amd_aql_env": amd_aql, "before": True, "after": True},
    "kernel_faults": [],
    "correctness": {"status": "PASS", "comparison": {"status": "pass", "mismatch_count": 0}},
    "timing": {
      "persistent_buffers": True, "preloaded_epochs": True, "stable_metadata_staging": True,
      "k_epoch_launches": prefix, "epoch_checks": [],
    },
    "artifacts": {
      "source_sha256": expected["source_sha256"], "binary_sha256": expected["binary_sha256"],
      "accumulation": ACCUMULATION, "no_fallback": True,
      "frozen_bundle": {
        **expected, "requires_recompile": False, "compile_performed": False,
      },
    },
    "runtime_evidence": {
      "amd_aql_env": amd_aql, "amd_aql_effective": amd_aql == "1",
      "queue_mode": "AQL" if amd_aql == "1" else "PM4",
      "launch_count": prefix, "intermediate_readback": False, "external_accumulation_add": False,
    },
  }


def test_differential_loads_once_and_reuses_isolated_probe_with_only_aql_env_difference(tmp_path: Path):
  artifact, loads, calls = _artifact(), [], []
  environ = {"KEEP": "same", "AMD_AQL": "caller-value"}

  def loader(path):
    loads.append(path)
    return artifact

  def runner(**kwargs):
    calls.append((dict(kwargs), dict(environ)))
    return _passing_result(artifact, kwargs)

  result = differential.run_pm4_aql_frozen_differential(
    tmp_path / "bundle", runner=runner, loader=loader, environ=environ)

  assert result["status"] == "PASS" and result["classification"] == "NO_DIFFERENTIAL_FAILURE"
  assert result["bundle_validations"] == 1 and result["compile_performed"] is False
  assert len(loads) == 1 and len(calls) == 4
  assert [(call["child_env_overrides"]["AMD_AQL"], call["epoch_limit"]) for call, _ in calls] == [
    ("0", 1), ("1", 1), ("0", 3), ("1", 3)]
  assert all(snapshot == {"KEEP": "same", "AMD_AQL": "caller-value"} for _, snapshot in calls)
  assert all(call["frozen_bundle"] == str((tmp_path / "bundle").resolve()) for call, _ in calls)
  assert all(call["in_kernel_accumulate"] and call["persistent_buffers"] and call["preloaded_epochs"]
             and call["stable_metadata_staging"] for call, _ in calls)
  assert all(not call["host_accumulate"] and not call["per_epoch_check"] for call, _ in calls)
  assert environ == {"KEEP": "same", "AMD_AQL": "caller-value"}


def test_differential_stops_faulted_mode_but_still_runs_other_mode(tmp_path: Path):
  artifact, calls = _artifact(), []

  def runner(**kwargs):
    calls.append((kwargs["child_env_overrides"]["AMD_AQL"], kwargs["epoch_limit"]))
    if kwargs["child_env_overrides"]["AMD_AQL"] == "0":
      row = _passing_result(artifact, kwargs)
      row.update({"status": "BLOCKED", "kernel_faults": ["SQC instruction fault"],
                  "health_after": True})
      return row
    return _passing_result(artifact, kwargs)

  result = differential.run_pm4_aql_frozen_differential(
    tmp_path / "bundle", runner=runner, loader=lambda _: artifact, environ={"UNCHANGED": "1"})
  assert calls == [("0", 1), ("1", 1)]
  assert result["status"] == "BLOCKED" and result["classification"] == "PM4_ONLY_FAILURE"
  assert result["modes"][0]["stopped_after_failure"] is True
  assert result["modes"][1]["status"] == "INCOMPLETE"
  assert "larger prefixes were not submitted" in result["escalation_stop_reason"]


def test_differential_stops_all_dispatch_after_uncontained_health_failure(tmp_path: Path):
  artifact, calls = _artifact(), []

  def runner(**kwargs):
    calls.append((kwargs["child_env_overrides"]["AMD_AQL"], kwargs["epoch_limit"]))
    row = _passing_result(artifact, kwargs)
    row.update({"status": "BLOCKED", "kernel_faults": ["GPU reset"], "health_after": False})
    return row

  result = differential.run_pm4_aql_frozen_differential(
    tmp_path / "bundle", runner=runner, loader=lambda _: artifact, environ={})
  assert calls == [("0", 1)]
  assert result["classification"] == "INCONCLUSIVE"
  assert result["modes"][1]["status"] == "INCOMPLETE"
  assert "did not leave a healthy" in result["escalation_stop_reason"]


def test_differential_fails_closed_when_no_recompile_or_mode_health_evidence_is_missing(tmp_path: Path):
  artifact, calls = _artifact(), []

  def runner(**kwargs):
    calls.append((kwargs["child_env_overrides"]["AMD_AQL"], kwargs["epoch_limit"]))
    row = _passing_result(artifact, kwargs)
    row["artifacts"]["frozen_bundle"].pop("compile_performed")
    row.pop("health_mode")
    return row

  result = differential.run_pm4_aql_frozen_differential(
    tmp_path / "bundle", runner=runner, loader=lambda _: artifact, environ={})
  assert calls == [("0", 1), ("1", 1)]
  assert result["classification"] == "BOTH_MODES_FAILED_SHARED_OR_KERNEL_LAYER"
  for mode in result["modes"]:
    errors = mode["attempts"][0]["validation_errors"]
    assert any("zero-recompile" in error for error in errors)
    assert any("health canary" in error for error in errors)


def test_differential_rejects_invalid_bundle_before_runner(tmp_path: Path):
  calls = []
  result = differential.run_pm4_aql_frozen_differential(
    tmp_path / "bad", runner=lambda **kwargs: calls.append(kwargs),
    loader=lambda _: (_ for _ in ()).throw(ValueError("hash mismatch")), environ={})
  assert result["status"] == "BLOCKED" and result["classification"] == "INCONCLUSIVE"
  assert "hash mismatch" in result["exact_blocker"]
  assert calls == []


def test_differential_numeric_failure_is_fail_closed_and_stops_prefix_escalation(tmp_path: Path):
  artifact, calls = _artifact(), []

  def runner(**kwargs):
    calls.append((kwargs["child_env_overrides"]["AMD_AQL"], kwargs["epoch_limit"]))
    row = _passing_result(artifact, kwargs)
    row["correctness"] = {"status": "BLOCKED", "comparison": {"status": "mismatch", "mismatch_count": 1}}
    return row

  result = differential.run_pm4_aql_frozen_differential(
    tmp_path / "bundle", runner=runner, loader=lambda _: artifact, environ={})
  assert calls == [("0", 1), ("1", 1)]
  assert result["status"] == "BLOCKED"
  assert all(any("numeric correctness" in error for error in mode["attempts"][0]["validation_errors"])
             for mode in result["modes"])
