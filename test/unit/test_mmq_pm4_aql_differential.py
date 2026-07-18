from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from extra.qk import mmq_pm4_aql_differential as differential
from extra.qk.mmq_exact_role_spec import DEFAULT_EXACT_ROLE_SPEC, ExactRoleSpec, exact_role_spec
from extra.qk.mmq_frozen_target_artifact import ACCUMULATION


def _artifact(role_spec: ExactRoleSpec = DEFAULT_EXACT_ROLE_SPEC):
  fixture = {"schema": "fixture.v1", "role": role_spec.role, "shape": list(role_spec.shape),
             "q4": {"sha256": "q4"}, "q8": {"sha256": "q8"}}
  manifest = {
    "schema": "frozen.v1", "state": "FROZEN", "compile_calls": 1,
    "accumulate": True, "accumulation": ACCUMULATION,
    "shape": list(role_spec.program.shape), "full_role_shape": list(role_spec.shape),
    "consumer": {"requires_recompile": False},
    "program": {"key": f"program-{role_spec.m}x{role_spec.n}x256", "function": "target",
                "global_size": list(role_spec.program.grid)},
    "files": {"fixture.json": {"sha256": "fixture-file-sha", "nbytes": 1}},
    "artifacts": {
      "source_sha256": "source-sha", "binary_sha256": "binary-sha",
      "serialized_program_sha256": "program-sha",
    },
  }
  return SimpleNamespace(manifest=manifest, fixture=fixture)


def _passing_result(artifact, kwargs):
  role_spec = kwargs["role_spec"]
  expected = differential._frozen_identity(artifact, role_spec)
  prefix = kwargs["epoch_limit"]
  amd_aql = kwargs["child_env_overrides"]["AMD_AQL"]
  epoch_staging_rows = [{
    "epoch": epoch, "source_q4_va": 0x1000 + epoch * 0x100,
    "source_values_va": 0x2000 + epoch * 0x100,
    "stage_q4_va": 0x3000, "stage_values_va": 0x4000,
  } for epoch in range(prefix)]
  epoch_staging = {
    "mode": "all_inputs_fixed_va_gpu_sdma", "fixed_va": True, "transfer": "gpu_sdma",
    "per_epoch_vas": epoch_staging_rows,
  }
  return {
    "status": "PASS", "role": role_spec.role, "shape": list(role_spec.shape),
    "accumulation": ACCUMULATION, "no_fallback": True,
    "health_before": True, "health_after": True,
    "health_mode": {"amd_aql_env": amd_aql, "before": True, "after": True},
    "kernel_faults": [],
    "correctness": {"status": "PASS", "comparison": {"status": "pass", "mismatch_count": 0}},
    "timing": {
      "persistent_buffers": True, "preloaded_epochs": True, "stable_metadata_staging": True,
      "stable_epoch_staging": True, "epoch_staging": epoch_staging,
      "k_epoch_launches": prefix, "total_k_epoch_launches": role_spec.epochs,
      "n_chunk_tiles": role_spec.program.grid[0], "epoch_checks": [],
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
      "launches": [{"global_size": list(role_spec.program.grid)} for _ in range(prefix)],
    },
    "epoch_staging": epoch_staging,
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
             and call["stable_metadata_staging"] and call["stable_epoch_staging"] for call, _ in calls)
  assert all(not call["host_accumulate"] and not call["per_epoch_check"] for call, _ in calls)
  assert all(call["role_spec"] == DEFAULT_EXACT_ROLE_SPEC and call["n_chunk_tiles"] == 136 for call, _ in calls)
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


@pytest.mark.parametrize("role,grid,epochs", [
  ("attn_kv", (8, 4, 1), 20), ("attn_qo", (40, 4, 1), 20), ("ffn_down", (40, 4, 1), 68)])
def test_differential_derives_role_grid_chunks_and_epochs_from_frozen_spec(tmp_path: Path, role, grid, epochs):
  role_spec, calls = exact_role_spec(role), []
  artifact = _artifact(role_spec)
  def runner(**kwargs):
    calls.append(kwargs)
    return _passing_result(artifact, kwargs)
  result = differential.run_pm4_aql_frozen_differential(
    tmp_path / role, runner=runner, loader=lambda _: artifact, environ={})
  assert result["status"] == "PASS" and result["role"] == role and result["shape"] == list(role_spec.shape)
  assert result["bundle"]["program_grid"] == list(grid) and result["bundle"]["total_epochs"] == epochs
  assert all(call["role_spec"] == role_spec and call["n_chunk_tiles"] == grid[0] for call in calls)


def test_differential_shared_qo_down_program_retains_distinct_full_role_and_mismatch_fails_closed(tmp_path: Path):
  qo, down = exact_role_spec("attn_qo"), exact_role_spec("ffn_down")
  qo_artifact, down_artifact = _artifact(qo), _artifact(down)
  assert qo_artifact.manifest["program"] == down_artifact.manifest["program"]
  assert qo_artifact.manifest["full_role_shape"] != down_artifact.manifest["full_role_shape"]
  shared_identity = differential._frozen_identity(qo_artifact, down)
  assert shared_identity["role"] == shared_identity["artifact_role"] == "attn_qo"
  assert shared_identity["artifact_full_role_shape"] == list(qo.shape)
  assert shared_identity["execution_role"] == "ffn_down"
  assert shared_identity["execution_full_role_shape"] == list(down.shape)
  assert shared_identity["shared_program_geometry"] is True
  assert shared_identity["fixture_relationship"] == "distinct_full_role_shared_program_geometry"
  assert shared_identity["fixture_sha256"] == shared_identity["artifact_fixture_sha256"]
  calls = []
  result = differential.run_pm4_aql_frozen_differential(
    tmp_path / "down", role_spec=qo, runner=lambda **kwargs: calls.append(kwargs),
    loader=lambda _: down_artifact, environ={})
  assert result["status"] == "BLOCKED" and "requested exact role differs" in result["exact_blocker"]
  assert calls == []
