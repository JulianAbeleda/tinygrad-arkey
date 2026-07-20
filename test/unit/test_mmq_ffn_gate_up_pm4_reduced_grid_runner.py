from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
from pathlib import Path

import pytest

from extra.qk import mmq_ffn_gate_up_pm4_reduced_grid_runner as runner
from extra.qk.mmq_ffn_gate_up_guarded_correctness import (
  ENVELOPE_SCHEMA, FFN_REDUCED_GRID_SCHEMA,
  build_production_candidate_prefix_runtime,
)


def _sid(label: str) -> str:
  return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


def _inputs(tmp_path: Path) -> dict[str, Path]:
  bundle = tmp_path / "bundle"
  bundle.mkdir()
  paths = {
    "frozen_bundle": bundle,
    "staged_family_manifest": tmp_path / "family.json",
    "execution_fixture_v2": tmp_path / "fixture.json",
    "pm4_c4": tmp_path / "c4.json",
    "output": tmp_path / "result.json",
  }
  for name, path in paths.items():
    if name not in ("frozen_bundle", "output"):
      path.write_text("{}\n")
  return paths


def _config(paths: dict[str, Path]) -> dict:
  return {
    "frozen_bundle": str(paths["frozen_bundle"].resolve()),
    "staged_family_manifest":
      str(paths["staged_family_manifest"].resolve()),
    "execution_fixture_v2":
      str(paths["execution_fixture_v2"].resolve()),
    "runtime_canary_isolation": str(paths["pm4_c4"].resolve()),
    "candidate_executable_identity": _sid("candidate"),
  }


def _clear_fault_evidence() -> dict:
  return {
    "schema": "tinygrad.amd_kernel_fault_evidence.v1",
    "status": "CLEAR", "source": "kernel_journal_window", "blocks": [],
    "relevant_line_count": 0, "retained_line_count": 0,
    "truncated": False,
    "limits": {
      "max_blocks": 8, "max_lines": 32, "max_line_chars": 512},
  }


def _envelope(
    paths: dict[str, Path], status: str,
    grid: tuple[int, int, int] = (1, 1, 1),
    ) -> dict:
  config = _config(paths)
  result = None
  if status == "PASS":
    child_payload = {
      "diagnostic_global_size": list(grid),
      "target_dispatch_submitted": True,
      "promotion_evidence_eligible": False,
    }
    result = {
      **child_payload, "evidence_identity": runner._identity(child_payload)}
  payload = {
    "schema": ENVELOPE_SCHEMA, "status": status,
    "exact_blocker": None if status == "PASS" else "test blocker",
    "queue_mode": "PM4", "operation_schema": FFN_REDUCED_GRID_SCHEMA,
    "health_before": True, "health_after": True,
    "kernel_faults": [], "kernel_fault_evidence": _clear_fault_evidence(),
    "launched": True, "spawn_count": 1,
    "child_status": "passed" if status == "PASS" else "failed",
    "timed_out": False,
    "error": None if status == "PASS" else "test blocker",
    "elapsed_seconds": 0.25, "result": result,
    "no_retry": True, "retry_count": 0,
    "no_queue_fallback": True, "promotion_evidence_eligible": False,
    "request_identity": runner._request_identity(config, grid),
    "config_identity": runner._identity(config),
  }
  return {**payload, "evidence_identity": runner._identity(payload)}


def _run(paths, guarded_stage, *, grid_x=1, grid_y=1, **overrides):
  return runner.run_pm4_reduced_grid(
    **paths, grid_x=grid_x, grid_y=grid_y, timeout_seconds=7,
    semantic_preflight=lambda *_: _sid("candidate"),
    guarded_stage=guarded_stage, **overrides)


def _identity_forensic_validator(value, **_):
  return dict(value)


def test_1x1_maps_to_exact_origin_128x128_rectangle():
  assert runner.validate_diagnostic_global_size(1, 1) == (1, 1, 1)
  assert runner.touched_output_rectangle((1, 1, 1)) == {
    "row_start": 0, "row_stop_exclusive": 128,
    "column_start": 0, "column_stop_exclusive": 128,
    "row_count": 128, "column_count": 128, "element_count": 16384,
  }


@pytest.mark.parametrize(
  "grid_x,grid_y", [
    (True, 1), (1, False), (137, 1), (136, 4), (3, 1), (40, 3),
  ])
def test_invalid_bool_oversize_or_unlisted_grid_never_claims_or_spawns(
    tmp_path, grid_x, grid_y):
  paths, calls = _inputs(tmp_path), []
  errors = io.StringIO()
  assert _run(
    paths, lambda **kwargs: calls.append(kwargs),
    grid_x=grid_x, grid_y=grid_y, error_stream=errors) == 2
  assert calls == []
  assert not paths["output"].exists()
  assert not runner._claim_path(paths["output"]).exists()
  assert "prelaunch failure" in errors.getvalue()


def test_missing_artifact_preflight_failure_never_claims_or_spawns(tmp_path):
  paths, calls = _inputs(tmp_path), []
  paths["pm4_c4"].unlink()
  assert _run(paths, lambda **kwargs: calls.append(kwargs)) == 2
  assert calls == []
  assert not paths["output"].exists()
  assert not runner._claim_path(paths["output"]).exists()


def test_exactly_one_guarded_call_has_fixed_environment_grid_and_builder(
    tmp_path, monkeypatch):
  paths, calls = _inputs(tmp_path), []
  envelope = _envelope(paths, "PASS", (40, 4, 1))
  monkeypatch.setenv("DEV", "CPU")
  monkeypatch.setenv("AMD_AQL", "1")
  monkeypatch.setenv("PROFILE", "9")

  def guarded(**kwargs):
    assert {
      key: os.environ[key] for key in ("DEV", "AMD_AQL", "PROFILE")
    } == {"DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"}
    calls.append(kwargs)
    return envelope

  receipts = io.StringIO()
  assert _run(
    paths, guarded, grid_x=40, grid_y=4,
    forensic_validator=_identity_forensic_validator,
    receipt_stream=receipts) == 0
  assert calls == [{
    "config": _config(paths),
    "runtime_builder": build_production_candidate_prefix_runtime,
    "diagnostic_global_size": (40, 4, 1),
    "timeout_seconds": 7.0,
  }]
  assert {
    key: os.environ[key] for key in ("DEV", "AMD_AQL", "PROFILE")
  } == {"DEV": "CPU", "AMD_AQL": "1", "PROFILE": "9"}
  receipt = json.loads(receipts.getvalue())
  assert receipt["schema"] == runner.RECEIPT_SCHEMA
  assert receipt["diagnostic_global_size"] == [40, 4, 1]
  assert receipt["touched_output_rectangle"] == {
    "row_start": 0, "row_stop_exclusive": 512,
    "column_start": 0, "column_stop_exclusive": 5120,
    "row_count": 512, "column_count": 5120,
    "element_count": 2621440,
  }
  assert receipt["target_dispatch_submitted"] is True
  assert receipt["kernel_faults"] == []
  assert receipt["kernel_fault_evidence_status"] == "CLEAR"
  assert receipt["promotion_evidence_eligible"] is False
  assert receipt["full_grid_correctness_claimed"] is False


@pytest.mark.parametrize("status,expected_rc", [("PASS", 0), ("BLOCKED", 1)])
def test_pass_and_blocked_are_immutable_forensic_round_trips(
    tmp_path, status, expected_rc):
  paths = _inputs(tmp_path)
  envelope = _envelope(paths, status)
  receipts = io.StringIO()
  assert _run(
    paths, lambda **_: envelope,
    forensic_validator=_identity_forensic_validator,
    receipt_stream=receipts) == expected_rc
  assert json.loads(paths["output"].read_bytes()) == envelope
  assert not runner._claim_path(paths["output"]).exists()
  receipt = json.loads(receipts.getvalue())
  assert receipt["status"] == status
  assert receipt["file_sha256"] == \
    hashlib.sha256(paths["output"].read_bytes()).hexdigest()
  assert receipt["promotion_evidence_eligible"] is False
  assert receipt["full_grid_correctness_claimed"] is False


def test_forensic_validator_accepts_blocked_without_promotion(tmp_path):
  paths = _inputs(tmp_path)
  envelope = _envelope(paths, "BLOCKED")
  assert runner.validate_pm4_reduced_grid_forensic_envelope(
    envelope, config=_config(paths),
    diagnostic_global_size=(1, 1, 1)) == envelope


def test_existing_output_and_claim_exclude_without_guarded_call(tmp_path):
  paths, calls = _inputs(tmp_path), []
  paths["output"].write_text("occupied")
  assert _run(paths, lambda **kwargs: calls.append(kwargs)) == 2
  paths["output"].unlink()
  claim = runner._claim_path(paths["output"])
  claim.write_text("owned elsewhere")
  assert _run(paths, lambda **kwargs: calls.append(kwargs)) == 2
  assert calls == [] and claim.read_text() == "owned elsewhere"


def test_output_created_during_claim_acquisition_is_not_replaced(tmp_path):
  paths, calls = _inputs(tmp_path), []

  def interleaved_claim(output, *, diagnostic_global_size):
    claim = runner._acquire_claim(
      output, diagnostic_global_size=diagnostic_global_size)
    output.write_text("racing writer")
    return claim

  assert _run(
    paths, lambda **kwargs: calls.append(kwargs),
    claim_acquirer=interleaved_claim) == 2
  assert calls == []
  assert paths["output"].read_text() == "racing writer"
  assert not runner._claim_path(paths["output"]).exists()


def test_guarded_failure_is_not_retried_and_retains_claim(tmp_path):
  paths, calls = _inputs(tmp_path), []

  def fail(**kwargs):
    calls.append(kwargs)
    raise RuntimeError("one terminal attempt")

  errors = io.StringIO()
  assert _run(paths, fail, error_stream=errors) == 3
  assert len(calls) == 1
  assert runner._claim_path(paths["output"]).exists()
  assert not paths["output"].exists()
  assert "postlaunch failure" in errors.getvalue()


def test_runner_has_no_correctness_freezer_or_retry_surface():
  source = inspect.getsource(runner)
  assert "freeze_correctness_evidence" not in source
  assert "pass_freezer" not in source
  signature = inspect.signature(runner.run_pm4_reduced_grid)
  assert "retry" not in signature.parameters
  assert "fallback" not in signature.parameters


def test_main_maps_four_artifacts_output_grid_and_timeout(tmp_path):
  paths, calls = _inputs(tmp_path), []
  argv = [
    "--frozen-bundle", str(paths["frozen_bundle"]),
    "--staged-family-manifest", str(paths["staged_family_manifest"]),
    "--execution-fixture-v2", str(paths["execution_fixture_v2"]),
    "--pm4-c4", str(paths["pm4_c4"]),
    "--output", str(paths["output"]),
    "--grid-x", "41", "--grid-y", "4",
    "--timeout-seconds", "12.5",
  ]
  assert runner.main(
    argv, runner=lambda **kwargs: calls.append(kwargs) or 17) == 17
  assert calls == [{
    **{key: str(value) for key, value in paths.items()},
    "grid_x": 41, "grid_y": 4, "timeout_seconds": 12.5,
  }]
