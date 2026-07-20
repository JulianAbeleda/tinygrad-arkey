from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
from pathlib import Path

import pytest

from extra.qk.mmq_ffn_gate_up_guarded_correctness import (
  CANDIDATE_SCHEMA, ENVELOPE_SCHEMA,
  build_production_candidate_prefix_runtime,
)
from extra.qk.mmq_ffn_gate_up_pm4_prefix1_runner import (
  _acquire_claim, _claim_path, _identity, main, run_pm4_prefix1,
  validate_pm4_prefix1_semantic_preflight,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_ARTIFACT = REPO_ROOT / \
  "docs/artifacts/qwen3-14b-prefill-ffn-gate-up-staged-3fa4cd619-20260719"
REAL_PATHS = {
  "frozen_bundle": REAL_ARTIFACT / "bundle",
  "staged_family_manifest": REAL_ARTIFACT / "evidence" /
    "qk-ffn-gate-up-staged-3fa4cd619-r1-20260719-family.json",
  "execution_fixture_v2": REAL_ARTIFACT / "evidence" /
    "qk-ffn-gate-up-staged-8cad0c4ba-execution-fixture-v2-20260719.json",
  "pm4_c4": REAL_ARTIFACT / "evidence" /
    "qk-ffn-gate-up-staged-8cad0c4ba-c4-pm4-20260719.json",
}


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


def _request_identity(config: dict) -> str:
  from extra.qk.mmq_ffn_gate_up_guarded_correctness import SCHEMA
  return _identity({
    "schema": f"{SCHEMA}.candidate_request",
    "queue_mode": "PM4", "prefix_epochs": 1,
    "config_identity": _identity(config),
    "prior_evidence_identity": None,
    "cross_queue_admission_identity": None,
  })


def _envelope(paths: dict[str, Path], status: str) -> dict:
  config = _config(paths)
  stage_payload = {"status": status}
  stage = {
    **stage_payload, "evidence_identity": _identity(stage_payload)}
  payload = {
    "schema": ENVELOPE_SCHEMA, "status": status,
    "exact_blocker": None if status == "PASS" else "test blocker",
    "queue_mode": "PM4", "operation_schema": CANDIDATE_SCHEMA,
    "health_before": True, "health_after": True,
    "kernel_faults": [], "kernel_fault_evidence": {},
    "launched": True, "spawn_count": 1, "child_status": "passed",
    "timed_out": False, "error": None, "elapsed_seconds": 1.0,
    "result": stage, "no_retry": True, "retry_count": 0,
    "no_queue_fallback": True, "promotion_evidence_eligible": False,
    "request_identity": _request_identity(config),
    "config_identity": _identity(config),
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _run(paths, guarded_stage, **overrides):
  return run_pm4_prefix1(
    **paths, timeout_seconds=7,
    semantic_preflight=lambda *_: _sid("candidate"),
    guarded_stage=guarded_stage, **overrides)


def test_prelaunch_missing_input_never_invokes_guarded_stage(tmp_path):
  paths = _inputs(tmp_path)
  paths["pm4_c4"].unlink()
  calls = []
  errors = io.StringIO()
  rc = _run(
    paths, lambda **kwargs: calls.append(kwargs),
    error_stream=errors)
  assert rc == 2 and calls == []
  assert not _claim_path(paths["output"]).exists()
  assert "prelaunch failure" in errors.getvalue()


def test_semantic_preflight_failure_never_reaches_claim_or_guard(tmp_path):
  paths = _inputs(tmp_path)
  guarded_calls = []

  def fail_preflight(*_):
    raise ValueError("semantic authority differs")

  rc = run_pm4_prefix1(
    **paths, semantic_preflight=fail_preflight,
    guarded_stage=lambda **kwargs: guarded_calls.append(kwargs))
  assert rc == 2 and guarded_calls == []
  assert not paths["output"].exists()
  assert not _claim_path(paths["output"]).exists()


def test_existing_output_and_existing_claim_exclude_without_spawn(tmp_path):
  paths = _inputs(tmp_path)
  calls = []
  paths["output"].write_text("occupied")
  assert _run(paths, lambda **kwargs: calls.append(kwargs)) == 2
  paths["output"].unlink()
  claim = _claim_path(paths["output"])
  claim.write_text("owned elsewhere")
  assert _run(paths, lambda **kwargs: calls.append(kwargs)) == 2
  assert calls == [] and claim.read_text() == "owned elsewhere"


def test_output_created_during_claim_acquisition_blocks_before_guard(tmp_path):
  paths, calls = _inputs(tmp_path), []

  def interleaved_claim(output):
    claim = _acquire_claim(output)
    output.write_text("racing writer")
    return claim

  assert _run(
    paths, lambda **kwargs: calls.append(kwargs),
    claim_acquirer=interleaved_claim) == 2
  assert calls == []
  assert paths["output"].read_text() == "racing writer"
  assert not _claim_path(paths["output"]).exists()


@pytest.mark.parametrize("malformed", ("execution_fixture_v2", "pm4_c4"))
def test_readable_malformed_semantic_artifact_blocks_before_claim_and_guard(
    tmp_path, malformed):
  paths = {**REAL_PATHS, "output": tmp_path / "result.json"}
  paths[malformed] = tmp_path / f"malformed-{malformed}.json"
  paths[malformed].write_text("{}\n")
  guarded_calls = []
  errors = io.StringIO()
  rc = run_pm4_prefix1(
    **paths, guarded_stage=lambda **kwargs: guarded_calls.append(kwargs),
    error_stream=errors)
  assert rc == 2 and guarded_calls == []
  assert not paths["output"].exists()
  assert not _claim_path(paths["output"]).exists()
  assert "prelaunch failure" in errors.getvalue()


def test_real_semantic_preflight_opens_no_device():
  from tinygrad.device import Device

  assert not Device._opened_devices
  identity = validate_pm4_prefix1_semantic_preflight(**REAL_PATHS)
  assert identity == \
    "sha256:48b8d87229e53cd323677fbdcdeb8681772e4fa31948c13a8f7849c3a7e02ccb"
  assert not Device._opened_devices


def test_invokes_exactly_one_hard_coded_pm4_prefix1_stage(
    tmp_path, monkeypatch):
  paths, calls = _inputs(tmp_path), []
  envelope = _envelope(paths, "PASS")
  monkeypatch.setenv("DEV", "CPU")
  monkeypatch.setenv("AMD_AQL", "1")

  def guarded(**kwargs):
    assert os.environ["DEV"] == "AMD"
    assert os.environ["AMD_AQL"] == "0"
    calls.append(kwargs)
    return envelope

  published = {}
  class Ref: pass
  ref = Ref()
  receipts = io.StringIO()

  def freeze(path, value):
    path.write_text(json.dumps(value))
    published.update(path=path, value=value)
    return ref

  rc = _run(
    paths, guarded, pass_validator=lambda value: dict(value),
    pass_freezer=freeze,
    pass_loader=lambda value: (
      envelope if value is ref else (_ for _ in ()).throw(AssertionError())),
    receipt_stream=receipts)
  assert rc == 0 and len(calls) == 1
  assert calls[0] == {
    "config": _config(paths), "queue_mode": "PM4", "prefix_epochs": 1,
    "runtime_builder": build_production_candidate_prefix_runtime,
    "prior_evidence": None, "cross_queue_admission": None,
    "timeout_seconds": 7.0,
  }
  assert published == {"path": paths["output"], "value": envelope}
  assert not _claim_path(paths["output"]).exists()
  assert os.environ["DEV"] == "CPU" and os.environ["AMD_AQL"] == "1"
  receipt = json.loads(receipts.getvalue())
  assert receipt == {
    "status": "PASS", "output": str(paths["output"]),
    "file_sha256": hashlib.sha256(paths["output"].read_bytes()).hexdigest(),
    "outer_evidence_identity": envelope["evidence_identity"],
    "nested_stage_evidence_identity":
      envelope["result"]["evidence_identity"],
    "launched": True, "spawn_count": 1, "blocker": None,
  }


def test_blocked_envelope_is_preserved_as_non_admissible_forensics(tmp_path):
  paths = _inputs(tmp_path)
  envelope = _envelope(paths, "BLOCKED")
  pass_calls = []
  receipts = io.StringIO()
  rc = _run(
    paths, lambda **_: envelope,
    pass_validator=lambda value: pass_calls.append(value),
    pass_freezer=lambda *_: pass_calls.append("freeze"),
    pass_loader=lambda *_: pass_calls.append("load"),
    receipt_stream=receipts)
  assert rc == 1 and pass_calls == []
  assert json.loads(paths["output"].read_bytes()) == envelope
  assert not _claim_path(paths["output"]).exists()
  receipt = json.loads(receipts.getvalue())
  assert receipt["status"] == "BLOCKED"
  assert receipt["output"] == str(paths["output"])
  assert receipt["file_sha256"] == \
    hashlib.sha256(paths["output"].read_bytes()).hexdigest()
  assert receipt["outer_evidence_identity"] == envelope["evidence_identity"]
  assert receipt["nested_stage_evidence_identity"] == \
    envelope["result"]["evidence_identity"]
  assert receipt["launched"] is True and receipt["spawn_count"] == 1
  assert receipt["blocker"] == "test blocker"


def test_pass_round_trip_mismatch_is_exit3_and_retains_claim(tmp_path):
  paths = _inputs(tmp_path)
  envelope = _envelope(paths, "PASS")
  errors = io.StringIO()
  rc = _run(
    paths, lambda **_: envelope,
    pass_validator=lambda value: dict(value),
    pass_freezer=lambda *_: object(),
    pass_loader=lambda _: {"different": True},
    error_stream=errors)
  assert rc == 3
  assert _claim_path(paths["output"]).exists()
  assert "postlaunch failure" in errors.getvalue()


def test_blocked_persistence_failure_is_exit3_and_retains_claim(tmp_path):
  paths = _inputs(tmp_path)
  envelope = _envelope(paths, "BLOCKED")

  def fail_publish(*_):
    raise OSError("disk full")

  rc = _run(
    paths, lambda **_: envelope,
    forensic_publisher=fail_publish)
  assert rc == 3 and _claim_path(paths["output"]).exists()
  assert not paths["output"].exists()


def test_receipt_failure_retains_durable_output_and_claim(tmp_path):
  paths = _inputs(tmp_path)
  envelope = _envelope(paths, "BLOCKED")

  class BrokenReceipt:
    def write(self, _):
      raise BrokenPipeError("receipt consumer closed")

    def flush(self):
      raise AssertionError("unreachable")

  rc = _run(
    paths, lambda **_: envelope, receipt_stream=BrokenReceipt())
  assert rc == 3
  assert json.loads(paths["output"].read_bytes()) == envelope
  assert _claim_path(paths["output"]).exists()


def test_main_dependency_injection_maps_all_cli_arguments(tmp_path):
  paths, calls = _inputs(tmp_path), []
  argv = [
    "--frozen-bundle", str(paths["frozen_bundle"]),
    "--staged-family-manifest", str(paths["staged_family_manifest"]),
    "--execution-fixture-v2", str(paths["execution_fixture_v2"]),
    "--pm4-c4", str(paths["pm4_c4"]), "--output", str(paths["output"]),
    "--timeout-seconds", "12.5",
  ]
  assert main(argv, runner=lambda **kwargs: calls.append(kwargs) or 17) == 17
  assert calls == [{
    **{key: str(value) for key, value in paths.items()},
    "timeout_seconds": 12.5,
  }]


def test_module_entry_and_spawn_targets_are_picklable():
  from extra.qk import mmq_ffn_gate_up_pm4_prefix1_runner as runner
  for target in (
      runner.main, runner.run_pm4_prefix1,
      runner.derive_candidate_executable_identity,
      build_production_candidate_prefix_runtime):
    assert pickle.loads(pickle.dumps(target)) is target
