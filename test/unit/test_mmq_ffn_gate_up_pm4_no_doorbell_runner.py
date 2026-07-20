from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import pytest

from extra.qk.mmq_ffn_gate_up_guarded_correctness import (
  CANDIDATE_EXECUTABLE_SCHEMA, ENVELOPE_SCHEMA,
  PM4_NO_DOORBELL_RECEIPT_SCHEMA, PM4_NO_DOORBELL_SCHEMA,
  build_production_candidate_prefix_runtime,
)
from extra.qk.mmq_ffn_gate_up_pm4_no_doorbell_runner import (
  RECEIPT_SCHEMA, _acquire_claim, _claim_path, _identity, main,
  run_pm4_no_doorbell, validate_pm4_no_doorbell_forensic_envelope,
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
    "schema": f"{SCHEMA}.pm4_no_doorbell_request",
    "queue_mode": "PM4", "prefix_epochs": 1,
    "submit_policy": "snapshot_only",
    "config_identity": _identity(config),
  })


def _clear_fault_evidence() -> dict:
  return {
    "schema": "tinygrad.amd_kernel_fault_evidence.v1",
    "status": "CLEAR", "source": "kernel_journal_window", "blocks": [],
    "relevant_line_count": 0, "retained_line_count": 0,
    "truncated": False,
    "limits": {
      "max_blocks": 8, "max_lines": 32, "max_line_chars": 512},
  }


def _no_doorbell_receipt() -> dict:
  pre_submit_checks = {
    "queue_device_matches_submit_device": True,
    "runtime_device_matches_submit_device": True,
    "args_state_program_matches_runtime": True,
    "exact_five_argument_buffers": True,
    "exact_five_kernarg_qwords": True,
    "five_qwords_match_constructed_buffers": True,
    "pm4_command_words_concrete": True,
    "pm4_command_stream_nonempty": True,
    "pm4_packet_stream_decoded": True,
    "pm4_kernarg_user_data_found_once": True,
    "pm4_kernarg_uses_user_data_0": True,
    "pm4_kernarg_user_data_matches_kernarg_va": True,
  }
  vas = [0x10000, 0x20000, 0x30000, 0x40000, 0x50000]
  kernarg_va = 0x70000
  pre_submit = {
    "schema": "tinygrad.mmq_q4k_q8_1.pm4_pre_submit_snapshot.v1",
    "capture_point":
      "AMDComputeQueue._submit_after_complete_command_construction_"
      "before_ring_copy_and_doorbell",
    "runtime_object_identity": 123,
    "runtime_class": "tinygrad.runtime.ops_amd.AMDProgram",
    "runtime_name": "mmq_llama_five_buffer_full_grid_accumulate",
    "runtime_device": "AMD",
    "kernarg_va": kernarg_va, "kernarg_nbytes": 40,
    "kernarg_qwords": vas,
    "argument_buffers": [
      {"slot": slot, "va": va, "size": size}
      for slot, (va, size) in enumerate(zip(
        vas, (35651584, 2506752, 131072, 16384, 16384)))],
    "pm4_kernarg_user_data": {
      "packet_dword_offset": 1, "register_index": 0,
      "low_dword": kernarg_va, "high_dword": 0,
      "pointer": kernarg_va,
    },
    "pm4_dword_count": 16, "pm4_sha256": "3" * 64,
    "checks": pre_submit_checks, "all_checks_pass": True,
  }
  checks = {
    "private_stop_raised_at_target_submit": True,
    "private_stop_caught_by_owner": True,
    "pre_submit_snapshot_passed": True,
    "native_submit_not_called": True,
    "timeline_advanced_exactly_once": True,
    "prof_exec_counter_advanced_exactly_once": True,
    "timeline_rollback_restored": True,
    "prof_exec_counter_rollback_restored": True,
    "timeline_signal_unchanged": True,
    "error_state_unchanged": True,
    "submit_hook_restored": True,
    "fill_kernargs_hook_restored": True,
  }
  return {
    "schema": PM4_NO_DOORBELL_RECEIPT_SCHEMA,
    "status": "CAPTURED_NO_SUBMIT", "submit_policy": "snapshot_only",
    "pre_submit": pre_submit,
    "target_dispatch_submitted": False, "native_submit_call_count": 0,
    "ring_copy_performed": False, "doorbell_rung": False,
    "timeline_value_before": 7, "timeline_value_after_runtime_unwind": 8,
    "timeline_value_after_rollback": 7, "timeline_rollback_applied": True,
    "prof_exec_counter_before": 11,
    "prof_exec_counter_after_runtime_unwind": 12,
    "prof_exec_counter_after_rollback": 11,
    "timeline_signal_value_before": 6,
    "timeline_signal_value_after": 6, "terminal_child_required": True,
    "promotion_evidence_eligible": False, "checks": checks,
    "all_checks_pass": True,
  }


def _stage(config: dict) -> dict:
  family = _sid("family")
  program_key, binary_sha256 = "1" * 64, "2" * 64
  executable = _identity({
    "schema": CANDIDATE_EXECUTABLE_SCHEMA,
    "family_identity": family, "program_key": program_key,
    "binary_sha256": binary_sha256,
  })
  receipt = _no_doorbell_receipt()
  attempt = {
    "phase": "complete", "invocation_count": 1, "receipt_count": 1,
    "receipt": receipt,
  }
  payload = {
    "schema": PM4_NO_DOORBELL_SCHEMA, "status": "PASS",
    "exact_blocker": None, "queue_mode": "PM4", "prefix_epochs": 1,
    "submit_policy": "snapshot_only", "no_retry": True, "retry_count": 0,
    "no_fallback": True, "compile_performed": False,
    "requires_recompile": False, "promotion_evidence_eligible": False,
    "config_identity": _identity(config),
    "request_identity": _request_identity(config),
    "environment": {"DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"},
    "family_identity": family, "fixture_identity": _sid("fixture"),
    "workload_identity": _sid("workload"),
    "input_identity": _sid("input"),
    "logical_q4_identity": _sid("logical-q4"),
    "resident_fp16_activation_identity": _sid("resident-fp16"),
    "candidate_executable_identity": executable,
    "program_key": program_key, "binary_sha256": binary_sha256,
    "c4_canary_identity": _sid("c4"),
    "invocation_count": 1, "receipt_count": 1, "receipt": receipt,
    "target_dispatch_submitted": False, "native_submit_call_count": 0,
    "timeline_rollback_applied": True, "terminal_child_required": True,
    "readback_performed": False, "numeric_validation_performed": False,
    "attestation_performed": False,
    "producer_attestation_performed": False, "attempt": attempt,
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _envelope(paths: dict[str, Path], status: str) -> dict:
  config = _config(paths)
  stage = _stage(config) if status == "PASS" else None
  payload = {
    "schema": ENVELOPE_SCHEMA, "status": status,
    "exact_blocker": None if status == "PASS" else "test blocker",
    "queue_mode": "PM4", "operation_schema": PM4_NO_DOORBELL_SCHEMA,
    "health_before": True, "health_after": True,
    "kernel_faults": [], "kernel_fault_evidence": _clear_fault_evidence(),
    "launched": status == "PASS", "spawn_count": int(status == "PASS"),
    "child_status": "passed" if status == "PASS" else None,
    "timed_out": False, "error": None,
    "elapsed_seconds": 1.0 if status == "PASS" else None,
    "result": stage, "no_retry": True, "retry_count": 0,
    "no_queue_fallback": True, "promotion_evidence_eligible": False,
    "request_identity": _request_identity(config),
    "config_identity": _identity(config),
  }
  return {**payload, "evidence_identity": _identity(payload)}


def _run(paths, guarded_stage, **overrides):
  return run_pm4_no_doorbell(
    **paths, timeout_seconds=7,
    semantic_preflight=lambda *_: _sid("candidate"),
    guarded_stage=guarded_stage, **overrides)


def test_real_semantic_preflight_is_reused_without_opening_device():
  from tinygrad.device import Device
  from extra.qk.mmq_ffn_gate_up_pm4_no_doorbell_runner import \
    validate_pm4_prefix1_semantic_preflight

  assert not Device._opened_devices
  identity = validate_pm4_prefix1_semantic_preflight(**REAL_PATHS)
  assert identity == \
    "sha256:48b8d87229e53cd323677fbdcdeb8681772e4fa31948c13a8f7849c3a7e02ccb"
  assert not Device._opened_devices


def test_preflight_failure_never_claims_or_invokes(tmp_path):
  paths, calls = _inputs(tmp_path), []
  paths["pm4_c4"].unlink()
  errors = io.StringIO()
  assert _run(
    paths, lambda **kwargs: calls.append(kwargs),
    error_stream=errors) == 2
  assert calls == [] and not paths["output"].exists()
  assert not _claim_path(paths["output"]).exists()
  assert "prelaunch failure" in errors.getvalue()


def test_exactly_one_guarded_call_has_fixed_environment_and_builder(
    tmp_path, monkeypatch):
  paths, calls = _inputs(tmp_path), []
  envelope = _envelope(paths, "PASS")
  monkeypatch.setenv("DEV", "CPU")
  monkeypatch.setenv("AMD_AQL", "1")
  monkeypatch.setenv("PROFILE", "9")

  def guarded(**kwargs):
    assert {key: os.environ[key] for key in
            ("DEV", "AMD_AQL", "PROFILE")} == {
              "DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"}
    calls.append(kwargs)
    return envelope

  receipts = io.StringIO()
  assert _run(paths, guarded, receipt_stream=receipts) == 0
  assert calls == [{
    "config": _config(paths),
    "runtime_builder": build_production_candidate_prefix_runtime,
    "timeout_seconds": 7.0,
  }]
  assert {key: os.environ[key] for key in
          ("DEV", "AMD_AQL", "PROFILE")} == {
            "DEV": "CPU", "AMD_AQL": "1", "PROFILE": "9"}
  receipt = json.loads(receipts.getvalue())
  assert receipt["schema"] == RECEIPT_SCHEMA
  assert receipt["diagnostic"] == "PM4_NO_DOORBELL"
  assert receipt["status"] == "PASS"
  assert receipt["target_dispatch_submitted"] is False
  assert receipt["native_submit_call_count"] == 0
  assert receipt["promotion_evidence_eligible"] is False


@pytest.mark.parametrize("status,expected_rc", [("PASS", 0), ("BLOCKED", 1)])
def test_pass_and_blocked_are_immutable_forensic_round_trips(
    tmp_path, status, expected_rc):
  paths = _inputs(tmp_path)
  envelope = _envelope(paths, status)
  receipts = io.StringIO()
  assert _run(
    paths, lambda **_: envelope, receipt_stream=receipts) == expected_rc
  assert json.loads(paths["output"].read_bytes()) == envelope
  assert validate_pm4_no_doorbell_forensic_envelope(
    json.loads(paths["output"].read_bytes()),
    config=_config(paths)) == envelope
  assert not _claim_path(paths["output"]).exists()
  receipt = json.loads(receipts.getvalue())
  assert receipt["status"] == status
  assert receipt["file_sha256"] == \
    hashlib.sha256(paths["output"].read_bytes()).hexdigest()
  assert receipt["promotion_evidence_eligible"] is False


def test_existing_output_and_claim_exclude_without_guarded_call(tmp_path):
  paths, calls = _inputs(tmp_path), []
  paths["output"].write_text("occupied")
  assert _run(paths, lambda **kwargs: calls.append(kwargs)) == 2
  paths["output"].unlink()
  claim = _claim_path(paths["output"])
  claim.write_text("owned elsewhere")
  assert _run(paths, lambda **kwargs: calls.append(kwargs)) == 2
  assert calls == [] and claim.read_text() == "owned elsewhere"


def test_output_created_during_claim_acquisition_is_not_replaced(tmp_path):
  paths, calls = _inputs(tmp_path), []

  def interleaved_claim(output):
    claim = _acquire_claim(output)
    output.write_text("racing writer")
    return claim

  assert _run(
    paths, lambda **kwargs: calls.append(kwargs),
    claim_acquirer=interleaved_claim) == 2
  assert calls == [] and paths["output"].read_text() == "racing writer"
  assert not _claim_path(paths["output"]).exists()


def test_guarded_failure_is_not_retried_and_retains_claim(tmp_path):
  paths, calls = _inputs(tmp_path), []

  def fail(**kwargs):
    calls.append(kwargs)
    raise RuntimeError("one terminal attempt")

  errors = io.StringIO()
  assert _run(paths, fail, error_stream=errors) == 3
  assert len(calls) == 1
  assert _claim_path(paths["output"]).exists()
  assert not paths["output"].exists()
  assert "postlaunch failure" in errors.getvalue()


def test_main_maps_the_same_four_artifacts_output_and_timeout(tmp_path):
  paths, calls = _inputs(tmp_path), []
  argv = [
    "--frozen-bundle", str(paths["frozen_bundle"]),
    "--staged-family-manifest", str(paths["staged_family_manifest"]),
    "--execution-fixture-v2", str(paths["execution_fixture_v2"]),
    "--pm4-c4", str(paths["pm4_c4"]),
    "--output", str(paths["output"]), "--timeout-seconds", "12.5",
  ]
  assert main(argv, runner=lambda **kwargs: calls.append(kwargs) or 17) == 17
  assert calls == [{
    **{key: str(value) for key, value in paths.items()},
    "timeout_seconds": 12.5,
  }]
