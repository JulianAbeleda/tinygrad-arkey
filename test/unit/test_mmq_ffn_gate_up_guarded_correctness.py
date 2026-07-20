from __future__ import annotations

import copy
import hashlib
import inspect
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import extra.qk.mmq_ffn_gate_up_guarded_correctness as gc


REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / (
  "docs/artifacts/qwen3-14b-prefill-ffn-gate-up-staged-"
  "3fa4cd619-20260719/evidence/"
  "qk-ffn-gate-up-staged-8cad0c4ba-execution-fixture-v2-20260719.json")


def _sid(seed: str) -> str:
  return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()


def _rehash(row, identity_field="evidence_identity"):
  row[identity_field] = gc._identity({
    key: value for key, value in row.items() if key != identity_field})
  return row


def _comparison(_got, _reference):
  return {
    "status": "pass", "rtol": 3e-3, "atol": 3e-3,
    "got_shape": list(gc.OUTPUT_SHAPE),
    "reference_shape": list(gc.OUTPUT_SHAPE),
    "mismatch_count": 0, "first_mismatch_index": None,
    "first_mismatch_got": None, "first_mismatch_reference": None,
    "got_size": gc.OUTPUT_ELEMENTS, "reference_size": gc.OUTPUT_ELEMENTS,
    "nan_got": 0, "nan_reference": 0, "inf_got": 0,
    "inf_reference": 0, "joint_finite": gc.OUTPUT_ELEMENTS,
    "max_abs_error": 0.0, "mean_abs_error": 0.0,
  }


class FakeSession:
  def __init__(self, queue, prefix):
    self.queue, self.prefix, self.pending = queue, prefix, None

  def invoke(self, prefix_epochs):
    assert prefix_epochs == self.prefix and self.pending is None
    self.pending = SimpleNamespace(output=np.array([prefix_epochs]))
    return self.pending

  def attest_post_sync(self, invocation, queue):
    assert invocation is self.pending
    self.pending = None
    payload = {
      "schema": gc.LOW_LEVEL_ATTESTATION_SCHEMA,
      "status": "PASS", "queue_mode": queue,
      "family_identity": _sid("family"),
      "candidate_executable_identity":
        gc._candidate_executable_identity_from_parts(
          _sid("family"), "1" * 64, "2" * 64),
      "input_identity": _sid("input"), "program_key": "1" * 64,
      "binary_sha256": "2" * 64, "runtime_class": "FakeRuntime",
      "runtime_name": "fake_kernel", "runtime_device": "AMD",
      "runtime_object_identity": 1, "runtime_device_identity_exact": True,
      "runtime_cache_binding_exact": True, "library_va": 1000,
      "library_nbytes": 1000, "entry_va": 1100,
      "fixed_five_vas": [1, 2, 3, 4, 5],
      "launch_count": self.prefix}
    return SimpleNamespace(
      **payload, observation_identity=gc._identity(payload))


def fake_candidate_builder(config, *, queue_mode, prefix_epochs):
  def producer(prefix):
    payload = {
      "schema": gc.PRODUCER_SCHEMA, "status": "PASS",
      "queue_mode": queue_mode, "prefix_epochs": prefix,
      "family_identity": _sid("family"), "input_identity": _sid("input"),
      "metadata_storage_dtype": "float32",
      "captured_for_consumer_reference": True,
      "consumer_reference_q8_sha256": {
        "values": "6" * 64, "scales": "7" * 64, "sums": "8" * 64},
      "fixture_diagnostic": {
        "q8_values_exact": False, "q8_scales_numeric_match": False,
        "q8_sums_numeric_match": False,
        "observed_sha256": {
          "values": "6" * 64, "scales": "7" * 64, "sums": "8" * 64},
        "fixture_sha256": {
          "values": "9" * 64, "scales": "a" * 64, "sums": "b" * 64},
        "scales_comparison": _mismatch_comparison(),
        "sums_comparison": _mismatch_comparison()},
      "promotion_evidence_eligible": False,
    }
    return {**payload, "evidence_identity": gc._identity(payload)}
  return gc.CandidatePrefixRuntime(
    queue_mode, prefix_epochs, _sid("family"), _sid("fixture"),
    _sid("workload"), _sid("input"), _sid("logical-q4"),
    _sid("resident-fp16"),
    gc._candidate_executable_identity_from_parts(
      _sid("family"), "1" * 64, "2" * 64),
    "1" * 64, "2" * 64,
    _sid(f"{queue_mode}-c4"), FakeSession(queue_mode, prefix_epochs),
    producer, lambda: None, lambda output: output,
    lambda prefix: np.array([prefix]), _comparison)


def _no_doorbell_receipt():
  argument_vas = [
    0x00007F8800001000, 0x00007F8804001000,
    0x00007F8808001000, 0x00007F880C001000,
    0x00007F8810001000]
  kernarg_va = 0x00007F88ABCDEF00
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
  pre_submit = {
    "schema": gc.PM4_PRE_SUBMIT_SCHEMA,
    "capture_point": gc.PM4_PRE_SUBMIT_CAPTURE_POINT,
    "runtime_object_identity": 12345,
    "runtime_class": gc.PM4_RUNTIME_CLASS,
    "runtime_name": gc.PM4_RUNTIME_NAME,
    "runtime_device": "AMD",
    "kernarg_va": kernarg_va, "kernarg_nbytes": 40,
    "kernarg_qwords": argument_vas,
    "argument_buffers": [
      {"slot": slot, "va": va, "size": size}
      for slot, (va, size) in enumerate(
        zip(argument_vas, gc.PM4_ARGUMENT_NBYTES))],
    "pm4_kernarg_user_data": {
      "packet_dword_offset": 12, "register_index": 0,
      "low_dword": kernarg_va & 0xffffffff,
      "high_dword": kernarg_va >> 32, "pointer": kernarg_va,
    },
    "pm4_dword_count": 64, "pm4_sha256": "c" * 64,
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
    "schema": gc.PM4_NO_DOORBELL_RECEIPT_SCHEMA,
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


class FakeNoDoorbellSession(FakeSession):
  def __init__(self, sink, receipts):
    super().__init__("PM4", 1)
    self.sink, self.receipts, self.invoke_count = sink, receipts, 0

  def invoke(self, prefix_epochs):
    self.invoke_count += 1
    assert prefix_epochs == 1
    self.sink.extend(copy.deepcopy(self.receipts))
    return SimpleNamespace(output=object())


def fake_no_doorbell_builder(
    config, *, queue_mode, prefix_epochs, pm4_submit_policy="execute",
    pm4_no_doorbell_receipt_sink=None):
  assert config == {}
  assert (queue_mode, prefix_epochs, pm4_submit_policy) == \
    ("PM4", 1, "snapshot_only")
  assert isinstance(pm4_no_doorbell_receipt_sink, list)
  runtime = fake_candidate_builder(
    config, queue_mode=queue_mode, prefix_epochs=prefix_epochs)
  session = FakeNoDoorbellSession(
    pm4_no_doorbell_receipt_sink, [_no_doorbell_receipt()])
  return gc.replace(
    runtime, session=session,
    producer_attest=lambda _prefix: pytest.fail(
      "producer attestation must not run"),
    synchronize=lambda: pytest.fail("extra synchronization must not run"),
    readback=lambda _output: pytest.fail("readback must not run"),
    reference=lambda _prefix: pytest.fail("reference must not run"),
    comparator=lambda _got, _reference: pytest.fail(
      "numeric comparison must not run"))


def _candidate_request(prefix=1, prior=None, queue="PM4", cross=None):
  return gc.CandidatePrefixRequest(
    {}, queue, prefix, prior, cross, fake_candidate_builder)


def _isolated(function, *, args, **_kwargs):
  return SimpleNamespace(
    status="passed", result=function(*args), timed_out=False,
    error=None, elapsed_seconds=0.1)


def _faults(_started):
  return [], _clear_fault_evidence()


def _clear_fault_evidence():
  return {
    "schema": "tinygrad.amd_kernel_fault_evidence.v1",
    "status": "CLEAR", "source": "kernel_journal_window", "blocks": [],
    "relevant_line_count": 0, "retained_line_count": 0,
    "truncated": False,
    "limits": {"max_blocks": 8, "max_lines": 32,
               "max_line_chars": 512},
  }


def _mismatch_comparison():
  row = _comparison(None, None)
  row.update({
    "status": "mismatch", "mismatch_count": 1,
    "first_mismatch_index": [0, 0],
    "first_mismatch_got": 1.0, "first_mismatch_reference": 0.0,
    "max_abs_error": 1.0, "mean_abs_error": 1.0})
  return row


def _pass_envelope(stage):
  payload = {
    "schema": gc.ENVELOPE_SCHEMA, "status": "PASS",
    "exact_blocker": None, "queue_mode": stage["queue_mode"],
    "operation_schema": stage["schema"], "health_before": True,
    "health_after": True, "kernel_faults": [],
    "kernel_fault_evidence": _clear_fault_evidence(),
    "launched": True, "spawn_count": 1, "child_status": "passed",
    "timed_out": False, "error": None, "elapsed_seconds": 0.1,
    "result": stage, "no_retry": True, "retry_count": 0,
    "no_queue_fallback": True, "promotion_evidence_eligible": False,
    "request_identity": stage["request_identity"],
    "config_identity": stage["config_identity"],
  }
  return {**payload, "evidence_identity": gc._identity(payload)}


def _freeze(value):
  directory = Path(tempfile.mkdtemp(prefix="ffn-correctness-test-"))
  return gc.freeze_correctness_evidence(directory / "artifact.json", value)


def _guard(prefix=1, prior=None, queue="PM4", cross=None, **kwargs):
  return gc.run_guarded_candidate_prefix(
    config={}, queue_mode=queue, prefix_epochs=prefix,
    prior_evidence=prior, cross_queue_admission=cross,
    runtime_builder=fake_candidate_builder,
    isolated_runner=kwargs.pop("isolated_runner", _isolated),
    health_probe=kwargs.pop("health_probe", lambda _env: True),
    fault_collector=kwargs.pop("fault_collector", _faults), **kwargs)


def _guard_no_doorbell(runtime_builder=fake_no_doorbell_builder, **kwargs):
  return gc.run_guarded_pm4_no_doorbell(
    config={}, runtime_builder=runtime_builder,
    isolated_runner=kwargs.pop("isolated_runner", _isolated),
    health_probe=kwargs.pop("health_probe", lambda _env: True),
    fault_collector=kwargs.pop("fault_collector", _faults), **kwargs)


def test_pm4_no_doorbell_is_one_terminal_diagnostic_child_with_no_submit():
  envelope = _guard_no_doorbell()
  result = gc.validate_pm4_no_doorbell_evidence(envelope["result"])
  assert gc.validate_guarded_envelope(envelope) == envelope
  assert envelope["status"] == result["status"] == "PASS"
  assert envelope["spawn_count"] == 1
  assert result["environment"] == {
    "DEV": "AMD", "AMD_AQL": "0", "PROFILE": "0"}
  assert result["invocation_count"] == result["receipt_count"] == 1
  assert result["target_dispatch_submitted"] is False
  assert result["native_submit_call_count"] == 0
  assert result["timeline_rollback_applied"] is True
  assert result["terminal_child_required"] is True
  assert result["readback_performed"] is False
  assert result["numeric_validation_performed"] is False
  assert result["attestation_performed"] is False
  assert result["producer_attestation_performed"] is False
  assert result["promotion_evidence_eligible"] is False


def test_pm4_no_doorbell_accepts_exact_real_harness_receipt_shape():
  from extra.qk.mmq_frozen_staged_low_level_session import \
    PM4_NO_DOORBELL_CHECK_KEYS
  receipt = _no_doorbell_receipt()
  assert set(receipt["checks"]) == PM4_NO_DOORBELL_CHECK_KEYS
  assert gc._validate_pm4_no_doorbell_receipt(receipt) == receipt


@pytest.mark.parametrize("mutation", (
  "schema", "runtime_name", "argument_va", "qword", "user_data_pointer",
  "check_key", "pm4_hash",
))
def test_rehashed_pm4_no_doorbell_nested_pre_submit_tamper_is_rejected(
    mutation):
  child = copy.deepcopy(_guard_no_doorbell()["result"])
  pre_submit = child["receipt"]["pre_submit"]
  if mutation == "schema":
    pre_submit["schema"] = "tampered"
  elif mutation == "runtime_name":
    pre_submit["runtime_name"] = "other_kernel"
  elif mutation == "argument_va":
    pre_submit["argument_buffers"][0]["va"] += 0x1000
  elif mutation == "qword":
    pre_submit["kernarg_qwords"][0] += 0x1000
  elif mutation == "user_data_pointer":
    pre_submit["pm4_kernarg_user_data"]["pointer"] += 0x1000
  elif mutation == "check_key":
    pre_submit["checks"].pop("pm4_packet_stream_decoded")
  elif mutation == "pm4_hash":
    pre_submit["pm4_sha256"] = "C" * 64
  child["evidence_identity"] = gc._identity({
    key: value for key, value in child.items() if key != "evidence_identity"})
  with pytest.raises(ValueError, match="PM4 no-doorbell"):
    gc.validate_pm4_no_doorbell_evidence(child)


@pytest.mark.parametrize("receipts,mutation", [
  ([], None),
  ([_no_doorbell_receipt(), _no_doorbell_receipt()], None),
  ([_no_doorbell_receipt()], "bad"),
])
def test_pm4_no_doorbell_zero_two_or_bad_receipts_block(
    receipts, mutation):
  receipts = copy.deepcopy(receipts)
  if mutation == "bad":
    receipts[0]["all_checks_pass"] = False

  def builder(config, *, queue_mode, prefix_epochs, pm4_submit_policy,
                  pm4_no_doorbell_receipt_sink):
    runtime = fake_no_doorbell_builder(
      config, queue_mode=queue_mode, prefix_epochs=prefix_epochs,
      pm4_submit_policy=pm4_submit_policy,
      pm4_no_doorbell_receipt_sink=pm4_no_doorbell_receipt_sink)
    return gc.replace(
      runtime, session=FakeNoDoorbellSession(
        pm4_no_doorbell_receipt_sink, receipts))

  envelope = _guard_no_doorbell(builder)
  assert envelope["status"] == "BLOCKED"
  assert envelope["result"]["status"] == "BLOCKED"
  assert "failed closed" in envelope["result"]["exact_blocker"]
  assert envelope["result"]["retry_count"] == 0


def test_pm4_no_doorbell_health_and_fault_containment_fail_closed():
  launches = []
  preflight = _guard_no_doorbell(
    isolated_runner=lambda *args, **kwargs:
      launches.append((args, kwargs)) or pytest.fail("must not launch"),
    health_probe=lambda _env: False)
  assert preflight["status"] == "BLOCKED"
  assert preflight["spawn_count"] == 0 and launches == []

  faults = _guard_no_doorbell(
    fault_collector=lambda _started: (
      ["amdgpu fault"], {
        **_clear_fault_evidence(), "status": "FAULTS",
        "blocks": [{"lines": ["amdgpu fault"]}],
        "relevant_line_count": 1, "retained_line_count": 1}))
  assert faults["status"] == "BLOCKED"
  assert faults["result"]["status"] == "PASS"
  assert "kernel fault/reset marker" in faults["exact_blocker"]


def test_production_candidate_builder_diagnostic_defaults_are_nonexecuting_opt_in():
  signature = inspect.signature(gc.build_production_candidate_prefix_runtime)
  assert signature.parameters["pm4_submit_policy"].default == "execute"
  assert signature.parameters[
    "pm4_no_doorbell_receipt_sink"].default is None
  source = inspect.getsource(gc.build_production_candidate_prefix_runtime)
  assert "pm4_submit_policy=pm4_submit_policy" in source
  assert "pm4_no_doorbell_receipt_sink=pm4_no_doorbell_receipt_sink" in source


def test_pm4_prefix1_is_standalone_one_child_with_no_later_prerequisites():
  envelope = _guard(1)
  result = envelope["result"]
  assert envelope["status"] == result["status"] == "PASS"
  assert envelope["spawn_count"] == 1
  assert result["queue_mode"] == "PM4" and result["prefix_epochs"] == 1
  assert result["predecessor_evidence_identity"] is None
  assert result["c4_canary_identity"] == _sid("PM4-c4")
  assert "joint_c7" not in result and "AQL" not in json.dumps(result)
  assert result["promotion_evidence_eligible"] is False


def test_pm4_prefix1_production_builder_has_no_direct_route_dependency():
  source = inspect.getsource(gc.build_production_candidate_prefix_runtime)
  assert "build_direct_packed_objects" not in source
  assert "mmq_attn_qo_c8_runtime" not in source
  assert "resident_fp16 = Tensor(" in source
  assert "common_resident_fp16=resident_fp16" in source


def test_prefix3_and_full_are_separate_children_with_exact_predecessors():
  p1_envelope = _guard(1)
  p1, p1_ref = p1_envelope["result"], _freeze(p1_envelope)
  p3_envelope = _guard(3, prior=p1_ref)
  p3 = p3_envelope["result"]
  p3_ref = _freeze(p3_envelope)
  full_envelope = _guard(20, prior=p3_ref)
  full = full_envelope["result"]
  assert p3_envelope["spawn_count"] == full_envelope["spawn_count"] == 1
  assert p3["predecessor_evidence_identity"] == \
    p1_ref.envelope_evidence_identity
  assert full["predecessor_evidence_identity"] == \
    p3_ref.envelope_evidence_identity
  assert [p1["prefix_epochs"], p3["prefix_epochs"], full["prefix_epochs"]] == \
    [1, 3, 20]


def test_persisted_predecessor_authority_mismatch_blocks_before_invocation():
  p1_envelope = _guard(1)
  called = []
  def mismatched_builder(config, *, queue_mode, prefix_epochs):
    runtime = fake_candidate_builder(
      config, queue_mode=queue_mode, prefix_epochs=prefix_epochs)
    return gc.replace(runtime, workload_identity=_sid("other-workload"),
                      session=FakeSession(queue_mode, prefix_epochs),
                      readback=lambda output: called.append(output) or output)
  result = gc.run_candidate_prefix_child(gc.CandidatePrefixRequest(
    {}, "PM4", 3, _freeze(p1_envelope), None, mismatched_builder))
  assert result["status"] == "BLOCKED"
  assert "differs from persisted predecessor" in result["exact_blocker"]
  assert called == []


@pytest.mark.parametrize(
  "prefix,prior,error",
  ((3, None, "frozen reference"),
   (20, None, "frozen reference")),
)
def test_escalation_without_persisted_predecessor_fails_before_runtime(
    prefix, prior, error):
  called = []
  request = gc.CandidatePrefixRequest(
    {}, "PM4", prefix, prior, None,
    lambda *args, **kwargs: called.append(True))
  result = gc.run_candidate_prefix_child(request)
  assert result["status"] == "BLOCKED" and error in result["exact_blocker"]
  assert called == []


def test_first_aql_prefix_requires_complete_pm4_artifact():
  blocked = gc.run_candidate_prefix_child(_candidate_request(queue="AQL"))
  assert blocked["status"] == "BLOCKED"
  p1_envelope = _guard(1)
  p3_envelope = _guard(3, prior=_freeze(p1_envelope))
  full_envelope = _guard(20, prior=_freeze(p3_envelope))
  passed = gc.run_candidate_prefix_child(
    _candidate_request(queue="AQL", cross=_freeze(full_envelope)))
  assert passed["status"] == "PASS"
  assert passed["predecessor_evidence_identity"] == \
    full_envelope["evidence_identity"]


@pytest.mark.parametrize(
  "mode,blocker",
  (("timeout", "timed out"), ("fault", "kernel fault/reset"),
   ("post_health", "postflight health failed")),
)
def test_each_stage_has_own_timeout_fault_health_envelope(mode, blocker):
  health_calls = {"count": 0}
  def health(_env):
    health_calls["count"] += 1
    return not (mode == "post_health" and health_calls["count"] == 2)
  def isolated(function, *, args, **kwargs):
    if mode == "timeout":
      return SimpleNamespace(
        status="timed_out", result=None, timed_out=True,
        error="timeout", elapsed_seconds=1.0)
    return _isolated(function, args=args, **kwargs)
  def faults(started):
    return (["fault"], {}) if mode == "fault" else _faults(started)
  result = _guard(
    isolated_runner=isolated, health_probe=health, fault_collector=faults)
  assert result["status"] == "BLOCKED"
  assert blocker in result["exact_blocker"]


def test_fault_window_contains_preflight_child_and_postflight(monkeypatch):
  events = []
  monkeypatch.setattr(
    gc.time, "time",
    lambda: events.append("fault_window_start") or 123.0)

  def health(_env):
    events.append("health")
    return True

  def isolated(function, *, args, **kwargs):
    events.append("child")
    return _isolated(function, args=args, **kwargs)

  def faults(started):
    events.append(("collect_since", started))
    return ["preflight-period fault"], {
      **_clear_fault_evidence(), "status": "FAULTS"}

  envelope = _guard(
    isolated_runner=isolated, health_probe=health,
    fault_collector=faults)
  assert events == [
    "fault_window_start", "health", "child", "health",
    ("collect_since", 123.0)]
  assert envelope["status"] == "BLOCKED"
  assert "kernel fault/reset" in envelope["exact_blocker"]


def test_tampered_stage_child_identity_is_rejected():
  def isolated(function, *, args, **kwargs):
    result = _isolated(function, args=args, **kwargs)
    result.result["prefix_epochs"] = 3
    return result
  envelope = _guard(isolated_runner=isolated)
  assert envelope["status"] == "BLOCKED"
  assert "content identity differs" in envelope["exact_blocker"]


def test_rehashed_candidate_with_nested_mismatch_is_rejected():
  child = copy.deepcopy(_guard(1)["result"])
  child["comparison"]["status"] = "mismatch"
  child["comparison"]["mismatch_count"] = 1
  child["evidence_identity"] = gc._identity({
    key: value for key, value in child.items() if key != "evidence_identity"})
  def isolated(_function, *, args, **_kwargs):
    return SimpleNamespace(
      status="passed", result=child, timed_out=False,
      error=None, elapsed_seconds=0.1)
  envelope = _guard(isolated_runner=isolated)
  assert envelope["status"] == "BLOCKED"
  assert "nested child evidence failed closed" in envelope["exact_blocker"]


def test_failed_candidate_attempt_retains_raw_mismatch_and_hashes():
  def mismatch_builder(config, *, queue_mode, prefix_epochs):
    runtime = fake_candidate_builder(
      config, queue_mode=queue_mode, prefix_epochs=prefix_epochs)
    return gc.replace(runtime, comparator=lambda _got, _ref:
                      _mismatch_comparison())
  result = gc.run_candidate_prefix_child(gc.CandidatePrefixRequest(
    {}, "PM4", 1, None, None, mismatch_builder))
  assert result["status"] == "BLOCKED"
  attempt = result["failed_attempt"]
  assert attempt["phase"] == "output_readback_and_comparison"
  assert attempt["comparison"]["status"] == "mismatch"
  assert attempt["comparison"]["mismatch_count"] == 1
  assert len(attempt["output_sha256"]) == 64
  assert len(attempt["reference_sha256"]) == 64


def test_failed_candidate_attempt_retains_low_level_dispatch_subphase():
  class FaultingSession:
    def attest_post_sync(self, invocation, queue):
      raise AssertionError("faulting invocation must not reach attestation")

    def invoke(self, prefix_epochs):
      failure = RuntimeError("injected target dispatch fault")
      failure.frozen_staged_low_level_failure = {
        "schema": gc.LOW_LEVEL_INVOCATION_FAILURE_SCHEMA,
        "phase": "epoch_dispatch",
        "subphase":
          "runtime_call_raised_after_kernarg_capture_before_return",
        "epoch": 0, "queue_mode": "PM4",
        "family_identity": _sid("family"),
        "candidate_executable_identity":
          gc._candidate_executable_identity_from_parts(
            _sid("family"), "1" * 64, "2" * 64),
        "input_identity": _sid("input"), "program_key": "1" * 64,
        "binary_sha256": "2" * 64,
        "runtime_observation": {
          "queue_mode": "PM4", "runtime_class": "FakeRuntime",
          "runtime_name": "fake_kernel", "runtime_device": "AMD",
          "runtime_object_identity": 7,
          "runtime_device_identity_exact": True,
          "runtime_cache_binding_exact": True,
          "program_key": "1" * 64, "binary_sha256": "2" * 64,
          "library_va": 0x800000, "library_nbytes": 0x2000,
          "entry_va": 0x800100,
        },
        "fixed_five_vas": [
          0x7F8800001000, 0x7F8800002000, 0x7F8800003000,
          0x7F8800004000, 0x7F8800005000],
        "dispatch_failure": {
          "schema": "tinygrad.mmq_q4k_q8_1.runtime_dispatch_failure.v1",
          "failure_boundary":
            "runtime_call_raised_after_kernarg_capture_before_return",
          "wait": True, "epoch": 0,
          "launch": {
            "epoch": 0, "global_size": [136, 4, 1],
            "local_size": [256, 1, 1],
            "arguments": [], "kernarg": {
              "pointer_words": [
                0x7F8800001000, 0x7F8800002000, 0x7F8800003000,
                0x7F8800004000, 0x7F8800005000]}},
          "authoritative_qword_snapshot": "pre_submit",
          "pre_submit": {
            "schema":
              "tinygrad.mmq_q4k_q8_1.pm4_pre_submit_snapshot.v1",
            "capture_point":
              "AMDComputeQueue._submit_after_complete_command_construction_"
              "before_ring_copy_and_doorbell",
            "runtime_object_identity": 7,
            "runtime_class": "FakeRuntime",
            "runtime_name": "fake_kernel", "runtime_device": "AMD",
            "kernarg_va": 0x7F88ABCDEF00,
            "pm4_kernarg_user_data": {
              "packet_dword_offset": 11, "register_index": 0,
              "low_dword": 0xABCDEF00, "high_dword": 0x7F88,
              "pointer": 0x7F88ABCDEF00,
            },
            "kernarg_qwords": [
              0x7F8800001000, 0x7F8800002000, 0x7F8800003000,
              0x7F8800004000, 0x7F8800005000],
            "argument_buffers": [
              {"slot": slot, "va": va, "size": 4096}
              for slot, va in enumerate((
                0x7F8800001000, 0x7F8800002000, 0x7F8800003000,
                0x7F8800004000, 0x7F8800005000))],
            "checks": {"pre_submit_exact": True},
            "all_checks_pass": True,
          },
          "checks": {}, "all_authority_checks_pass": True,
        },
      }
      raise failure

  def faulting_builder(config, *, queue_mode, prefix_epochs):
    runtime = fake_candidate_builder(
      config, queue_mode=queue_mode, prefix_epochs=prefix_epochs)
    return gc.replace(runtime, session=FaultingSession())

  result = gc.run_candidate_prefix_child(gc.CandidatePrefixRequest(
    {}, "PM4", 1, None, None, faulting_builder))
  assert result["status"] == "BLOCKED"
  assert result["exception"] == "RuntimeError"
  assert "injected target dispatch fault" in result["exact_blocker"]
  attempt = result["failed_attempt"]
  assert attempt["phase"] == "invocation"
  assert attempt["invocation_subphase"] == \
    "runtime_call_raised_after_kernarg_capture_before_return"
  failure = attempt["invocation_failure"]
  assert failure["phase"] == "epoch_dispatch"
  assert failure["runtime_observation"]["runtime_object_identity"] == 7
  assert failure["dispatch_failure"]["launch"]["global_size"] == \
    [136, 4, 1]
  assert failure["dispatch_failure"]["launch"]["kernarg"]["pointer_words"] == \
    failure["fixed_five_vas"]
  assert failure["dispatch_failure"]["pre_submit"]["kernarg_qwords"] == \
    failure["fixed_five_vas"]
  assert result["no_retry"] is result["no_fallback"] is True
  assert result["retry_count"] == 0


def _direct_evidence(queue="PM4", candidate_full_ref=None):
  manifest_payload = {
    "artifact_schema": "fake", "queue_mode": queue,
    "variables": {}, "calls": [], "programs": [{}],
    "direct_packed_program_ordinals": [0],
    "aggregate_binary_sha256": "3" * 64}
  manifest = {
    **manifest_payload, "executable_identity": gc._identity(manifest_payload)}
  observation_payload = {
    "schema": "observation", "status": "PASS", "queue_mode": queue,
    "manifest": manifest, "runtime_programs": [],
    "runtime_cache_join_verified": True,
    "profile_code_objects_verified": 0,
    "post_sync_attestation": True}
  observation = {
    **observation_payload,
    "observation_identity": gc._identity(observation_payload)}
  executable_payload = {
    "schema": "fallback", "status": "PASS", "queue_mode": queue,
    "input_identity": _sid("input"), "workload_identity": _sid("workload"),
    "executable_identity": manifest["executable_identity"]}
  executable = {
    **executable_payload,
    "evidence_identity": gc._identity(executable_payload)}
  comparison = _comparison(None, None)
  attempt = {
    "phase": "complete", "executable_observation": observation,
    "executable_evidence": executable, "comparison": comparison,
    "output_sha256": "4" * 64, "reference_sha256": "5" * 64}
  payload = {
    "schema": gc.DIRECT_SCHEMA, "queue_mode": queue, "status": "PASS",
    "exact_blocker": None, "family_identity": _sid("family"),
    "no_retry": True, "retry_count": 0, "no_fallback": True,
    "fixture_identity": _sid("fixture"),
    "workload_identity": _sid("workload"),
    "input_identity": _sid("input"), "logical_q4_identity": _sid("logical-q4"),
    "resident_fp16_activation_identity": _sid("resident-fp16"),
    "c4_canary_identity": _sid(f"{queue}-direct-c4"),
    "direct_executable_identity": manifest["executable_identity"],
    "request_identity": _sid(f"{queue}-direct-request"),
    "config_identity": gc._identity({}),
    "candidate_full_evidence_identity":
      _sid("candidate-full") if candidate_full_ref is None else
      candidate_full_ref.envelope_evidence_identity,
    "comparison_authority":
      "independent_dense_fp16_activation_q4k_dequant_oracle_v1",
    "post_sync_before_observation_and_readback": True,
    "readback_performed": True, "comparison": comparison,
    "output_sha256": "4" * 64, "reference_sha256": "5" * 64,
    "executable_observation": observation,
    "executable_evidence": executable, "attempt": attempt,
    "promotion_evidence_eligible": False,
  }
  return {**payload, "evidence_identity": gc._identity(payload)}


def fake_transition_worker(
    _config, *, queue_mode, sequence, candidate_evidence, direct_evidence):
  steps = []
  for index, (route, prefix) in enumerate(sequence):
    payload = {
      "ordinal": index, "route": route, "prefix_epochs": prefix,
      "status": "PASS", "comparison_status": "pass",
      "mismatch_count": 0, "post_route_sync": True}
    steps.append({**payload, "evidence_identity": gc._identity(payload)})
  payload = {
    "status": "PASS", "queue_mode": queue_mode,
    "sequence": [
      {"route": route, "prefix_epochs": prefix}
      for route, prefix in sequence],
    "all_outputs_correct": True, "post_route_sync_each_step": True,
    "steps": steps,
  }
  return {**payload, "evidence_identity": gc._identity(payload)}


def test_every_transition_call_uses_a_fresh_child():
  p1_envelope = _guard(1)
  p3_envelope = _guard(3, prior=_freeze(p1_envelope))
  full_envelope = _guard(20, prior=_freeze(p3_envelope))
  full_ref = _freeze(full_envelope)
  direct = _direct_evidence(candidate_full_ref=full_ref)
  direct_ref = _freeze(_pass_envelope(direct))
  launches = []
  def isolated(function, *, args, **kwargs):
    launches.append(args[0].sequence_name)
    return _isolated(function, args=args, **kwargs)
  results = {
    name: gc.run_guarded_transition(
      config={}, queue_mode="PM4", sequence_name=name,
      candidate_full_evidence=full_ref, direct_evidence=direct_ref,
      worker=fake_transition_worker, isolated_runner=isolated,
      health_probe=lambda _env: True, fault_collector=_faults)
    for name in gc.TRANSITION_SEQUENCES}
  assert launches == list(gc.TRANSITION_SEQUENCES)
  assert all(row["status"] == "PASS" and row["spawn_count"] == 1
             for row in results.values())


def test_rehashed_transition_with_nested_bad_step_is_rejected():
  p1_envelope = _guard(1)
  p3_envelope = _guard(3, prior=_freeze(p1_envelope))
  full_envelope = _guard(20, prior=_freeze(p3_envelope))
  full_ref = _freeze(full_envelope)
  direct = _direct_evidence(candidate_full_ref=full_ref)
  direct_ref = _freeze(_pass_envelope(direct))
  row = gc.run_transition_child(gc.TransitionRequest(
    {}, "PM4", "candidate_candidate", full_ref, direct_ref,
    fake_transition_worker))
  row["raw_transition"]["steps"][0]["comparison_status"] = "mismatch"
  step = row["raw_transition"]["steps"][0]
  step["evidence_identity"] = gc._identity({
    key: value for key, value in step.items() if key != "evidence_identity"})
  raw = row["raw_transition"]
  raw["evidence_identity"] = gc._identity({
    key: value for key, value in raw.items() if key != "evidence_identity"})
  row["attempt"]["raw_transition"] = raw
  row["evidence_identity"] = gc._identity({
    key: value for key, value in row.items() if key != "evidence_identity"})
  with pytest.raises(ValueError, match="transition step 0 differs"):
    gc.validate_transition_evidence(
      row, queue_mode="PM4", sequence_name="candidate_candidate")


def test_rehashed_direct_with_false_runtime_join_is_rejected():
  row = _direct_evidence()
  row["executable_observation"]["runtime_cache_join_verified"] = False
  observation = row["executable_observation"]
  observation["observation_identity"] = gc._identity({
    key: value for key, value in observation.items()
    if key != "observation_identity"})
  row["attempt"]["executable_observation"] = observation
  row["evidence_identity"] = gc._identity({
    key: value for key, value in row.items() if key != "evidence_identity"})
  with pytest.raises(ValueError, match="executable observation differs"):
    gc.validate_direct_evidence(row, queue_mode="PM4")


def test_exact_prefix1_candidate_oracle_hash():
  from extra.qk.mmq_exact_role_spec import exact_role_spec
  from extra.qk.mmq_ffn_gate_up_c8_runtime import \
    rebuild_ffn_gate_up_v2_fixture
  fixture = rebuild_ffn_gate_up_v2_fixture(
    exact_role_spec("ffn_gate_up"), json.loads(FIXTURE.read_text()))
  output = gc.ffn_gate_up_consumer_prefix_reference(fixture, 1)
  assert hashlib.sha256(output.tobytes()).hexdigest() == \
    "5bd7a149630fda2da375c33306425398618bb8e58b8ae02fcc982803622a8d7b"


def test_direct_child_requires_real_capture_not_configured_identity():
  p1_envelope = _guard(1)
  p3_envelope = _guard(3, prior=_freeze(p1_envelope))
  full_envelope = _guard(20, prior=_freeze(p3_envelope))
  called = []
  def bad_builder(_config, *, queue_mode):
    called.append(queue_mode)
    return SimpleNamespace(
      queue_mode=queue_mode, family_identity=_sid("family"),
      input_identity=_sid("input"), c4_canary_identity=_sid("c4"),
      # Deliberately no capture or bindings.
      direct_executable_identity=_sid("configured"))
  result = gc.run_direct_correctness_child(
    gc.DirectCorrectnessRequest(
      {}, "PM4", _freeze(full_envelope), bad_builder))
  assert called == ["PM4"] and result["status"] == "BLOCKED"
  assert "typed runtime" in result["exact_blocker"]


def test_exhaustive_composer_is_cpu_only_and_requires_complete_artifacts():
  p1_envelope = _guard(1)
  p3_envelope = _guard(3, prior=_freeze(p1_envelope))
  full_envelope = _guard(20, prior=_freeze(p3_envelope))
  transitions = {}
  for name, sequence in gc.TRANSITION_SEQUENCES.items():
    payload = {
      "schema": gc.TRANSITION_SCHEMA, "queue_mode": "PM4",
      "sequence_name": name, "status": "PASS",
      "no_retry": True, "retry_count": 0, "no_fallback": True,
      "sequence": [
        {"route": route, "prefix_epochs": prefix}
        for route, prefix in sequence],
      "promotion_evidence_eligible": False,
    }
    transitions[name] = {
      **payload, "evidence_identity": gc._identity(payload)}
  # Missing AQL is rejected without calling an execution function.
  with pytest.raises(ValueError, match="both queue modes"):
    gc.compose_guarded_correctness_artifacts(
      candidate_by_queue={"PM4": {
        1: _freeze(p1_envelope), 3: _freeze(p3_envelope),
        20: _freeze(full_envelope)}},
      direct_by_queue={"PM4": None},
      transitions_by_queue={"PM4": {}},
      joint_c7_evidence={"status": "PASS",
                         "promotion_evidence_eligible": False})


def test_real_spawn_roundtrip_for_standalone_pm4_prefix1():
  from tinygrad.runtime.process_isolated import run_isolated
  isolated = run_isolated(
    gc.run_candidate_prefix_child, args=(_candidate_request(),),
    timeout_seconds=10, start_method="spawn")
  assert isolated.status == "passed" and isolated.timed_out is False
  assert isolated.result["status"] == "PASS"


def test_frozen_stage_evidence_is_no_replace_and_detects_file_tampering(
    tmp_path):
  candidate = _guard(1)
  path = tmp_path / "candidate.json"
  reference = gc.freeze_correctness_evidence(path, candidate)
  assert gc.load_frozen_correctness_evidence(reference) == candidate
  assert reference.envelope_evidence_identity == \
    candidate["evidence_identity"]
  with pytest.raises(FileExistsError, match="refusing to replace"):
    gc.freeze_correctness_evidence(path, candidate)
  path.write_text(path.read_text().replace('"status": "PASS"',
                                           '"status": "BLOCKED"', 1))
  with pytest.raises(ValueError, match="file content differs"):
    gc.load_frozen_correctness_evidence(reference)


def test_child_stage_or_blocked_envelope_can_never_freeze(tmp_path):
  passed = _guard(1)
  with pytest.raises(ValueError, match="envelope"):
    gc.freeze_correctness_evidence(
      tmp_path / "raw-child.json", passed["result"])
  blocked = _guard(
    health_probe=lambda _env: False,
    isolated_runner=lambda *args, **kwargs: pytest.fail("must not launch"))
  assert blocked["status"] == "BLOCKED"
  with pytest.raises(ValueError, match="identity/state"):
    gc.freeze_correctness_evidence(
      tmp_path / "blocked-envelope.json", blocked)


@pytest.mark.parametrize("mutation,match", [
  (lambda row: row.update(health_before=False), "containment"),
  (lambda row: row.update(timed_out=True), "containment"),
  (lambda row: row["kernel_fault_evidence"].update(status="FAULTS"),
   "exact CLEAR"),
  (lambda row: row.update(request_identity=_sid("wrong-request")),
   "request_identity binding"),
])
def test_rehashed_unhealthy_fault_timeout_or_binding_envelope_never_admits(
    tmp_path, mutation, match):
  envelope = copy.deepcopy(_guard(1))
  mutation(envelope)
  _rehash(envelope)
  with pytest.raises(ValueError, match=match):
    gc.freeze_correctness_evidence(tmp_path / "forged.json", envelope)


@pytest.mark.parametrize("target", [
  "candidate", "attempt", "producer", "diagnostic", "hashes", "attestation",
])
def test_rehashed_extra_nested_candidate_fields_are_rejected(tmp_path, target):
  envelope = copy.deepcopy(_guard(1))
  child = envelope["result"]
  if target == "candidate":
    child["junk"] = True
  elif target == "attempt":
    child["attempt"]["junk"] = True
  elif target == "producer":
    child["q8_producer"]["junk"] = True
    _rehash(child["q8_producer"])
    child["attempt"]["q8_producer"] = child["q8_producer"]
  elif target == "diagnostic":
    child["q8_producer"]["fixture_diagnostic"]["junk"] = True
    _rehash(child["q8_producer"])
    child["attempt"]["q8_producer"] = child["q8_producer"]
  elif target == "hashes":
    child["q8_producer"]["consumer_reference_q8_sha256"]["junk"] = "c" * 64
    child["consumer_reference_q8_sha256"] = \
      child["q8_producer"]["consumer_reference_q8_sha256"]
    _rehash(child["q8_producer"])
    child["attempt"]["q8_producer"] = child["q8_producer"]
  else:
    attestation = child["attestation"]
    attestation["junk"] = True
    _rehash(attestation, "observation_identity")
    child["attempt"]["attestation"] = attestation
  _rehash(child)
  _rehash(envelope)
  with pytest.raises(ValueError, match="fields differ"):
    gc.freeze_correctness_evidence(tmp_path / f"{target}.json", envelope)


def test_rehashed_huge_tolerance_and_forged_executable_are_rejected(tmp_path):
  envelope = copy.deepcopy(_guard(1))
  child = envelope["result"]
  child["comparison"]["rtol"] = 1e9
  child["attempt"]["comparison"] = child["comparison"]
  _rehash(child)
  _rehash(envelope)
  with pytest.raises(ValueError, match="rtol"):
    gc.freeze_correctness_evidence(tmp_path / "tolerance.json", envelope)

  envelope = copy.deepcopy(_guard(1))
  child = envelope["result"]
  forged = _sid("forged-candidate-executable")
  child["candidate_executable_identity"] = forged
  child["attempt"]["candidate_executable_identity"] = forged
  child["attestation"]["candidate_executable_identity"] = forged
  _rehash(child["attestation"], "observation_identity")
  child["attempt"]["attestation"] = child["attestation"]
  _rehash(child)
  _rehash(envelope)
  with pytest.raises(ValueError, match="executable derivation"):
    gc.freeze_correctness_evidence(tmp_path / "executable.json", envelope)
