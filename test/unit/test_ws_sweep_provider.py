from __future__ import annotations

import pytest

from extra.qk.ws_sweep_provider import (IsolationAttestation, PointOutcome, SweepRequestError,
                                        consume_ws_sweep_request)


def request():
  return {"schema": "boltbeam.ws_sweep_request.v1", "target_id": "gfx1100",
          "execution": {"isolated": True, "health_preflight": True, "health_postflight": True,
                        "record_system_clocks_compiler": True, "warmups": 2},
          "points": [{"bytes": 4096, "kind": "copy", "mode": "bandwidth", "traffic": "read",
                      "temperatures": ["warm", "cold"], "repeats": 2}],
          "lds_probe": {"bytes": 32768, "kind": "lds_resident", "mode": "bandwidth", "traffic": "mixed",
                        "temperatures": ["warm"], "repeats": 2, "reuse": 256}}


class FakeRunner:
  isolation = IsolationAttestation(True, "spawn")
  def __init__(self, outcomes=None): self.outcomes, self.calls = list(outcomes or []), []
  def run_point(self, point, *, timeout_seconds):
    self.calls.append((point, timeout_seconds))
    if self.outcomes: return self.outcomes.pop(0)
    samples = ({"sustained_gbs": [10, 11], "cold_gbs": [8, 9]} if "cold" in point.temperatures
               else {"sustained_gbs": [20, 21]})
    return PointOutcome("passed", samples, "snap-1", {"name": "clang", "version": "1"},
                        {"unit": "MHz", "values": [1000]}, {"unit": "C", "values": [50]},
                        {"status": "healthy"}, {"status": "healthy"})


def test_emits_classifier_shape_identity_units_and_evidence():
  runner = FakeRunner()
  result = consume_ws_sweep_request(request(), runner, timeout_seconds=3)
  assert result["schema"] == "boltbeam.ws_sweep_samples.v1"
  assert result["system_snapshot_id"] == result["identity"]["system_snapshot_id"] == "snap-1"
  assert result["points"][0]["sustained_gbs"] == [10.0, 11.0]
  assert result["lds_probe"]["sustained_gbs"] == [20.0, 21.0]
  assert result["units"]["bandwidth"] == "GB/s"
  assert len(result["measurement_evidence"]) == 2
  assert all(call[1] == 3 for call in runner.calls)


def test_timeout_is_typed_unsupported_and_does_not_fabricate_samples():
  timeout = PointOutcome("timeout", error="hard deadline exceeded")
  runner = FakeRunner([timeout, timeout])
  result = consume_ws_sweep_request(request(), runner)
  assert result["status"] == "unsupported" and result["points"] == []
  assert {row["code"] for row in result["unsupported_outcomes"]} == {"timeout"}


def test_health_evidence_is_required_and_failed_health_is_typed_unsupported():
  bad = PointOutcome("passed", {"sustained_gbs": [1, 2], "cold_gbs": [1, 2]}, "snap", {"name": "c"},
                     {"unit": "MHz"}, {"unit": "C"}, {"status": "healthy"}, None)
  with pytest.raises(SweepRequestError, match="health_postflight"):
    consume_ws_sweep_request(request(), FakeRunner([bad]))
  unhealthy = PointOutcome("passed", {"sustained_gbs": [1, 2], "cold_gbs": [1, 2]}, "snap", {"name": "c"},
                           {"unit": "MHz"}, {"unit": "C"}, {"status": "healthy"},
                           {"status": "failed", "reason": "device vanished"})
  result = consume_ws_sweep_request(request(), FakeRunner([unhealthy, unhealthy]))
  assert result["status"] == "unsupported"
  assert {row["code"] for row in result["unsupported_outcomes"]} == {"health_postflight_failed"}


def test_unregistered_adapter_and_unsafe_runner_are_fail_closed():
  result = consume_ws_sweep_request(request(), None)
  assert result["status"] == "unsupported"
  assert result["unsupported_outcomes"][0]["code"] == "adapter_unregistered"
  runner = FakeRunner(); runner.isolation = IsolationAttestation(True, "fork")
  with pytest.raises(SweepRequestError, match="spawn isolation"):
    consume_ws_sweep_request(request(), runner)


def test_request_safety_requirements_are_validated_before_runner_call():
  req = request(); req["execution"]["health_preflight"] = False
  runner = FakeRunner()
  with pytest.raises(SweepRequestError, match="pre/post health"):
    consume_ws_sweep_request(req, runner)
  assert runner.calls == []
