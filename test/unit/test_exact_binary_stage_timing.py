import argparse, json

import numpy as np
import pytest

from extra.qk.prefill import exact_binary_stage_timing as timing


class _Dispatch:
  def __init__(self, elapsed): self.elapsed, self.calls = elapsed, []
  def dispatch(self, *args): self.calls.append(args); return self.elapsed
  def close(self): pass


class _Buffer:
  def __init__(self, name): self.name, self.freed = name, False
  def get_buf(self, device): return (self.name, device)
  def is_allocated(self): return not self.freed
  def deallocate(self): self.freed = True


def test_stage_executable_records_separate_device_times_and_exact_sum():
  producer, mmq = _Dispatch(0.00125), _Dispatch(0.0035)
  intermediates = [_Buffer(name) for name in ("values", "scales", "sums")]
  target = timing._StageTimedExecutable(producer, mmq, *intermediates, "AMD")
  total = target.dispatch(_Buffer("out"), _Buffer("q4"), _Buffer("activation"))
  assert target.last_stage_times == {"producer": 0.00125, "mmq": 0.0035, "total": total}
  assert total == target.last_stage_times["producer"] + target.last_stage_times["mmq"]
  assert producer.calls[0][-1] == ("activation", "AMD")
  assert mmq.calls[0][0] == ("out", "AMD")


@pytest.mark.parametrize("value", (None, float("nan"), -1.0))
def test_stage_executable_rejects_missing_or_dishonest_device_time(value):
  target = timing._StageTimedExecutable(_Dispatch(value), _Dispatch(1.0),
    _Buffer("v"), _Buffer("s"), _Buffer("sum"), "AMD")
  with pytest.raises(ValueError, match="device time"):
    target.dispatch(_Buffer("out"), _Buffer("q4"), _Buffer("activation"))


def test_run_reports_raw_stage_samples_and_binary_resource_identity(monkeypatch, tmp_path):
  candidate = tmp_path / "candidate.json"
  candidate.write_text(json.dumps({"payload": {"opaque": True}, "canonical_identity": "cid"}))
  admission = object()
  evidence = {"pipeline_binary_sha256": "pipe", "producer_binary_sha256": "prod", "binary_sha256": "mmq",
    "producer_resource_summary": {"vgpr": 12}, "resource_summary": {"vgpr": 24}}
  monkeypatch.setattr(timing, "prepare_q4k_q8_five_buffer_pipeline_compile",
    lambda payload, identity: (argparse.Namespace(admission=admission), evidence))
  monkeypatch.setattr(timing, "load_q4k_q8_five_buffer_pipeline_npz", lambda path, adm: (
    {"q4_packed_words": np.arange(2, dtype=np.uint32), "activation": np.arange(2, dtype=np.float32)},
    np.zeros(2, dtype=np.float32), {"input_artifact_sha256": "i", "reference_sha256": "r",
      "content_sha256": {"q4_packed_words": "q", "activation": "a", "reference": "r"}}))
  monkeypatch.setattr(timing, "make_tiny_health_probe", lambda **kwargs: object())
  values = iter(((1.0, 2.0), (9.0, 8.0), (0.1, 0.2), (0.4, 0.5)))
  def launch(**kwargs):
    producer, mmq = next(values); stages = {"producer": producer, "mmq": mmq, "total": producer + mmq}
    return {"passed": True, "guarded": {"stage_device_seconds": stages}}
  args = argparse.Namespace(candidate=str(candidate), input_npz="input.npz", warmups=1, rounds=2,
                            timeout_seconds=4.0, rtol=0.0, atol=0.0)
  report = timing.run(args, launch=launch)
  assert report["passed"] is True
  assert report["measurement"]["samples"] == [
    {"producer": .1, "mmq": .2, "total": .1 + .2},
    {"producer": .4, "mmq": .5, "total": .4 + .5}]
  assert report["identity"]["producer_binary_sha256"] == "prod"
  assert report["identity"]["mmq_binary_sha256"] == "mmq"
  assert report["resources"] == {"producer": {"vgpr": 12}, "mmq": {"vgpr": 24}}


def test_candidate_contract_and_run_bounds_fail_closed(tmp_path):
  candidate = tmp_path / "candidate.json"; candidate.write_text(json.dumps({"payload": {}}))
  with pytest.raises(ValueError, match="exactly"):
    timing._candidate(candidate)
  args = argparse.Namespace(candidate=str(candidate), input_npz="x", warmups=0, rounds=timing.MAX_ROUNDS + 1,
                            timeout_seconds=1.0, rtol=0.0, atol=0.0)
  with pytest.raises(ValueError, match="bounded"):
    timing.run(args)


def test_bundle_rejects_resource_drift_before_runtime(monkeypatch):
  required = {"canonical_identity": "cid", "abi_digest": "abi", "compile_target": "target", "target": "gfx",
    "source_sha256": "source", "binary_sha256": "mmq", "producer_source_sha256": "producer-source",
    "producer_binary_sha256": "producer", "producer_resource_summary": {"vgpr": 1},
    "pipeline_binary_sha256": "pipeline", "program_count": 2, "execution_input_format": "fp32_activation"}
  contract = {**required, "enabled": True, "reject_sha256_mismatch_before_dispatch": True}
  parent = {**required, "resource_summary": {"vgpr": 2},
            "child_recompile_binary_identity_contract": contract}
  child = {**required, "resource_summary": {"vgpr": 3}}
  monkeypatch.setattr(timing, "prepare_q4k_q8_five_buffer_pipeline_compile",
                      lambda *args, **kwargs: (object(), child))
  with pytest.raises(ValueError, match="resource_summary differs"):
    timing.build_stage_timed_bundle(payload={}, canonical_identity="cid", compile_evidence=parent,
                                    compile_target="target")
