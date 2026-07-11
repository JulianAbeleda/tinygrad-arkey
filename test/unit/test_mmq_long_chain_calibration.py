import pytest
import json
from pathlib import Path

from extra.qk.mmq_long_chain_calibration import join_long_chain_modes, validate_long_chain_calibration


def _mode(mode, ms, binary="a" * 64, counters=None):
  return {"schema": "tinygrad.mmq_long_chain_calibration.v1", "provenance_class": "generated_microbenchmark",
    "mode": mode, "system_snapshot_id": "sha256:" + "b" * 64, "cases": [{"case_id": "dependent_valu.wg96.n1024",
    "family": "dependent_valu", "chain_length": 1024, "binary_sha256": binary, "median_ms": ms,
    "counters": counters}], "candidate_timing_used_for_fit": False, "production_dispatch_changed": False}


def test_mode_join_derives_sq_to_wall_without_candidate_data():
  counters = {"SQ_WAVES": 96, "SQ_WAVE_CYCLES": 960000, "SQ_BUSY_CYCLES": 480000, "SQ_WAIT_ANY": 900000}
  result = join_long_chain_modes(_mode("auto", 1.0), _mode("profile_standard", 2.0, counters=counters))
  validate_long_chain_calibration(result)
  assert result["joined_cases"][0]["profile_to_auto_ratio"] == 2.0
  assert result["joined_cases"][0]["aggregate_wave_cycles_per_wall_ns"] == 0.48


def test_mode_join_rejects_system_and_binary_mismatch():
  profile = _mode("profile_standard", 1.0, counters={"SQ_WAVES": 1, "SQ_WAVE_CYCLES": 1, "SQ_BUSY_CYCLES": 1, "SQ_WAIT_ANY": 1})
  auto = _mode("auto", 1.0); auto["system_snapshot_id"] = "sha256:" + "c" * 64
  with pytest.raises(ValueError, match="system snapshot"): join_long_chain_modes(auto, profile)
  auto["system_snapshot_id"] = profile["system_snapshot_id"]; auto["cases"][0]["binary_sha256"] = "c" * 64
  with pytest.raises(ValueError, match="binary mismatch"): join_long_chain_modes(auto, profile)


def test_committed_long_chain_artifact_preserves_samples_and_compile_blockers():
  path = Path(__file__).resolve().parents[2] / "bench/prefill-14b-mmq-machine-search/long-chain-calibration-v1-20260711.json"
  artifact = json.loads(path.read_text())
  validate_long_chain_calibration(artifact)
  assert artifact["protocol"]["wall_samples"] == 30
  assert [row["chain_length"] for row in artifact["joined_cases"]] == [128, 256, 512]
  assert all(len(row["samples_ms"]) == 30 for mode in artifact["modes"].values() for row in mode["cases"])
  blocked = {row["chain_length"]: row for row in artifact["requested_long_chains"]}
  assert blocked[1024]["auto_status"] == "blocked_compile_sigsegv"
  assert blocked[4096]["auto_status"] == blocked[16384]["auto_status"] == "blocked_compile_timeout"
  assert artifact["candidate_timing_used_for_fit"] is False
