from __future__ import annotations

import json

import pytest

from extra.qk.mmq_amd_counter_capability import (
  SCHEMA as CAPABILITY_SCHEMA, parse_rocprof_counter_names, probe_amd_counter_capabilities, validate_counter_capability,
)
from extra.qk.mmq_amd_pmc import (
  DEFAULT_GROUPS, SCHEMA as PMC_SCHEMA, classify_liveness, collect_kernel_pmc, collect_mmq_pmc, probe_rocprof_fallback, validate_pmc_result,
)
from extra.qk.mmq_amd_telemetry import (
  SCHEMA as TELEMETRY_SCHEMA, collect_mmq_kernel_window_telemetry, collect_process_telemetry, collect_telemetry, read_sensor, validate_telemetry,
)
from extra.qk.mmq_amd_probes import SCHEMA as PROBE_SCHEMA, summarize_store_calibration, validate_differential_probe


def test_parse_rocprof_counter_names_filters_prose_and_preserves_metrics():
  text = "GPU 0 SQ_BUSY_CYCLES SQ_INSTS_VALU GL2C_HIT OccupancyPercent random-word"
  assert parse_rocprof_counter_names(text) == ("GL2C_HIT", "OccupancyPercent", "SQ_BUSY_CYCLES", "SQ_INSTS_VALU")


def test_capability_probe_distinguishes_advertisement_from_liveness(tmp_path):
  artifact = probe_amd_counter_capabilities(requested=("SQ_BUSY_CYCLES", "NOT_REAL"), rocm_root=tmp_path)
  validate_counter_capability(artifact)
  rows = {row["name"]: row for row in artifact["counters"]}
  assert artifact["schema"] == CAPABILITY_SCHEMA
  assert rows["SQ_BUSY_CYCLES"]["status"] == "advertised"
  assert rows["SQ_BUSY_CYCLES"]["status"] != "live"
  assert rows["NOT_REAL"]["status"] == "unsupported"


@pytest.mark.parametrize(("negative", "positive", "status"), [
  ([0, 0], [0, 0], "zero_suspect"),
  ([5, 5], [5, 5], "zero_suspect"),
  ([1, 2, 1], [10, 11, 9], "live"),
  ([10, 9], [1, 2], "zero_suspect"),
  ([], [1, 2], "blocked"),
])
def test_liveness_classification_requires_repeated_directional_controls(negative, positive, status):
  assert classify_liveness(negative, positive)[0] == status


def test_pmc_validation_rejects_status_and_invalid_samples():
  artifact = {"schema": PMC_SCHEMA, "passes": [{"metrics": [{"status": "live", "negative_samples": [1], "positive_samples": [2]}]}]}
  validate_pmc_result(artifact)
  artifact["passes"][0]["metrics"][0]["status"] = "measured"
  with pytest.raises(ValueError, match="status is invalid"): validate_pmc_result(artifact)
  artifact["passes"][0]["metrics"][0].update(status="live", positive_samples=[-1])
  with pytest.raises(ValueError, match="positive_samples is invalid"): validate_pmc_result(artifact)


def test_identity_bound_candidate_collection_rejects_missing_or_fake_binary():
  candidate = {"candidate_id": "c", "backend": "b", "shape": {"M": 16, "N": 16, "K": 256}}
  with pytest.raises(ValueError, match="binary_sha256"):
    collect_kernel_pmc(candidate, ["SQ_BUSY_CYCLES"], 1, command=["true"], system_snapshot_id="s", binary_sha256="short")
  with pytest.raises(ValueError, match="candidate_id"):
    collect_kernel_pmc({}, ["SQ_BUSY_CYCLES"], 1, command=["true"], system_snapshot_id="s", binary_sha256="a" * 64)


def test_default_groups_isolate_native_unsupported_ta_block():
  assert any(all(name.startswith("TA_") for name in group) for group in DEFAULT_GROUPS)
  assert all(not (any(name.startswith("TA_") for name in group) and any(name.startswith("GL2C_") for name in group)) for group in DEFAULT_GROUPS)


def test_mmq_collection_requires_canonical_writeback_mode_before_execution():
  with pytest.raises(ValueError, match="writeback_mode"):
    collect_mmq_pmc({"candidate_id": "c", "knobs": {"writeback_mode": "other"}}, ["SQ_BUSY_CYCLES"], 1,
                    system_snapshot_id="s", binary_sha256="a" * 64)


def test_rocprof_fallback_missing_tool_is_unsupported():
  result = probe_rocprof_fallback(["true"], ["SQ_BUSY_CYCLES"], rocprof="/not/a/tool")
  assert result["schema"] == PMC_SCHEMA and result["status"] == "unsupported"


def test_sensor_reads_live_unsupported_and_blocked_without_coercing_to_zero(tmp_path, monkeypatch):
  live = tmp_path / "live"; live.write_text("42\n")
  assert read_sensor(live) == {"status": "live", "value": 42}
  assert read_sensor(tmp_path / "missing")["status"] == "unsupported"
  def blocked(_self): raise OSError(16, "busy")
  monkeypatch.setattr(type(live), "read_text", blocked)
  result = read_sensor(live)
  assert result["status"] == "blocked" and result["value"] is None and result["errno"] == 16


def test_telemetry_trace_preserves_identity_and_failures(tmp_path):
  good = tmp_path / "good"; good.write_text("7")
  artifact = collect_telemetry("candidate-window", samples=2,
    sensors={"good": str(good), "missing": str(tmp_path / "missing")},
    system_snapshot_id="sys", experiment_id="exp")
  validate_telemetry(artifact)
  assert artifact["schema"] == TELEMETRY_SCHEMA
  assert artifact["system_snapshot_id"] == "sys" and artifact["experiment_id"] == "exp"
  assert all(row["sensors"]["good"]["status"] == "live" for row in artifact["samples"])
  assert all(row["sensors"]["missing"]["status"] == "unsupported" for row in artifact["samples"])


def test_process_telemetry_binds_candidate_binary_and_samples_window(tmp_path):
  sensor = tmp_path / "sensor"; sensor.write_text("9")
  artifact = collect_process_telemetry(["sleep", "0.04"], interval_s=0.005, sensors={"test": str(sensor)},
    system_snapshot_id="sys", experiment_id="exp", candidate_id="candidate", binary_sha256="a" * 64)
  validate_telemetry(artifact)
  assert artifact["status"] == "live" and artifact["sample_count"] >= 1
  assert artifact["candidate_id"] == "candidate" and artifact["binary_sha256"] == "a" * 64


def test_kernel_window_telemetry_validates_identity_before_execution():
  with pytest.raises(ValueError, match="writeback_mode"):
    collect_mmq_kernel_window_telemetry("other", repetitions=1, interval_s=.01, system_snapshot_id="s",
      experiment_id="e", candidate_id="c", binary_sha256="a" * 64)
  with pytest.raises(ValueError, match="binary_sha256"):
    collect_mmq_kernel_window_telemetry("direct_owner_v0", repetitions=1, interval_s=.01, system_snapshot_id="s",
      experiment_id="e", candidate_id="c", binary_sha256="short")


def test_observability_artifacts_are_json_serializable(tmp_path):
  artifacts = [probe_amd_counter_capabilities(rocm_root=tmp_path),
               collect_telemetry("test", sensors={"missing": str(tmp_path / "none")})]
  for artifact in artifacts: json.loads(json.dumps(artifact))


def test_store_calibration_only_derives_transaction_rule_when_every_sample_matches():
  points = [{"samples": [{"status": "live", "unique_64b_lines": 4, "counters": {"GL2C_MC_WRREQ": 4}},
                          {"status": "live", "unique_64b_lines": 8, "counters": {"GL2C_MC_WRREQ": 8}}]}]
  result = summarize_store_calibration(points)
  assert result == {"status": "live", "truth_status": "derived",
                    "rule": "GL2C_MC_WRREQ equals unique touched 64B output lines",
                    "supporting_samples": 2, "all_samples_exact": True}
  points[0]["samples"][1]["counters"]["GL2C_MC_WRREQ"] = 7
  assert summarize_store_calibration(points)["status"] == "zero_suspect"


def test_differential_probe_validation_requires_system_identity_and_known_status():
  artifact = {"schema": PROBE_SCHEMA, "system_snapshot_id": "sys", "points": [{"status": "live"}]}
  validate_differential_probe(artifact)
  artifact["system_snapshot_id"] = None
  with pytest.raises(ValueError, match="system_snapshot_id"): validate_differential_probe(artifact)
