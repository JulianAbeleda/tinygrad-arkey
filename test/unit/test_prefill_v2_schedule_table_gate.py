import json

from extra.qk import prefill_v2_schedule_table_gate as gate
from extra.qk.prefill_v2_schedule_table_gate import build_report


def test_prefill_v2_schedule_table_gate_static_report_uses_local_for_representative_shapes():
  report = build_report(run_amd=False, artifact=False)
  assert report["schema"] == "prefill-v2-schedule-table-gate.v1"
  assert report["route_id"] == "prefill_v2_scheduler_matmul_default"
  assert report["verdict"] == "PREFILL_V2_SCHEDULE_TABLE_APPLIES_PASS"
  assert report["evidence"]["table_exists"] is True
  assert report["evidence"]["all_selected_shapes_present"] is True
  assert report["evidence"]["all_selected_shapes_use_local"] is True
  assert report["evidence"]["run_amd"] is False
  assert report["evidence"]["pin_clock"] is False
  # The frozen warmstart table was narrowed to a single measured 5120x5120 row (commit 8ab3ee80c).
  assert report["shapes"] == ["5120x5120"]
  assert report["missing_shapes"] == []
  assert all(row["params"]["loc"] > 0 for row in report["rows"])
  assert all(row["table_tflops"] > row["table_default_tflops"] for row in report["rows"])
  json.dumps(report)


def test_prefill_v2_schedule_table_gate_reports_dropped_shape_as_coverage_gap_not_crash():
  # 4096x4096 (+8 other) rows were dropped from the frozen table in commit 8ab3ee80c. A request for a
  # dropped shape must produce an honest BLOCKED coverage-gap verdict, not a KeyError crash.
  report = build_report(run_amd=False, artifact=False, shapes=("4096x4096", "5120x5120"))
  assert report["verdict"] == "PREFILL_V2_SCHEDULE_TABLE_BLOCKED"
  assert report["missing_shapes"] == ["4096x4096"]
  assert report["shapes"] == ["5120x5120"]
  assert report["evidence"]["all_selected_shapes_present"] is False
  assert "4096x4096" in report["remaining_blocker"]


def test_prefill_v2_schedule_table_gate_cli_can_skip_artifact(monkeypatch):
  seen = {}

  def fake_build_report(**kwargs):
    seen.update(kwargs)
    return {"verdict": "ok"}

  monkeypatch.setattr(gate, "build_report", fake_build_report)
  gate.main(["--compact", "--no-artifact"])
  assert seen["artifact"] is False
