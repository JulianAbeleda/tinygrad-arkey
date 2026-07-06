import json
import pathlib

from extra.qk import prefill_performance_lowering_registry as registry
from extra.qk import prefill_performance_lowering_report as report


def _is_promotion_row(row: dict[str, object]) -> bool:
  return row.get("phase_name") == "promotion" or str(row["id"]).endswith("_promotion")


def _row_sort_key(row: dict[str, object]) -> tuple[str, int, str]:
  return (str(row["target"]), int(row["phase_order"]), str(row["id"]))


def _derive_note_rows(rows: list[dict[str, object]], note_type: str) -> list[str]:
  assert note_type in {"active_blocker", "evidence", "sidecar"}
  return sorted(
    f"{row['id']}: {blocker}"
    for row in rows
    for blocker in row["blockers"]
    if report._classify_note(blocker, row["status"]) == note_type
  )


def test_prefill_performance_registry_rows_are_valid_and_ordered():
  rows = registry.rows()
  assert rows
  assert len(rows) == len(set(r["id"] for r in rows))
  assert len(registry.ids()) == len(rows)
  assert pathlib.Path(pathlib.Path(registry.DOC_PATH)).name == "prefill-performance-lowering-scope-20260706.md"

  by_target: dict[str, list[int]] = {}
  for row in rows:
    assert row["id"].startswith("prefill_performance_target_")
    assert row["phase"] == row["phase_order"]
    assert row["status"] in registry.VALID_STATUSES
    assert row["owner_area"] in registry.VALID_OWNER_AREAS
    assert row["scope_doc"] == registry.DOC_PATH
    assert pathlib.Path(row["scope_doc"]).exists()
    for reuse_file in row["reuse_files"]:
      assert pathlib.Path(reuse_file).exists()
    by_target.setdefault(row["target"], []).append(row["phase"])

  for target, phases in by_target.items():
    ordered = sorted(phases)
    assert ordered[0] == 0
    assert ordered == list(range(ordered[0], ordered[0] + len(ordered)))
    assert len(ordered) >= 6


def test_prefill_performance_registry_build_shape():
  payload = registry.build()
  assert payload["schema"] == "prefill-performance-lowering-registry.v1"
  assert payload["total_rows"] == len(registry.rows())
  assert payload["scope_doc"] == registry.DOC_PATH
  assert payload["targets"] == sorted(payload["targets"])
  json.dumps(payload)


def test_prefill_performance_report_prints_json_and_can_filter_target():
  full = report.build_prefill_performance_lowering_report()
  rows = registry.rows()
  expected_targets = sorted({r["target"] for r in rows})
  assert full["schema"] == "prefill-performance-lowering-report.v1"
  assert full["row_count"] == len(rows)
  assert full["target_count"] == len(expected_targets)
  assert sorted(full["targets"]) == expected_targets
  assert full["scope_doc"] == registry.DOC_PATH
  assert full["pre_promotion_only"] is False
  assert full["row_scope"] == "full"
  assert full["promotion_rows"] == [row["id"] for row in rows if _is_promotion_row(row)]
  assert "scope_files" in full
  assert isinstance(full["scope_files"], list)
  for target in expected_targets:
    expected_rows = [r["id"] for r in sorted((r for r in rows if r["target"] == target), key=lambda r: r["phase_order"])]
    assert full["targets"][target]["rows"] == expected_rows
  assert not full["targets"]["target_1"]["done"]
  assert any("single_operand_stage" in blocker for blocker in full["blocker_list"])
  baseline_row = registry.row("prefill_performance_target_1_baseline")
  assert "prefill_v2_schedule_table_gate" in baseline_row["gates"]
  assert "extra/qk/prefill_v2_schedule_table_gate.py" in baseline_row["reuse_files"]
  assert "bench/prefill-v2-schedule-table/latest.json" in baseline_row["reuse_files"]
  stage_row = registry.row("prefill_performance_target_1_single_operand_stage")
  assert "prefill_graph_gemm_fp16_single_operand_stage_gate" in stage_row["gates"]
  assert "prefill_graph_gemm_route_bound_no_raw_ops_ins_gate" in stage_row["gates"]
  assert "prefill_graph_gemm_route_bound_stage_gate" in stage_row["gates"]
  assert "extra/qk/prefill_graph_gemm_fp16_stage_gate.py" in stage_row["reuse_files"]
  assert "extra/qk/prefill_graph_gemm_route_bound_stage_gate.py" in stage_row["reuse_files"]
  assert "bench/prefill-graph-gemm-route-bound-stage/latest.json" in stage_row["reuse_files"]
  assert any("fp16 route-bound gate" in criterion for criterion in stage_row["success_criteria"])
  both_row = registry.row("prefill_performance_target_1_both_operands_stage")
  assert "prefill_graph_gemm_fp16_both_operands_stage_gate" in both_row["gates"]
  assert "bench/prefill-graph-gemm-fp16-both-operands-stage/latest.json" in both_row["reuse_files"]

  filtered = report.build_prefill_performance_lowering_report("target_1")
  assert filtered["target_count"] == 1
  assert filtered["scope_doc"] == registry.DOC_PATH
  assert all(r["target"] == "target_1" for r in filtered["rows"])


def test_prefill_performance_report_pre_promotion_view_excludes_promotion_rows():
  rows = registry.rows()
  full = report.build_prefill_performance_lowering_report()
  pre_promotion = report.build_prefill_performance_lowering_report(pre_promotion_only=True)
  expected_promotion_rows = [row["id"] for row in rows if _is_promotion_row(row)]
  expected_non_promotion_rows = [row["id"] for row in rows if not _is_promotion_row(row)]

  assert pre_promotion["pre_promotion_only"] is True
  assert pre_promotion["row_scope"] == "pre_promotion"
  assert pre_promotion["promotion_rows"] == expected_promotion_rows
  assert len(pre_promotion["rows"]) == len(expected_non_promotion_rows)
  assert [row["id"] for row in pre_promotion["rows"]] == expected_non_promotion_rows
  assert pre_promotion["row_count"] == full["row_count"] - len(expected_promotion_rows)
  assert all(_is_promotion_row(r) is False for r in pre_promotion["rows"])
  assert set(full["promotion_rows"]) == set(pre_promotion["promotion_rows"])
  assert all(r not in pre_promotion["rows"] for r in full["promotion_rows"])


def test_prefill_performance_report_orchestration_is_consistent_with_registry_rows():
  rows = registry.rows()
  full = report.build_prefill_performance_lowering_report()
  orchestration = full["orchestration"]

  owners = sorted(set(row["owner_area"] for row in rows))
  assert sorted(orchestration["summary"]["owner_areas"]) == owners
  assert orchestration["summary"]["active_blocker_count"] == len(_derive_note_rows(rows, "active_blocker"))
  assert orchestration["summary"]["evidence_note_count"] == len(_derive_note_rows(rows, "evidence"))
  assert orchestration["summary"]["sidecar_blocker_count"] == len(_derive_note_rows(rows, "sidecar"))

  expected_owner_rows = {
    owner: [row["id"] for row in rows if row["owner_area"] == owner]
    for owner in owners
  }
  expected_by_owner = {
    owner: {
      "rows": [row["id"] for row in sorted((r for r in rows if r["owner_area"] == owner), key=_row_sort_key)],
      "row_count": len(expected_owner_rows[owner]),
      "parallel_ready_rows": [
        row["id"]
        for row in rows
        if row["owner_area"] == owner
        and row["status"] in {"pending", "not_started"}
        and all(report._classify_note(blocker, row["status"]) != "active_blocker" for blocker in row["blockers"])
      ],
    }
    for owner in owners
  }
  for owner, owner_payload in expected_by_owner.items():
    bucket = orchestration["by_owner_area"][owner]
    assert bucket["rows"] == owner_payload["rows"]
    assert bucket["row_count"] == owner_payload["row_count"]
    assert bucket["parallel_ready_rows"] == owner_payload["parallel_ready_rows"]
    assert set(bucket["status_counts"].keys()) == set(row["status"] for row in rows if row["owner_area"] == owner)
    assert bucket["gates"] == sorted(set(gate for row in rows if row["owner_area"] == owner for gate in row["gates"]))

  expected_gate_rows: dict[str, list[str]] = {}
  for row in rows:
    for gate in row["gates"]:
      expected_gate_rows.setdefault(gate, []).append((row["target"], row["phase_order"], row["id"]))
  expected_gate_rows = {
    gate: [row_id for _, _, row_id in sorted(ids)]
    for gate, ids in expected_gate_rows.items()
  }
  assert orchestration["gates"] == {gate: ids for gate, ids in sorted(expected_gate_rows.items())}

  for row in rows:
    has_active_blocker = any(report._classify_note(blocker, row["status"]) == "active_blocker" for blocker in row["blockers"])
    if has_active_blocker:
      assert row["id"] in orchestration["active_blocker_rows"]
    if row["status"] == "blocked":
      assert row["id"] in orchestration["status_blocked_rows"]
    if row["status"] in {"pending", "not_started"} and not has_active_blocker:
      assert row["id"] in orchestration["parallel_ready_rows"]


def test_prefill_performance_report_orchestration_cli_mode(monkeypatch, capsys):
  expected_rows = report.build_prefill_performance_lowering_report()["rows"]
  report.main(["--orchestration", "--compact"])
  parsed = json.loads(capsys.readouterr().out.strip())
  assert "by_owner_area" in parsed
  assert "gates" in parsed
  assert "notes" in parsed
  assert "parallel_ready_rows" in parsed
  assert parsed["summary"]["active_blocker_count"] == len(_derive_note_rows(expected_rows, "active_blocker"))


def test_prefill_performance_report_pre_promotion_cli_mode(monkeypatch, capsys):
  report.main(["--pre-promotion", "--orchestration", "--compact"])
  parsed = json.loads(capsys.readouterr().out.strip())
  assert "by_owner_area" in parsed
  assert "notes" in parsed
  assert "parallel_ready_rows" in parsed
  rows = registry.rows()
  promotion_ids = {row["id"] for row in rows if _is_promotion_row(row)}
  assert not any(row_id in promotion_ids for row_id in parsed["parallel_ready_rows"])
  assert not any(row_id in promotion_ids for row_id in parsed["active_blocker_rows"])
  assert not any(row_id in promotion_ids for row_id in parsed["status_blocked_rows"])


def test_prefill_performance_report_uses_current_route_bound_blocker():
  filtered = report.build_prefill_performance_lowering_report("target_1")
  payload = json.dumps(filtered)
  assert "current CFG/control-flow dependency lowering" not in payload
  assert "outside warp+reduce" in payload


def test_prefill_performance_report_cli_modes(monkeypatch, capsys):
  report.main(["--compact"])
  compact = capsys.readouterr().out.strip()
  parsed = json.loads(compact)
  assert parsed["schema"] == "prefill-performance-lowering-report.v1"
  assert parsed["scope_doc"] == registry.DOC_PATH

  report.main([])
  expanded = capsys.readouterr().out.strip()
  assert json.loads(expanded)["schema"] == "prefill-performance-lowering-report.v1"
