import json
import pathlib

from extra.qk import prefill_performance_lowering_registry as registry
from extra.qk import prefill_performance_lowering_report as report


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
  assert full["schema"] == "prefill-performance-lowering-report.v1"
  assert full["row_count"] == len(registry.rows())
  assert full["target_count"] == 2
  assert full["scope_doc"] == registry.DOC_PATH
  assert "scope_files" in full
  assert isinstance(full["scope_files"], list)
  assert full["targets"]["target_1"]["rows"] == [
    "prefill_performance_target_1_fp16_recovery",
    "prefill_performance_target_1_baseline",
    "prefill_performance_target_1_single_operand_stage",
    "prefill_performance_target_1_both_operands_stage",
    "prefill_performance_target_1_coop_partition",
    "prefill_performance_target_1_optional_double_buffer",
    "prefill_performance_target_1_promotion",
  ]
  assert full["targets"]["target_2"]["rows"] == [
    "prefill_performance_target_2_packed_mmq_recovery",
    "prefill_performance_target_2_baseline",
    "prefill_performance_target_2_tile_contract",
    "prefill_performance_target_2_wmma_surface_decision",
    "prefill_performance_target_2_small_lifecycle",
    "prefill_performance_target_2_synthetic_shape",
    "prefill_performance_target_2_model_authority",
    "prefill_performance_target_2_q6_residual_decision",
    "prefill_performance_target_2_promotion",
  ]
  assert not full["targets"]["target_1"]["done"]
  assert any("single_operand_stage" in blocker for blocker in full["blocker_list"])
  stage_row = registry.row("prefill_performance_target_1_single_operand_stage")
  assert "prefill_graph_gemm_fp16_single_operand_stage_gate" in stage_row["gates"]
  assert "prefill_graph_gemm_route_bound_no_raw_ops_ins_gate" in stage_row["gates"]
  assert "extra/qk/prefill_graph_gemm_fp16_stage_gate.py" in stage_row["reuse_files"]
  assert any("fp16 route-bound gate" in criterion for criterion in stage_row["success_criteria"])
  both_row = registry.row("prefill_performance_target_1_both_operands_stage")
  assert "prefill_graph_gemm_fp16_both_operands_stage_gate" in both_row["gates"]
  assert "bench/prefill-graph-gemm-fp16-both-operands-stage/latest.json" in both_row["reuse_files"]

  filtered = report.build_prefill_performance_lowering_report("target_1")
  assert filtered["target_count"] == 1
  assert filtered["scope_doc"] == registry.DOC_PATH
  assert all(r["target"] == "target_1" for r in filtered["rows"])


def test_prefill_performance_report_cli_modes(monkeypatch, capsys):
  report.main(["--compact"])
  compact = capsys.readouterr().out.strip()
  parsed = json.loads(compact)
  assert parsed["schema"] == "prefill-performance-lowering-report.v1"
  assert parsed["scope_doc"] == registry.DOC_PATH

  report.main([])
  expanded = capsys.readouterr().out.strip()
  assert json.loads(expanded)["schema"] == "prefill-performance-lowering-report.v1"
