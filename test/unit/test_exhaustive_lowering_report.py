import json
import sys
import types

from extra.qk import exhaustive_lowering_report as report
from extra.qk import pure_kernel_surface_audit as audit


def _sample_audit_report():
  return {
    "schema": "pure_kernel_surface_audit.v1",
    "verdict": "PURE_KERNEL_SURFACE_AUDIT_DEBT_FOUND",
    "strict_default_purity": {
      "blockers": [
        {
          "route_id": "decode_q4k_g3_generated",
          "surface_class": "route_local_custom_kernel",
          "status": "promoted_default",
          "strict_pure": False,
        },
      ],
    },
    "audit_blockers": {
      "strict_default_route_blockers": ["decode_q4k_g3_generated"],
      "unmanifested_runtime_surfaces": ["prefill_q6k_direct_packed_default_capable"],
    },
    "unmanifested_runtime_surfaces": [
      {
        "surface_id": "prefill_q6k_direct_packed_default_capable",
        "surface_class": "route_local_custom_kernel",
        "writer_files": ["tinygrad/llm/prefill_routes.py", "extra/qk/quant/q6_k_gemv_primitive.py"],
        "reason": "runtime-capable hand-written surface",
        "replacement_scope": "manifest or replace",
      }
    ],
    "routes": [{"route_id": "decode_q4k_g3_generated"}],
  }


def test_build_report_contains_blockers_and_work_queue(monkeypatch):
  monkeypatch.setattr(report.audit, "build", lambda: _sample_audit_report())
  monkeypatch.setattr(report, "_load_phase_lookup", lambda: {})
  monkeypatch.setattr(report, "_load_done_criteria_lookup", lambda: {})
  out = report.build_exhaustive_lowering_report()

  assert out["schema"] == "exhaustive-lowering-report.v1"
  assert out["audit_verdict"] == "PURE_KERNEL_SURFACE_AUDIT_DEBT_FOUND"
  assert out["blockers"]["strict_default_route_blockers"] == ["decode_q4k_g3_generated"]
  assert out["blockers"]["unmanifested_runtime_surfaces"] == ["prefill_q6k_direct_packed_default_capable"]
  assert len(out["work_queue"]) == 2
  kinds = {i["work_item_type"] for i in out["work_queue"]}
  assert kinds == {"strict_default_route_blocker", "unmanifested_runtime_surface"}
  strict_item = next(i for i in out["work_queue"] if i["work_item_type"] == "strict_default_route_blocker")
  runtime_item = next(i for i in out["work_queue"] if i["work_item_type"] == "unmanifested_runtime_surface")
  assert strict_item["work_item_id"] == "decode_q4k_g3_generated"
  assert runtime_item["work_item_id"] == "prefill_q6k_direct_packed_default_capable"
  assert "done_criteria" not in strict_item
  assert "done_criteria" not in runtime_item


def test_report_prints_json_payload(monkeypatch, capsys):
  sample = _sample_audit_report()
  monkeypatch.setattr(report.audit, "build", lambda: sample)
  report.main(["--compact"])
  out = capsys.readouterr().out.strip()
  parsed = json.loads(out)
  assert parsed["schema"] == "exhaustive-lowering-report.v1"
  assert parsed["audit_report"]["verdict"] == sample["verdict"]


def test_phase_registry_enrichment(monkeypatch):
  fake_phase_registry = types.ModuleType("extra.qk.lowering_phase_registry")
  fake_phase_registry.rows = lambda: [
    {
      "id": "decode_q4k_g3_generated",
      "phase": 2,
      "phase_name": "descriptor_replacement",
      "target_lowering_level": "L3",
      "next_action": "move to generated substrate",
      "route_fact_that_should_not_be_copied": "do not copy",
    },
    {
      "id": "prefill_q6k_direct_packed_default_capable",
      "phase": 2,
      "phase_name": "direct_packed_prefill",
      "target_lowering_level": "L3",
      "next_action": "manifest or replace route",
    },
    {
      "id": "prefill_pipe_global_rollback",
      "phase": 5,
      "phase_name": "rollback_and_quarantine",
      "target_lowering_level": "L4",
      "next_action": "quarantine fixture",
    },
  ]

  monkeypatch.setitem(sys.modules, "extra.qk.lowering_phase_registry", fake_phase_registry)
  monkeypatch.setattr(report.audit, "build", lambda: _sample_audit_report())

  out = report.build_exhaustive_lowering_report()
  strict_item = next(i for i in out["work_queue"] if i["work_item_type"] == "strict_default_route_blocker")
  runtime_item = next(i for i in out["work_queue"] if i["work_item_type"] == "unmanifested_runtime_surface")

  assert strict_item["phase"] == 2
  assert strict_item["phase_name"] == "descriptor_replacement"
  assert strict_item["target_lowering_level"] == "L3"
  assert strict_item["next_action"] == "move to generated substrate"
  assert "route_fact_that_should_not_be_copied" not in strict_item
  assert runtime_item["phase"] == 2
  assert runtime_item["phase_name"] == "direct_packed_prefill"
  assert runtime_item["target_lowering_level"] == "L3"
  assert runtime_item["next_action"] == "manifest or replace route"
  phase_item = next(i for i in out["work_queue"] if i["work_item_id"] == "prefill_pipe_global_rollback")
  assert phase_item["work_item_type"] == "phase_registry_item"
  assert phase_item["phase_name"] == "rollback_and_quarantine"


def test_report_enriches_done_criteria_from_dynamic_module(monkeypatch):
  fake_phase_registry = types.ModuleType("extra.qk.lowering_phase_registry")
  fake_phase_registry.rows = lambda: [
    {
      "id": "decode_q4k_g3_generated",
      "phase": 2,
      "phase_name": "descriptor_replacement",
      "target_lowering_level": "L3",
      "next_action": "move to generated substrate",
      "route_fact_that_should_not_be_copied": "do not copy",
    },
    {
      "id": "prefill_q6k_direct_packed_default_capable",
      "phase": 2,
      "phase_name": "direct_packed_prefill",
      "target_lowering_level": "L4",
      "next_action": "manifest or replace route",
    },
    {
      "id": "prefill_pipe_global_rollback",
      "phase": 5,
      "phase_name": "rollback_and_quarantine",
      "target_lowering_level": "L4",
      "next_action": "quarantine fixture",
    },
  ]
  fake_done_criteria = types.ModuleType("extra.qk.lowering_done_criteria")
  fake_done_criteria.rows = lambda: [
    {"target_lowering_level": "L3", "required_criteria": ["descriptor_owned_substrate"]},
    {"target_lowering_level": "L4", "required_criteria": ["ordinary_tinygrad_graph"]},
  ]

  monkeypatch.setitem(sys.modules, "extra.qk.lowering_phase_registry", fake_phase_registry)
  monkeypatch.setitem(sys.modules, "extra.qk.lowering_done_criteria", fake_done_criteria)
  monkeypatch.setattr(report.audit, "build", lambda: _sample_audit_report())

  out = report.build_exhaustive_lowering_report()
  strict_item = next(i for i in out["work_queue"] if i["work_item_type"] == "strict_default_route_blocker")
  runtime_item = next(i for i in out["work_queue"] if i["work_item_type"] == "unmanifested_runtime_surface")
  rollback_item = next(i for i in out["work_queue"] if i["work_item_id"] == "prefill_pipe_global_rollback")

  assert strict_item["done_criteria"] == ["descriptor_owned_substrate"]
  assert runtime_item["done_criteria"] == ["ordinary_tinygrad_graph"]
  assert rollback_item["done_criteria"] == ["ordinary_tinygrad_graph"]


def test_done_criteria_loader_uses_required_criteria_schema(monkeypatch):
  fake_done_criteria = types.ModuleType("extra.qk.lowering_done_criteria")
  fake_done_criteria.rows = lambda: [
    {"target_lowering_level": "L3", "required_criteria": ["descriptor_owned_substrate"]},
    {"target_lowering_level": "L4", "done_criteria": ["legacy_shape_should_not_load"]},
    {"target_lowering_level": "L5", "required_criteria": ["valid", 1]},
  ]

  monkeypatch.setitem(sys.modules, "extra.qk.lowering_done_criteria", fake_done_criteria)

  assert report._load_done_criteria_lookup() == {"L3": ["descriptor_owned_substrate"]}


def test_report_build_integrates_current_audit_output():
  real_report = audit.build()
  out = report.build_exhaustive_lowering_report()
  strict_blockers = {b["route_id"] for b in real_report["strict_default_purity"]["blockers"]}
  runtime_blockers = {s["surface_id"] for s in real_report["unmanifested_runtime_surfaces"]}
  queue_blockers = {i["work_item_id"] for i in out["work_queue"]}

  assert out["audit_verdict"] == real_report["verdict"]
  assert strict_blockers.issubset(queue_blockers)
  assert runtime_blockers.issubset(queue_blockers)
  assert out["blockers"]["strict_default_route_blockers"] == sorted(strict_blockers)
  enriched = {i["work_item_id"]: i for i in out["work_queue"] if "phase_name" in i}
  assert "prefill_q6k_direct_packed_default_capable" in enriched
  assert enriched["prefill_q6k_direct_packed_default_capable"]["phase_name"] == "direct_packed_prefill"
  assert enriched["prefill_q6k_direct_packed_default_capable"]["target_lowering_level"] == "L3"
  phase_registry_ids = {r["id"] for r in report._load_phase_lookup().values()}
  assert phase_registry_ids.issubset(queue_blockers)
  done_criteria_lookup = report._load_done_criteria_lookup()
  if done_criteria_lookup:
    for item in out["work_queue"]:
      target_level = item.get("target_lowering_level")
      if not isinstance(target_level, str):
        continue
      if target_level in done_criteria_lookup:
        assert item["done_criteria"] == done_criteria_lookup[target_level]
      else:
        assert "done_criteria" not in item
