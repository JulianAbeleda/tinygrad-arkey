import pytest

from extra.qk import pure_kernel_surface_audit as audit
from extra.qk.pure_search_guard import effective_routes, assert_pure_machine_search


def test_strict_surface_audit_flags_default_blockers():
  report = audit.strict_default_purity_report()
  assert report["verdict"] == "STRICT_DEFAULT_PURITY_FAIL"
  blockers = {r["route_id"]: r for r in report["blockers"]}
  assert blockers["prefill_pipe_role_selective_generated"]["surface_class"] == "external_raw_or_binary"
  assert blockers["decode_flash_live_split_g4_8b_kvboth"]["surface_class"] == "route_local_custom_kernel"
  assert report["manifest_contradictions"] == []


def test_route_surface_rows_classify_known_surfaces():
  assert audit.route_surface_row("decode_q4k_g3_generated")["strict_pure"] is True
  assert audit.route_surface_row("decode_q6k_coop_generated")["strict_pure"] is True
  assert audit.route_surface_row("prefill_q4k_direct_tile4x4_default")["surface_class"] == "route_local_custom_kernel"
  assert "Ops.INS" in audit.route_surface_row("prefill_pipe_role_selective_generated")["markers"]["extra/qk/prefill_graph_gemm_route.py"]


def test_unmanifested_runtime_surfaces_are_explicit():
  got = {s["surface_id"] for s in audit.build()["unmanifested_runtime_surfaces"]}
  assert "prefill_q6k_direct_packed_default_capable" in got
  assert "decode_q4k_smallk_batched" in got
  assert "decode_q6k_smallk_batched" in got


def test_missing_writer_file_evidence_is_reported_in_rows(tmp_path, monkeypatch):
  route_id = "prefill_q4k_direct_tile4x4_default"
  existing = "tinygrad/llm/prefill_routes.py"
  missing = "extra/qk/does_not_exist.py"
  (tmp_path / existing).parent.mkdir(parents=True, exist_ok=True)
  (tmp_path / existing).write_text("from tinygrad import Tensor\n_ = Tensor.custom_kernel\n_ = 'Ops.CUSTOM'\n")

  monkeypatch.setattr(audit, "ROOT", tmp_path)
  surface = audit.ROUTE_SURFACES[route_id]
  monkeypatch.setattr(audit, "ROUTE_SURFACES", {
    **audit.ROUTE_SURFACES,
    route_id: audit.RouteSurface(route_id, surface.surface_class, (existing, missing), surface.reason,
                                replacement_scope=surface.replacement_scope, descriptor_artifact=surface.descriptor_artifact),
  })

  row = audit.route_surface_row(route_id)
  assert row["writer_file_exists"][existing] is True
  assert row["writer_file_exists"][missing] is False
  assert row["missing_writer_files"] == [missing]
  assert "Tensor.custom_kernel" in row["markers"][existing]


def test_missing_writer_files_are_reported_in_summary(tmp_path, monkeypatch):
  route_id = "prefill_pipe_role_selective_generated"
  existing = "extra/qk/prefill_graph_gemm_route.py"
  missing = "extra/qk/missing_prefill_writer.py"
  (tmp_path / existing).parent.mkdir(parents=True, exist_ok=True)
  (tmp_path / existing).write_text("from tinygrad import Tensor\n_ = Tensor.custom_kernel\n")

  monkeypatch.setattr(audit, "ROOT", tmp_path)
  surface = audit.ROUTE_SURFACES[route_id]
  monkeypatch.setattr(audit, "ROUTE_SURFACES", {
    **audit.ROUTE_SURFACES,
    route_id: audit.RouteSurface(route_id, surface.surface_class, (existing, missing), surface.reason,
                                replacement_scope=surface.replacement_scope, descriptor_artifact=surface.descriptor_artifact),
  })

  report = audit.build()
  assert missing in report["summary"]["missing_writer_files"]
  assert route_id in report["summary"]["routes_with_missing_writer_files"]


def test_pure_search_guard_uses_strict_surface_classification():
  routes = {r["family"]: r for r in effective_routes({})}
  assert routes["decode_q4k_gemv"]["pure"] is True
  assert routes["decode_q6k_gemv"]["pure"] is True
  assert routes["prefill_gemm"]["pure"] is False
  assert routes["decode_attention"]["pure"] is False
  with pytest.raises(RuntimeError, match="surface=external_raw_or_binary"):
    assert_pure_machine_search({"PURE_MACHINE_SEARCH_ONLY": "1"})
