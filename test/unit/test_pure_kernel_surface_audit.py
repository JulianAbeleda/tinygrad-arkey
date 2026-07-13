import pytest

from extra.qk import pure_kernel_surface_audit as audit
from extra.qk import generated_route_registry as registry
from extra.qk import runtime_surface_registry
from extra.qk.pure_search_guard import effective_routes, assert_pure_machine_search


def test_strict_surface_audit_flags_default_blockers():
  report = audit.strict_default_purity_report()
  assert report["verdict"] == "STRICT_DEFAULT_PURITY_PASS"
  blockers = {r["route_id"]: r for r in report["blockers"]}
  assert "prefill_pipe_role_selective_generated" not in blockers
  assert "decode_flash_live_split_g4_8b_kvboth" not in blockers
  assert "decode_flash_block_tile_g5_konly" not in blockers
  assert report["manifest_contradictions"] == []


def test_route_surface_rows_classify_known_surfaces():
  assert audit.route_surface_row("decode_q4k_g3_generated")["strict_pure"] is True
  assert audit.route_surface_row("decode_q6k_coop_generated")["strict_pure"] is True
  assert audit.route_surface_row("prefill_v2_scheduler_matmul_default")["surface_class"] == "ordinary_tinygrad_graph"
  assert audit.route_surface_row("prefill_wmma_pipe_primitive_generated")["surface_class"] == "ordinary_tinygrad_graph"
  assert audit.route_surface_row("prefill_wmma_pipe_lds_dbuf_primitive_generated")["surface_class"] == "compiler_primitive_spec_owned_asm_backend_atom"
  assert audit.route_surface_row("prefill_wmma_pipe_lds_dbuf_primitive_generated")["strict_pure"] is False
  assert audit.route_surface_row("prefill_q4k_direct_tile4x4_default")["surface_class"] == "descriptor_owned_uop_codegen"
  assert "preassembled_linear" in audit.route_surface_row("prefill_pipe_role_selective_generated")["markers"]["extra/qk/prefill_graph_gemm_route.py"]


def test_surface_rows_distinguish_backend_asm_from_hand_kernel_authorship():
  generated = audit.route_surface_row("decode_q4k_g3_generated")
  assert generated["strict_pure"] is True
  assert generated["asm_usage"] == "backend_emitted_from_descriptor"
  assert generated["kernel_authorship"] == "descriptor_generated"
  assert generated["surface_policy"] == "generated_default_allowed"

  prefill_oracle = audit.route_surface_row("prefill_pipe_role_selective_generated")
  assert prefill_oracle["strict_pure"] is False
  assert prefill_oracle["asm_usage"] == "raw_instruction_or_binary_injection"
  assert prefill_oracle["kernel_authorship"] == "hand_authored_full_kernel_schedule"
  assert prefill_oracle["surface_policy"] == "oracle_or_escape_hatch_only"

  s10_route = audit.route_surface_row("prefill_wmma_pipe_lds_dbuf_primitive_generated")
  assert s10_route["strict_pure"] is False
  assert s10_route["asm_usage"] == "asm_backend_atom"
  assert s10_route["kernel_authorship"] == "compiler_primitive_spec_owned"
  assert s10_route["surface_policy"] == "compiler_primitive_not_strict_pure_generated"
  assert "attn_kv uses generated pipe transport with local staging disabled" in s10_route["reason"]

  s10_mixed_route = audit.route_surface_row("prefill_wmma_lds_dbuf_primitive_mixed")
  assert s10_mixed_route["strict_pure"] is False
  assert s10_mixed_route["asm_usage"] == "asm_backend_atom_and_raw_pipe_oracle"
  assert s10_mixed_route["kernel_authorship"] == "mixed_compiler_primitive_and_raw_fallback"
  assert s10_mixed_route["surface_policy"] == "compiler_primitive_with_explicit_raw_pipe_fallback"


def test_s10_composed_manifest_records_attn_kv_generated_no_local_stage():
  manifest = audit.route_manifest.ROUTES["prefill_wmma_pipe_lds_dbuf_primitive_generated"]
  role_primitives = {g["role"]: g["primitive"] for g in manifest["shape_guards"] if "role" in g}
  assert role_primitives == {
    "attn_qo": "pipe",
    "attn_kv": "generated_pipe_no_local_stage",
    "ffn_down": "pipe",
    "ffn_gate_up": "lds_dbuf",
  }
  assert "attn_kv uses generated pipe transport with local staging disabled" in manifest["route_attribution"]
  assert "retaining resource-gated raw fallback" in manifest["shape_guards"][-1]["note"]
  assert any("outside explicit safety fallback" in k for k in manifest["forbidden_kernels"])


def test_l3_descriptor_surfaces_are_derived_from_registry():
  for route_id in registry.route_ids():
    reg = registry.row(route_id)
    row = audit.route_surface_row(route_id)
    assert route_id not in audit.ROUTE_SURFACES
    assert row["surface_class"] == "descriptor_owned_uop_codegen"
    assert row["descriptor_artifact"] == reg["descriptor_artifact"]
    assert row["writer_files"] == reg["writer_files"]


def test_route_rows_expose_expected_kernel_bindings_for_generated_and_handwritten_routes():
  route_id = "decode_q4k_g3_generated"
  reg_row = registry.row(route_id)
  route_row = audit.route_surface_row(route_id)
  assert route_row["expected_kernel_patterns"] == reg_row["emitted_kernel_patterns"]
  assert route_row["has_expected_kernel_binding"] is True

  generated_research_route = "prefill_q4k_reduce_out_research"
  manifest_row = audit.route_manifest.ROUTES[generated_research_route]
  generated_surface_row = audit.route_surface_row(generated_research_route)
  assert generated_surface_row["expected_kernel_patterns"] == list(manifest_row["expected_kernels"])
  assert generated_surface_row["has_expected_kernel_binding"] is True


def test_unmanifested_runtime_surfaces_are_explicit():
  report = audit.build()
  got = {s["surface_id"] for s in report["unmanifested_runtime_surfaces"]}
  assert got == set(runtime_surface_registry.surface_ids())
  assert set(report["audit_blockers"]["unmanifested_runtime_surfaces"]) == got
  assert report["summary"]["unmanifested_runtime_surface_blockers"] == sorted(got)


def test_unmanifested_runtime_surfaces_are_derived_from_registry():
  by_surface = {s["surface_id"]: s for s in audit.unmanifested_runtime_surface_rows()}
  for surface_id in runtime_surface_registry.surface_ids():
    reg = runtime_surface_registry.row(surface_id)
    row = by_surface[surface_id]
    assert row["surface_class"] == reg["surface_class"]
    assert row["writer_files"] == reg["writer_files"]
    assert row["reason"] == reg["reason"]
    assert row["replacement_scope"] == reg["replacement_scope"]
    assert "expected_kernel_patterns" not in row
    assert "has_expected_kernel_binding" not in row


def test_missing_writer_file_evidence_is_reported_in_rows(tmp_path, monkeypatch):
  route_id = "prefill_q4k_direct_tile4x4_default"
  existing = "tinygrad/llm/prefill_routes.py"
  missing = "extra/qk/does_not_exist.py"
  (tmp_path / existing).parent.mkdir(parents=True, exist_ok=True)
  (tmp_path / existing).write_text("from tinygrad import Tensor\n_ = Tensor.custom_kernel\n_ = 'Ops.CUSTOM'\n")

  monkeypatch.setattr(audit, "ROOT", tmp_path)
  surface = audit.route_surface(route_id)
  monkeypatch.setattr(audit, "ROUTE_SURFACES", {
    **audit.ROUTE_SURFACES,
    route_id: audit.RouteSurface(route_id, surface.surface_class, (existing, missing), surface.reason,
                                replacement_scope=surface.replacement_scope, descriptor_artifact=surface.descriptor_artifact),
  })

  row = audit.route_surface_row(route_id)
  assert row["writer_file_exists"][existing] is True
  assert row["writer_file_exists"][missing] is False
  assert row["writer_files_present"] is False
  assert row["missing_writer_files"] == [missing]
  assert "Tensor.custom_kernel" in row["markers"][existing]


def test_missing_writer_file_blocks_default_pure_route(tmp_path, monkeypatch):
  route_id = "decode_q4k_g3_generated"
  existing = "extra/qk/gemv_g3_codegen_lowering.py"
  missing = "extra/qk/missing_generated_writer.py"
  (tmp_path / existing).parent.mkdir(parents=True, exist_ok=True)
  (tmp_path / existing).write_text("def emit():\n  return 'q4k_g3_lanemap_gemv_kernel'\n")

  monkeypatch.setattr(audit, "ROOT", tmp_path)
  surface = audit.route_surface(route_id)
  monkeypatch.setattr(audit, "ROUTE_SURFACES", {
    **audit.ROUTE_SURFACES,
    route_id: audit.RouteSurface(route_id, surface.surface_class, (existing, missing), surface.reason,
                                replacement_scope=surface.replacement_scope, descriptor_artifact=surface.descriptor_artifact),
  })
  monkeypatch.setattr(audit.route_manifest, "default_routes", lambda: [route_id])

  row = audit.route_surface_row(route_id)
  assert row["surface_pure"] is True
  assert row["strict_pure"] is False
  assert row["contradiction"] is True
  report = audit.strict_default_purity_report()
  assert report["verdict"] == "STRICT_DEFAULT_PURITY_FAIL"
  assert [r["route_id"] for r in report["missing_writer_file_blockers"]] == [route_id]
  assert [r["route_id"] for r in report["blockers"]] == [route_id]


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
  assert route_id in report["audit_blockers"]["missing_writer_file_routes"]
  assert report["verdict"] == "PURE_KERNEL_SURFACE_AUDIT_DEBT_FOUND"


def test_unmanifested_runtime_surfaces_block_otherwise_passing_audit(monkeypatch):
  route_id = "decode_q4k_g3_generated"
  monkeypatch.setattr(audit.route_manifest, "ROUTES", {route_id: audit.route_manifest.ROUTES[route_id]})
  monkeypatch.setattr(audit, "ROUTE_SURFACES", {})
  monkeypatch.setattr(audit.runtime_surface_registry, "rows", lambda: [{
    "surface_id": "synthetic_runtime_capable_surface", "surface_class": "route_local_custom_kernel",
    "writer_files": ["tinygrad/llm/prefill_routes.py"], "reason": "runtime-capable handwritten test surface",
    "replacement_scope": "manifest or replace",
  }])

  report = audit.build()
  assert report["strict_default_purity"]["verdict"] == "STRICT_DEFAULT_PURITY_PASS"
  assert report["verdict"] == "PURE_KERNEL_SURFACE_AUDIT_DEBT_FOUND"
  assert report["audit_blockers"]["strict_default_route_blockers"] == []
  assert report["audit_blockers"]["unmanifested_runtime_surfaces"] == ["synthetic_runtime_capable_surface"]


def test_pure_search_guard_uses_strict_surface_classification():
  routes = {r["family"]: r for r in effective_routes({})}
  assert routes["decode_q4k_gemv"]["pure"] is True
  assert routes["decode_q6k_gemv"]["pure"] is True
  assert routes["prefill_gemm"]["pure"] is True
  assert routes["decode_attention"]["pure"] is True
  assert_pure_machine_search({"PURE_MACHINE_SEARCH_ONLY": "1"})
  with pytest.raises(RuntimeError, match="surface=external_raw_or_binary"):
    assert_pure_machine_search({"PURE_MACHINE_SEARCH_ONLY": "1", "PREFILL_GRAPH_GEMM": "1"})


def test_pure_search_guard_distinguishes_generated_pipe_primitive_from_raw_oracle():
  default_route = {r["family"]: r for r in effective_routes({})}["prefill_gemm"]
  assert default_route["effective_route"] == "prefill_v2_scheduler_matmul_default"
  assert default_route["rolled_back_to_oracle"] is False
  assert default_route["pure"] is True

  raw_route = {r["family"]: r for r in effective_routes({"PREFILL_GRAPH_GEMM": "1"})}["prefill_gemm"]
  assert raw_route["effective_route"] == "prefill_pipe_role_selective_generated"
  assert raw_route["rolled_back_to_oracle"] is True
  assert raw_route["pure"] is False

  primitive_route = {r["family"]: r for r in effective_routes({
    "PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_PIPE_PRIMITIVE": "1"})}["prefill_gemm"]
  assert primitive_route["effective_route"] == "prefill_wmma_pipe_primitive_generated"
  assert primitive_route["rolled_back_to_oracle"] is False
  assert primitive_route["provenance"] == "tinygrad_scheduler_generated"
  assert primitive_route["pure"] is True

  lds_only_route = {r["family"]: r for r in effective_routes({
    "PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1", "PREFILL_DBUF": "1"})}["prefill_gemm"]
  assert lds_only_route["effective_route"] == "prefill_wmma_lds_dbuf_primitive_mixed"
  assert lds_only_route["rolled_back_to_oracle"] is False
  assert lds_only_route["provenance"] == "compiler_primitive_spec_owned"
  assert lds_only_route["surface_class"] == "compiler_primitive_spec_owned_mixed_raw_pipe"
  assert lds_only_route["pure"] is False

  lds_without_dbuf_route = {r["family"]: r for r in effective_routes({
    "PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1"})}["prefill_gemm"]
  assert lds_without_dbuf_route["effective_route"] == "prefill_pipe_role_selective_generated"
  assert lds_without_dbuf_route["rolled_back_to_oracle"] is True

  pipe_lds_without_dbuf_route = {r["family"]: r for r in effective_routes({
    "PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_PIPE_PRIMITIVE": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1"})}["prefill_gemm"]
  assert pipe_lds_without_dbuf_route["effective_route"] == "prefill_wmma_pipe_primitive_generated"
  assert pipe_lds_without_dbuf_route["rolled_back_to_oracle"] is False
  assert pipe_lds_without_dbuf_route["pure"] is True

  composed_route = {r["family"]: r for r in effective_routes({
    "PREFILL_GRAPH_GEMM": "1", "PREFILL_WMMA_PIPE_PRIMITIVE": "1", "PREFILL_WMMA_LDS_PRIMITIVE": "1",
    "PREFILL_DBUF": "1"})}["prefill_gemm"]
  assert composed_route["effective_route"] == "prefill_wmma_pipe_lds_dbuf_primitive_generated"
  assert composed_route["rolled_back_to_oracle"] is False
  assert composed_route["provenance"] == "compiler_primitive_spec_owned"
  assert composed_route["surface_class"] == "compiler_primitive_spec_owned_asm_backend_atom"
  assert composed_route["pure"] is False
