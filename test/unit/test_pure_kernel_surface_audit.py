import pytest

from extra.qk import pure_kernel_surface_audit as audit
from extra.qk.pure_search_guard import effective_routes, assert_pure_machine_search


def test_strict_surface_audit_flags_manifest_contradictions():
  report = audit.strict_default_purity_report()
  assert report["verdict"] == "STRICT_DEFAULT_PURITY_FAIL"
  contradictions = {r["route_id"]: r for r in report["manifest_contradictions"]}
  assert contradictions["prefill_pipe_role_selective_generated"]["surface_class"] == "external_raw_or_binary"
  assert contradictions["decode_flash_live_split_g4_8b_kvboth"]["surface_class"] == "route_local_custom_kernel"


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


def test_pure_search_guard_uses_strict_surface_classification():
  routes = {r["family"]: r for r in effective_routes({})}
  assert routes["decode_q4k_gemv"]["pure"] is True
  assert routes["decode_q6k_gemv"]["pure"] is True
  assert routes["prefill_gemm"]["pure"] is False
  assert routes["decode_attention"]["pure"] is False
  with pytest.raises(RuntimeError, match="surface=external_raw_or_binary"):
    assert_pure_machine_search({"PURE_MACHINE_SEARCH_ONLY": "1"})
