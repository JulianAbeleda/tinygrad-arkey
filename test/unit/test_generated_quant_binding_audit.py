import textwrap

from extra.qk import generated_quant_binding_audit as audit


def test_route_rows_classify_manifest_defaults_and_debt():
  rows = {r["route_id"]: r for r in audit.route_rows()}
  assert rows["prefill_q4k_int8_wmma_generated_research"]["classification"] == "allowed.generated"
  assert rows["prefill_q4k_direct_tile4x4_default"]["classification"] == "transitional.hand_authored_uop"
  assert rows["decode_q4k_owned_warp"]["classification"] == "banned.not_default_but_reachable_or_ledgered"


def test_scan_file_detects_custom_source_and_bindings(tmp_path, monkeypatch):
  root = tmp_path
  p = root / "sample.py"
  p.write_text(textwrap.dedent("""
    from tinygrad.uop.ops import UOp, Ops
    def k(x):
      y = x.custom_kernel(fxn=lambda out: out)
      z = UOp(Ops.CUSTOM, arg="asm volatile(\\"v_dot4_u32_u8\\")")
      return y, z
  """))
  monkeypatch.setattr(audit, "ROOT", root)
  found = audit.scan_file("sample.py")
  kinds = {f.kind for f in found}
  assert "binding.custom_kernel" in kinds
  assert "binding.ops_custom" in kinds
  assert "source.inline_asm" in kinds
  assert "source_builder.inline_asm" in kinds


def test_build_report_shape():
  out = audit.build()
  assert out["verdict"] == "GENERATED_QUANT_BINDING_AUDIT_READY"
  assert out["routes"]
  assert out["candidates"]
  assert out["bindings"]
  assert "routes_by_classification" in out["summary"]
  assert "bindings_by_kind" in out["summary"]
  assert out["summary"]["candidates"]["non_generated"] == []
  assert out["summary"]["candidates"]["unknown_routes"] == []
