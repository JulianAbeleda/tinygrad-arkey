"""E1/E2/E3/K11/E5: a promotion record the reader cannot trust refuses; it never admits every shape."""
import json, pathlib
from types import SimpleNamespace
import pytest

from tinygrad.llm import model_route_plan as mrp
from tinygrad.llm.decode_routes import DECODE_ROUTE_SHAPE_PROBES
from tinygrad.llm.model import _decode_route_promotions, _model_weight_formats
from tinygrad.llm.model_route_plan import DecodeRouteShapeGate, DecodeRouteShapes

GENERATED = pathlib.Path(mrp.__file__).with_name("generated")
NV = ("NV", "sm_120")


def _write(tmp_path, route, record):
  (tmp_path / mrp.route_policy_file(route)).write_text(json.dumps(record))
  mrp.route_record.cache_clear()


# ---- E1 ----

def test_a_missing_record_refuses(tmp_path):
  assert not mrp.shape_measured("decode_nothing_here", {"dim": 4096}, tmp_path)
  assert mrp.measured_entry("decode_nothing_here", {"dim": 4096}, field="measured_leases", directory=tmp_path) is None
  with pytest.raises(ValueError, match="no promotion record"): mrp.route_measured_shapes("decode_nothing_here", tmp_path)


def test_a_misspelled_record_field_refuses(tmp_path):
  _write(tmp_path, "decode_r", {"schema": "boltbeam.route_policy.v1", "route": "decode_r", "description": "", "promoted_targets": [],
                                "measured_model": [{"dim": 4096}]})  # misspelled: it used to admit every shape
  assert not mrp.shape_measured("decode_r", {"dim": 5120}, tmp_path)
  assert not mrp.shape_measured("decode_r", {"dim": 4096}, tmp_path)
  with pytest.raises(ValueError, match="misspelled"): mrp.route_measured_shapes("decode_r", tmp_path)


def test_a_misspelled_field_argument_refuses(tmp_path):
  _write(tmp_path, "decode_r", {"schema": "boltbeam.route_policy.v1", "promoted_targets": [], "measured_models": [{"dim": 4096}]})
  assert mrp.shape_measured("decode_r", {"dim": 4096}, tmp_path)
  assert not mrp.shape_measured("decode_r", {"dim": 4096}, tmp_path, field="measured_model")


def test_a_record_without_a_shape_field_leaves_the_kernels_validate_as_the_rule(tmp_path):
  _write(tmp_path, "decode_r", {"schema": "boltbeam.route_policy.v1", "promoted_targets": []})
  assert mrp.shape_measured("decode_r", {"dim": 1}, tmp_path)


def test_every_record_in_the_tree_uses_known_field_names():
  records = sorted(GENERATED.glob("*-route-policy.json"))
  assert records
  for path in records:
    assert set(json.loads(path.read_text())) <= mrp.ROUTE_RECORD_FIELDS, path.name


def test_every_route_the_gate_asks_about_has_a_record():
  gate = _decode_route_promotions(SimpleNamespace(dim=4096, hidden_dim=12288, n_heads=32, n_kv_heads=8, head_dim=128,
                                                  vocab_size=151936, norm_eps=1e-6), NV, getenv_fn=lambda *a, **k: 0)
  for route in gate.decisions: mrp.route_record(route)  # raises when missing or misspelled


# ---- E2 / E3 ----

def test_an_empty_formats_list_promotes_no_format(tmp_path):
  path = tmp_path / "r.json"
  path.write_text(json.dumps({"formats": []}))
  assert mrp.load_decode_q8_0_gemv_formats(str(path)) == frozenset()
  path.write_text(json.dumps({}))
  with pytest.raises(ValueError, match="'formats' must be a list"): mrp.load_decode_q8_0_gemv_formats(str(path))


def test_an_unknown_format_is_one_clear_error_everywhere(tmp_path):
  path = tmp_path / "r.json"
  path.write_text(json.dumps({"formats": ["Q8_O"]}))
  with pytest.raises(ValueError, match=r"unknown block-quant format\(s\) \['Q8_O'\]"): mrp.load_decode_q8_0_gemv_formats(str(path))
  with pytest.raises(ValueError, match="TINYGRAD_BLOCK_QUANT_FORMATS_DISABLE"):
    mrp.decode_q8_0_gemv_formats(lambda k, d=0: "Q8_O")
  from tinygrad.llm.qk_primitives import _install_block_quant_primitives
  with pytest.raises(ValueError, match="installer"):
    _install_block_quant_primitives(None, {"tensor_infos": []}, None, frozenset({"Q8_O"}))


# ---- K11 ----

def test_the_model_gate_checks_only_the_formats_the_model_has(monkeypatch):
  from tinygrad.llm import decode_routes
  monkeypatch.setattr(decode_routes, "decode_q8_0_gemv_formats", lambda *a, **k: frozenset({"Q8_0", "Q5_K"}))
  shapes = dict(dim=3136, hidden_dim=12544, n_heads=40, n_kv_heads=8, head_dim=128, vocab=131072, eps=1e-6, target=NV)
  probe = DECODE_ROUTE_SHAPE_PROBES["decode_q8_0_gemv"]
  with pytest.raises(ValueError): probe(DecodeRouteShapes(**shapes))  # formats unknown: every promoted format is checked
  with pytest.raises(ValueError): probe(DecodeRouteShapes(**shapes, weight_formats=frozenset({"Q8_0", "Q5_K"})))
  probe(DecodeRouteShapes(**shapes, weight_formats=frozenset({"Q8_0", "Q6_K"})))  # no Q5_K weight: Q5_K cannot refuse it


def test_model_weight_formats_reads_the_2d_weights():
  meta = {"tensor_infos": [("a.weight", [4096, 4096], 8, 0), ("b.weight", [4096, 1024], 14, 0), ("n.weight", [4096], 0, 0)]}
  assert _model_weight_formats(meta) == frozenset({"Q8_0", "Q6_K"})
  assert _model_weight_formats(None) is None


# ---- E5 ----

def test_a_refused_flash_route_is_on_the_plan():
  gate = DecodeRouteShapeGate(DecodeRouteShapes(4096, 12288, 48, 1, 128, 151936, 1e-6, NV), {})
  from tinygrad.llm.flash_decode_attention import flash_decode_route_refusal
  gate.refuse("decode_flash_live_split", flash_decode_route_refusal(48, 1, 128))
  plan = mrp.build_model_route_plan(shape_gate=gate)
  assert not gate.admitted("decode_flash_live_split")
  assert any(line.startswith("decode_flash_live_split: ") and "threads per block" in line for line in plan.report_lines())
