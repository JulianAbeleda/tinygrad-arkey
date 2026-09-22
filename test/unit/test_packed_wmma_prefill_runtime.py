from pathlib import Path
from types import SimpleNamespace

from tinygrad import Tensor, dtypes
from tinygrad.llm import packed_wmma_prefill as runtime
from tinygrad.llm.qk_layout import Q4_K, Q6_K


EXPECTED = {
  ("Q4_K", "attn_qo", (512, 5120, 5120)): (128, 32, 32, 4, 1, 1),
  ("Q4_K", "attn_kv", (512, 1024, 5120)): (64, 32, 32, 2, 1, 1),
  ("Q4_K", "ffn_gate_up", (512, 17408, 5120)): (256, 64, 32, 8, 1, 1),
  ("Q4_K", "ffn_down", (512, 5120, 17408)): (256, 128, 32, 8, 2, 2),
  ("Q6_K", "attn_kv", (512, 1024, 5120)): (64, 32, 32, 2, 1, 1),
  ("Q6_K", "ffn_down", (512, 5120, 17408)): (256, 64, 32, 8, 1, 1),
}


def _spec(quant: str, role: str, shape: tuple[int, int, int]):
  return SimpleNamespace(quant=Q4_K if quant == "Q4_K" else Q6_K, role=role,
                         m=shape[0], n=shape[1], k=shape[2])


def setup_function(): runtime.set_packed_wmma_canary_verifier(None)


def test_exact_six_selected_rows_and_geometry_are_frozen():
  assert len(runtime.PACKED_WMMA_ROUTES) == 6
  assert {(r.quant.name, r.role, r.shape): r.geometry for r in runtime.PACKED_WMMA_ROUTES} == EXPECTED
  assert runtime.PACKED_WMMA_GEOM == {(r.quant, r.role): r.geom for r in runtime.PACKED_WMMA_ROUTES}
  assert all(len(r.canonical_identity) == 64 for r in runtime.PACKED_WMMA_ROUTES)


def test_selector_matches_every_selected_row_and_declines_unknown_shapes():
  for quant, role, shape in EXPECTED:
    selected = runtime.select_packed_wmma_prefill_candidate(object(), _spec(quant, role, shape))
    assert selected is not None and selected.quant.name == quant
  assert runtime.select_packed_wmma_prefill_candidate(object(), _spec("Q4_K", "attn_qo", (512, 4096, 4096))) is None
  assert runtime.select_packed_wmma_prefill_candidate(object(), _spec("Q6_K", "ffn_gate_up", (512, 17408, 5120))) is None


def test_gate_is_once_per_exact_row_and_fails_closed():
  calls = []
  runtime.set_packed_wmma_canary_verifier(lambda row: (calls.append(row.canonical_identity) is None, 0.0))
  quant, role, shape = next(iter(EXPECTED))
  fmt = Q4_K if quant == "Q4_K" else Q6_K
  assert runtime.gate_combo(fmt, role, shape)
  assert runtime.gate_combo(fmt, role, shape)
  assert len(calls) == 1
  assert runtime.gate_result(fmt, role, shape) == (True, 0.0)
  assert not runtime.gate_combo(Q4_K, "attn_qo", (128, 5120, 5120))


def test_verifier_error_declines_without_entry():
  runtime.set_packed_wmma_canary_verifier(lambda row: (_ for _ in ()).throw(RuntimeError("canary")))
  quant, role, shape = next(iter(EXPECTED))
  fmt = Q4_K if quant == "Q4_K" else Q6_K
  assert not runtime.gate_combo(fmt, role, shape)
  try: runtime.warmstart_entry(fmt, role, shape)
  except ValueError: pass
  else: raise AssertionError("failed canary must not produce a warmstart entry")


def test_exp_qualification_adapter_installs_on_the_production_seam(monkeypatch):
  from extra.llm_research.prefill import packed_wmma_production_canary as adapter
  seen = []
  monkeypatch.setattr(adapter, "verify_production_row",
    lambda row, *, timeout_seconds, device, use_lane=False: (seen.append((row, timeout_seconds, device)) is None, 0.0))
  adapter.install_production_qualification_verifier(timeout_seconds=17.0)
  row = runtime.PACKED_WMMA_ROUTES[0]
  assert runtime.gate_combo(row.quant, row.role, row.shape)
  assert seen == [(row, 17.0, "AMD")]


def test_exp_qualification_adapter_threads_an_explicit_device(monkeypatch):
  from extra.llm_research.prefill import packed_wmma_production_canary as adapter
  seen = []
  monkeypatch.setattr(adapter, "verify_production_row",
    lambda row, *, timeout_seconds, device, use_lane=False: (seen.append((row, timeout_seconds, device)) is None, 0.0))
  adapter.install_production_qualification_verifier(timeout_seconds=17.0, device="METAL")
  row = runtime.PACKED_WMMA_ROUTES[0]
  assert runtime.gate_combo(row.quant, row.role, row.shape)
  assert seen == [(row, 17.0, "METAL")]


def test_exp_qualification_adapter_lane_path_is_opt_in_and_uses_the_canonical_probe(monkeypatch):
  # C4: the lane qualifies a production row through run_precontract_probe (the canonical
  # three-axis probe) with the row's own quant/role/shape/geometry and the requested device;
  # run_canary stays the default and is not invoked on the lane path.
  from extra.llm_research.prefill import packed_wmma_production_canary as adapter
  from extra.llm_research.prefill import precontract_probe_lane as lane
  calls = []

  class _Result:
    passed = True
    max_abs_error = 0.001

  monkeypatch.setattr(lane, "run_precontract_probe", lambda config: (calls.append(config), _Result())[1])
  monkeypatch.setattr(adapter, "run_canary", lambda *a, **k: pytest.fail("run_canary must not run on the lane path"))
  row = runtime.PACKED_WMMA_ROUTES[0]
  passed, error = adapter.verify_production_row(row, device="AMD", use_lane=True)
  assert passed is True and error == 0.001
  assert len(calls) == 1
  config = calls[0]
  assert (config.quant, config.role, config.shape, config.geometry, config.device) == \
    (row.quant.name, row.role, row.shape, row.geometry, "AMD")


def test_exp_qualification_adapter_default_path_still_routes_through_run_canary(monkeypatch):
  from extra.llm_research.prefill import packed_wmma_production_canary as adapter
  from extra.llm_research.prefill import precontract_probe_lane as lane
  calls = []
  monkeypatch.setattr(adapter, "build_artifact", lambda *a, **k: None)
  monkeypatch.setattr(adapter, "run_canary", lambda *a, **k: (calls.append(a), {"passed": True, "guarded": {"max_abs_error": 0.0}})[1])
  monkeypatch.setattr(lane, "run_precontract_probe", lambda config: pytest.fail("lane must not run on the default path"))
  row = runtime.PACKED_WMMA_ROUTES[0]
  passed, error = adapter.verify_production_row(row, device="AMD", use_lane=False)
  assert passed is True and error == 0.0
  assert len(calls) == 1


def test_warmstart_context_preserves_geometry_identity_and_packed_semantics():
  for row in runtime.PACKED_WMMA_ROUTES:
    entry = runtime.warmstart_entry(row.quant, row.role, row.shape)
    assert (entry["m"], entry["n"], entry["k"]) == row.shape
    assert entry["canonical_identity"] == row.canonical_identity
    assert entry["context"].canonical_identity == row.canonical_identity
    assert entry["context"].geometry.tile == row.geometry[:3]
    assert entry["context"].geometry.waves == row.geometry[3:5]
    assert (entry["context"].pipeline.buffer_count if entry["context"].pipeline is not None else 1) == row.geometry[5]
    assert entry["transform"].quant_format == row.quant
    assert entry["transform"].storage_dtype == (dtypes.uint32 if row.quant is Q4_K else dtypes.uint16)
    assert entry["one_buffer"] is False


def test_packed_carrier_is_movement_only_and_retains_storage_source():
  transform = runtime.warmstart_entry(Q4_K, "attn_kv", (512, 1024, 5120))["transform"]
  source = Tensor.empty(transform.packed_bytes // transform.storage_width, dtype=transform.storage_dtype, device="CPU")
  carrier = runtime.packed_half_carrier(source, transform, transform.rows, transform.k)
  assert carrier.shape == (transform.rows, transform.k)
  assert carrier.dtype == dtypes.half
  assert source.uop in carrier.uop.backward_slice


def test_production_module_has_no_research_import():
  source = Path(runtime.__file__).read_text()
  assert "extra.llm_research" not in source
