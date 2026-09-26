import json

import pytest

from tinygrad import Tensor, dtypes
from tinygrad.llm import dense_candidate_gemm as dense
from tinygrad.llm.prefill_candidate_runtime import NV_ARTIFACT, expand_compact_candidate_set, promoted_candidate_set

NV = {"backend":"NV", "arch":"sm_120", "wave_size":32}
NEMOTRON = {"ssm_in":(17536, 3136), "ssm_out":(3200, 7680), "attn_q":(5120, 3136), "attn_kv":(1024, 3136),
            "attn_o":(3200, 5120), "ffn_up":(12544, 3136), "ffn_down":(3200, 12544), "output":(131072, 3136)}


def _raw(): return json.loads(dense.DENSE_BF16_ARTIFACT.read_text())


def test_artifact_is_reproduced_by_the_mint_from_the_checked_in_selection():
  from extra.llm_research.mint_typed_candidate_template import mint_dense_bf16
  assert mint_dense_bf16() == _raw()


def test_every_set_is_the_promoted_nv_family_with_bf16_operands():
  raw, nv = _raw(), json.loads(NV_ARTIFACT.read_text())
  assert raw["target"] == nv["target"] == NV
  from extra.llm_research.runtime_specs import NV_SM120_ASYNC_RING_CAPABILITY
  fixed = ("lane_ownership", "wmma", "dependency_policy", "numerical_mode", "epilogue")
  for compact in raw["sets"]:
    template, promoted = compact["template"], nv["template"]
    assert template["dtypes"] == {"a":"bf16", "b":"bf16", "accumulator":"fp32", "c":"fp32"}
    assert {key:template["schedule"][key] for key in fixed} == {key:promoted["schedule"][key] for key in fixed}
    if template["schedule"]["pipeline"].get("async_copy"):
      # cp.async ring family: launch-sized LDS bounded by the async-ring capability row, not the 48 KB static limit
      assert template["static_constraints"]["max_lds_bytes"] == NV_SM120_ASYNC_RING_CAPABILITY.max_lds_bytes
      assert template["schedule"]["pipeline"]["epoch_graph"] == promoted["schedule"]["pipeline"]["epoch_graph"]
      continue
    assert template["static_constraints"] == promoted["static_constraints"]
    assert template["schedule"]["pipeline"] == promoted["schedule"]["pipeline"]
    if (template["schedule"]["tile"], template["schedule"]["waves"]) == (promoted["schedule"]["tile"], promoted["schedule"]["waves"]):
      assert template["schedule"] == promoted["schedule"]   # the promoted geometry is reused verbatim


def test_routes_cover_only_exact_padded_nemotron_projections():
  routes = dense.dense_bf16_routes(**NV)
  assert routes
  for (role, m, n, k), route in routes.items():
    assert NEMOTRON[role] == (n, k) and n % dense.WEIGHT_ROW_TILE == 0 and k % route.split_k == 0
    assert route.admission.normalized_payload["workload"]["shape"] == {"m":m, "n":n, "k":k // route.split_k}
  assert dense.dense_bf16_routes("NV", "sm_89", 32) is None and dense.dense_bf16_routes("AMD", "gfx1100", 32) is None


def test_route_identity_drift_fails_closed():
  raw = _raw()
  raw["routes"][0]["split_k"] += 1
  with pytest.raises(ValueError, match="does not match|unknown"): dense.load_routes(raw, **NV)
  raw = _raw()
  raw["routes"][0]["canonical_identity"] = "0" * 64
  with pytest.raises(ValueError, match="unknown candidate"): dense.load_routes(raw, **NV)
  raw = _raw()
  raw["sets"][0]["template"]["schedule"]["tile"]["k"] *= 2
  with pytest.raises(ValueError, match="identity drifted"): dense.load_routes(raw, **NV)


def test_qwen_compact_artifact_still_requires_its_four_rows():
  assert len(promoted_candidate_set(**NV).entries) == 4
  raw = json.loads(NV_ARTIFACT.read_text())
  raw["entries"] = raw["entries"][:3]
  with pytest.raises(ValueError, match="must contain 4 rows"): expand_compact_candidate_set(raw, **NV, row_count=4)


def _fake_routes(counts, split=1):
  return {("r", m, 256, 1024): dense.DenseRoute("r", m, 256, 1024, split, None) for m in counts}


def test_plan_rows_pads_to_the_nearest_promoted_count_and_chunks_above_the_largest():
  routes = _fake_routes((32, 128, 256))
  assert [(s, t, r.m) for s, t, r in dense.plan_rows(routes, "r", 256, 1024, 8)] == [(0, 8, 32)]
  assert [(s, t, r.m) for s, t, r in dense.plan_rows(routes, "r", 256, 1024, 200)] == [(0, 200, 256)]
  assert [(s, t, r.m) for s, t, r in dense.plan_rows(routes, "r", 256, 1024, 600)] == [(0, 256, 256), (256, 256, 256), (512, 88, 128)]
  assert dense.plan_rows(routes, "r", 512, 1024, 8) is None


def test_route_declines_without_a_promoted_target_or_exact_row(monkeypatch):
  x, w = Tensor.ones(4, 3136), Tensor.ones(5120, 3136, dtype=dtypes.bfloat16)
  monkeypatch.setattr(dense, "device_target", lambda device: {"backend":"CPU", "arch":"x86_64", "wave_size":None})
  assert dense.route_dense_bf16(x, w, "attn_q", 5120) is None
  monkeypatch.setattr(dense, "_routes_for", lambda device: _fake_routes((32,)))
  assert dense.route_dense_bf16(Tensor.ones(4, 1024), Tensor.ones(256, 1024, dtype=dtypes.bfloat16), "r", 256, min_rows=8) is None
  assert dense.route_dense_bf16(Tensor.ones(4, 1024), Tensor.ones(256, 1024, dtype=dtypes.half), "r", 256) is None
  assert dense.route_dense_bf16(Tensor.ones(4, 1024), Tensor.ones(512, 1024, dtype=dtypes.bfloat16), "r", 512) is None
  assert dense.route_dense_bf16(Tensor.ones(4, 1000), Tensor.ones(256, 1024, dtype=dtypes.bfloat16), "r", 256) is None


def test_split_and_unsplit_routes_build_views_and_materialize_before_the_slice(monkeypatch):
  from tinygrad.uop.ops import Ops
  installed = []
  monkeypatch.setattr(dense, "_install", lambda admission, dims, reduce: installed.append((frozenset(dims), reduce)))
  for split in (1, 3):
    routes = {("ssm_out", 128, 3200, 7680): dense.DenseRoute("ssm_out", 128, 3200, 7680, split, None)}
    monkeypatch.setattr(dense, "_routes_for", lambda device, routes=routes: routes)
    out = dense.route_dense_bf16(Tensor.ones(1, 100, 7680), Tensor.ones(3200, 7680, dtype=dtypes.bfloat16), "ssm_out", 3136)
    assert out.shape == (1, 100, 3136) and out.dtype == dtypes.float
    product = (split, 128, 3200) if split > 1 else (128, 3200)
    assert any(u.op is Ops.CONTIGUOUS and u.shape == product for u in out.uop.toposort())
  assert installed == [(frozenset({128, 3200}), 7680), (frozenset({3, 128, 3200}), 2560)]


def test_hilo_route_stacks_both_bf16_terms_and_sums_them(monkeypatch):
  seen = []
  def fake(x, weight, role, n_out, *, min_rows=1):
    seen.append((x.shape, x.dtype, min_rows))
    return x.float().sum(-1, keepdim=True).expand(x.shape[0], n_out)
  monkeypatch.setattr(dense, "route_dense_bf16", fake)
  x = Tensor([[1.0 + 2**-12, 3.0]])
  out = dense.route_dense_bf16_hilo(x, Tensor.ones(8, 2, dtype=dtypes.bfloat16), "r", 8, min_rows=4)
  assert seen == [((2, 2), dtypes.bfloat16, 8)]
  from tinygrad.uop.ops import Ops
  # the hi + lo sum is its own buffer: fused into a Mamba-scan consumer it hit a lowering KeyError on NV
  assert any(u.op is Ops.CONTIGUOUS and u.shape == (1, 8) for u in out.uop.toposort())
  assert out.shape == (1, 8) and out.numpy()[0, 0] == 4.0 + 2**-12   # hi + lo recovers what one bf16 rounding loses


def test_bind_wraps_projections_shares_the_padded_buffer_and_falls_back(monkeypatch):
  from tinygrad import nn
  monkeypatch.setattr(dense, "_routes_for", lambda device: _fake_routes((32,)))
  class Block: pass
  class Model: pass
  model, block = Model(), Block()
  block.ffn_down = nn.Linear(4, 256, bias=False)
  block.ffn_down.weight = original = Tensor.empty(256, 4, dtype=dtypes.bfloat16)
  block.ffn_up = nn.Linear(4, 8)   # biased: left alone
  model.blk, model.output = [block], nn.Linear(4, 8, bias=False)   # fp32 weight: left alone
  assert dense.bind_candidate_linears(model, {"ffn_down":"r", "ffn_up":"r"}) == 1
  wrapped = block.ffn_down
  assert isinstance(wrapped, dense.CandidateLinear) and wrapped.padded is original   # tile-aligned: no copy
  assert wrapped.fallback.weight is wrapped.weight is original
  calls = []
  wrapped.fallback = lambda x: calls.append(x.shape) or x
  assert wrapped(Tensor.ones(3, 4)).shape == (3, 4) and calls == [(3, 4)]   # no route for K=4: fallback


def test_bind_projection_stays_out_of_the_state_dict_and_routes_hilo(monkeypatch):
  from tinygrad import nn
  from tinygrad.nn.state import get_state_dict
  lin = nn.Linear(1024, 256, bias=False)
  lin.weight = original = Tensor.empty(256, 1024, dtype=dtypes.bfloat16)
  monkeypatch.setattr(dense, "_routes_for", lambda device: None)
  assert dense.bind_projection(lin, "r") is None and dense.route_bound(lin, Tensor.ones(20, 1024)) is None
  monkeypatch.setattr(dense, "_routes_for", lambda device: _fake_routes((64,)))
  binding = dense.bind_projection(lin, "r")
  assert binding.padded is original and lin.weight is original            # tile-aligned: shared, no copy
  assert set(get_state_dict(lin)) == {"weight"}                             # the binding adds no state
  seen = []
  monkeypatch.setattr(dense, "route_dense_bf16", lambda x, w, role, n, *, min_rows=1: seen.append((x.shape, role, n, min_rows)) or
                      Tensor.zeros(x.shape[0], n))
  assert dense.route_bound(lin, Tensor.ones(1, 20, 1024), min_rows=17).shape == (1, 20, 256)
  assert seen == [((40, 1024), "r", 256, 34)]                               # hi and lo rows stacked
  assert dense.route_bound(lin, Tensor.ones(1, 20, 1024), max_rows=16) is None and len(seen) == 1


@pytest.mark.parametrize("split", [1, 2])
def test_route_scratch_shares_buffers_and_matches_the_unshared_route(monkeypatch, split):
  import numpy as np
  from tinygrad import Device
  monkeypatch.setattr(dense, "_install", lambda *args: None)
  monkeypatch.setattr(dense, "_routes_for", lambda device: {("r", 8, 32, 16): dense.DenseRoute("r", 8, 32, 16, split, None)})
  monkeypatch.setattr(dense, "_SCRATCH_BUFFERS", {})
  rng = np.random.default_rng(0)
  w1, w2 = (Tensor(rng.standard_normal((32, 16), dtype=np.float32)).cast(dtypes.bfloat16).realize() for _ in range(2))
  x = Tensor(rng.standard_normal((1, 4, 16), dtype=np.float32)).realize()
  unshared = [dense.route_dense_bf16_hilo(x, w, "r", 30).numpy() for w in (w1, w2)]
  assert dense._SCRATCH_BUFFERS == {}                         # off by default
  with dense.route_scratch():
    shared = [dense.route_dense_bf16_hilo(x, w, "r", 30).numpy() for w in (w1, w2)]
  assert len(dense._SCRATCH_BUFFERS) == 2                      # one operand + one product buffer for both projections
  for a, b in zip(unshared, shared): np.testing.assert_array_equal(a, b)


def test_route_scratch_lifts_the_row_cap(monkeypatch):
  seen = []
  monkeypatch.setattr(dense, "route_dense_bf16_hilo", lambda x, w, role, n, *, min_rows=1: seen.append(x.shape) or x)
  class Lin: pass
  lin = Lin(); lin.weight = Tensor.empty(8, 4); lin._candidate = dense.CandidateBinding("r", lin.weight)
  assert dense.route_bound(lin, Tensor.ones(200, 4), max_rows=128) is None and seen == []
  with dense.route_scratch(): assert dense.route_bound(lin, Tensor.ones(200, 4), max_rows=128) is not None
  assert seen == [(200, 4)]
