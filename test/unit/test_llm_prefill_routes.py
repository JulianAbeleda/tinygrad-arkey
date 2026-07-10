import os
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def clean_prefill_route_env():
  old = {k: os.environ.get(k) for k in ("PREFILL_ROUTE", "PREFILL_QK_DIRECT", "PREFILL_ROUTE_STRICT",
                                        "QK_GENERATED_POLICY_STRICT", "PREFILL_DIRECT_QUANTS",
                                        "PREFILL_DIRECT_TENSORS", "PREFILL_DIRECT_SKIP_TENSORS",
                                        "PREFILL_Q4K_PACKED_LOAD", "PREFILL_Q6K_PACKED_LOAD",
                                        "PREFILL_DIRECT_B_UPCAST", "PREFILL_DIRECT_OUT", "PREFILL_DIRECT_PARTS",
                                        "PREFILL_DIRECT_Q4K_PARTS", "PREFILL_DIRECT_Q6K_PARTS",
                                        "PREFILL_DIRECT_FFN_GATE_UP_PARTS", "PREFILL_DIRECT_FFN_DOWN_PARTS",
                                        "PREFILL_Q4K_Q8", "PREFILL_Q4K_REDUCE_OUT", "PREFILL_Q4K_DIRECT_OPTS",
                                        "PREFILL_Q4K_DIRECT_EXTRA_OPTS", "PREFILL_Q6K_DIRECT_OPTS",
                                        "PREFILL_Q6K_DIRECT_EXTRA_OPTS", "PREFILL_DIRECT_FFN_GATE_UP_OPTS",
                                        "PREFILL_DIRECT_FFN_GATE_UP_EXTRA_OPTS", "PREFILL_Q4K_DIRECT_SCHEDULE",
                                        "PREFILL_Q4K_WMMA_TILED_M_TILE", "PREFILL_Q4K_WMMA_TILED_N_TILE",
                                        "PREFILL_Q4K_WMMA_TILED_GROUP_TILE", "PREFILL_Q4K_WMMA_TILED_MAX_RAW_TILE_STEPS",
                                        "PREFILL_QK_GENERATED_TILE",
                                        "PREFILL_QK_GENERATED_TILE_ROLES", "PREFILL_QK_GENERATED_TILE_MODE",
                                        "PREFILL_QK_GENERATED_TILE_ROWS", "PREFILL_QK_GENERATED_TILE_TOKENS",
                                        "PREFILL_UBATCH")}
  for k in old: os.environ.pop(k, None)
  yield
  for k, v in old.items():
    if v is None: os.environ.pop(k, None)
    else: os.environ[k] = v


def test_prefill_route_policy_defaults_auto():
  from tinygrad.llm.prefill_routes import prefill_route_policy
  assert prefill_route_policy() == "auto"


def test_prefill_qk_direct_alias_selects_direct_packed():
  from tinygrad.llm.prefill_routes import prefill_route_policy
  os.environ["PREFILL_QK_DIRECT"] = "1"
  assert prefill_route_policy() == "direct_packed"


def test_prefill_route_rejects_unknown_policy():
  from tinygrad.llm.prefill_routes import prefill_route_policy
  os.environ["PREFILL_ROUTE"] = "hardcoded_14b"
  with pytest.raises(ValueError):
    prefill_route_policy()


def test_auto_keeps_resident_fp16_when_it_fits():
  from tinygrad.llm.prefill_routes import prefill_route_wants_resident_fp16
  assert prefill_route_wants_resident_fp16(est_gb=12.0, budget_gb=18.0, has_direct_packed=True, prefill_chunked=False)


def test_auto_skips_resident_fp16_when_direct_packed_exists_and_fp16_exceeds_budget():
  from tinygrad.llm.prefill_routes import prefill_route_wants_resident_fp16
  assert not prefill_route_wants_resident_fp16(est_gb=24.0, budget_gb=18.0, has_direct_packed=True, prefill_chunked=False)


def test_fp16_policy_keeps_resident_fp16_even_over_budget():
  from tinygrad.llm.prefill_routes import prefill_route_wants_resident_fp16
  os.environ["PREFILL_ROUTE"] = "fp16"
  assert prefill_route_wants_resident_fp16(est_gb=24.0, budget_gb=18.0, has_direct_packed=True, prefill_chunked=False)


def test_direct_policy_skips_resident_fp16_for_8b_experiments_too():
  from tinygrad.llm.prefill_routes import prefill_route_wants_resident_fp16
  os.environ["PREFILL_ROUTE"] = "direct_packed"
  assert not prefill_route_wants_resident_fp16(est_gb=12.0, budget_gb=18.0, has_direct_packed=True, prefill_chunked=False)


def test_direct_packed_quant_selector():
  from tinygrad.llm.prefill_routes import _direct_packed_enabled_for
  lin = type("Lin", (), {"name": "blk.0.ffn_down.weight"})()
  os.environ["PREFILL_DIRECT_QUANTS"] = "Q4_K"
  assert _direct_packed_enabled_for(lin, "Q4_K")
  assert not _direct_packed_enabled_for(lin, "Q6_K")


def test_direct_packed_tensor_selector():
  from tinygrad.llm.prefill_routes import _direct_packed_enabled_for
  lin = type("Lin", (), {"name": "blk.0.ffn_down.weight"})()
  os.environ["PREFILL_DIRECT_TENSORS"] = "ffn_gate,attn_q"
  assert not _direct_packed_enabled_for(lin, "Q4_K")
  os.environ["PREFILL_DIRECT_TENSORS"] = "ffn_down"
  assert _direct_packed_enabled_for(lin, "Q4_K")


def test_direct_packed_parts_prefers_role_then_quant_then_global():
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_parts
  lin = type("Lin", (), {"parts": 1})()
  spec = PrefillLinearRouteSpec("direct_packed", "q4k", "ffn_gate_up", 512, 17408, 5120)
  os.environ["PREFILL_DIRECT_PARTS"] = "2"
  assert _direct_packed_parts(lin, spec) == 2
  os.environ["PREFILL_DIRECT_Q4K_PARTS"] = "3"
  assert _direct_packed_parts(lin, spec) == 3
  os.environ["PREFILL_DIRECT_FFN_GATE_UP_PARTS"] = "4"
  assert _direct_packed_parts(lin, spec) == 4


def test_direct_packed_q4_ffn_down_defaults_to_single_part():
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_parts
  lin = type("Lin", (), {"parts": 4, "name": "blk.0.ffn_down.weight"})()
  spec = PrefillLinearRouteSpec("direct_packed", "q4k", "", 512, 5120, 17408)
  assert _direct_packed_parts(lin, spec) == 1
  os.environ["PREFILL_DIRECT_FFN_DOWN_PARTS"] = "2"
  assert _direct_packed_parts(lin, spec) == 2


def test_direct_packed_q6_defaults_to_single_part():
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_parts
  lin = type("Lin", (), {"parts": 4, "name": "blk.0.ffn_down.weight"})()
  spec = PrefillLinearRouteSpec("direct_packed", "q6k", "", 512, 5120, 17408)
  assert _direct_packed_parts(lin, spec) == 1
  os.environ["PREFILL_DIRECT_Q6K_PARTS"] = "2"
  assert _direct_packed_parts(lin, spec) == 2


def test_direct_packed_q4_opts_override_and_extra():
  from tinygrad.codegen.opt import OptOps
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec, _direct_packed_opts
  lin = type("Lin", (), {"opts": ()})()
  spec = PrefillLinearRouteSpec("direct_packed", "q4k", "ffn_gate_up", 512, 17408, 5120)
  opts = _direct_packed_opts(lin, spec)
  assert [(x.op, x.axis, x.arg) for x in opts] == [
    (OptOps.LOCAL, 0, 16), (OptOps.LOCAL, 1, 16), (OptOps.UPCAST, 0, 4), (OptOps.UPCAST, 1, 4)]
  os.environ["PREFILL_Q4K_DIRECT_SCHEDULE"] = "legacy"
  os.environ["PREFILL_Q4K_DIRECT_EXTRA_OPTS"] = "UPCAST:0:4"
  opts = _direct_packed_opts(lin, spec)
  assert opts[-2].op is OptOps.UPCAST and opts[-2].axis == 1 and opts[-2].arg == 4
  assert opts[-1].op is OptOps.UPCAST and opts[-1].axis == 0 and opts[-1].arg == 4
  os.environ.pop("PREFILL_Q4K_DIRECT_SCHEDULE")
  os.environ["PREFILL_Q4K_DIRECT_OPTS"] = "LOCAL:0:16,UPCAST:1:4"
  opts = _direct_packed_opts(lin, spec)
  assert [(x.op, x.axis, x.arg) for x in opts] == [(OptOps.LOCAL, 0, 16), (OptOps.UPCAST, 1, 4)]
  os.environ["PREFILL_DIRECT_FFN_GATE_UP_OPTS"] = "LOCAL:0:64,GROUP:0:10,UPCAST:1:4"
  opts = _direct_packed_opts(lin, spec)
  assert [(x.op, x.axis, x.arg) for x in opts] == [(OptOps.LOCAL, 0, 64), (OptOps.GROUP, 0, 10), (OptOps.UPCAST, 1, 4)]


def test_prefill_q4k_q8_legacy_gemm_flag_is_rejected():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode
  os.environ["PREFILL_Q4K_Q8"] = "1"
  with pytest.raises(ValueError, match="PREFILL_Q4K_Q8"):
    prefill_q4k_q8_mode()


def test_prefill_q4k_q8_wmma_flag_is_valid_route_env():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode, prefill_route_policy
  os.environ["PREFILL_Q4K_Q8"] = "wmma"
  assert prefill_route_policy() == "auto"
  assert prefill_q4k_q8_mode() == "wmma"


def test_prefill_q4k_q8_mmq_direct_flag_is_rejected():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode
  os.environ["PREFILL_Q4K_Q8"] = "mmq_direct"
  with pytest.raises(ValueError, match="PREFILL_Q4K_Q8"):
    prefill_q4k_q8_mode()


def test_prefill_q4k_q8_wmma_tiled_flag_is_valid_but_explicit():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode, prefill_route_policy
  os.environ["PREFILL_Q4K_Q8"] = "wmma_tiled"
  assert prefill_route_policy() == "auto"
  assert prefill_q4k_q8_mode() == "wmma_tiled"


def test_prefill_q4k_q8_rejects_unknown_mode():
  from tinygrad.llm.prefill_routes import prefill_q4k_q8_mode
  os.environ["PREFILL_Q4K_Q8"] = "surprise_tensorcore"
  with pytest.raises(ValueError, match="PREFILL_Q4K_Q8"):
    prefill_q4k_q8_mode()


def test_direct_packed_route_spec_exports_runtime_op_spec():
  from tinygrad.llm.prefill_routes import PrefillLinearRouteSpec
  q4 = PrefillLinearRouteSpec("direct_packed", "q4k", "ffn_gate_up", 512, 17408, 5120).runtime_op_spec()
  assert q4.family == "QuantizedLinear"
  assert q4.phase == "prefill"
  assert q4.role == "ffn_gate_up"
  assert q4.weight.format == "Q4_K"
  assert q4.activation.format == "fp16"
  assert q4.shape == {"M": 512, "N": 17408, "K": 5120}
  q6 = PrefillLinearRouteSpec("direct_packed", "q6k", "", 512, 5120, 17408).runtime_op_spec()
  assert q6.role == "unknown"
  assert q6.weight.format == "Q6_K"


class _PrefillTensorStub:
  shape = (1, 512, 256)
  device = "CPU"

  def __getitem__(self, _idx):
    return self

  def cast(self, *_args, **_kwargs):
    return self

  def contiguous(self):
    return self

  def reshape(self, *_args, **_kwargs):
    return self

  def custom_kernel(self, *_args, **_kwargs):
    return (self,)

  def sum(self, *_args, **_kwargs):
    return self

  def transpose(self, *_args, **_kwargs):
    return self


class _TensorFactoryStub(_PrefillTensorStub):
  @classmethod
  def empty(cls, *_args, **_kwargs):
    return cls()


class _SmallPrefillTensorStub(_PrefillTensorStub):
  shape = (1, 37, 384)


class _Q6PrefillWeight:
  def to(self, *_args, **_kwargs):
    return self


class _Q4PrefillWeight:
  def to(self, *_args, **_kwargs):
    return self


def _q4_prefill_linear(parts=1):
  return SimpleNamespace(
    bias=None, in_features=256, out_features=16, parts=parts, opts=(), name="blk.0.ffn_gate.weight",
    q4k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q4PrefillWeight())


def _q6_prefill_linear(parts=1):
  return SimpleNamespace(
    bias=None, in_features=256, out_features=16, parts=parts, opts=(), name="blk.0.ffn_down.weight",
    q6k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q6PrefillWeight())


def test_direct_packed_q4_request_facts_are_built_from_fake_module():
  from tinygrad.llm.prefill_routes import build_direct_packed_prefill_request

  lin = SimpleNamespace(
    bias=None, in_features=384, out_features=96, parts=1, opts=(), name="custom.layers.7.ffn_gate_proj",
    _prefill_graph_role="mlp_expand", q4k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q4PrefillWeight())
  req = build_direct_packed_prefill_request(lin, _SmallPrefillTensorStub(), ubatch=123)
  assert req is not None
  assert req.route_facts == {
    "quant": "Q4_K", "role": "mlp_expand", "M": 37, "N": 96, "K": 384, "bias": False, "ubatch": 123}


def test_direct_packed_q6_shadow_request_facts_are_built_from_fake_module():
  from tinygrad.llm.prefill_routes import select_direct_packed_prefill_shadow_request

  lin = SimpleNamespace(
    bias=object(), in_features=384, out_features=80, parts=1, opts=(), name="toy.block.3.ffn_down",
    q6k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q6PrefillWeight())
  req = select_direct_packed_prefill_shadow_request(lin, _SmallPrefillTensorStub(), ubatch=64)
  assert req is not None
  assert req.route_facts == {
    "quant": "Q6_K", "role": "ffn_down", "M": 37, "N": 80, "K": 384, "bias": True, "ubatch": 64}


def test_q4_direct_packed_prefill_default_uses_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q4k_gemm_packed_load_direct_out_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q4_K packed prefill default must not use q4k_gemm_packed_load_direct_out_kernel")))
  monkeypatch.setattr(prefill_routes.qk_ops, "q4k_gemm_packed_load_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q4_K packed prefill default must not use q4k_gemm_packed_load_kernel")))

  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q4k_packed_prefill_generated", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_q4_prefill_linear(), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "direct_out"


def test_q4_direct_packed_prefill_partials_use_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  os.environ["PREFILL_DIRECT_Q4K_PARTS"] = "2"
  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q4k_gemm_packed_load_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q4_K packed prefill partials must not use q4k_gemm_packed_load_kernel")))

  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q4k_packed_prefill_generated", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_q4_prefill_linear(parts=2), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "partials"


def test_q4_reduce_out_uses_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  os.environ["PREFILL_Q4K_REDUCE_OUT"] = "1"
  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q4k_gemm_packed_load_reduce_out_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q4_K reduce_out should use Q4KPrefillRouteSpec")))
  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q4k_packed_prefill_generated", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_q4_prefill_linear(), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "reduce_out"


def test_q4_generated_tile_flag_is_retired(monkeypatch):
  from tinygrad.llm import prefill_routes

  os.environ["PREFILL_QK_GENERATED_TILE"] = "1"
  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  with pytest.raises(RuntimeError, match="PREFILL_QK_GENERATED_TILE was retired"):
    prefill_routes.route_direct_packed_prefill(_q4_prefill_linear(), _PrefillTensorStub())


def test_q4_wmma_tiled_small_multitile_uses_lifecycle(monkeypatch):
  from tinygrad.llm import prefill_routes

  os.environ["PREFILL_Q4K_Q8"] = "wmma_tiled"
  os.environ["PREFILL_Q4K_WMMA_TILED_MAX_RAW_TILE_STEPS"] = "1024"
  calls = []
  monkeypatch.setattr(prefill_routes.qk_ops, "q8_1_quantize", lambda x: ("xq", "xscales"))

  def one_tile(*_args, **_kwargs):
    calls.append("one_tile")
    raise NotImplementedError("multi tile")

  def lifecycle(*_args, **_kwargs):
    calls.append("lifecycle")
    return _PrefillTensorStub()

  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_int8_wmma_tiled_prefill_tensor", one_tile)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_int8_wmma_tiled_lifecycle_tensor", lifecycle)
  out = prefill_routes.route_direct_packed_prefill(SimpleNamespace(
    bias=None, in_features=256, out_features=32, parts=1, opts=(), name="blk.0.ffn_gate.weight",
    q4k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q4PrefillWeight()), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls == ["one_tile", "lifecycle"]


def test_q4_wmma_tiled_large_shape_keeps_explicit_blocker(monkeypatch):
  from tinygrad.llm import prefill_routes

  class _LargePrefillTensorStub(_PrefillTensorStub):
    shape = (1, 512, 5120)

  os.environ["PREFILL_Q4K_Q8"] = "wmma_tiled"
  monkeypatch.setattr(prefill_routes.qk_ops, "q8_1_quantize", lambda x: ("xq", "xscales"))
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_int8_wmma_tiled_prefill_tensor",
                      lambda *_args, **_kwargs: (_ for _ in ()).throw(NotImplementedError("full role")))
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q4k_int8_wmma_tiled_lifecycle_tensor",
                      lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must stay blocked")))
  with pytest.raises(RuntimeError, match="raw_tile_steps=.*max_raw_tile_steps"):
    prefill_routes.route_direct_packed_prefill(SimpleNamespace(
      bias=None, in_features=5120, out_features=5120, parts=1, opts=(), name="blk.0.attn_q.weight",
      q4k_storage=SimpleNamespace(), prefill_packed_weight=lambda: _Q4PrefillWeight()), _LargePrefillTensorStub())


def test_q6_direct_packed_prefill_default_uses_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q6k_gemm_packed_load_direct_out_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q6_K packed prefill default must not use q6k_gemm_packed_load_direct_out_kernel")))
  monkeypatch.setattr(prefill_routes.qk_ops, "q6k_gemm_packed_load_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q6_K packed prefill default must not use q6k_gemm_packed_load_kernel")))

  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q6k_packed_prefill", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q6k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_q6_prefill_linear(), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "direct_out"


def test_q6_direct_packed_prefill_partials_use_generated_descriptor(monkeypatch):
  from tinygrad.llm import prefill_routes

  os.environ["PREFILL_DIRECT_Q6K_PARTS"] = "2"
  monkeypatch.setattr(prefill_routes, "Tensor", _TensorFactoryStub)
  monkeypatch.setattr(prefill_routes.qk_ops, "q6k_gemm_packed_load_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("Q6_K packed prefill partials must not use q6k_gemm_packed_load_kernel")))

  calls = []
  def describe(*args, **kwargs):
    calls.append(("describe", args, kwargs))
    return SimpleNamespace(output_layout=kwargs["output_layout"])
  monkeypatch.setattr(prefill_routes.qk_ops, "describe_q6k_packed_prefill", describe)
  monkeypatch.setattr(prefill_routes.qk_ops, "emit_q6k_packed_prefill_kernel", lambda spec: ("generated", spec.output_layout))

  out = prefill_routes.route_direct_packed_prefill(_q6_prefill_linear(parts=2), _PrefillTensorStub())
  assert isinstance(out, _PrefillTensorStub)
  assert calls[0][2]["output_layout"] == "partials"
