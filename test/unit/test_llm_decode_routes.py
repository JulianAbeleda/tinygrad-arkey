from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from tinygrad import Tensor, UOp
from tinygrad.uop.ops import Ops
from tinygrad.llm import decode_routes
from tinygrad.llm.model import Transformer, _generation_input_slice


def test_generation_decode_slice_retains_lazy_symbolic_jit_contract():
  tokens = Tensor.empty(1, 32)
  start = UOp.variable("start", 0, 31).bind(7)
  extent = UOp.variable("extent", 1, 8).bind(1)
  decode = _generation_input_slice(tokens, start, extent, 1)
  # Do not eagerly materialize the prompt-side token. TinyJit normalizes this
  # lazy view to the same input contract as decode feedback during preparation.
  assert decode.uop.op is Ops.SHRINK and decode.shape[1].vmax == 8
  assert _generation_input_slice(tokens, start, UOp.variable("chunk", 1, 8), 8).shape[1].vmax == 8


def test_reset_generation_state_forgets_dense_request_cache():
  model = object.__new__(Transformer)
  model.blk, model._cached_tokens = [], [1, 2, 3]
  model.reset_generation_state()
  assert model._cached_tokens == []


class _TensorShapeOnly:
  def __init__(self, shape):
    self.shape = shape


class _MockTensor:
  def __init__(self):
    self.shape = (1, 1, 256)
    self.device = "CPU"
    self.calls = []

  def __getitem__(self, idx):
    self.calls.append(("getitem", idx))
    return self

  def reshape(self, *shape):
    self.calls.append(("reshape", shape))
    self.shape = shape
    return self

  def cast(self, dtype):
    self.calls.append(("cast", dtype))
    return self

  def contiguous(self):
    self.calls.append(("contiguous",))
    return self


class _MockPartial:
  def __init__(self, result):
    self.result = result

  def sum(self, axis):
    self.axis = axis
    return self

  def reshape(self, *shape):
    self.reshape_shape = shape
    return self.result


class _MockPartials:
  def __init__(self, result):
    self.result = result
    self.kernel_calls = []

  def custom_kernel(self, *args, **kwargs):
    self.kernel_calls.append((args, kwargs))
    return (self.result,)


class _Words:
  def to(self, *_args, **_kwargs):
    return self

  def contiguous(self):
    return self


def test_q4k_smallk_batched_routes_to_fallback(monkeypatch):
  monkeypatch.setattr(decode_routes.qk_ops, "q4k_gemm_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("batched K!=1 should not use q4k_gemm_kernel")))
  linear = SimpleNamespace(decode_enabled=True, bias=None, in_features=8, name="batched_test_linear")
  x = _TensorShapeOnly(shape=(1, 4, 8))

  called = []
  def fallback(xarg):
    called.append(xarg)
    return "fallback"

  out = decode_routes.q4k_primitive_linear_call(linear, x, fallback, True)
  assert out == "fallback"
  assert called == [x]


def test_q4k_single_token_keeps_generated_g3_path(monkeypatch):
  monkeypatch.setenv("BUBBLEBEAM_FUTURESIGHT", "0")
  monkeypatch.setenv("Q4K_GEMV_SCHEDULER", "1")
  monkeypatch.setenv("DECODE_Q4K_G3_ANYSHAPE", "0")
  monkeypatch.setenv("DECODE_Q4K_INKERNEL_COMBINE_KV", "0")
  monkeypatch.setenv("DECODE_Q4K_SPLIT_K_KV", "0")
  g3_calls = {"n": 0}
  monkeypatch.setattr(decode_routes.qk_ops, "q4k_g3_lanemap_gemv_kernel",
                      lambda *_args, **_kwargs: g3_calls.__setitem__("n", g3_calls["n"] + 1) or "kernel")
  monkeypatch.setattr(decode_routes.qk_ops, "q4k_gemm_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("single-token decode should not use q4k_gemm_kernel")))

  class TensorStub:
    device = "CPU"

    @classmethod
    def empty(cls, *_args, **_kwargs):
      return cls()

    def custom_kernel(self, *_args, **_kwargs):
      return (self.__class__(),)

    def __getitem__(self, _idx):
      return self

    def reshape(self, *_args, **_kwargs):
      return self

    def cast(self, *_args, **_kwargs):
      return self

    def contiguous(self):
      return self

  monkeypatch.setattr(decode_routes, "Tensor", TensorStub)
  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=1024, out_features=32, parts=1, opts=(), kernel_mode="partial",
    name="decode_g3_test_linear", q4k_storage=SimpleNamespace(mode="sidecar", words=_Words()))
  x = TensorStub()
  x.shape = (1, 1, 1024)

  def fallback(_x):
    raise AssertionError("fallback must not be used for K==1")

  decode_routes.q4k_primitive_linear_call(linear, x, fallback, True)
  assert g3_calls["n"] == 1


def test_q4k_candidate_binds_explicit_quant_shape_target_requirements():
  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=2048, out_features=96,
    q4k_storage=SimpleNamespace(mode="sidecar", words=_Words()), name="blk.0.attn_k.weight")
  x = _TensorShapeOnly(shape=(1, 1, 2048))

  binding = decode_routes.Q4K_DECODE_CANDIDATE.bind(linear, x, arch_ok=True)

  assert binding is not None
  assert (binding.quant, binding.target, binding.B, binding.T, binding.K, binding.N) == \
    ("Q4_K", "amd_gfx1100", 1, 1, 2048, 96)
  with pytest.raises(FrozenInstanceError): binding.N = 32


def test_q4k_binding_does_not_depend_on_model_role_label():
  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=2048, out_features=96, route_role="attn_qo",
    q4k_storage=SimpleNamespace(mode="sidecar", words=_Words()), name="blk.0.attn_k.weight")
  x = _TensorShapeOnly(shape=(1, 1, 2048))

  labeled = decode_routes.Q4K_DECODE_CANDIDATE.bind(linear, x, arch_ok=True)
  linear.route_role, linear.name = "lm_head", "output.weight"
  relabeled = decode_routes.Q4K_DECODE_CANDIDATE.bind(linear, x, arch_ok=True)

  assert labeled == relabeled


def test_q4k_candidate_rejects_unsupported_shapes_and_bias():
  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=2048, out_features=96,
    q4k_storage=SimpleNamespace(mode="sidecar", words=_Words()), name="blk.0.attn_k.weight")

  assert decode_routes.Q4K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(2, 1, 2048)), True) is None
  assert decode_routes.Q4K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(1, 2, 2048)), True) is None
  assert decode_routes.Q4K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(1, 1, 1024)), True) is None
  assert decode_routes.Q4K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(1, "T", 2048)), True) is None
  assert decode_routes.Q4K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(1, 1, 2048)), False) is None

  biased = SimpleNamespace(**{**linear.__dict__, "bias": object()})
  assert decode_routes.Q4K_DECODE_CANDIDATE.bind(biased, _TensorShapeOnly(shape=(1, 1, 2048)), True) is None
  non_block_shape = SimpleNamespace(**{**linear.__dict__, "in_features": 1536})
  assert decode_routes.Q4K_DECODE_CANDIDATE.bind(non_block_shape, _TensorShapeOnly(shape=(1, 1, 1536)), True) is None


def test_q6k_smallk_batched_routes_to_fallback(monkeypatch):
  monkeypatch.setattr(decode_routes.qk_ops, "q6k_gemm_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("batched K!=1 should not use q6k_gemm_kernel")))
  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=256, out_features=16, q6k_storage=SimpleNamespace(halfs=0),
    parts=1, opts=(), name="ffn_down.weight")
  x = _TensorShapeOnly(shape=(1, 8, 256))

  called = []
  def fallback(xarg):
    called.append(xarg)
    return "fallback"

  out = decode_routes.q6k_primitive_linear_call(linear, x, fallback, True)
  assert out == "fallback"
  assert called == [x]


def test_q6k_candidate_binds_explicit_quant_shape_target_requirements():
  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=256, out_features=16, q6k_storage=SimpleNamespace(halfs=0),
    parts=1, opts=(), name="blk.0.ffn_down.weight")
  x = _TensorShapeOnly(shape=(1, 1, 256))

  binding = decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, x, arch_ok=True)

  assert binding is not None
  assert (binding.quant, binding.target, binding.B, binding.T, binding.K, binding.N) == \
    ("Q6_K", "amd_gfx1100", 1, 1, 256, 16)
  assert (binding.parts, binding.row_tile, binding.use_coop) == (1, 4, True)


def test_q6k_candidate_rejects_unsupported_shapes_and_bias():
  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=256, out_features=16, q6k_storage=SimpleNamespace(halfs=0),
    parts=1, opts=(), name="ffn_down.weight")

  assert decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(2, 1, 256)), True) is None
  assert decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(1, 2, 256)), True) is None
  assert decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(1, 1, 8)), True) is None
  assert decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(1, "T", 256)), True) is None
  assert decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, _TensorShapeOnly(shape=(1, 1, 256)), False) is None

  biased = SimpleNamespace(**{**linear.__dict__, "bias": object()})
  assert decode_routes.Q6K_DECODE_CANDIDATE.bind(biased, _TensorShapeOnly(shape=(1, 1, 256)), True) is None
  non_block_shape = SimpleNamespace(**{**linear.__dict__, "in_features": 8})
  assert decode_routes.Q6K_DECODE_CANDIDATE.bind(non_block_shape, _TensorShapeOnly(shape=(1, 1, 8)), True) is None


def test_q6k_single_token_keeps_generated_path(monkeypatch):
  monkeypatch.setenv("Q6K_COOP_RT", "7")
  monkeypatch.setenv("Q6K_FFN_DOWN_COOP", "0")
  out_obj = "generated-path-result"
  partial = _MockPartial(out_obj)
  partials = _MockPartials(partial)

  class _HalfStorage:
    def to(self, *_, **__):
      return self

  monkeypatch.setattr(decode_routes, "Tensor", SimpleNamespace(empty=lambda *_, **__: partials), raising=True)
  spec = SimpleNamespace(partial_axis_extent=8)
  spec_calls = []
  monkeypatch.setattr(decode_routes.qk_ops, "q6k_spec_for_role",
                      lambda *_args, **kwargs: spec_calls.append(kwargs) or spec)
  monkeypatch.setattr(decode_routes.qk_ops, "emit_q6k_gemv_kernel", lambda *_args, **_kwargs: "kernel")

  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=256, out_features=16, q6k_storage=SimpleNamespace(halfs=_HalfStorage()),
    parts=1, opts=(), name="ffn_down.weight")
  x = _MockTensor()

  def fallback(_x):
    raise AssertionError("fallback must not be used for K==1")

  out = decode_routes.q6k_primitive_linear_call(linear, x, fallback, True)
  assert out == out_obj
  assert spec_calls == [{"parts": 1, "row_tile": 4, "use_coop": True, "opts": ()}]
  assert partials.kernel_calls
  assert isinstance(partials.kernel_calls[0][1]["fxn"], str)


def test_q6k_binding_is_independent_of_role_and_size_thresholds():
  linear = SimpleNamespace(decode_enabled=True, bias=None, in_features=4096, out_features=128,
    q6k_storage=SimpleNamespace(halfs=0), parts=1, opts=(), name="unknown.weight")
  x = _TensorShapeOnly(shape=(1, 1, 4096))
  first = decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, x, True)
  linear.name, linear.route_role = "lm_head.weight", "ffn_down"
  assert decode_routes.Q6K_DECODE_CANDIDATE.bind(linear, x, True) == first


def test_flash_decode_binding_has_fixed_production_parameters(monkeypatch):
  monkeypatch.setenv("DECODE_LIVE_SPLIT", "0")
  monkeypatch.setenv("DECODE_LIVE_SPLIT_S", "7")
  monkeypatch.setenv("DECODE_LIVE_SPLIT_STAGING", "K_ONLY")
  binding = decode_routes.FLASH_DECODE_CANDIDATE.bind(1, 32, 8, 128, "AMD:0")
  assert binding is not None
  assert (binding.target, binding.split_size, binding.staging) == ("AMD", 48, "KV_BOTH")
  assert decode_routes.FLASH_DECODE_CANDIDATE.bind(2, 32, 8, 128, "AMD") is None
  assert decode_routes.FLASH_DECODE_CANDIDATE.bind(1, 32, 8, 128, "CPU") is None
