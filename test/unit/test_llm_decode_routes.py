from __future__ import annotations

from types import SimpleNamespace

from tinygrad.llm import decode_routes


class _TensorShapeOnly:
  def __init__(self, shape):
    self.shape = shape


class _MockTensor:
  def __init__(self):
    self.shape = (1, 1, 4)
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


def test_q6k_smallk_batched_routes_to_fallback(monkeypatch):
  monkeypatch.setattr(decode_routes.qk_ops, "q6k_gemm_kernel", lambda *_args, **_kwargs: (_ for _ in ()).throw(
    AssertionError("batched K!=1 should not use q6k_gemm_kernel")))
  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=4, out_features=16, q6k_storage=SimpleNamespace(halfs=0),
    parts=1, opts=(), name="ffn_down.weight")
  x = _TensorShapeOnly(shape=(1, 8, 4))

  called = []
  def fallback(xarg):
    called.append(xarg)
    return "fallback"

  out = decode_routes.q6k_primitive_linear_call(linear, x, fallback, True)
  assert out == "fallback"
  assert called == [x]


def test_q6k_single_token_keeps_generated_path(monkeypatch):
  out_obj = "generated-path-result"
  partial = _MockPartial(out_obj)
  partials = _MockPartials(partial)

  class _HalfStorage:
    def to(self, *_, **__):
      return self

  monkeypatch.setattr(decode_routes, "Tensor", SimpleNamespace(empty=lambda *_, **__: partials), raising=True)
  spec = SimpleNamespace(partial_axis_extent=8)
  monkeypatch.setattr(decode_routes.qk_ops, "q6k_spec_for_role", lambda *_args, **_kwargs: spec)
  monkeypatch.setattr(decode_routes.qk_ops, "emit_q6k_gemv_kernel", lambda *_args, **_kwargs: "kernel")

  linear = SimpleNamespace(
    decode_enabled=True, bias=None, in_features=4, out_features=16, q6k_storage=SimpleNamespace(halfs=_HalfStorage()),
    parts=1, opts=(), name="ffn_down.weight")
  x = _MockTensor()

  def fallback(_x):
    raise AssertionError("fallback must not be used for K==1")

  out = decode_routes.q6k_primitive_linear_call(linear, x, fallback, True)
  assert out == out_obj
  assert partials.kernel_calls
  assert isinstance(partials.kernel_calls[0][1]["fxn"], str)
