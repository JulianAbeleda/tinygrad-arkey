import pytest
from types import SimpleNamespace

from extra.qk import prefill_graph_gemm_route as route
from extra.qk import prefill_schedule_spec as spec
from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp


def _walk_uops(uop):
  if not isinstance(uop, UOp):
    return
  yield uop
  for child in uop.src:
    yield from _walk_uops(child)


def _prefill_schedule(route_family: str, out_f: int, in_f: int) -> spec.PrefillGEMMScheduleSpec:
  return spec.PrefillGEMMScheduleSpec(
    m=512, n=out_f, k=in_f, route_family=route_family, tile_m=128, tile_n=256, tile_k=32, waves_m=2, waves_n=2,
    wm=4, wn=4, pipe_tm=2, pipe_tn=2, pipeline_depth=2, threads=256, dbuf=1, plra=0, plrab=0, pad=16, leanaddr=0,
    role="ffn_down"
  )


def test_prefill_pipe_role_selective_zero_is_retired(monkeypatch):
  route._resolve_schedule.cache_clear()
  monkeypatch.setenv("PREFILL_PIPE_ROLE_SELECTIVE", "0")
  with pytest.raises(RuntimeError, match="PREFILL_PIPE_ROLE_SELECTIVE=0 global-pipe rollback was retired"):
    route._resolve_schedule(4096, 4096)
  route._resolve_schedule.cache_clear()


def test_route_pf16_graph_gemm_current_lowering_path_still_wraps_ops_ins(monkeypatch):
  route._resolve_schedule.cache_clear()
  captured = {}

  def fake_describe(out_f: int, in_f: int, role: str | None = None):
    captured["described"] = (out_f, in_f, role)
    return _prefill_schedule("pipe", out_f, in_f)

  def fake_emit(_spec):
    captured["emitted"] = _spec
    return (("ins",), 1, _spec.tile_m, _spec.tile_n, _spec.threads, _spec.kernel_name)

  class _Tensor:
    def __init__(self, shape, *, base_dtype=dtypes.float16):
      self.shape = shape
      self.device = "CPU"
      self.base = UOp.placeholder(shape, base_dtype, len(shape))
    @property
    def ndim(self):
      return len(self.shape)
    def __getitem__(self, _idx):
      return self
    def reshape(self, *shape):
      self.shape = shape
      return self
    def cast(self, *_args, **_kwargs):
      return self
    def contiguous(self):
      return self
    @classmethod
    def empty(cls, *shape, **_kwargs):
      return cls(shape, base_dtype=dtypes.float16)
    def custom_kernel(self, *args, **kwargs):
      program = kwargs["fxn"](self, *args)
      captured["program"] = program
      return (self, self, self)

  lin = SimpleNamespace(
    _pf16_w=_Tensor((4096, 2560), base_dtype=dtypes.float16),
    bias=None)
  x = _Tensor((1, 512, 2560), base_dtype=dtypes.float16)

  old_tensor = route.Tensor
  try:
    route.Tensor = _Tensor
    monkeypatch.setattr(spec, "describe_prefill_schedule", fake_describe)
    monkeypatch.setattr(spec, "emit_prefill_gemm_from_spec", fake_emit)
    out = route.route_pf16_graph_gemm(lin, x)
  finally:
    route.Tensor = old_tensor

  assert out is not None
  assert captured["described"] == (4096, 2560, getattr(lin, "_prefill_graph_role", None))
  assert captured["emitted"].route_family == "pipe"
  assert any(uop.op is Ops.INS for uop in _walk_uops(captured["program"]))
