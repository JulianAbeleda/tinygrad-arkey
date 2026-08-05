import importlib.util, pathlib
import pytest

from tinygrad import Tensor, TinyJit
from tinygrad.engine.jit import JitError

PATH = pathlib.Path(__file__).parents[2] / "scratchpad" / "nv_decode_predispatch_ab.py"
SPEC = importlib.util.spec_from_file_location("nv_decode_predispatch_ab", PATH)
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)


def test_structural_cache_is_identity_strict():
  c, a, b = MOD.StructuralInputCache(), object(), object()
  assert c.lookup([a]) is None
  c.store([a], "descriptor")
  assert c.lookup([a]) == "descriptor"
  assert c.lookup([b]) is None


def test_cached_prepare_revalidates_shape_and_dtype_fail_closed():
  @TinyJit
  def f(x): return (x+1).realize()
  metrics = {}
  with MOD.installed_candidates(True, False, metrics):
    x = Tensor.ones(4).realize()
    for _ in range(3): assert f(x).numpy().tolist() == [2.0]*4
    # A different allocation with the same contract must bind freshly, not use
    # a cached concrete buffer.
    assert f(Tensor.full((4,), 3.0).realize()).numpy().tolist() == [4.0]*4
    # A different concrete extent can validly bind the capture's symbolic view;
    # a different rank and dtype cannot and must reach the existing oracle.
    with pytest.raises(JitError): f(Tensor.ones(2, 2))
    with pytest.raises(JitError): f(Tensor.ones(4).cast("int32"))
    other_device = "CPU" if x.device != "CPU" else "PYTHON"
    with pytest.raises(JitError): f(Tensor.ones(4, device=other_device))
  assert any(metrics["a_hits"])


def test_shadow_candidate_keeps_the_copy_command():
  # Guard the load-bearing rule directly: the reusable buffer may replace the
  # allocation, never the copy or run_linear dependency edge.
  src = PATH.read_text()
  body = src[src.index("def captured(self"):src.index("tj._prepare_jit_inputs", src.index("def captured(self"))]
  assert "u.copy_to_device(u.device).call(shadow, u" in body
  assert "tj.run_linear(UOp(Ops.LINEAR, src=(call,)))" in body
