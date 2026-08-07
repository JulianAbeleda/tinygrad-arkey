"""Nested precompile bodies must resolve at callify time, not land raw in composites.

The M4 resadd fold admits a precompile block-output into the next block's precompile
body, so the open-arm schedule contains precompile ``FUNCTION`` nodes nested inside
other precompile ``FUNCTION`` bodies.  ``graph_rewrite`` does not enter FUNCTION bodies
by default, and ``rangeify.resolve_function`` deliberately skips precompile bodies, so
those nested functions used to survive raw into every composite: the composite's SINK
embedded the whole 3733-node body plus its weakint ``SPECIAL``s and value ``AFTER``s, and
the NV render crashed with ``UOp verification failed ... on Ops.SPECIAL dtypes.weakint``
(M4 S4 gate record section 7).  These tests lock the scheduler-side resolution: after
callify no raw precompile ``FUNCTION``/``GETTUPLE`` may remain anywhere in the graph
(bodies included), and the scheduled kernels must render clean on the NV renderer.
"""

from tinygrad import Tensor, UOp, dtypes
from tinygrad.function import function
from tinygrad.schedule import create_linear_with_vars
from tinygrad.uop.ops import KernelInfo, Ops


N = 16


def _reader_kernel(out: UOp, words: UOp) -> UOp:
  row = UOp.special(N, "gidx0")
  return out[row].store(words[row] * 2.0).sink(arg=KernelInfo(name="k_reader"))


@function(precompile=True)
def _writer(x): return (x * 2 + 1).contiguous()


@function(precompile=True, allow_implicit=True)
def _consumer(w):
  out = Tensor.empty(N, dtype=dtypes.float32).contiguous()
  ret = UOp.custom_kernel(out.uop, w.uop, fxn=_reader_kernel)
  return Tensor(ret[0]).contiguous()


@function(precompile=True, allow_implicit=True)
def _wrapper(x):
  # The nested precompiled calls are the load-bearing shape: a precompile body whose
  # values call two more precompile functions (mirrors the open-arm block-output chain).
  return _consumer(_writer(x))


def _schedule(out: Tensor):
  from tinygrad.callify import transform_to_call
  from tinygrad.tensor import _apply_map_to_tensors
  big_sink, becomes = transform_to_call(UOp.sink(out.uop))
  _apply_map_to_tensors(becomes, name="buffers")
  return create_linear_with_vars(big_sink)


def _raw_precompile_nodes(big_sink: UOp) -> list[UOp]:
  return [x for x in big_sink.toposort()
          if (x.op is Ops.FUNCTION and getattr(x.arg, "precompile", False)) or
             (x.op is Ops.GETTUPLE and x.src[0].op is Ops.FUNCTION)]


def test_nested_precompile_functions_resolve_at_callify():
  """No raw precompile FUNCTION/GETTUPLE may survive callify, including inside the
  transformed CALL bodies (toposort enters bodies by default)."""
  w_raw = Tensor.empty(N, dtype=dtypes.float32).contiguous()
  from tinygrad.callify import transform_to_call
  big_sink, _ = transform_to_call(UOp.sink(_wrapper(w_raw).uop))
  assert _raw_precompile_nodes(big_sink) == []


def test_nested_precompile_composites_render():
  """The scheduled kernels must render on the NV renderer; previously the composite
  carrying the nested precompile body crashed with the weakint SPECIAL type_verify
  failure, and no kernel body may contain a raw precompile artifact."""
  from tinygrad.helpers import Target
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.codegen import to_program

  w_raw = Tensor.empty(N, dtype=dtypes.float32).contiguous()
  linear, _ = _schedule(_wrapper(w_raw))
  ren = CUDARenderer(Target.parse("NV:CUDA:sm_120"))
  assert len(linear.src) >= 1
  for item in linear.src:
    ast = item.src[0]
    if ast.op is not Ops.SINK: continue
    assert not any(x.op in (Ops.FUNCTION, Ops.GETTUPLE) for x in ast.toposort(enter_calls=False)), \
      f"raw precompile artifact in kernel body: {ast}"
    to_program(ast, ren)  # must not raise (was: weakint SPECIAL type_verify)
