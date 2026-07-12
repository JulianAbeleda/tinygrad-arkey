"""Host contract tests for the future pure-pipe graph ABI."""
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.dtype import dtypes
from tinygrad.renderer import Target
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import KernelCandidateContext, KernelInfo, UOp


def _abi_sink(identity):
  # ABI is A, B, output in that order; shapes encode the row/column strides.
  a = UOp.param(0, dtypes.half.ptr(128 * 32))
  b = UOp.param(1, dtypes.half.ptr(32 * 128))
  out = UOp.param(2, dtypes.float.ptr(128 * 128))
  value = out.index(UOp.const(dtypes.int, 0), ptr=True).store(UOp.const(dtypes.float, 0))
  return UOp.sink(a, b, value, arg=KernelInfo(name="pure_pipe_abi",
    candidate_context=KernelCandidateContext("boltbeam.full_kernel_candidate.v1", identity)))


def test_pure_pipe_abi_preserves_arg_order_dtypes_and_strides():
  sink = _abi_sink("1" * 64)
  params = sorted((u.arg.slot, u.dtype) for u in sink.toposort() if u.op.name == "PARAM")
  assert [slot for slot, _ in params] == [0, 1, 2]
  assert [dtype for _, dtype in params] == [dtypes.half.ptr(4096), dtypes.half.ptr(4096), dtypes.float.ptr(16384)]

  program = to_program(sink, PythonRenderer(Target("PYTHON")))
  assert [u.arg.slot for u in program.src[0].toposort() if u.op.name == "PARAM"] == [0, 1, 2]


def test_pure_pipe_abi_candidate_identity_separates_graph_cache():
  to_program_cache.clear()
  first = to_program(_abi_sink("2" * 64), PythonRenderer(Target("PYTHON")))
  second = to_program(_abi_sink("3" * 64), PythonRenderer(Target("PYTHON")))
  assert first is not second
  assert first.key != second.key
  assert first.src[0].arg.candidate_context.canonical_identity == "2" * 64
  assert second.src[0].arg.candidate_context.canonical_identity == "3" * 64
