from tinygrad import dtypes
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.tc import cuda_81632_i8, get_cuda
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops


def test_nv_int8_tensor_core_descriptor():
  tc = cuda_81632_i8[0]
  assert tc.dims == (8, 16, 32) and tc.threads == 32
  assert tc.elements_per_thread == (16, 8, 4)
  assert tc.dtype_in == dtypes.char and tc.dtype_out == dtypes.int
  tc.lane_map.validate()


def test_nv_int8_tensor_core_is_arch_gated():
  assert cuda_81632_i8[0] not in get_cuda("sm_80")
  assert cuda_81632_i8[0] in get_cuda("sm_89")
  assert cuda_81632_i8[0] in get_cuda("sm_120")


def test_cuda_renderer_exposes_nv_int8_tensor_core():
  renderer = CUDARenderer(Target("MOCK", "NV", "sm_120"))
  assert any(x.dtype_in == dtypes.char and x.dtype_out == dtypes.int for x in renderer.tensor_cores)


def test_nv_int8_tensor_core_renders_signed_imma():
  renderer = CUDARenderer(Target("MOCK", "NV", "sm_120"))
  a, b = Tensor.empty(16, 32, dtype=dtypes.char), Tensor.empty(32, 8, dtype=dtypes.char)
  linear = a.dot(b, dtype=dtypes.int).schedule_linear()
  ast = next(u for u in linear.toposort() if u.op is Ops.SINK)
  tc_index = renderer.tensor_cores.index(cuda_81632_i8[0])
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, 0, (tc_index, 0, 1)),)))
  sink = full_rewrite_to_sink(ast, renderer, optimize=True)
  assert sum(u.op is Ops.WMMA for u in sink.toposort()) == 1
  source = renderer.render(line_rewrite(linearize(sink), pm_linearize_cleanups))
  assert "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32" in source
  assert "__dp4a" not in source
