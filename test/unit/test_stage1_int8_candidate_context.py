from dataclasses import replace

from tinygrad import Tensor, dtypes
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import KernelCandidateContext, KernelLDSWindow, KernelTileGeometry, Ops


def test_dense_int8_candidate_context_threads_staged_abi_and_accumulator_dtype():
  geometry = KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 4096, 32), KernelLDSWindow("B", 4096, 8192, 32)))
  pipeline = KernelStage1PipelinePlan(2, 8192)
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "b" * 64, geometry, pipeline)
  a, b = Tensor.empty(128, 256, dtype=dtypes.char), Tensor.empty(256, 128, dtype=dtypes.char)
  sink = next(u for u in a.matmul(b, dtype=dtypes.int).schedule_linear().toposort() if u.op is Ops.SINK)
  sink = sink.replace(arg=replace(sink.arg, opts_to_apply=(Opt(OptOps.TC, 0, (3, 0, 1)),
                                                           Opt(OptOps.UNROLL, 0, 0)), candidate_context=context))

  lowered = full_rewrite_to_sink(sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100")), optimize=True)
  nodes = lowered.toposort()
  wmmas = [u for u in nodes if u.op is Ops.WMMA]
  lds = [u for u in nodes if u.op is Ops.DEFINE_LOCAL]
  accumulators = [u for u in nodes if u.op is Ops.DEFINE_REG]
  params = [u for u in nodes if u.op is Ops.PARAM]

  assert geometry.tile == (128, 128, 32) and geometry.waves == (4, 2)
  assert geometry.lds_windows == (KernelLDSWindow("A", 0, 4096, 32), KernelLDSWindow("B", 4096, 8192, 32))
  assert len(lds) == 1 and lds[0].ptrdtype.base == dtypes.char
  assert lds[0].ptrdtype.size * dtypes.char.itemsize == pipeline.active_lds_bytes == 16_384
  assert len(accumulators) == 1 and accumulators[0].ptrdtype.base == dtypes.int
  assert len(wmmas) == 48
  assert all(u.dtype == u.src[2].dtype == dtypes.int.vec(8) for u in wmmas)
  assert all(u.src[0].dtype == u.src[1].dtype == dtypes.char.vec(16) for u in wmmas)
  assert sum(u.src[2].op is Ops.WMMA for u in wmmas) == len(wmmas) // 2  # two K16 substeps per K32 group
  assert [u.ptrdtype.base for u in params].count(dtypes.char) == 2
  assert all(u.ptrdtype.base != dtypes.half for u in params)
  assert lowered.arg.candidate_context == context
