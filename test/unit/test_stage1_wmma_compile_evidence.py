from dataclasses import replace

from tinygrad import Tensor
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import KernelCandidateContext, KernelLDSWindow, KernelTileGeometry, Ops


def test_stage1_128x128x256_reaches_amd_wmma_with_lds_resource_evidence():
  geometry = KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
    (KernelLDSWindow("A", 0, 10240, 80), KernelLDSWindow("B", 10240, 20480, 80)))
  pipeline = KernelStage1PipelinePlan(2, 20480)
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "a" * 64, geometry, pipeline)
  a, b = Tensor.empty(128, 256, dtype="half"), Tensor.empty(256, 128, dtype="half")
  sink = next(u for u in (a @ b).schedule_linear().toposort() if u.op is Ops.SINK)
  sink = sink.replace(arg=replace(sink.arg, opts_to_apply=(Opt(OptOps.TC, 0, (0, 0, 1)),
                                                           Opt(OptOps.UNROLL, 0, 0)), candidate_context=context))

  lowered = full_rewrite_to_sink(sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100")), optimize=True)
  nodes = lowered.toposort()
  wmmas = [u for u in nodes if u.op is Ops.WMMA]
  barriers = [u for u in nodes if u.op is Ops.BARRIER]
  lds = [u for u in nodes if u.op is Ops.DEFINE_LOCAL]

  assert len(wmmas) == 48
  assert len(barriers) == 2
  assert len(lds) == 1 and lds[0].ptrdtype.size * 2 == pipeline.active_lds_bytes == 40960
  assert any(isinstance(u.tag, tuple) and u.tag[0] == "pipeline_body_join" for u in barriers)
  assert lowered.arg.candidate_context.pipeline.active_lds_bytes == 40960
