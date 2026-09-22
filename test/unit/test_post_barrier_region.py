import pytest

from tinygrad import dtypes
from tinygrad.codegen import full_rewrite_to_sink, line_rewrite, pm_linearize_cleanups, to_program, validate_post_barrier_regions
from tinygrad.codegen.late.linearizer import linearize
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.renderer.ptx import PTXRenderer
from tinygrad.uop.ops import KernelInfo, Ops, PostBarrierRegion, UOp


def _ast():
  out=UOp.placeholder((64,),dtypes.float32,0)
  lid=UOp.special(64,"lidx0")
  lane=lid%32
  smem=UOp.placeholder((32,),dtypes.float32,10,addrspace=AddrSpace.LOCAL)
  published=smem[lane].store(lid.cast(dtypes.float32),lid>=32)
  ready=UOp.barrier(UOp.group(published))
  consumer=ready.post_barrier_region(lid<32)
  loaded=smem.after(consumer)[lane]
  stored=out[lid].store(loaded,lid<32)
  return consumer.end_region(stored).sink(arg=KernelInfo(name="post_barrier_region_test",opts_to_apply=()))


def _linear(renderer):
  prg=to_program(_ast(),renderer)
  return next(u.src for u in prg.src if u.op is Ops.LINEAR)


def test_post_barrier_region_orders_barrier_if_body_and_endif():
  linear=_linear(CUDARenderer(Target.parse("NV:CUDA:sm_120")))
  mif=next(u for u in linear if u.op is Ops.IF and isinstance(u.arg,PostBarrierRegion))
  mend=next(u for u in linear if u.op is Ops.ENDIF and isinstance(u.arg,PostBarrierRegion))
  barrier=next(u for u in linear if u.op is Ops.BARRIER)
  lds=next(u for u in linear if u.op is Ops.LOAD and u.src[0].addrspace is AddrSpace.LOCAL)
  assert linear.index(barrier)<linear.index(mif)<linear.index(lds)<linear.index(mend)
  assert mif.src[1] is barrier and mend.src[0] is mif


def test_post_barrier_region_renders_a_ptx_branch_around_shared_load():
  # Rendering itself is hermetic; bypass PTXRenderer.__init__, which owns a
  # compiler/JIT-link handle not needed for this source-level proof.
  ren=object.__new__(PTXRenderer); ren.target=Target.parse("NV:MOCK:sm_120"); ren.tensor_cores=[]
  sink=full_rewrite_to_sink(_ast(),ren)
  linear=linearize(sink); validate_post_barrier_regions(linear,ren)
  linear=line_rewrite(linear,pm_linearize_cleanups)
  src=ren.render(linear)
  bar=src.index("bar.sync")
  outer_branch=src.index("bra IF_",bar)
  lds=src.index("ld.shared",outer_branch)
  outer_label=src.index("IF_",lds)
  assert bar<outer_branch<lds<outer_label


def test_post_barrier_region_builders_fail_closed():
  barrier=UOp(Ops.BARRIER,dtypes.void)
  with pytest.raises(ValueError,match="bool"): barrier.post_barrier_region(UOp.const(dtypes.int,1))
  with pytest.raises(ValueError,match="anchored"): UOp.const(dtypes.int,0).post_barrier_region(UOp.const(dtypes.bool,True))
  region=barrier.post_barrier_region(UOp.const(dtypes.bool,True))
  with pytest.raises(ValueError,match="body root"): region.end_region()


def test_post_barrier_region_rejects_unsupported_renderer_and_inner_barrier():
  gate=UOp.const(dtypes.bool,True); barrier=UOp(Ops.BARRIER,dtypes.void)
  region=barrier.post_barrier_region(gate)
  body=UOp.const(dtypes.float32,1).after(region)
  end=region.end_region(body)
  class Unsupported: supports_post_barrier_regions=False
  with pytest.raises(RuntimeError,match="does not support"): validate_post_barrier_regions([gate,barrier,region,body,end],Unsupported())
  inner=UOp(Ops.BARRIER,dtypes.void,(region,))
  body2=body.after(inner); end2=region.end_region(body2)
  class Supported: supports_post_barrier_regions=True
  with pytest.raises(RuntimeError,match="forbidden inside"): validate_post_barrier_regions([gate,barrier,region,inner,body2,end2],Supported())


def test_workgroup_uniform_region_admits_inner_barrier_and_rejects_local_gate():
  anchor=UOp(Ops.BARRIER,dtypes.void); gidx=UOp.special(64,"gidx0"); gate=gidx<32
  region=anchor.post_barrier_region(gate,workgroup_uniform=True)
  inner=UOp(Ops.BARRIER,dtypes.void,(region,)); body=UOp.const(dtypes.float32,1).after(inner); end=region.end_region(body)
  class Supported: supports_post_barrier_regions=True
  validate_post_barrier_regions([gidx,gate,anchor,region,inner,body,end],Supported())
  lgate=UOp.special(64,"lidx0")<32; bad=anchor.post_barrier_region(lgate,workgroup_uniform=True)
  bad_body=UOp.const(dtypes.float32,1).after(bad); bad_end=bad.end_region(bad_body)
  with pytest.raises(RuntimeError,match="not proved uniform"):
    validate_post_barrier_regions([lgate,anchor,bad,bad_body,bad_end],Supported())


def test_post_barrier_region_requires_body_dependency_and_untyped_if_stays_banned():
  gate=UOp.const(dtypes.bool,True); barrier=UOp(Ops.BARRIER,dtypes.void)
  region=barrier.post_barrier_region(gate)
  independent=UOp.const(dtypes.float32,1)
  end=region.end_region(independent)
  class Supported: supports_post_barrier_regions=True
  with pytest.raises(RuntimeError,match="body root must depend"): validate_post_barrier_regions([gate,barrier,region,independent,end],Supported())
  untyped=UOp(Ops.IF,dtypes.void,(gate,UOp.const(dtypes.int,0)))
  with pytest.raises(RuntimeError,match="if not allowed"): line_rewrite([untyped],pm_linearize_cleanups)


def test_unused_cleanup_path_is_identity():
  out=UOp.placeholder((1,),dtypes.float32,0)
  store=out[0].store(1.0)
  assert line_rewrite([store],pm_linearize_cleanups)==[store]
  ren=CUDARenderer(Target.parse("NV:CUDA:sm_120"))
  sink=full_rewrite_to_sink(store.sink(arg=KernelInfo(name="no_region",opts_to_apply=())),ren)
  assert not any(isinstance(u.arg,PostBarrierRegion) for u in sink.toposort())
