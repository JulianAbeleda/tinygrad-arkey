import pytest

from tinygrad import dtypes
from tinygrad.codegen.late.devectorizer import ReduceContext, pm_reduce
from tinygrad.codegen.late.expander import expander, pm_group_for_reduce, pm_pre_expander
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite
from tinygrad.uop.symbolic import gep_pushing, sym


def _expanded_pipeline_accumulator(axis_ids):
  sm=UOp.range(2,axis_ids[0],AxisType.UPCAST); sn=UOp.range(4,axis_ids[1],AxisType.UPCAST)
  caxes=tuple(UOp.range(2,axis_ids[2+i],AxisType.UPCAST) for i in range(3))
  elem=(caxes[0]*2+caxes[1])*2+caxes[2]
  reg=UOp.placeholder((64,),dtypes.float,9300,addrspace=AddrSpace.REG)
  acc_idx=(sm*4+sn)*8+elem
  c=UOp(Ops.CONTRACT,dtypes.float.vec(8),(reg.index(acc_idx).load(),),tuple((x.arg[0],2) for x in caxes))
  a=UOp.const(dtypes.half.vec(16),0.0); b=UOp.const(dtypes.half.vec(16),0.0)
  tc_arg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD",32,
          (((101,2),(102,2),(103,2),(104,2)),)*2+(tuple((x.arg[0],2) for x in caxes),),())
  wmma=UOp(Ops.WMMA,dtypes.float.vec(8),(a,b,c),tc_arg)
  unroll=UOp(Ops.UNROLL,dtypes.float,(wmma,),tuple((x.arg[0],2) for x in caxes))
  update=reg.index(acc_idx).store(unroll)
  sink=UOp.sink(update)
  expanded=graph_rewrite(sink,sym+pm_pre_expander+pm_group_for_reduce+expander,name="pipeline expansion spike")
  return graph_rewrite(expanded,pm_reduce+gep_pushing,ctx=ReduceContext(),name="pipeline reduce spike")


@pytest.mark.xfail(strict=True,reason="expander retains one vec64 REG store; no 2x4 ownership-aware vec8 grouping pass exists")
def test_rdna3_2x4_pipeline_accumulator_expands_to_eight_disjoint_vec8_slices():
  snapshots=[]
  for ids in ((40,41,42,43,44),(91,17,73,6,55),(5,4,3,2,1)):
    out=_expanded_pipeline_accumulator(ids); topo=out.toposort()
    regs=[x for x in topo if x.op is Ops.DEFINE_REG]
    stores=[x for x in topo if x.op is Ops.STORE and regs and regs[0] in x.src[0].backward_slice]
    assert len(regs)==1 and regs[0].ptrdtype.size==64
    assert not any(x.op is Ops.REDUCE for x in topo)
    assert len(stores)==8 and all(x.src[1].dtype==dtypes.float.vec(8) for x in stores)
    offsets=sorted(x.src[0].src[1].simplify().arg for x in stores)
    assert offsets==list(range(0,64,8))
    snapshots.append((offsets,len([x for x in topo if x.op is Ops.WMMA])))
  assert snapshots[0]==snapshots[1]==snapshots[2]
