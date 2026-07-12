import pytest

from tinygrad import dtypes
from tinygrad.codegen.late.devectorizer import ReduceContext, _group_wmma_reg_store, pm_group_wmma_reg_store, pm_reduce
from tinygrad.codegen.late.expander import do_contract, do_expand, expander, pm_group_for_reduce, pm_pre_expander
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops, UOp, graph_rewrite
from tinygrad.uop.symbolic import gep_pushing, sym


def test_contract_preserves_matching_vector_carrier_and_rejects_mismatched_vector():
  carrier = UOp.const(dtypes.half.vec(16), 0.0)
  matching = UOp(Ops.CONTRACT, dtypes.half.vec(16), (carrier,), ((101, 16),))
  assert do_contract(matching) is carrier

  mismatched = UOp(Ops.CONTRACT, dtypes.half.vec(16), (UOp.const(dtypes.half.vec(8), 0.0),), ((101, 16),))
  with pytest.raises(ValueError, match="scalar source or matching vector"):
    do_contract(mismatched)

def test_reg_index_load_expansion_keeps_scalar_loads_under_stack():
  reg = UOp.placeholder((8,), dtypes.float, 9400, addrspace=AddrSpace.REG)
  idxs = tuple(reg.index(UOp.const(dtypes.weakint, i)) for i in (0, 1))
  pointers = UOp(Ops.STACK, idxs[0].dtype.vec(2), idxs)
  unroll = UOp(Ops.UNROLL, pointers.dtype, (pointers,), ())
  load = UOp(Ops.LOAD, dtypes.float.vec(2), (unroll,))
  expanded = do_expand(load)
  assert expanded is not None and expanded.op is Ops.UNROLL
  carrier = expanded.src[0]
  assert carrier.op is Ops.STACK and carrier.dtype == dtypes.float.vec(2)
  assert all(x.op is Ops.LOAD and x.dtype == dtypes.float for x in carrier.src)


def _expanded_pipeline_accumulator(axis_ids,m=2,n=4,group=True):
  sm=UOp.range(m,axis_ids[0],AxisType.UPCAST); sn=UOp.range(n,axis_ids[1],AxisType.UPCAST)
  caxes=tuple(UOp.range(2,axis_ids[2+i],AxisType.UPCAST) for i in range(3))
  elem=(caxes[0]*2+caxes[1])*2+caxes[2]
  reg=UOp.placeholder((m*n*8,),dtypes.float,9300,addrspace=AddrSpace.REG)
  acc_idx=(sm*n+sn)*8+elem
  c=UOp(Ops.CONTRACT,dtypes.float.vec(8),(reg.index(acc_idx).load(),),tuple((x.arg[0],2) for x in caxes))
  a=UOp.const(dtypes.half.vec(16),0.0); b=UOp.const(dtypes.half.vec(16),0.0)
  tc_arg=("WMMA_16_16_16_half_float",(16,16,16),dtypes.half,dtypes.float,"AMD",32,
          (((101,2),(102,2),(103,2),(104,2)),)*2+(tuple((x.arg[0],2) for x in caxes),),())
  wmma=UOp(Ops.WMMA,dtypes.float.vec(8),(a,b,c),tc_arg)
  unroll=UOp(Ops.UNROLL,dtypes.float,(wmma,),tuple((x.arg[0],2) for x in caxes))
  update=reg.index(acc_idx).store(unroll)
  sink=UOp.sink(update)
  expanded=graph_rewrite(sink,sym+pm_pre_expander+pm_group_for_reduce+expander,name="pipeline expansion spike")
  reduced=graph_rewrite(expanded,pm_reduce+gep_pushing,ctx=ReduceContext(),name="pipeline reduce spike")
  return graph_rewrite(reduced,pm_group_wmma_reg_store,name="pipeline ownership groups") if group else reduced


@pytest.mark.parametrize(("m","n"),((1,1),(1,4),(4,1),(2,4)))
def test_pipeline_accumulator_expands_to_disjoint_vec8_ownership_groups(m,n):
  snapshots=[]
  for ids in ((40,41,42,43,44),(91,17,73,6,55),(5,4,3,2,1)):
    out=_expanded_pipeline_accumulator(ids,m,n); topo=out.toposort()
    regs=[x for x in topo if x.op is Ops.DEFINE_REG]
    stores=[x for x in topo if x.op is Ops.STORE and regs and regs[0] in x.src[0].backward_slice]
    assert len(regs)==1 and regs[0].ptrdtype.size==m*n*8
    assert not any(x.op is Ops.REDUCE for x in topo)
    assert len(stores)==m*n and all(x.src[1].dtype==dtypes.float.vec(8) for x in stores)
    offsets=sorted(x.src[0].src[1].simplify().arg for x in stores)
    assert offsets==list(range(0,m*n*8,8))
    snapshots.append((offsets,len([x for x in topo if x.op is Ops.WMMA])))
  assert snapshots[0]==snapshots[1]==snapshots[2]


def _raw_store():
  out=_expanded_pipeline_accumulator((40,41,42,43,44),2,4,False)
  return next(x for x in out.toposort() if x.op is Ops.STORE)


def test_wmma_reg_grouping_rejects_nonownership_store_shapes():
  store=_raw_store(); tgt,val=store.src
  assert _group_wmma_reg_store(tgt,UOp.const(dtypes.float.vec(64),0.0)) is None
  assert _group_wmma_reg_store(tgt.replace(src=tgt.src[:-1]+(tgt.src[-2],)),val) is None
  dynamic=tgt.src[0].src[0].index(UOp.range(64,999,AxisType.LOOP))
  assert _group_wmma_reg_store(tgt.replace(src=(dynamic,)+tgt.src[1:]),val) is None
  gap=tgt.src[-1].replace(src=(tgt.src[-1].src[0],UOp.const(dtypes.weakint,65)))
  assert _group_wmma_reg_store(tgt.replace(src=tgt.src[:-1]+(gap,)),val) is None


def test_wmma_reg_grouping_rejects_wrong_contract_and_nonreg_targets():
  store=_raw_store(); tgt,val=store.src
  wmma=val if val.op is Ops.WMMA else val.src[0]
  bad_arg=wmma.arg[:6]+(wmma.arg[6][:2]+((),),)+wmma.arg[7:]
  bad_wmma=wmma.replace(arg=bad_arg)
  bad_val=bad_wmma if val.op is Ops.WMMA else val.replace(src=(bad_wmma,))
  assert _group_wmma_reg_store(tgt,bad_val) is None
  global_buf=UOp.param(12,dtypes.float.ptr(64))
  global_tgt=tgt.replace(src=tuple(global_buf.index(UOp.const(dtypes.weakint,i)) for i in range(64)))
  assert _group_wmma_reg_store(global_tgt,val) is None
