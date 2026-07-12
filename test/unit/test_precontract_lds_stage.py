import pytest

from tinygrad import dtypes
from tinygrad.codegen.opt.kernel_lds import (PrecontractContractSpec, PrecontractKAxis, PrecontractOperandTemplate,
  PrecontractThreadAxes, build_precontract_lds_stage)
from tinygrad.codegen.opt.tc import amd_rdna3
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelLDSWindow, KernelTileGeometry, Ops, UOp

def _geometry(): return KernelTileGeometry((128,128,32),(4,2),256,32,
  (KernelLDSWindow("A",0,10240,80),KernelLDSWindow("B",10240,20480,80)))
def _tc(): return next(tc for tc in amd_rdna3 if tc.dtype_in == dtypes.half and tc.dtype_out == dtypes.float)

def _fixture():
  ra,rb,ka,kb=(UOp.range(512,20,AxisType.LOOP),UOp.range(12288,21,AxisType.LOOP),
               UOp.range(4096,22,AxisType.REDUCE),UOp.range(4096,23,AxisType.REDUCE))
  a,b=UOp.param(0,dtypes.half.ptr(512*4096)),UOp.param(1,dtypes.half.ptr(12288*4096))
  ops=(PrecontractOperandTemplate("A",a.index(ra*4096+ka).load(),ra,ka,UOp.const(dtypes.weakint,128)),
       PrecontractOperandTemplate("B",b.index(rb*4096+kb).load(),rb,kb,UOp.const(dtypes.weakint,384)))
  threads=PrecontractThreadAxes(UOp.range(4,30,AxisType.LOCAL),UOp.range(2,31,AxisType.LOCAL),UOp.range(32,-1,AxisType.WARP))
  tile_owner=UOp.range(128,32,AxisType.REDUCE); substep_owner=UOp.range(2,43,AxisType.UNROLL)
  kaxis=PrecontractKAxis(tile_owner,substep_owner,tile_owner*32,substep_owner)
  sm,sn=UOp.range(2,33,AxisType.UPCAST),UOp.range(4,34,AxisType.UPCAST)
  contracts=[]
  for operand_idx,role in enumerate(("A","B")):
    axes=tuple(UOp.range(2,35+operand_idx*4+i,AxisType.UPCAST) for i in range(4))
    elem=((axes[0]*2+axes[1])*2+axes[2])*2+axes[3]
    contracts.append(PrecontractContractSpec(role,axes,tuple((a.arg[0],2) for a in axes),elem,
      tuple(_tc().lane_map.remaps()[operand_idx].items())))
  allocation=UOp.placeholder((10240,),dtypes.half,994,addrspace=AddrSpace.LOCAL)
  return allocation,ops,threads,kaxis,sm,sn,tuple(contracts)

def _stage(**overrides):
  allocation,ops,threads,kaxis,sm,sn,contracts=_fixture()
  values={"allocation":allocation,"operands":ops,"threads":threads,"k_axis":kaxis,
          "subtile_m":sm,"subtile_n":sn,"contracts":contracts}|overrides
  return build_precontract_lds_stage(_geometry(),tc=_tc(),**values)

def test_real_range_stage_structure_and_contract_args():
  stage=_stage()
  assert stage.allocation.op is Ops.DEFINE_LOCAL and stage.allocation.ptrdtype.addrspace is AddrSpace.LOCAL
  assert stage.allocation.ptrdtype.size*2 == 20480
  assert stage.producer.op is Ops.GROUP and stage.barrier.src == (stage.producer,)
  assert stage.fragment_a.arg == ((35,2),(36,2),(37,2),(38,2))
  assert stage.fragment_b.arg == ((39,2),(40,2),(41,2),(42,2))
  assert all(stage.barrier in x.backward_slice for x in UOp.sink(stage.fragment_a,stage.fragment_b).backward_slice
             if x.op is Ops.LOAD and stage.allocation in x.backward_slice)
  assert not any(x.op in (Ops.END,Ops.SPECIAL) for x in UOp.sink(stage.producer,stage.fragment_a,stage.fragment_b).backward_slice)

def test_nonzero_row_and_k_tile_bases_survive_producer_templates():
  stage=_stage(); stores=[x for x in stage.producer.backward_slice_with_self if x.op is Ops.STORE]
  global_loads=[x for x in UOp.sink(*stores).backward_slice if x.op is Ops.LOAD and stage.allocation not in x.backward_slice]
  rendered=[x.src[0].src[1].render() for x in global_loads]
  assert any("524288" in x for x in rendered) and any("1572864" in x for x in rendered)
  owner=next(x for x in stage.producer.backward_slice if x.op is Ops.RANGE and x.arg[0] == 32)
  assert all(owner in x.backward_slice for x in global_loads)

def test_local4_local2_warp32_scalar_store_coverage():
  stage=_stage(); stores=[x for x in stage.producer.backward_slice_with_self if x.op is Ops.STORE]
  indices=[x.src[0].src[1] for x in stores]
  wm,wn,lane=_fixture()[2].wave_m,_fixture()[2].wave_n,_fixture()[2].lane
  # Use the axes from the built graph, not equal-looking detached fixture axes.
  wm=next(x for x in stage.producer.backward_slice if x.op is Ops.RANGE and x.arg[0] == 30)
  wn=next(x for x in stage.producer.backward_slice if x.op is Ops.RANGE and x.arg[0] == 31)
  lane=next(x for x in stage.producer.backward_slice if x.op is Ops.RANGE and x.arg[0] == -1)
  for m in range(4):
    for n in range(2):
      for l in range(32):
        repl={wm:UOp.const(dtypes.weakint,m),wn:UOp.const(dtypes.weakint,n),lane:UOp.const(dtypes.weakint,l)}
        actual={x.substitute(repl).simplify().arg for x in indices}; tid=(m*2+n)*32+l; row,vec=tid//4,tid%4
        expected={(base//2)+r*40+vec*8+e for base in (0,10240) for r in (row,row+64) for e in range(8)}
        assert actual == expected

def test_fail_closed_detached_axes_contract_and_allocation():
  allocation,ops,threads,kaxis,sm,sn,contracts=_fixture()
  detached_tile=PrecontractKAxis(kaxis.tile_owner,kaxis.substep_owner,UOp.const(dtypes.weakint,0),kaxis.substep)
  with pytest.raises(ValueError,match="K tile owner"): _stage(k_axis=detached_tile)
  detached_substep=PrecontractKAxis(kaxis.tile_owner,kaxis.substep_owner,kaxis.tile_base,UOp.const(dtypes.weakint,0))
  with pytest.raises(ValueError,match="K substep owner"): _stage(k_axis=detached_substep)
  swapped=PrecontractKAxis(kaxis.substep_owner,kaxis.tile_owner,kaxis.tile_base,kaxis.substep)
  with pytest.raises(ValueError,match="K tile owner"): _stage(k_axis=swapped)
  a_contract,b_contract=contracts
  bad_contract=PrecontractContractSpec("A",a_contract.axes,((99,16),),a_contract.element,a_contract.descriptor_remap)
  with pytest.raises(ValueError,match="actual descriptor"): _stage(contracts=(bad_contract,b_contract))
  with pytest.raises(ValueError,match="ordered A and B"): _stage(contracts=(b_contract,a_contract))
  bad_remap=PrecontractContractSpec("B",b_contract.axes,b_contract.arg,b_contract.element,a_contract.descriptor_remap)
  with pytest.raises(ValueError,match="actual descriptor"): _stage(contracts=(a_contract,bad_remap))
  wrong_element=PrecontractContractSpec("B",b_contract.axes,b_contract.arg,b_contract.axes[0],b_contract.descriptor_remap)
  with pytest.raises(ValueError,match="actual descriptor"): _stage(contracts=(a_contract,wrong_element))
  bad_threads=PrecontractThreadAxes(UOp.range(8,30,AxisType.LOCAL),threads.wave_n,threads.lane)
  with pytest.raises(ValueError,match="LOCAL4/LOCAL2/WARP32"): _stage(threads=bad_threads)
  with pytest.raises(ValueError,match="caller allocation"): _stage(allocation=UOp.placeholder((6144,),dtypes.half,994,addrspace=AddrSpace.LOCAL))
