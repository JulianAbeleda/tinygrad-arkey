from __future__ import annotations

import pytest
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt.kernel_lds import (PackedPrecontractOperandTemplate, PrecontractContractSpec, PrecontractKAxis,
  PrecontractOperandTemplate, PrecontractThreadAxes, build_precontract_lds_stage, fold_binary_axes,
  validate_precontract_operand_templates)
from tinygrad.codegen.opt.tc import cuda_81632_i8
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider, Q8ActivationRecordTransform,
                                                Q8Int8FragmentProvider, Q4KQ8GroupAccumulatorContract)
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry


def _operands(provider=True, dtype=dtypes.char):
  transform = PackedWeightTransform("Q4_K", 16, 256)
  n, m, k = UOp.range(16, 300, AxisType.LOOP), UOp.range(32, 301, AxisType.LOOP), UOp.range(256, 302, AxisType.REDUCE)
  q8 = UOp.param(1, dtype.ptr(32*256))
  words = UOp.param(2, dtypes.uint32.ptr(transform.packed_bytes//4))
  a = PrecontractOperandTemplate("A", q8.index(m*256+k).load(), m, k, UOp.const(dtypes.weakint, 0))
  b = PackedPrecontractOperandTemplate("B", words, transform, n, k, UOp.const(dtypes.weakint, 0),
    Q4KInt8FragmentProvider(transform) if provider else None)
  return a, b


def test_nv_packed_int8_precontract_requires_and_accepts_typed_provider():
  validate_precontract_operand_templates(_operands(True), dtype_in=dtypes.char, context="nv q4")
  with pytest.raises(ValueError, match="typed logical fragment provider"):
    validate_precontract_operand_templates(_operands(False), dtype_in=dtypes.char, context="nv q4")


def test_typed_int8_provider_is_not_silently_used_for_fp16_wmma():
  with pytest.raises(ValueError, match="fp16 packed template"):
    validate_precontract_operand_templates(_operands(True, dtypes.half), dtype_in=dtypes.half, context="amd q4")


def test_q8_record_offsets_sign_and_group_metadata_on_python():
  transform, provider = Q8ActivationRecordTransform(1, 64), Q8Int8FragmentProvider(Q8ActivationRecordTransform(1, 64))
  assert provider.transform == transform
  values = np.arange(-32, 32, dtype=np.int8)
  scales, sums = np.asarray([0.5, 1.5], np.float32), np.asarray([-7.0, 11.0], np.float32)
  record_np = np.frombuffer(values.tobytes()+scales.tobytes()+sums.tobytes(), np.uint32).copy()
  assert (transform.values_bytes, transform.scales_offset_words, transform.sums_offset_words,
          transform.storage_units) == (64, 16, 18, 20)
  record = Tensor(record_np, device="PYTHON").contiguous().realize()

  def values_kernel(out:UOp, source:UOp) -> UOp:
    stores = []
    for base in range(0, 64, 16):
      fragment = provider.fragment(source, 0, base, 16)
      stores.append(out.index(UOp.const(dtypes.weakint, base), dtype=dtypes.char.vec(16)).store(fragment.value))
    return UOp.sink(*stores, arg=KernelInfo(name="q8_record_values"))
  got_values = Tensor.empty(64, dtype=dtypes.int8, device="PYTHON").uop_program(record, fxn=values_kernel)[0].numpy()
  np.testing.assert_array_equal(got_values, values)

  def metadata_kernel(out:UOp, source:UOp) -> UOp:
    stores = []
    for group in range(2):
      scale, raw_sum, logical_group = transform.metadata(source, 0, group*32)
      assert logical_group == group
      stores += [out.index(UOp.const(dtypes.weakint, 2*group)).store(scale),
                 out.index(UOp.const(dtypes.weakint, 2*group+1)).store(raw_sum)]
    return UOp.sink(*stores, arg=KernelInfo(name="q8_record_metadata"))
  got_metadata = Tensor.empty(4, dtype=dtypes.float, device="PYTHON").uop_program(record, fxn=metadata_kernel)[0].numpy()
  np.testing.assert_array_equal(got_metadata, np.asarray([0.5,-7.0,1.5,11.0],np.float32))


def test_q8_provider_rejects_concrete_and_symbolic_cross_group_fragments():
  provider=Q8Int8FragmentProvider(Q8ActivationRecordTransform(1,256))
  source=lambda _:UOp.const(dtypes.uint32,0)
  provider.fragment(source,0,16,16)
  provider.fragment(source,0,UOp.range(8,330)*32+16,16)
  with pytest.raises(ValueError,match="K32 metadata boundary"): provider.fragment(source,0,24,16)
  with pytest.raises(ValueError,match="K32 metadata boundary"):
    provider.fragment(source,0,UOp.range(8,331)*32+24,16)


def _nv_staged_build(tile_base_offset:int):
  tc=cuda_81632_i8[0]
  geometry=KernelTileGeometry((16,8,64),(1,1),32,32,
    (KernelLDSWindow("A",0,16*80,80),KernelLDSWindow("B",16*80,24*80,80)))
  at,wt=Q8ActivationRecordTransform(16,256),PackedWeightTransform("Q4_K",8,256)
  ap,wp=Q8Int8FragmentProvider(at),Q4KInt8FragmentProvider(wt)
  m,n,k=UOp.range(16,340),UOp.range(8,341),UOp.range(256,342,AxisType.REDUCE)
  operands=(PackedPrecontractOperandTemplate("A",UOp.param(1,dtypes.uint32.ptr(at.storage_units)),at,m,k,
              UOp.const(dtypes.weakint,0),ap),
            PackedPrecontractOperandTemplate("B",UOp.param(2,dtypes.uint32.ptr(wt.packed_bytes//wt.storage_width)),wt,n,k,
              UOp.const(dtypes.weakint,0),wp))
  lane=UOp.range(32,-1,AxisType.WARP)
  threads=PrecontractThreadAxes(UOp.const(dtypes.weakint,0),UOp.const(dtypes.weakint,0),lane)
  tile_owner,substep=UOp.range(2,343,AxisType.REDUCE),UOp.range(2,344,AxisType.UNROLL)
  k_axis=PrecontractKAxis(tile_owner,substep,tile_owner*64+tile_base_offset,substep)
  contracts=[]
  for operand_idx,role in enumerate(("A","B")):
    count=(4,3)[operand_idx]
    axes=tuple(UOp.range(2,350+operand_idx*8+i,AxisType.UPCAST) for i in range(count))
    contracts.append(PrecontractContractSpec(role,axes,tuple((x.arg[0],2) for x in axes),fold_binary_axes(axes),
                                             tuple(tc.lane_map.remaps()[operand_idx].items())))
  allocation=UOp.placeholder((geometry.lds_bytes,),dtypes.char,399,addrspace=AddrSpace.LOCAL)
  return build_precontract_lds_stage(geometry,tc=tc,allocation=allocation,operands=operands,threads=threads,k_axis=k_axis,
    subtile_m=UOp.range(1,370,AxisType.UPCAST),subtile_n=UOp.range(1,371,AxisType.UPCAST),contracts=tuple(contracts),
    lds_bank_dwords=32,lds_bank_cycle_lanes=8,lds_read_before_next_write_ordered=False)


def test_staged_build_path_accepts_aligned_k64_and_rejects_cross_group_base():
  stage=_nv_staged_build(0)
  assert stage.fragment_a.dtype==dtypes.char.vec(16) and stage.fragment_b.dtype==dtypes.char.vec(8)
  with pytest.raises(ValueError,match="K32 metadata boundary"): _nv_staged_build(8)


def test_group_accumulator_staged_half_round_formula_on_python():
  weight = Q4KInt8FragmentProvider(PackedWeightTransform("Q4_K",1,256))
  activation = Q8Int8FragmentProvider(Q8ActivationRecordTransform(1,256))
  contract = Q4KQ8GroupAccumulatorContract(weight,activation)
  dot, wc, yc = 137, (np.float16(0.1875),np.float16(-0.3125)), (np.float16(1.25),np.float16(-9.0))
  def kernel(out:UOp, _dummy:UOp) -> UOp:
    weight_packet = UOp(Ops.STACK,dtypes.half.vec(2),tuple(UOp.const(dtypes.half,float(x)) for x in wc))
    activation_packet = UOp(Ops.STACK,dtypes.half.vec(2),tuple(UOp.const(dtypes.half,float(x)) for x in yc))
    return UOp.sink(out.index(UOp.const(dtypes.weakint,0)).store(
      contract.combine_staged(UOp.const(dtypes.int,dot),weight_packet,activation_packet)),arg=KernelInfo(name="q4_q8_half2_correction"))
  got = Tensor.empty(1,dtype=dtypes.float,device="PYTHON").uop_program(Tensor([0],device="PYTHON"),fxn=kernel)[0].item()
  expected = np.float32(wc[0])*np.float32(yc[0])*np.float32(dot)+np.float32(wc[1])*np.float32(yc[1])
  assert got == expected


def test_k64_metadata_packet_selection_keeps_group0_and_group1_distinct_on_python():
  # The cooperative stage owns two 16-byte vectors per K32 group. Each vector
  # writes a half2 packet; the accumulator must select vector 2*k_substep.
  transform = Q8ActivationRecordTransform(1,64)
  values = np.zeros(64,np.int8); scales=np.asarray([0.5,3.0],np.float32); sums=np.asarray([-5.0,17.0],np.float32)
  record_np=np.frombuffer(values.tobytes()+scales.tobytes()+sums.tobytes(),np.uint32).copy()
  record=Tensor(record_np,device="PYTHON").contiguous().realize()
  def kernel(out:UOp, source:UOp) -> UOp:
    # Mirror the stage's four vector owners: [g0,g0,g1,g1], then mirror the
    # correction reader's packet offsets vector 0 and vector 2.
    packets=[]
    for vector in range(4): packets.append(transform.metadata(source,0,vector*16)[:2])
    stores=[]
    for substep,vector in enumerate((0,2)):
      stores += [out.index(UOp.const(dtypes.weakint,2*substep)).store(packets[vector][0]),
                 out.index(UOp.const(dtypes.weakint,2*substep+1)).store(packets[vector][1])]
    return UOp.sink(*stores,arg=KernelInfo(name="q8_k64_packet_selection"))
  got=Tensor.empty(4,dtype=dtypes.float,device="PYTHON").uop_program(record,fxn=kernel)[0].numpy()
  np.testing.assert_array_equal(got,np.asarray([0.5,-5.0,3.0,17.0],np.float32))
