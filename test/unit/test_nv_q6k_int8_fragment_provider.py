from __future__ import annotations

import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q6KInt8FragmentProvider,
  Q8ActivationRecordTransform, Q8Int8FragmentProvider, Q6KQ8SubgroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_key
from tinygrad.uop.ops import KernelInfo, Ops, UOp


def _q6_block(codes:np.ndarray, scales:np.ndarray, d:float) -> np.ndarray:
  assert codes.shape == (16, 16) and scales.shape == (16,)
  raw = np.zeros(210, dtype=np.uint8)
  for group in range(16):
    half, payload_group = group//8, group%8
    for pos in range(16):
      q = int(codes[group, pos])+32
      ql_i = half*64+(payload_group%4)*16+pos
      if payload_group < 4: raw[ql_i] |= q&15
      else: raw[ql_i] |= (q&15)<<4
      qh_i, shift = 128+half*32+(payload_group%2)*16+pos, (payload_group//2)*2
      raw[qh_i] |= ((q>>4)&3)<<shift
  raw[192:208] = scales.view(np.uint8)
  raw[208:210] = np.asarray([d], np.float16).view(np.uint8)
  return raw


def test_q6_provider_adversarial_unpack_sign_and_all_k16_scale_owners_on_python():
  # Every bit plane and every signed scale differs. This catches ql/qh swaps,
  # sign loss, K16 scale aliases, and low/high-half permutation.
  codes = (((np.arange(256, dtype=np.int16)*29+7)%64)-32).astype(np.int8).reshape(16,16)
  scales = np.asarray([-128,-97,-65,-33,-17,-9,-3,-1,1,2,5,11,23,47,79,127],np.int8)
  raw = _q6_block(codes, scales, 0.03125)
  halfs = Tensor(raw.copy(),device="PYTHON").bitcast(dtypes.uint16).contiguous().realize()
  provider = Q6KInt8FragmentProvider(PackedWeightTransform("Q6_K",1,256))

  def kernel(out:UOp, source:UOp) -> UOp:
    stores=[]
    for group in range(16):
      fragment=provider.fragment(source,0,group*16,16)
      stores.append(out.index(UOp.const(dtypes.weakint,group*16),dtype=dtypes.char.vec(16)).store(fragment.value))
      stores.append(out.index(UOp.const(dtypes.weakint,256+group)).store(fragment.subgroup_scale.cast(dtypes.float)))
    return UOp.sink(*stores,arg=KernelInfo(name="q6_k_typed_s8_adversarial"))
  got=Tensor.empty(272,dtype=dtypes.float,device="PYTHON").uop_program(halfs,fxn=kernel)[0].numpy()
  np.testing.assert_array_equal(got[:256].astype(np.int8).reshape(16,16),codes)
  np.testing.assert_array_equal(got[256:].astype(np.int8),scales)


def test_q6_paired_accumulator_preserves_low_high_scale_roundpoints_on_python():
  codes=np.zeros((16,16),np.int8)
  scales=np.asarray([3,-11]+[1]*14,np.int8)
  raw=_q6_block(codes,scales,0.1875)
  halfs=Tensor(raw.copy(),device="PYTHON").bitcast(dtypes.uint16).contiguous().realize()
  weight=Q6KInt8FragmentProvider(PackedWeightTransform("Q6_K",1,256))
  activation=Q8Int8FragmentProvider(Q8ActivationRecordTransform(1,256))
  contract=Q6KQ8SubgroupAccumulatorContract(weight,activation)
  q8=np.zeros(256,np.int8); q8_scale=np.ones(8,np.float32); q8_scale[0]=0.625
  sums=np.zeros(8,np.float32)
  record_np=np.frombuffer(q8.tobytes()+q8_scale.tobytes()+sums.tobytes(),np.uint32).copy()
  record=Tensor(record_np,device="PYTHON").contiguous().realize()
  dot0,dot1=137,-53
  def kernel(out:UOp, weight_source:UOp, activation_source:UOp) -> UOp:
    value=contract.correct(weight_source,activation_source,row=0,column=0,k_base=0,
      integer_dots=(UOp.const(dtypes.int,dot0),UOp.const(dtypes.int,dot1)))
    return UOp.sink(out.index(UOp.const(dtypes.weakint,0)).store(value),arg=KernelInfo(name="q6_q8_k16_pair"))
  got=Tensor.empty(1,dtype=dtypes.float,device="PYTHON").uop_program(halfs,record,fxn=kernel)[0].item()
  wc0=np.float32(np.float16(np.float32(0.1875)*np.float32(scales[0])))
  wc1=np.float32(np.float16(np.float32(0.1875)*np.float32(scales[1])))
  yc=np.float32(np.float16(q8_scale[0]))
  expected=yc*(wc0*np.float32(dot0)+wc1*np.float32(dot1))
  swapped=yc*(wc1*np.float32(dot0)+wc0*np.float32(dot1))
  assert got == expected and got != swapped


def test_q6_provider_and_accumulator_fail_closed_by_format_and_k16_ownership():
  with pytest.raises(ValueError,match="Q6_K"):
    Q6KInt8FragmentProvider(PackedWeightTransform("Q4_K",1,256))
  provider=Q6KInt8FragmentProvider(PackedWeightTransform("Q6_K",1,256))
  source=lambda _:UOp.const(dtypes.uint16,0)
  provider.fragment(source,0,0,16)
  with pytest.raises(ValueError,match="K16 scale boundary"): provider.fragment(source,0,8,16)
  with pytest.raises(ValueError,match="K32-aligned"): provider.k32_metadata(source,0,16)
  activation=Q8Int8FragmentProvider(Q8ActivationRecordTransform(1,256))
  contract=Q6KQ8SubgroupAccumulatorContract(provider,activation)
  with pytest.raises(TypeError,match="separate low/high"):
    contract.combine_staged((UOp.const(dtypes.int,1),),UOp.const(dtypes.half.vec(2),(0,0)),
                            UOp.const(dtypes.half.vec(2),(0,0)))  # type: ignore[arg-type]


def test_q6_symbolic_fragment_retains_nk_and_storage_bounds():
  transform=PackedWeightTransform("Q6_K",2,512)
  provider=Q6KInt8FragmentProvider(transform)
  source=UOp.placeholder((transform.packed_bytes//2,),dtypes.uint16,720)
  row,base=UOp.range(2,721),UOp.range(32,722)*16
  fragment=provider.fragment(source,row,base,16)
  assert row in fragment.value.backward_slice_with_self and base in fragment.value.backward_slice_with_self
  indexes=[u for u in fragment.value.toposort() if u.op is Ops.INDEX and source in u.backward_slice_with_self]
  assert indexes and all(u.src[1].vmin>=0 and u.src[1].vmax<transform.packed_bytes//2 for u in indexes)


def test_q6_compact_record_warmstart_key_retains_both_storage_dtypes():
  key=warmstart_key({512,1024},4096,{dtypes.uint16,dtypes.uint32})
  assert key==(frozenset({512,1024}),4096,frozenset({dtypes.uint16,dtypes.uint32}))
  with pytest.raises(ValueError,match="canonical packed storage"):
    warmstart_key({512,1024},4096,{dtypes.uint16,dtypes.half})
