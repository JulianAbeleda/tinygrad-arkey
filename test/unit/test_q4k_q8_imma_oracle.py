import numpy as np
from extra.llm_research.prefill.q4k_q8_imma_oracle import *

def test_adversarial_integer_algebra_matches_dense_affine():
  f=adversarial_fixture(); got=q4k_q8_block(*f); ref,_=dense_reference(*f)
  assert got == ref
  # Stable vector catches nibble order, signed-nibble centering, scale/min unpack.
  assert got == np.float32(16505.28125)

def test_adversarial_fixture_rejects_common_q4_errors():
  dm,meta,packed,q8,d8,sums=adversarial_fixture(); right=q4k_q8_block(dm,meta,packed,q8,d8,sums)
  raw=np.frombuffer(packed,np.uint8)
  swapped=((raw>>4)|((raw&15)<<4)).astype(np.uint8).tobytes()
  assert q4k_q8_block(dm,meta,swapped,q8,d8,sums) != right
  # Incorrectly treating the unsigned Q4 nibble as a centered signed value.
  sc,mn=unpack_scales(meta); centered=unpack_qs(packed).astype(np.int32)-8
  wrong=np.float32(dm[0])*sum(d8*sc*(centered*q8.reshape(8,32)).sum(1))-np.float32(dm[1])*sum(mn*sums)
  assert np.float32(wrong) != right

def test_raw_sum_correction_is_not_quantized_sum():
  dm,meta,qs,q8,d8,sums=adversarial_fixture()
  a=q4k_q8_block(dm,meta,qs,q8,d8,sums)
  wrong=d8*np.asarray(q8,dtype=np.int8).reshape(8,32).astype(np.int32).sum(1)
  b=q4k_q8_block(dm,meta,qs,q8,d8,wrong)
  assert a != b

def test_m16n8k32_fragment_coordinates_are_bijective():
  A=[a_fragment_coord(l,r,b) for l in range(32) for r in range(4) for b in range(4)]
  B=[b_fragment_coord(l,r,b) for l in range(32) for r in range(2) for b in range(4)]
  D=[d_fragment_coord(l,r) for l in range(32) for r in range(4)]
  assert len(set(A))==16*32 and set(A)==set(np.ndindex(16,32))
  assert len(set(B))==32*8 and set(B)==set(np.ndindex(32,8))
  assert len(set(D))==16*8 and set(D)==set(np.ndindex(16,8))

def test_fragment_mma_reconstructs_dense_int32():
  A=((np.arange(16*32)*17+3)%16).reshape(16,32).astype(np.int8)
  B=((np.arange(32*8)*29+5)%255-127).reshape(32,8).astype(np.int8)
  ref=A.astype(np.int32)@B.astype(np.int32)
  got=np.empty_like(ref)
  for lane in range(32):
    for reg in range(4):
      i,j=d_fragment_coord(lane,reg); got[i,j]=sum(int(A[i,k])*int(B[k,j]) for k in range(32))
  np.testing.assert_array_equal(got,ref)

def test_typed_q4_fragment_provider_preserves_group_parity():
  b=0xD2
  assert [int(q4_typed_fragment_value(b,g)) for g in range(8)] == [2,13,2,13,2,13,2,13]
  # The observed blind postrange substitution agrees only for groups 0 and 1.
  assert [int(q4_broken_postrange_value(b,g)) for g in range(8)] == [2,13,13,13,13,13,13,13]
