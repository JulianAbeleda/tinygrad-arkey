import inspect
import numpy as np

from extra.llm_research.prefill.bench_nv_q6_oracle_kprefix_trace import _published
from extra.llm_research.prefill.nv_q6_oracle_broad_cta import Q6_STRIDE, SHARED_BYTES, q6_oracle_broad_cta_kernel


def _block(scales, d):
  raw=np.zeros(210,dtype=np.uint8)
  raw[192:208]=np.asarray(scales,dtype=np.int8).view(np.uint8)
  raw[208:210]=np.frombuffer(np.float16(d).tobytes(),dtype=np.uint8)
  return raw


def test_trusted_fp16_packed_76_word_abi_and_rounding():
  scales=np.asarray((-128,-127,-65,-33,-3,-1,0,1,2,3,17,31,63,65,126,127),dtype=np.int8)
  raw=_block(scales,np.float16(0.017578125));published=_published(raw,"packed")
  d=np.float32(np.float16(0.017578125));expected=np.float16(d*scales.astype(np.float32)).view(np.uint16)
  unpacked=np.empty(16,dtype=np.uint16)
  unpacked[0::2]=(published[65:73]&0xffff).astype(np.uint16)
  unpacked[1::2]=(published[65:73]>>16).astype(np.uint16)
  assert Q6_STRIDE == 76 and SHARED_BYTES == 57344 and published.shape == (76,)
  assert published[64] == np.asarray(d,dtype=np.float32).view(np.uint32)
  assert np.array_equal(unpacked,expected)
  assert np.array_equal(unpacked.view(np.float16).astype(np.float32),expected.view(np.float16).astype(np.float32))
  assert np.array_equal(published[73:76],np.full(3,0xdeadbeef,dtype=np.uint32))


def test_packed_contract_is_explicit_and_default_compatible():
  source=inspect.getsource(q6_oracle_broad_cta_kernel)
  assert 'weight_scale_contract:str="legacy"' in source
  assert 'trusted_fp16_packed' in source
  assert 'packed_weight_pair' in source
  assert '65+kphase*4+p' in source


def test_trusted_contract_exposes_legal_inner_and_outer_fma_modes():
  source=inspect.getsource(q6_oracle_broad_cta_kernel)
  assert '"trusted_inner","trusted_outer","trusted_both"' in source
  assert 'fp_fma(ws0,z0.cast(dtypes.float32),dot1)' in source
  assert 'fp_fma(yscale,weighted,carrier[0])' in source
  assert 'fp32_contraction not in trusted_contractions' in source
