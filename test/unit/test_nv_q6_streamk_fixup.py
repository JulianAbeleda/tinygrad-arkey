import numpy as np
from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.prefill.nv_q6_streamk_fixup import emit_fixup_kernel, fixup_numpy

def test_q6_streamk_fixup_is_owner_ordered_and_supports_two_or_three_contributors():
  rng=np.random.default_rng(20260831); partials=rng.standard_normal((5,128*128),dtype=np.float32)
  ids=np.array([[0,1,2],[3,4,-1]],dtype=np.int32)
  got=fixup_numpy(partials,ids)
  want=np.zeros((512,4096),np.float32)
  want[:128,:128]=partials[:3].sum(0).reshape(128,128)
  want[:128,128:256]=partials[3:5].sum(0).reshape(128,128)
  assert np.array_equal(got.reshape(512,4096),want)

def test_q6_streamk_fixup_emits_real_bounded_uop_kernel():
  p=lambda n,t,i: UOp.placeholder((n,),t,i)
  sink=emit_fixup_kernel(p(512*4096,dtypes.float32,0),p(340*16384,dtypes.float32,1),p(128*3,dtypes.int32,2))
  assert sum(u.op is Ops.STORE for u in sink.toposort()) == 64
  assert sum(u.op is Ops.SPECIAL for u in sink.toposort()) == 2
