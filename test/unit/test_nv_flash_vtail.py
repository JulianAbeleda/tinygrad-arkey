import pytest

from tinygrad import dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.llm.flash_decode_attention import flash_vec_llama_score_pv_kernel
from tinygrad.uop.ops import Ops, UOp


def test_wide_flash_tail_v_has_distinct_register_storage_and_name():
  pout=UOp.placeholder((32*6*130,),dtypes.float32,0)
  q=UOp.placeholder((32*64,),dtypes.uint32,1)
  cache=UOp.placeholder((2,1,8,768,64),dtypes.uint32,2)
  ast=flash_vec_llama_score_pv_kernel(128,32,8,768,6,UOp.const(dtypes.int,513),wide_kv=True,
                                      token_bound=768,v_pipeline_tail=1)(pout,q,cache)
  assert ast.arg.name.endswith("_widekv16_vtail1")
  assert any(u.op is Ops.DEFINE_REG and u.dtype.addrspace is AddrSpace.REG for u in ast.toposort())


def test_tail_v_rejects_non_wide_flash():
  with pytest.raises(ValueError,match="requires wide_kv"):
    flash_vec_llama_score_pv_kernel(128,32,8,768,6,UOp.const(dtypes.int,513),token_bound=768,v_pipeline_tail=1)
