from tinygrad import dtypes
import pytest

from tinygrad.llm.shared_q8_attention import (SharedQ8AttentionAdmission, _emit_q4_cooperative_pair,
  _emit_q4_cooperative_qkv, _emit_q4_cooperative_qkv_balanced, _emit_q4_cooperative_qkv_full,
  _emit_q4_q6_cooperative_pair, _emit_q4_q6_cooperative_qkv_full)
from tinygrad.llm.model_route_plan import (decode_shared_q8_q4kv_pair_promoted, load_decode_shared_q8_q4kv_pair_promotion,
  decode_shared_q8_q4q6_kv_pair_promoted, load_decode_shared_q8_q4q6_kv_pair_promotion,
  decode_shared_q8_q4q4_qkv_full_promoted, decode_q4k_q4q4_qkv_full_promoted)
from tinygrad.uop.ops import Ops, UOp


def test_shared_q8_pair_exact_body_shape():
  rows,k=1024,4096; words=rows*(k//256)*36; blocks=UOp.variable("pair_blocks_test",1,4)
  body=_emit_q4_cooperative_pair(rows,blocks)(UOp.placeholder((rows,),dtypes.float32,0),
    UOp.placeholder((rows,),dtypes.float32,1),UOp.placeholder((words,),dtypes.uint32,2),
    UOp.placeholder((words,),dtypes.uint32,3),UOp.placeholder((k//4+k//32,),dtypes.uint32,4))
  topo=body.toposort()
  assert body.arg.name == "q4k_warp_coop_q8_dp4a_pair_direct_1024_4096"
  assert sum(u.op is Ops.BARRIER for u in topo) == 1


def test_shared_q8_pair_rejects_non_kv_shape():
  try: _emit_q4_cooperative_pair(4096,UOp.variable("bad_pair_blocks",1,4))
  except ValueError: pass
  else: raise AssertionError("non-K/V pair shape must be rejected")


def test_shared_q8_qkv_research_bodies_keep_exact_output_topology():
  qrows,kvrows,k=4096,1024,4096; qwords=qrows*(k//256)*36; kvwords=kvrows*(k//256)*36
  packed=k//4+k//32; blocks=UOp.variable("qkv_blocks_test",1,4); p=UOp.placeholder
  collapsed=_emit_q4_cooperative_qkv(blocks)(p((qrows,),dtypes.float32,0),p((kvrows,),dtypes.float32,1),
    p((kvrows,),dtypes.float32,2),p((qwords,),dtypes.uint32,3),p((kvwords,),dtypes.uint32,4),
    p((kvwords,),dtypes.uint32,5),p((packed,),dtypes.uint32,6))
  balanced=_emit_q4_cooperative_qkv_balanced(blocks)(p((qrows,),dtypes.float32,0),p((kvrows*2,),dtypes.float32,1),
    p((qwords,),dtypes.uint32,2),p((kvwords*2,),dtypes.uint32,3),p((packed,),dtypes.uint32,4))
  full=_emit_q4_cooperative_qkv_full(blocks)(p((qrows,),dtypes.float32,0),p((kvrows,),dtypes.float32,1),
    p((kvrows,),dtypes.float32,2),p((qwords,),dtypes.uint32,3),p((kvwords*2,),dtypes.uint32,4),p((packed,),dtypes.uint32,5))
  assert collapsed.arg.name == "q4k_warp_coop_q8_dp4a_qkv_direct_4096_1024_4096"
  assert balanced.arg.name == "q4k_warp_coop_q8_dp4a_qkv_balanced_direct_4096_1024_4096"
  assert full.arg.name == "q4k_warp_coop_q8_dp4a_qkv_full_direct_4096_1024_4096"
  assert sum(u.op is Ops.BARRIER for u in collapsed.toposort()) == 1
  assert sum(u.op is Ops.BARRIER for u in balanced.toposort()) == 1
  assert sum(u.op is Ops.BARRIER for u in full.toposort()) == 3


def test_shared_q8_mixed_pair_exact_body_shape():
  rows,k=1024,4096; q4words=rows*(k//256)*36; q6halfs=rows*(k//256)*110
  body=_emit_q4_q6_cooperative_pair(rows,UOp.variable("mixed_pair_blocks_test",1,4))(
    UOp.placeholder((rows,),dtypes.float32,0),UOp.placeholder((rows,),dtypes.float32,1),
    UOp.placeholder((q4words,),dtypes.uint32,2),UOp.placeholder((q6halfs,),dtypes.uint16,3),
    UOp.placeholder((k//4+k//32,),dtypes.uint32,4))
  assert body.arg.name == "q4k_q6k_warp_coop_q8_dp4a_pair_direct_1024_4096"
  assert sum(u.op is Ops.BARRIER for u in body.toposort()) == 1


def test_shared_q8_mixed_qkv_full_exact_body_shape():
  qrows,rows,k=4096,1024,4096; q4words=rows*(k//256)*36; q6halfs=rows*(k//256)*110
  body=_emit_q4_q6_cooperative_qkv_full(UOp.variable("mixed_qkv_blocks_test",1,4))(
    UOp.placeholder((qrows,),dtypes.float32,0),UOp.placeholder((rows,),dtypes.float32,1),
    UOp.placeholder((rows,),dtypes.float32,2),UOp.placeholder((q4words*4,),dtypes.uint32,3),
    UOp.placeholder((q4words,),dtypes.uint32,4),UOp.placeholder((q6halfs,),dtypes.uint16,5),
    UOp.placeholder((k//4+k//32,),dtypes.uint32,6))
  assert body.arg.name == "q4k_q6k_warp_coop_q8_dp4a_qkv_full_direct_4096_1024_4096"
  assert sum(u.op is Ops.BARRIER for u in body.toposort()) == 5


def test_shared_q8_pair_admission_is_separate_and_closed():
  assert not SharedQ8AttentionAdmission(1,cooperative_q4=True,q4_direct_output=True).q4_kv_pair_output
  admitted=SharedQ8AttentionAdmission(1,cooperative_q4=True,q4_direct_output=True,q4_kv_pair_output=True)
  assert admitted.q4_kv_pair_output
  with pytest.raises(ValueError,match="requires cooperative Q4 direct output"):
    SharedQ8AttentionAdmission(1,q4_kv_pair_output=True)
  mixed=SharedQ8AttentionAdmission(1,cooperative_q4=True,q4_direct_output=True,q6_direct_output=True,q4_q6_kv_pair_output=True)
  assert mixed.q4_q6_kv_pair_output
  with pytest.raises(ValueError,match="requires cooperative Q4 and Q6 direct output"):
    SharedQ8AttentionAdmission(1,cooperative_q4=True,q4_direct_output=True,q4_q6_kv_pair_output=True)
  full=SharedQ8AttentionAdmission(1,cooperative_q4=True,q4_direct_output=True,q6_direct_output=True,
    q4_q6_qkv_triple_output=True)
  assert full.q4_q6_qkv_triple_output


def test_shared_q8_pair_policy_and_rollback(tmp_path):
  enabled=lambda _name,default=0: default
  disabled=lambda name,default=0: 1 if name=="TINYGRAD_SHARED_Q8_Q4KV_PAIR_DISABLE" else default
  assert decode_shared_q8_q4kv_pair_promoted(("NV","sm_120"),enabled)
  assert not decode_shared_q8_q4kv_pair_promoted(("AMD","gfx1100"),enabled)
  assert not decode_shared_q8_q4kv_pair_promoted(("NV","sm_120"),disabled)
  policy=tmp_path/"policy.json"; policy.write_text('{"schema":"boltbeam.route_policy.v1"}')
  assert load_decode_shared_q8_q4kv_pair_promotion(str(policy)) == frozenset()


def test_shared_q8_mixed_pair_policy_and_rollback(tmp_path):
  enabled=lambda _name,default=0: default
  disabled=lambda name,default=0: 1 if name=="TINYGRAD_SHARED_Q8_Q4Q6_KV_PAIR_DISABLE" else default
  assert decode_shared_q8_q4q6_kv_pair_promoted(("NV","sm_120"),enabled)
  assert not decode_shared_q8_q4q6_kv_pair_promoted(("AMD","gfx1100"),enabled)
  assert not decode_shared_q8_q4q6_kv_pair_promoted(("NV","sm_120"),disabled)
  policy=tmp_path/"mixed-policy.json"; policy.write_text('{"schema":"boltbeam.route_policy.v1"}')
  assert load_decode_shared_q8_q4q6_kv_pair_promotion(str(policy)) == frozenset()


def test_full_grid_policies_are_closed_after_installed_rollback():
  assert not decode_shared_q8_q4q4_qkv_full_promoted(("NV","sm_120"))
  assert not decode_q4k_q4q4_qkv_full_promoted(("NV","sm_120"))
