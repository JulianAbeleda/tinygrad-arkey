from tinygrad import dtypes
import pytest

from tinygrad.llm.shared_q8_attention import SharedQ8AttentionAdmission, _emit_q4_cooperative_pair
from tinygrad.llm.model_route_plan import decode_shared_q8_q4kv_pair_promoted, load_decode_shared_q8_q4kv_pair_promotion
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


def test_shared_q8_pair_admission_is_separate_and_closed():
  assert not SharedQ8AttentionAdmission(1,cooperative_q4=True,q4_direct_output=True).q4_kv_pair_output
  admitted=SharedQ8AttentionAdmission(1,cooperative_q4=True,q4_direct_output=True,q4_kv_pair_output=True)
  assert admitted.q4_kv_pair_output
  with pytest.raises(ValueError,match="requires cooperative Q4 direct output"):
    SharedQ8AttentionAdmission(1,q4_kv_pair_output=True)


def test_shared_q8_pair_policy_and_rollback(tmp_path):
  enabled=lambda _name,default=0: default
  disabled=lambda name,default=0: 1 if name=="TINYGRAD_SHARED_Q8_Q4KV_PAIR_DISABLE" else default
  assert decode_shared_q8_q4kv_pair_promoted(("NV","sm_120"),enabled)
  assert not decode_shared_q8_q4kv_pair_promoted(("AMD","gfx1100"),enabled)
  assert not decode_shared_q8_q4kv_pair_promoted(("NV","sm_120"),disabled)
  policy=tmp_path/"policy.json"; policy.write_text('{"schema":"boltbeam.route_policy.v1"}')
  assert load_decode_shared_q8_q4kv_pair_promotion(str(policy)) == frozenset()
