import pytest

from tinygrad import dtypes
from tinygrad.llm.q4k_kv_pair import Q4KKVPairAdmission, emit_q4k_kv_pair_vector, q4k_kv_pair_call
from tinygrad.llm.model_route_plan import decode_q4k_kv_pair_promoted, load_decode_q4k_kv_pair_promotion
from tinygrad.uop.ops import Ops, UOp


def test_q4k_kv_pair_exact_body():
  words = 1024 * 16 * 36
  body = emit_q4k_kv_pair_vector()(UOp.placeholder((1024,), dtypes.float32, 0),
    UOp.placeholder((1024,), dtypes.float32, 1), UOp.placeholder((words,), dtypes.uint32, 2),
    UOp.placeholder((words,), dtypes.uint32, 3), UOp.placeholder((4096,), dtypes.float16, 4))
  topo = body.toposort()
  assert body.arg.name == "q4k_g3_lanemap_gemv_pair_vec_1024_4096"
  # Two accumulator chains, two staged shuffle ladders, and two output stores.
  assert sum(u.op is Ops.STORE for u in topo) == 16
  assert sum(u.op is Ops.BARRIER for u in topo) == 0


def test_q4k_kv_pair_shape_is_closed():
  with pytest.raises(ValueError): emit_q4k_kv_pair_vector(2048, 4096)
  with pytest.raises(ValueError): emit_q4k_kv_pair_vector(1024, 8192)


def test_q4k_kv_pair_admission_is_closed():
  assert Q4KKVPairAdmission(0).block_index == 0
  with pytest.raises(ValueError): Q4KKVPairAdmission(-1)
  with pytest.raises(ValueError): Q4KKVPairAdmission(True)
  assert q4k_kv_pair_call(None, None, None, None) is None


def test_q4k_kv_pair_policy_and_rollback(tmp_path):
  enabled=lambda _name,default=0: default
  disabled=lambda name,default=0: 1 if name == "TINYGRAD_Q4K_KV_PAIR_DISABLE" else default
  assert decode_q4k_kv_pair_promoted(("NV","sm_120"),enabled)
  assert not decode_q4k_kv_pair_promoted(("AMD","gfx1100"),enabled)
  assert not decode_q4k_kv_pair_promoted(("NV","sm_120"),disabled)
  policy=tmp_path/"policy.json"; policy.write_text('{"schema":"boltbeam.route_policy.v1"}')
  assert load_decode_q4k_kv_pair_promotion(str(policy)) == frozenset()
