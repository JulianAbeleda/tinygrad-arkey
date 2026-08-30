import pytest
from extra.llm_research.prefill.nv_qkv_packed_pp512_binding import M,K,QN,KVN,supports,QKVCapture

def test_exact_qkv_admission_is_fail_closed():
  assert supports(model_family="qwen3_8b",role="attn_q",weight_type="Q4_K",m=M,n=QN,k=K,device="NV")
  for role,typ,n in (("attn_k","Q4_K",KVN),("attn_v","Q4_K",KVN),("attn_v","Q6_K",KVN)):
    assert supports(model_family="qwen3_8b",role=role,weight_type=typ,m=M,n=n,k=K,device="NV")
  assert not supports(model_family="qwen3_14b",role="attn_q",weight_type="Q4_K",m=M,n=QN,k=K,device="NV")

def test_capture_requires_epoch_and_keeps_buffers_lazy():
  c=QKVCapture(object())
  with pytest.raises(RuntimeError, match="begin_trace"): c.project_q6_v(None,None)
