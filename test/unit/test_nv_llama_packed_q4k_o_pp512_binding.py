import unittest
from extra.llm_research.prefill.nv_llama_packed_q4k_o_pp512_binding import (
  M,N,K,PROJECTIONS_PER_MODEL,supports,Binding,Capture)

class TestPackedQ4KAttentionOutput(unittest.TestCase):
  def test_exact_admission(self):
    self.assertTrue(supports(model_family="qwen3_8b",role="attn_output",weight_type="Q4_K",m=M,n=N,k=K,device="NV"))
  def test_fail_closed(self):
    base=dict(model_family="qwen3_8b",role="attn_output",weight_type="Q4_K",m=M,n=N,k=K,device="NV")
    for key,val in (("role","ffn_down"),("weight_type","Q6_K"),("m",511),("n",12288),("k",8192),("device","CPU")):
      self.assertFalse(supports(**(base|{key:val})))
  def test_exact_census_and_trace_gate(self):
    c=Capture(Binding(None,None,None))
    with self.assertRaises(RuntimeError): c.project(None,None,model_family="qwen3_8b",role="attn_output")
    c.begin_trace(); c.cursor=PROJECTIONS_PER_MODEL
    with self.assertRaises(RuntimeError): c.project(None,None,model_family="qwen3_8b",role="attn_output")

if __name__ == "__main__": unittest.main()
