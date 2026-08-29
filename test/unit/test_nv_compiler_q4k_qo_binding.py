import unittest

from extra.llm_research.prefill.nv_compiler_q4k_qo_binding import K,M,N,RECORD_BYTES,RECORD_U32,supports


class TestNVCompilerQ4KQOBinding(unittest.TestCase):
  def test_exact_qo_admission(self):
    for role in ("attn_q","attn_output"):
      self.assertTrue(supports(model_family="qwen3_8b",role=role,weight_type="Q4_K",m=M,n=N,k=K,device="NV"))

  def test_fail_closed(self):
    base=dict(model_family="qwen3_8b",role="attn_q",weight_type="Q4_K",m=M,n=N,k=K,device="NV")
    for mutation in ({"model_family":"qwen3_14b"},{"role":"ffn_gate"},{"weight_type":"Q6_K"},{"m":256},
                     {"n":12288},{"k":5120},{"device":"AMD"}):
      self.assertFalse(supports(**(base|mutation)),mutation)

  def test_compact_record_size(self):
    self.assertEqual(RECORD_BYTES,M*K+2*M*(K//32)*4)
    self.assertEqual(RECORD_U32*4,RECORD_BYTES)


if __name__=="__main__":unittest.main()
