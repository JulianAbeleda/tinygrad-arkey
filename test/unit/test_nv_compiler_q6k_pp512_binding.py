import unittest

from extra.llm_research.prefill.nv_compiler_q6k_pp512_binding import (_record_source, supports,
  ROLE_COUNTS, ROLE_SHAPES, PROJECTIONS_PER_MODEL)


class TestNVCompilerQ6KPP512Binding(unittest.TestCase):
  def test_exact_role_population(self):
    self.assertEqual(dict(ROLE_SHAPES), {"attn_v":(1024,4096), "ffn_down":(4096,12288)})
    self.assertEqual(dict(ROLE_COUNTS), {"attn_v":18, "ffn_down":18})
    self.assertEqual(PROJECTIONS_PER_MODEL, 36)

  def test_supports_fail_closed_by_format_role_shape_and_device(self):
    good=dict(model_family="qwen3_8b",role="attn_v",weight_type="Q6_K",m=512,n=1024,k=4096,device="NV")
    self.assertTrue(supports(**good))
    for key,bad in (("model_family","qwen3_14b"),("role","attn_k"),("weight_type","Q4_K"),
                    ("m",256),("n",4096),("k",12288),("device","CUDA")):
      self.assertFalse(supports(**{**good,key:bad}),key)
    self.assertTrue(supports(model_family="qwen3_8b",role="ffn_down",weight_type="Q6_K",
                             m=512,n=4096,k=12288,device="NV"))

  def test_down_producer_has_exact_k_and_record_abi(self):
    src=_record_source(12288,"q8_compact_record_fp16_q6_ffn_down")
    self.assertIn("base=row*12288+i;",src)
    self.assertIn("int g=row*384+seg*16+t/8;",src)
    self.assertIn("q+6291456",src)
    self.assertNotIn("base=row*4096+i;",src)


if __name__ == "__main__": unittest.main()
