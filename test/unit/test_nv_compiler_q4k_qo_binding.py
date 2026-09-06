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

  def test_live_census_rejects_same_name_with_wrong_geometry_or_binary(self):
    from dataclasses import replace
    from types import SimpleNamespace
    from tinygrad.uop.ops import UOp, Ops
    from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
    from extra.llm_research.prefill.nv_compiler_q4k_qo_72real_proxy import census
    expected=native_nv_program("projection",b"\x7fELFfixture",global_size=(32,8,1),local_size=(32,2,2),globals=())
    def captured(program):
      return SimpleNamespace(captured=SimpleNamespace(linear=UOp(Ops.LINEAR,src=(UOp(Ops.CALL,src=(program,)),))))
    self.assertTrue(census(captured(expected),[(expected,1)])["exact"])
    wrong_grid=expected.replace(arg=replace(expected.arg,global_size=(32,1,1)))
    wrong_binary=expected.replace(src=tuple(u.replace(arg=b"\x7fELFother") if u.op is Ops.BINARY else u for u in expected.src))
    for wrong in (wrong_grid,wrong_binary):
      self.assertFalse(census(captured(wrong),[(expected,1)])["exact"])
    self.assertFalse(census(captured(expected),[(expected,2)])["exact"])


if __name__=="__main__":unittest.main()
