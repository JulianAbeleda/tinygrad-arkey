import unittest
from extra.llm_research.prefill.nv_q4_imma_pp512_binding import supports
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from tinygrad.uop.ops import ProgramInfo, UOp


class TestPP512BindingAdmission(unittest.TestCase):
  def test_exact_gate_and_up(self):
    for role in ("ffn_gate", "ffn_up"):
      self.assertTrue(supports(model_family="qwen3_8b", role=role, weight_type="Q4_K",
                               m=512, n=12288, k=4096, device="NV"))

  def test_fail_closed(self):
    mutations = ({"role":"ffn_down"}, {"model_family":"qwen3_14b"}, {"weight_type":"Q6_K"},
                 {"m":511}, {"n":4096}, {"k":5120}, {"device":"CPU"})
    base = dict(model_family="qwen3_8b", role="ffn_gate", weight_type="Q4_K",
                m=512, n=12288, k=4096, device="NV")
    for mutation in mutations:
      self.assertFalse(supports(**(base | mutation)), mutation)

  def test_finalized_program_fixed_vals(self):
    fixed=UOp.variable("fixed_native_arg", 4096, 4096)
    self.assertEqual(ProgramInfo(vars=(fixed,)).vals({}), (4096,))

  def test_symbolic_program_val_still_requires_binding(self):
    symbolic=UOp.variable("symbolic_native_arg", 1, 4096)
    with self.assertRaises(KeyError): ProgramInfo(vars=(symbolic,)).vals({})

  def test_native_program_records_construction_provenance(self):
    program=native_nv_program("fixture", b"\x7fELFfixture", global_size=(1,1,1), local_size=(32,1,1), globals=())
    self.assertEqual(program.arg.provenance,
      ("native_precompiled", "extra.llm_research.prefill.nv_native_program_uop.native_nv_program", "NV"))


if __name__ == "__main__": unittest.main()
