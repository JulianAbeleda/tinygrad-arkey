import unittest

from extra.llm_research.prefill.nv_llama_fattn_mma_pp512_binding import program


class TestNVLlamaFattnMMAPP512Contract(unittest.TestCase):
  def test_native_program_contract(self):
    prg = program()
    self.assertEqual(prg.arg.name, "nv_llama_fattn_mma_pp512")
    self.assertEqual(prg.arg.global_size, (256, 1, 1))
    self.assertEqual(prg.arg.local_size, (32, 4, 1))
    self.assertEqual(prg.arg.globals, (0, 1, 2, 3, 4))
    self.assertEqual(prg.arg.outs, (4,))
    self.assertEqual(prg.arg.ins, (0, 1, 2, 3))
    # CUDA reports 37,120 dynamic bytes; QMD requires that plus the cubin's
    # 1,024-byte static .nv.shared allocation.
    self.assertEqual(prg.arg.aux, (38144,))
    self.assertEqual(prg.src[4].arg[:4], b"\x7fELF")


if __name__ == "__main__": unittest.main()
