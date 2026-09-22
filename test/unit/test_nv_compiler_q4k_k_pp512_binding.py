import unittest
from types import SimpleNamespace

from extra.llm_research.prefill.nv_compiler_q4k_k_pp512_binding import (
  K, M, N, PROJECTIONS_PER_MODEL, RECORD_BYTES, RECORD_U32, CompilerKPP512Binding, CompilerKPP512Capture, supports)
from tinygrad.llm.model import _nv_compiler_q4_imma_k_capture


class TestNVCompilerQ4KKPP512Binding(unittest.TestCase):
  def test_only_exact_k_role_is_admitted(self):
    base = dict(model_family="qwen3_8b", role="attn_k", weight_type="Q4_K", m=M, n=N, k=K, device="NV")
    self.assertTrue(supports(**base))
    for mutation in ({"model_family":"qwen3_14b"}, {"role":"attn_v"}, {"weight_type":"Q6_K"},
                     {"m":M-1}, {"n":N*2}, {"k":K+1}, {"device":"AMD"}):
      self.assertFalse(supports(**(base | mutation)), mutation)

  def test_record_is_exact_compact_packet(self):
    self.assertEqual(RECORD_BYTES, M*K + 2*M*(K//32)*4)
    self.assertEqual(RECORD_U32*4, RECORD_BYTES)

  def test_k_population_rejects_v_sized_census(self):
    CompilerKPP512Binding.prepare_records(SimpleNamespace(), PROJECTIONS_PER_MODEL)
    with self.assertRaisesRegex(ValueError, "exact K route"):
      CompilerKPP512Binding.prepare_records(SimpleNamespace(), 18)

  def test_capture_epoch_and_census_fail_closed(self):
    capture = CompilerKPP512Capture(SimpleNamespace())
    with self.assertRaisesRegex(RuntimeError, "begin_trace"):
      capture.project(None, None, model_family="qwen3_8b", role="attn_k")
    capture.begin_trace(); capture.cursor = PROJECTIONS_PER_MODEL
    with self.assertRaisesRegex(RuntimeError, "36-projection"):
      capture.project(None, None, model_family="qwen3_8b", role="attn_k")

  def test_model_and_jit_captures_are_independent(self):
    class Asset:
      def new_capture(self): return CompilerKPP512Capture(self)
    asset, model_a, model_b, jit_0, jit_1 = Asset(), SimpleNamespace(), SimpleNamespace(), object(), object()
    a0 = _nv_compiler_q4_imma_k_capture(model_a, jit_0, asset)
    a1 = _nv_compiler_q4_imma_k_capture(model_a, jit_1, asset)
    b0 = _nv_compiler_q4_imma_k_capture(model_b, jit_0, asset)
    self.assertIs(_nv_compiler_q4_imma_k_capture(model_a, jit_0, asset), a0)
    self.assertIsNot(a0, a1)
    self.assertIsNot(a0, b0)
    a0.begin_trace(); a0.cursor = 19
    a1.begin_trace(); b0.begin_trace()
    self.assertEqual((a0.cursor, a1.cursor, b0.cursor), (19, 0, 0))


if __name__ == "__main__": unittest.main()
