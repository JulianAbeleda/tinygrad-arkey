import unittest
from types import SimpleNamespace

from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import (M, N, K, RECORD_BYTES, RECORD_U32,
  PROJECTIONS_PER_MODEL, CompilerPP512Capture, supports)
from tinygrad import Tensor, dtypes
from tinygrad.llm.model import _nv_compiler_q4_imma_capture, _nv_compiler_q4_imma_pp512_qualified


class TestNVCompilerQ4KPP512Binding(unittest.TestCase):
  def test_exact_gate_up_admission(self):
    for role in ("ffn_gate", "ffn_up"):
      self.assertTrue(supports(model_family="qwen3_8b", role=role, weight_type="Q4_K",
                               m=M, n=N, k=K, device="NV"))

  def test_fail_closed(self):
    base = dict(model_family="qwen3_8b", role="ffn_gate", weight_type="Q4_K", m=M, n=N, k=K, device="NV")
    for mutation in ({"model_family":"qwen3_14b"}, {"role":"ffn_down"}, {"weight_type":"Q6_K"},
                     {"m":M-1}, {"n":4096}, {"k":5120}, {"device":"AMD"}):
      self.assertFalse(supports(**(base | mutation)), mutation)

  def test_record_is_exact_compact_packet(self):
    self.assertEqual(RECORD_BYTES, M*K + 2*M*(K//32)*4)
    self.assertEqual(RECORD_U32*4, RECORD_BYTES)

  def test_model_topology_is_exact(self):
    exact = dict(prefill_ubatch=512, num_blocks=36, dim=4096, hidden_dim=12288,
                 n_heads=32, n_kv_heads=8, head_dim=128, num_experts=0)
    self.assertTrue(_nv_compiler_q4_imma_pp512_qualified(SimpleNamespace(**exact)))
    for key, value in (("prefill_ubatch",256), ("num_blocks",40), ("dim",5120), ("hidden_dim",17408),
                       ("n_heads",40), ("n_kv_heads",4), ("head_dim",256), ("num_experts",128)):
      self.assertFalse(_nv_compiler_q4_imma_pp512_qualified(SimpleNamespace(**(exact | {key:value}))), key)

  def test_two_models_and_two_captures_have_independent_epochs(self):
    class Asset:
      def new_capture(self): return CompilerPP512Capture(self)
    asset, model_a, model_b, jit_0, jit_1 = Asset(), SimpleNamespace(), SimpleNamespace(), object(), object()
    model_a_capture_0 = _nv_compiler_q4_imma_capture(model_a, jit_0, asset)
    model_a_capture_1 = _nv_compiler_q4_imma_capture(model_a, jit_1, asset)
    model_b_capture_0 = _nv_compiler_q4_imma_capture(model_b, jit_0, asset)
    # Replay of one JIT must recover its stable state, while neither a second
    # JIT nor a second model is allowed to share that mutable trace identity.
    self.assertIs(_nv_compiler_q4_imma_capture(model_a, jit_0, asset), model_a_capture_0)
    self.assertIsNot(model_a_capture_0, model_a_capture_1)
    self.assertIsNot(model_a_capture_0, model_b_capture_0)
    model_a_capture_0.begin_trace()
    model_a_capture_0.cursor = 17
    model_a_capture_1.begin_trace()
    model_b_capture_0.begin_trace()
    self.assertEqual((model_a_capture_0.cursor, model_a_capture_1.cursor, model_b_capture_0.cursor), (17, 0, 0))
    model_a_capture_1.cursor = 9
    model_a_capture_0.begin_trace()
    self.assertEqual((model_a_capture_0.trace_epoch, model_a_capture_0.cursor), (2, 0))
    self.assertEqual((model_a_capture_1.trace_epoch, model_a_capture_1.cursor), (1, 9))
    self.assertEqual((model_b_capture_0.trace_epoch, model_b_capture_0.cursor), (1, 0))

  def test_capture_fails_closed_without_epoch_or_past_census(self):
    capture = CompilerPP512Capture(SimpleNamespace())
    with self.assertRaisesRegex(RuntimeError, "begin_trace"):
      capture.project(None, None, model_family="qwen3_8b", role="ffn_gate")
    capture.begin_trace(); capture.cursor = PROJECTIONS_PER_MODEL
    with self.assertRaisesRegex(RuntimeError, "72-projection"):
      capture.project(None, None, model_family="qwen3_8b", role="ffn_gate")

  def test_projection_buffers_are_lazy_and_capture_local(self):
    # This is the allocation contract used by _project: unique lazy buffers,
    # never a device-global mutable pool. A scheduler may plan/reuse these
    # after it sees the whole graph.
    def buffers(): return (Tensor.empty(RECORD_U32, dtype=dtypes.uint32, device="CPU"),
                           Tensor.empty(M*N, dtype=dtypes.float32, device="CPU"))
    a_record, a_output = buffers(); b_record, b_output = buffers()
    self.assertIsNot(a_record.uop.buf_uop, b_record.uop.buf_uop)
    self.assertIsNot(a_output.uop.buf_uop, b_output.uop.buf_uop)
    for tensor in (a_record, a_output, b_record, b_output):
      self.assertFalse(tensor.uop.buf_uop.buffer.is_allocated())


if __name__ == "__main__": unittest.main()
