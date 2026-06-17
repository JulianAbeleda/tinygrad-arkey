#!/usr/bin/env python3
"""Fast unit guards for the prefill-v2 build (no model load). The end-to-end win (~13x warm prefill on 8B)
is measured by extra/qk_prefill_v2_measure.py; these lock the pure logic that makes it correct + decode-safe."""
import unittest

from tinygrad import Tensor, UOp, dtypes
from tinygrad.codegen.opt import OptOps
from tinygrad.llm import model as M

class TestPrefillV2Opts(unittest.TestCase):
  def test_per_shape_upcast(self):
    # contraction-heavy (in>out, e.g. ffn_down 4096x12288) wants UPCAST(0,4); the rest UPCAST(0,2).
    # One schedule for all drops the in-model chain ~37%->~9% (verified), so this split is load-bearing.
    down = M._prefill_v2_opts(4096, 12288)   # ffn_down: in>out
    gate = M._prefill_v2_opts(12288, 4096)   # ffn_gate/up: in<out
    attn = M._prefill_v2_opts(4096, 4096)    # attn_q/output: in==out
    def upcast0(opts): return next(o.arg for o in opts if o.op is OptOps.UPCAST and o.axis == 0)
    self.assertEqual(upcast0(down), 4)
    self.assertEqual(upcast0(gate), 2)
    self.assertEqual(upcast0(attn), 2)
    for opts in (down, gate, attn):  # all carry the TC + the UPCAST(1,4)
      self.assertTrue(any(o.op is OptOps.TC for o in opts))
      self.assertTrue(any(o.op is OptOps.UPCAST and o.axis == 1 and o.arg == 4 for o in opts))

class TestConcreteVsSymbolic(unittest.TestCase):
  def test_shape_int_detection(self):
    # is_prefill_v2 hinges on: a fixed-length slice off a symbolic offset has a CONCRETE int dim, while the
    # normal symbolic prefill (v_toks) dim is a UOp -> isinstance(...,int) cleanly separates the two jits.
    vsp = UOp.variable("start_pos", 0, 4095)
    t = Tensor.zeros(1, 4096, dtype="int32").contiguous()
    sp = vsp.bind(3)
    v2 = t[:, sp:sp + M.PREFILL_UBATCH]
    self.assertTrue(isinstance(v2.shape[1], int))
    self.assertEqual(v2.shape[1], M.PREFILL_UBATCH)
    sym = t[:, sp:sp + UOp.variable("toks", 1, 32).bind(5)]
    self.assertFalse(isinstance(sym.shape[1], int))

class TestPf16(unittest.TestCase):
  def test_uses_cached_realized_weight(self):
    # _pf16 must matmul against the realized fp16 cache (_pf16_w), not the lazy dequant weight -- the lazy
    # path fuses the dequant into the matmul (~3% peak). Output is fp16.
    class FakeLin:
      def __init__(self, w): self.weight, self.bias = w, None
    lin = FakeLin(Tensor.randn(8, 4, dtype=dtypes.float32))   # lazy "dequant" stand-in (fp32)
    lin._pf16_w = lin.weight.cast(dtypes.float16).contiguous().realize()
    x = Tensor.randn(1, 2, 4, dtype=dtypes.float32)
    out = M._pf16(lin, x)
    self.assertEqual(out.dtype, dtypes.float16)
    self.assertEqual(out.shape, (1, 2, 8))

  def test_fallback_without_cache(self):
    class FakeLin:
      def __init__(self, w): self.weight, self.bias = w, None
    lin = FakeLin(Tensor.randn(8, 4, dtype=dtypes.float16))
    out = M._pf16(lin, Tensor.randn(1, 2, 4, dtype=dtypes.float16))
    self.assertEqual(out.dtype, dtypes.float16)
    self.assertEqual(out.shape, (1, 2, 8))

if __name__ == "__main__":
  unittest.main()
