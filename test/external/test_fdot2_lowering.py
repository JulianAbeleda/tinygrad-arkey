#!/usr/bin/env python3
"""P0.1: opt-in AMD fdot2 lowering for the exact fp16 dot2 idiom."""
import os, tempfile, unittest
import numpy as np

from tinygrad import Tensor, Device, dtypes
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import UOp, Ops, graph_rewrite
from extra.qk_fdot2_lowering import pm_fdot2, lower_fdot2_add

_DEV_OK = Device.DEFAULT == "AMD"

def _half2_symbol(name: str) -> UOp:
  seed = (sum(ord(c) for c in name) % 17) + 1
  return UOp(Ops.STACK, dtypes.half.vec(2), (UOp.const(dtypes.half, float(seed)), UOp.const(dtypes.half, float(seed + 1))))

def _lane(v: UOp, i: int) -> UOp:
  return UOp(Ops.INDEX, dtypes.half, (v, UOp.const(dtypes.int, i)))

def _term(a: UOp, b: UOp, i: int) -> UOp:
  return (_lane(a, i) * _lane(b, i)).cast(dtypes.float)

class TestFdot2LoweringStructural(unittest.TestCase):
  def test_rule_structural_pair(self):
    a, b = _half2_symbol("a"), _half2_symbol("b")
    out = graph_rewrite(_term(a, b, 0) + _term(a, b, 1), pm_fdot2)
    nodes = list(out.toposort())
    fdot = [u for u in nodes if u.op is Ops.CUSTOMI and "fdot2" in str(u.arg)]
    self.assertEqual(len(fdot), 1)
    self.assertFalse(any(u.op is Ops.MUL for u in nodes), "fdot2 pair should remove lane MULs")

  def test_rule_structural_with_acc(self):
    a, b = _half2_symbol("a"), _half2_symbol("b")
    acc = UOp.const(dtypes.float, 7.0)
    out = lower_fdot2_add(acc + (_term(a, b, 0) + _term(a, b, 1)))
    self.assertIsNotNone(out)
    fdot = [u for u in out.toposort() if u.op is Ops.CUSTOMI and "fdot2" in str(u.arg)]
    self.assertEqual(len(fdot), 1)
    self.assertIs(fdot[0].src[0], acc)

  def test_rule_declines_mismatched_sources(self):
    a, b, c = _half2_symbol("a"), _half2_symbol("b"), _half2_symbol("c")
    expr = _term(a, b, 0) + _term(a, c, 1)
    out = graph_rewrite(expr, pm_fdot2)
    self.assertFalse(any(u.op is Ops.CUSTOMI and "fdot2" in str(u.arg) for u in out.toposort()))

  def test_rule_declines_same_lane_twice(self):
    a, b = _half2_symbol("a"), _half2_symbol("b")
    expr = _term(a, b, 0) + _term(a, b, 0)
    out = graph_rewrite(expr, pm_fdot2)
    self.assertFalse(any(u.op is Ops.CUSTOMI and "fdot2" in str(u.arg) for u in out.toposort()))

@unittest.skipUnless(_DEV_OK, "fdot2 lowering is AMD/gfx1100 gated")
class TestFdot2LoweringPipeline(unittest.TestCase):
  M, K = 64, 2

  def setUp(self):
    from tinygrad.helpers import getenv
    self._old = os.environ.get("V_DOT2_LOWERING")
    os.environ["V_DOT2_LOWERING"] = "1"; getenv.cache_clear()
    Tensor.manual_seed(0)

  def tearDown(self):
    from tinygrad.helpers import getenv
    if self._old is None: os.environ.pop("V_DOT2_LOWERING", None)
    else: os.environ["V_DOT2_LOWERING"] = self._old
    getenv.cache_clear()

  def _expr(self, anp, bnp):
    return (Tensor(anp) * Tensor(bnp)).sum(axis=1)

  def test_pipeline_correct_and_emits_fdot2(self):
    rng = np.random.default_rng(20260625)
    anp = rng.standard_normal((self.M, self.K)).astype(np.float16)
    bnp = rng.standard_normal((self.M, self.K)).astype(np.float16)
    out = self._expr(anp, bnp)
    got = out.numpy()
    ref = (anp * bnp).sum(axis=1)
    rel = np.linalg.norm(got.astype(np.float32) - ref.astype(np.float32)) / max(1e-12, np.linalg.norm(ref.astype(np.float32)))
    self.assertLessEqual(rel, 1e-2)

    srcs, bins = [], []
    for c in compile_linear(out.schedule_linear()).src:
      if c.src[0].op is Ops.PROGRAM:
        srcs.append(next((u.arg for u in c.src[0].toposort() if u.op is Ops.SOURCE), ""))
        bins.append(next((u.arg for u in c.src[0].toposort() if u.op is Ops.BINARY), None))
    src = "\n".join(srcs)
    self.assertIn("__builtin_amdgcn_fdot2", src)
    with tempfile.NamedTemporaryFile(suffix=".co") as f:
      f.write(next(b for b in bins if b)); f.flush()
      from extra.qk_amdgpu_isa_primitive_audit import audit
      isa = audit(f.name)
    self.assertTrue(isa.get("flags", {}).get("has_v_dot2"), isa)

if __name__ == "__main__":
  unittest.main()
