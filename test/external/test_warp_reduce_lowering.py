#!/usr/bin/env python3
"""Milestone 5 first pass: AUTO-lowering of a warp-axis Ops.REDUCE to the ds_bpermute ladder
(extra/qk_warp_reduce_lowering.pm_warp_reduce). Distinct from test_amd_warp_reduce.py, which tests the
hand-CALLED ladder; here a generic REDUCE over an AxisType.WARP range is auto-rewritten by a PatternMatcher.

- test_rule_structural: pure-graph, no GPU -- proves the rewrite logic (REDUCE over a full-warp axis -> the
  log2(32)=5-step ds_bpermute ladder, no REDUCE left). This is the deterministic capability proof.
- test_rule_numerical_* / test_emits_bpermute: AMD-gated end-to-end through the production renderer (the lane
  must bind to a real lidx via gpudims; this exercises the integration risk).
"""
import os, unittest
import numpy as np

from tinygrad import Tensor, dtypes, Device
from tinygrad.uop.ops import UOp, Ops, AxisType, KernelInfo, graph_rewrite
from extra.qk_warp_reduce_lowering import pm_warp_reduce, WARP

_DEV_OK = Device.DEFAULT == "AMD"
NB = 8

class TestWarpReduceLoweringStructural(unittest.TestCase):
  def _build_reduce(self, alu):
    lane = UOp.range(WARP, 0, AxisType.WARP)
    val = lane.cast(dtypes.float32)            # a float value that depends on the lane
    return val.reduce(lane, arg=alu), lane

  def _check_ladder(self, alu):
    red, _ = self._build_reduce(alu)
    out = graph_rewrite(red, pm_warp_reduce)
    nodes = list(out.toposort())
    self.assertFalse(any(u.op is Ops.REDUCE for u in nodes), "REDUCE not lowered")
    bperm = [u for u in nodes if u.op is Ops.CUSTOMI and "ds_bpermute" in str(u.arg)]
    self.assertEqual(len(bperm), 5, f"expected log2(32)=5 ds_bpermute shuffles, got {len(bperm)}")

  def test_rule_structural_sum(self): self._check_ladder(Ops.ADD)
  def test_rule_structural_max(self): self._check_ladder(Ops.MAX)

  def test_rule_skips_non_warp_axis(self):
    # a plain LOOP/REDUCE axis must NOT be lowered (rule only claims full-warp AxisType.WARP axes)
    lane = UOp.range(WARP, 0, AxisType.REDUCE)
    red = lane.cast(dtypes.float32).reduce(lane, arg=Ops.ADD)
    out = graph_rewrite(red, pm_warp_reduce)
    self.assertTrue(any(u.op is Ops.REDUCE for u in out.toposort()), "non-warp REDUCE wrongly rewritten")

  def test_rule_skips_non_pow2_width(self):
    lane = UOp.range(24, 0, AxisType.WARP)     # non-power-of-2 -> first pass declines (xor-ladder needs pow2)
    red = lane.cast(dtypes.float32).reduce(lane, arg=Ops.ADD)
    out = graph_rewrite(red, pm_warp_reduce)
    self.assertTrue(any(u.op is Ops.REDUCE for u in out.toposort()), "non-pow2 REDUCE wrongly rewritten")

  def test_rule_width16(self):
    # GROUPTOP(16) is what tinygrad's heuristic emits -> 4-step ladder
    lane = UOp.range(16, 0, AxisType.GROUP_REDUCE)
    out = graph_rewrite(lane.cast(dtypes.float32).reduce(lane, arg=Ops.ADD), pm_warp_reduce)
    bperm = [u for u in out.toposort() if u.op is Ops.CUSTOMI and "ds_bpermute" in str(u.arg)]
    self.assertEqual(len(bperm), 4, f"expected log2(16)=4 ds_bpermute shuffles, got {len(bperm)}")


@unittest.skipUnless(_DEV_OK, "ds_bpermute is AMD wave32 (gfx1100) gated")
class TestWarpReduceLoweringPipeline(unittest.TestCase):
  """End-to-end through the REAL codegen pipeline with WARP_REDUCE_LOWERING=1: a generic Tensor matvec whose
  K-reduce the heuristic maps to a single pow2 GROUP (here K=16 -> GROUPTOP(16), no serial remainder) auto-lowers
  to the ds_bpermute ladder -- no hand-written kernel. (A custom_kernel can't host this: it forbids open ranges;
  the lane must be bound to a lidx by gpudims inside the pipeline.) Note the lowering must run in the SAME process
  with the flag set before compile -- the to_program cache key includes WARP_REDUCE_LOWERING."""
  M, K = 256, 16   # K==group width so the reduce is a single GROUP axis (no serial remainder)

  @classmethod
  def setUpClass(cls):
    from tinygrad.helpers import getenv
    cls._old = os.environ.get("WARP_REDUCE_LOWERING")
    os.environ["WARP_REDUCE_LOWERING"] = "1"; getenv.cache_clear()
    rng = np.random.default_rng(0)
    cls.anp = rng.standard_normal((cls.M, cls.K)).astype(np.float32)
    cls.bnp = rng.standard_normal((cls.K,)).astype(np.float32)

  @classmethod
  def tearDownClass(cls):
    from tinygrad.helpers import getenv
    if cls._old is None: os.environ.pop("WARP_REDUCE_LOWERING", None)
    else: os.environ["WARP_REDUCE_LOWERING"] = cls._old
    getenv.cache_clear()

  def _matvec(self, op):
    t = Tensor(self.anp) * Tensor(self.bnp)
    return (t.sum(axis=1) if op == "sum" else t.max(axis=1))

  def test_pipeline_sum_correct(self):
    got = self._matvec("sum").numpy(); ref = (self.anp * self.bnp).sum(1)
    self.assertTrue(np.allclose(got, ref, atol=1e-4), f"auto-lowered warp sum wrong, max_err {np.abs(got-ref).max()}")

  def test_pipeline_max_correct(self):
    got = self._matvec("max").numpy(); ref = (self.anp * self.bnp).max(1)
    self.assertTrue(np.allclose(got, ref), f"auto-lowered warp max wrong, max_err {np.abs(got-ref).max()}")

  def test_pipeline_emits_bpermute_no_lds(self):
    from tinygrad.engine.realize import compile_linear
    out = self._matvec("sum")
    srcs = [next((u.arg for u in c.src[0].toposort() if u.op is Ops.SOURCE), "")
            for c in compile_linear(out.schedule_linear()).src if c.src[0].op is Ops.PROGRAM]
    red_src = next((s for s in srcs if "ds_bpermute" in s), "")
    self.assertNotEqual(red_src, "", "auto-lowered warp reduce did not emit ds_bpermute")
    self.assertNotIn("__attribute__((shared", red_src, "ds_bpermute path should not stage LDS for the reduce")
    self.assertNotIn("s_barrier", red_src, "ds_bpermute path should not emit s_barrier for the reduce")

  def test_pipeline_mixed_reduce_k4096(self):
    # MIXED reduce: a real GEMV K=4096 -> matvec heuristic adds GROUP(8) lane + serial-K/8 + LOCAL block(4).
    # The serial+group split must lower correctly AND the lane-packing must be right while a LOCAL block shares
    # the wave (the agent-flagged correctness subtlety). Garbage here = wrong lidx packing.
    rng = np.random.default_rng(7); M, K = 256, 4096
    anp = rng.standard_normal((M, K)).astype(np.float32); bnp = rng.standard_normal((K,)).astype(np.float32)
    got = (Tensor(anp) * Tensor(bnp)).sum(axis=1).numpy()
    ref = (anp.astype(np.float64) * bnp.astype(np.float64)).sum(1)
    self.assertTrue(np.allclose(got, ref, rtol=2e-3, atol=5e-2), f"mixed-reduce K=4096 wrong, max_err {np.abs(got-ref).max()}")

  def test_pipeline_mixed_reduce_emits_bpermute(self):
    from tinygrad.engine.realize import compile_linear
    rng = np.random.default_rng(7); M, K = 256, 4096
    out = (Tensor(rng.standard_normal((M, K)).astype(np.float32)) * Tensor(rng.standard_normal((K,)).astype(np.float32))).sum(axis=1)
    srcs = [next((u.arg for u in c.src[0].toposort() if u.op is Ops.SOURCE), "")
            for c in compile_linear(out.schedule_linear()).src if c.src[0].op is Ops.PROGRAM]
    self.assertTrue(any("ds_bpermute" in s for s in srcs), "mixed K=4096 reduce did not emit ds_bpermute")

if __name__ == "__main__":
  unittest.main()
