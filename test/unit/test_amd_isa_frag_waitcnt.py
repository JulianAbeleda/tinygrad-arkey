import unittest
from types import SimpleNamespace
from tinygrad.renderer.isa.amd import _frag_base, FRAG_BASE, FRAG_TOP, AMDISARenderer
from tinygrad.renderer.isa.amd import AMDOps, isel_typed_wait, lower_inst
from tinygrad.codegen.opt.compiler_policies import WaitCount
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp


def _fake_ctx():
  # minimal stand-in for IselContext: _frag_base only touches getattr/setattr on the ctx object.
  return SimpleNamespace()


class TestFragAllocator(unittest.TestCase):
  def test_non_overlapping_aligned_ranges(self):
    ctx = _fake_ctx()
    # three fragments of 8, aligned to 4: bases must be 4-aligned, contiguous, non-overlapping.
    a = _frag_base(ctx, "A", 8, align=4)
    b = _frag_base(ctx, "B", 8, align=4)
    c = _frag_base(ctx, "C", 8, align=4)
    self.assertEqual(a, FRAG_BASE)                 # 200, already 4-aligned
    self.assertEqual(b, a + 8)                     # 208
    self.assertEqual(c, b + 8)                     # 216
    for base in (a, b, c):
      self.assertEqual(base % 4, 0)                # aligned
      self.assertLessEqual(base + 7, 237)          # cap base+7 at most 237 (v>=238 trap)
    # ranges are disjoint
    ranges = [set(range(base, base + 8)) for base in (a, b, c)]
    self.assertEqual(len(ranges[0] & ranges[1]), 0)
    self.assertEqual(len(ranges[1] & ranges[2]), 0)

  def test_stable_per_key(self):
    ctx = _fake_ctx()
    first = _frag_base(ctx, "A", 8)
    _frag_base(ctx, "B", 8)
    self.assertEqual(_frag_base(ctx, "A", 8), first)   # same key -> same base

  def test_alignment_rounds_up(self):
    ctx = _fake_ctx()
    _frag_base(ctx, "A", 3)                         # top -> 203
    b = _frag_base(ctx, "B", 8, align=8)            # round 203 up to 208
    self.assertEqual(b, 208)
    self.assertEqual(b % 8, 0)

  def test_returns_none_past_frag_top(self):
    ctx = _fake_ctx()
    # region is [200, 238) == 38 regs. Fill it, then the next request must return None.
    self.assertEqual(_frag_base(ctx, "big", 38), FRAG_BASE)   # 200..237 exactly
    self.assertIsNone(_frag_base(ctx, "overflow", 1))         # nothing left
    # a single request larger than the region is refused outright
    ctx2 = _fake_ctx()
    self.assertIsNone(_frag_base(ctx2, "toobig", FRAG_TOP - FRAG_BASE + 1))

  def test_frag_top_bound(self):
    self.assertEqual((FRAG_BASE, FRAG_TOP), (200, 238))   # base+7 for a base<=230 stays <= 237


class TestWaitcntSimm16(unittest.TestCase):
  def test_vm3_packs_correct_bits(self):
    # vm=bits[15:10], lgkm=bits[9:4], exp=bits[2:0]; unspecified fields default to maxed (don't-wait).
    got = AMDISARenderer._waitcnt_simm16(vm=3)
    expected = (3 << 10) | (63 << 4) | 7
    self.assertEqual(got, expected)
    # decode the field back out
    self.assertEqual((got >> 10) & 0x3F, 3)
    self.assertEqual((got >> 4) & 0x3F, 63)
    self.assertEqual(got & 0x7, 7)

  def test_defaults_are_maxed(self):
    self.assertEqual(AMDISARenderer._waitcnt_simm16(), (63 << 10) | (63 << 4) | 7)

  def test_full_drain_is_zero(self):
    # the value the rerouted _insert_waitcnt sites use must equal the old literal simm16=0.
    self.assertEqual(AMDISARenderer._waitcnt_simm16(0, 0, 0), 0)

  def test_typed_wait_reaches_native_s_waitcnt(self):
    wait = UOp(Ops.WAIT, dtypes.void, (), WaitCount(vmcnt=8))
    lowered = isel_typed_wait(wait)
    self.assertIs(lowered.op, Ops.INS)
    self.assertIs(lowered.arg, AMDOps.TYPED_WAIT)
    inst, _ = lower_inst(lowered)
    self.assertEqual(inst.arg.simm16, WaitCount(vmcnt=8).simm16)


if __name__ == "__main__":
  unittest.main()
