import itertools, unittest
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import Target
from tinygrad.renderer.isa import IselContext
from tinygrad.renderer.isa.amd import AMDISARenderer, FRAG_BASE, FRAG_TOP
from tinygrad.codegen import full_rewrite_to_sink, to_program


def _tc_matmul_ast():
  # a forced-TC 16x16x16 half matmul -> the AST sink, with the amd_rdna3 TC opt planned.
  a = Tensor.empty(16, 16, dtype="half"); b = Tensor.empty(16, 16, dtype="half")
  lin = (a @ b).schedule_linear()
  ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
  return ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, axis=0, arg=(0, 0, 1)),)))


def _tc_matmul_ast_k64():
  # a forced-TC 16x16x64 half matmul with the K axis UNROLLed -> a 4-tile WMMA K-reduction chain (K=64 -> 4 K-tiles).
  a = Tensor.empty(16, 64, dtype="half"); b = Tensor.empty(64, 16, dtype="half")
  lin = (a @ b).schedule_linear()
  ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
  opts = (Opt(OptOps.TC, axis=0, arg=(0, 0, 1)), Opt(OptOps.UNROLL, axis=0, arg=0))
  return ast.replace(arg=replace(ast.arg, opts_to_apply=opts))


class TestAMDISAWmmaStructuralGate(unittest.TestCase):
  # DEV=PYTHON structural gate for B0.L7. NO numerical check (needs DEV=AMD -> parent's 16x16x16 bit-exact gate).
  def setUp(self):
    self.ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))

  def test_reaches_isel_wmma_and_allocates_three_ranges(self):
    ast = _tc_matmul_ast()
    fs = full_rewrite_to_sink(ast, self.ren, optimize=True)
    self.assertEqual(len([u for u in fs.toposort() if u.op is Ops.WMMA]), 1, "TC opt must build exactly one Ops.WMMA")
    fs = graph_rewrite(fs, self.ren.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre isel", bottom_up=True)
    ictx = IselContext(fs)
    fs = graph_rewrite(fs, self.ren.isel_matcher, ctx=ictx, name="isel", bottom_up=True)
    # (a) reached isel_wmma -> a V_WMMA INS exists
    wmmas = [u for u in fs.toposort() if u.op is Ops.INS and getattr(u.arg, "name", None) == "V_WMMA"]
    self.assertEqual(len(wmmas), 1, "isel_wmma must emit exactly one V_WMMA INS")
    # (b) exactly 3 non-overlapping 8-VGPR ranges inside [FRAG_BASE, FRAG_TOP)
    bases = sorted(getattr(ictx, "_frag", {}).values())
    self.assertEqual(len(bases), 3, f"expected 3 fragment ranges, got {bases}")
    for base in bases:
      self.assertGreaterEqual(base, FRAG_BASE)
      self.assertLess(base + 7, FRAG_TOP)                 # base+7 <= 237 (v>=238 trap)
    ranges = [set(range(b, b + 8)) for b in bases]
    for i in range(len(ranges)):
      for j in range(i + 1, len(ranges)):
        self.assertEqual(len(ranges[i] & ranges[j]), 0, f"fragment ranges {bases} overlap")

  def test_renders_v_wmma_without_raising(self):
    # (c) full to_program: py_compile-clean module + render to an instruction list with a v_wmma, no exception.
    prg = to_program(_tc_matmul_ast(), self.ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    insts = lin_uop.src
    self.assertTrue(all(u.op is Ops.INS for u in insts), f"non-INS leaked into linear list: {[u.op for u in insts if u.op is not Ops.INS]}")
    mns = [str(u.arg).split("(", 1)[0] for u in insts if not isinstance(u.arg, tuple)]
    self.assertIn("v_wmma_f32_16x16x16_f16", mns, "rendered instruction list must contain a v_wmma")
    self.assertEqual(sum(1 for m in mns if m == "v_pack_b32_f16"), 16, "8 A + 8 B fragment packs expected")
    # and it assembled to a non-empty binary
    self.assertTrue(any(u.op is Ops.BINARY and len(u.arg) > 0 for u in prg.src), "assemble_linear produced no binary")


class TestAMDISAWmmaKReduceGate(unittest.TestCase):
  # B0.K DEV=PYTHON structural gate: a K=64 (4 K-tiles) half matmul must lower to 4 accumulating v_wmma that share ONE
  # in-place accumulator range. NO numerical check (that is the parent's 16x16x64 bit-exact gate on DEV=AMD).
  def setUp(self):
    self.ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))

  def test_k64_chain_reaches_isel_and_shares_one_accumulator(self):
    ast = _tc_matmul_ast_k64()
    fs = full_rewrite_to_sink(ast, self.ren, optimize=True)
    self.assertEqual(len([u for u in fs.toposort() if u.op is Ops.WMMA]), 4, "K=64 UNROLL must build a 4-node WMMA chain")
    fs = graph_rewrite(fs, self.ren.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre isel", bottom_up=True)
    ictx = IselContext(fs)
    fs = graph_rewrite(fs, self.ren.isel_matcher, ctx=ictx, name="isel", bottom_up=True)
    # (a) every K-tile reached isel_wmma -> 4 V_WMMA INS
    wmmas = [u for u in fs.toposort() if u.op is Ops.INS and getattr(u.arg, "name", None) == "V_WMMA"]
    self.assertEqual(len(wmmas), 4, "isel must emit exactly 4 V_WMMA INS (one per K-tile)")
    # (b) ONE accumulator range reused across the 4 tiles (not 4 separate C ranges): 3 fragment ranges total (A,B,C)
    bases = sorted(getattr(ictx, "_frag", {}).values())
    self.assertEqual(len(bases), 3, f"expected exactly 3 fragment ranges (shared A,B,C), got {bases}")
    for base in bases:
      self.assertGreaterEqual(base, FRAG_BASE)
      self.assertLess(base + 7, FRAG_TOP)

  def test_k64_renders_four_v_wmma_sharing_vdst_src2(self):
    prg = to_program(_tc_matmul_ast_k64(), self.ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    lines = [str(u.arg) for u in lin_uop.src if not isinstance(u.arg, tuple)]
    wmma_lines = [l for l in lines if l.startswith("v_wmma_f32_16x16x16_f16")]
    # (c) 4 v_wmma, ALL with vdst == src2 (in-place) and ALL identical operand ranges (shared accumulator + reused A/B)
    self.assertEqual(len(wmma_lines), 4, f"expected 4 rendered v_wmma, got {len(wmma_lines)}")
    self.assertEqual(len(set(wmma_lines)), 1, f"all 4 v_wmma must share the same vdst/src0/src1/src2 ranges: {wmma_lines}")
    inner = wmma_lines[0].split("(", 1)[1]
    vdst, src0, src1, src2 = [s.strip() for s in inner.rstrip(")").split(",")]
    self.assertEqual(vdst, src2, f"in-place accumulate requires vdst==src2, got {vdst} vs {src2}")
    # (d) assembled to a non-empty binary without raising
    self.assertTrue(any(u.op is Ops.BINARY and len(u.arg) > 0 for u in prg.src), "assemble_linear produced no binary")


if __name__ == "__main__":
  unittest.main()
