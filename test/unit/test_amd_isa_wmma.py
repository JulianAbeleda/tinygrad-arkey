import itertools, unittest
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import Target
from tinygrad.renderer.isa import IselContext
from tinygrad.renderer.isa.amd import AMDISARenderer, FRAG_BASE, FRAG_TOP, WMMA_ACC_BASE, _vpool, _acc_top
from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.renderer.amd.dsl import Reg


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


def _tc_matmul_ast_k64_rolled():
  # a forced-TC 16x16x64 half matmul with the K axis LEFT ROLLED (no UNROLL) -> ONE Ops.WMMA in a RANGE loop with a
  # reduce accumulator (reduce_to_acc). wmma.src[2] is an 8-lane carrier of LOADs from the accumulator DEFINE_REG.
  a = Tensor.empty(16, 64, dtype="half"); b = Tensor.empty(64, 16, dtype="half")
  lin = (a @ b).schedule_linear()
  ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
  return ast.replace(arg=replace(ast.arg, opts_to_apply=(Opt(OptOps.TC, axis=0, arg=(0, 0, 1)),)))


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


class TestAMDISAWmmaRolledKGate(unittest.TestCase):
  # ROLLED-K DEV=PYTHON structural gate: a K=64 (RANGE loop, NOT unrolled) half matmul must lower to ONE in-place
  # v_wmma in the loop body over a FIXED zero-initialised C fragment -- no per-iteration accumulator movs, no LDS.
  # NO numerical check (that is the parent's 16x16x64 + 64x64x64 bit-exact gate on DEV=AMD).
  def setUp(self):
    self.ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))

  def test_rolled_reaches_isel_and_one_accumulator(self):
    ast = _tc_matmul_ast_k64_rolled()
    fs = full_rewrite_to_sink(ast, self.ren, optimize=True)
    self.assertEqual(len([u for u in fs.toposort() if u.op is Ops.WMMA]), 1, "rolled K must keep exactly one Ops.WMMA")
    self.assertEqual(len([u for u in fs.toposort() if u.op is Ops.RANGE]), 1, "rolled K must keep the reduce RANGE loop")
    fs = graph_rewrite(fs, self.ren.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre isel", bottom_up=True)
    ictx = IselContext(fs)
    fs = graph_rewrite(fs, self.ren.isel_matcher, ctx=ictx, name="isel", bottom_up=True)
    # (a) reached the ROLLED isel_wmma path (C lane is Ops.LOAD, NOT the CONST-seed fail-loud) -> exactly one V_WMMA INS
    wmmas = [u for u in fs.toposort() if u.op is Ops.INS and getattr(u.arg, "name", None) == "V_WMMA"]
    self.assertEqual(len(wmmas), 1, "rolled isel_wmma must emit exactly one V_WMMA INS")
    # (b) ONE cbase accumulator range -> 3 fragment ranges total (shared A,B,C), all inside [FRAG_BASE, FRAG_TOP)
    bases = sorted(getattr(ictx, "_frag", {}).values())
    self.assertEqual(len(bases), 3, f"expected exactly 3 fragment ranges (A,B,C), got {bases}")
    for base in bases:
      self.assertGreaterEqual(base, FRAG_BASE)
      self.assertLess(base + 7, FRAG_TOP)

  def test_rolled_one_inplace_v_wmma_in_loop_zero_movs(self):
    prg = to_program(_tc_matmul_ast_k64_rolled(), self.ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    insts = list(lin_uop.src)
    def is_tuple(u): return isinstance(u.arg, tuple)
    def mn(u): return None if is_tuple(u) else str(u.arg).split("(", 1)[0]
    # split the linear list into pre-loop / loop-body via the top/out label markers
    top_i = next(i for i, u in enumerate(insts) if is_tuple(u) and u.arg[0] == "label" and u.arg[1][0] == "top")
    out_i = next(i for i, u in enumerate(insts) if is_tuple(u) and u.arg[0] == "label" and u.arg[1][0] == "out")
    pre, loop = insts[:top_i], insts[top_i:out_i]
    cbase = 200                                             # first fragment allocated -> the C accumulator
    def is_cbase_init(u):
      return mn(u) == "v_mov_b32_e32" and "LIT" in str(u.arg) and any(f"v[{cbase+i}]" in str(u.arg) for i in range(8))
    # (c) exactly ONE v_wmma in the loop body, with vdst == src2 == the C fragment (in-place accumulate)
    wmma_lines = [str(u.arg) for u in loop if mn(u) == "v_wmma_f32_16x16x16_f16"]
    self.assertEqual(len(wmma_lines), 1, f"expected exactly one v_wmma in the loop body, got {len(wmma_lines)}")
    vdst, _s0, _s1, src2 = [s.strip() for s in wmma_lines[0].split("(", 1)[1].rstrip(")").split(",")]
    self.assertEqual(vdst, src2, f"in-place accumulate requires vdst==src2, got {vdst} vs {src2}")
    self.assertEqual(vdst, f"v[{cbase}:{cbase+7}]", f"C fragment must be v[{cbase}:{cbase+7}], got {vdst}")
    # (d) 8 V_CONST 0.0 inits to the C fragment PRE-loop, and ZERO inits inside the loop (init exactly once)
    self.assertEqual(sum(1 for u in pre if is_cbase_init(u)), 8, "expected 8 pre-loop V_CONST inits to the C fragment")
    self.assertEqual(sum(1 for u in loop if is_cbase_init(u)), 0, "no accumulator init may sink into the loop")
    # KEY INSIGHT check: NO accumulator movs in the loop (v_wmma accumulates in place)
    loop_acc_mov = [u for u in loop if mn(u) == "v_mov_b32_e32" and any(f"v[{cbase+i}]" in str(u.arg) for i in range(8))]
    self.assertEqual(len(loop_acc_mov), 0, f"no accumulator movs allowed in the loop, got {len(loop_acc_mov)}")
    # (e) NO ds_store/ds_load for the accumulator anywhere
    self.assertFalse(any(mn(u) and mn(u).startswith(("ds_store", "ds_load")) for u in insts), "accumulator must not touch LDS")
    # (f) assembled to a non-empty binary without raising
    self.assertTrue(any(u.op is Ops.BINARY and len(u.arg) > 0 for u in prg.src), "assemble_linear produced no binary")


def _tc_matmul_ast_multitile(m_up:int):
  # a forced-TC 64x64x64 half matmul with the M/N output UPCAST into a WM x WN grid of 16x16 subtiles per warp. Each
  # UPCAST(axis=0, arg=4) quadruples the per-warp output tile -> one UPCAST = 4 subtiles (32 acc VGPRs), two = 16 subtiles
  # (WM=WN=4, 128 acc VGPRs). ROLLED K (no UNROLL) -> ONE reduce DEFINE_REG of width WM*WN*8 split by no_vectorized_wmma
  # into WM*WN distinct Ops.WMMA, each src[2] an 8-lane accumulator slice (idx.arg == subtile*8).
  a = Tensor.empty(64, 64, dtype="half"); b = Tensor.empty(64, 64, dtype="half")
  lin = (a @ b).schedule_linear()
  ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
  opts = (Opt(OptOps.TC, axis=0, arg=(0, 0, 1)),) + (Opt(OptOps.UPCAST, axis=0, arg=4),) * m_up
  return ast.replace(arg=replace(ast.arg, opts_to_apply=opts))


class TestAMDISAWmmaMultiOutputTileGate(unittest.TestCase):
  # B0.M DEV=PYTHON structural gate for the multi-output-tile register model. A hand_coded M/N>16 upcasts the output into
  # a WM x WN grid of 16x16 subtiles -> WM*WN accumulators. The bug: keying the C base on id(dreg) ALONE aliased all
  # subtiles onto ONE 8-VGPR run (and isel_index walked cbase+idx.arg off the run). The fix pins each subtile its OWN
  # fixed, contiguous, 8-aligned, LOW 8-VGPR run (loop-carried, read+written in place by v_wmma). NO numerical check
  # (the parent's DEV=AMD gate). See amd.py _n_c_runs / _acc_base / _c_low / _vpool.
  def setUp(self):
    self.ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))

  def _isel(self, ast):
    fs = full_rewrite_to_sink(ast, self.ren, optimize=True)
    n_wmma = len([u for u in fs.toposort() if u.op is Ops.WMMA])
    fs = graph_rewrite(fs, self.ren.pre_isel_matcher, ctx=itertools.count(-1, -1), name="pre isel", bottom_up=True)
    ictx = IselContext(fs)
    fs = graph_rewrite(fs, self.ren.isel_matcher, ctx=ictx, name="isel", bottom_up=True)   # (d) no isel NotImplementedError
    return fs, ictx, n_wmma

  def test_16_subtile_register_model(self):
    # 64x64x64 WM=WN=4 -> 16 output subtiles, 128 accumulator VGPRs (the bug's 4x4=128 case).
    fs, ictx, n_wmma = self._isel(_tc_matmul_ast_multitile(2))
    self.assertEqual(n_wmma, 16, "64x64 UPCASTx2 must build a 16-subtile WM*WN grid")
    # (a) one V_WMMA INS per subtile
    vwmma = [u for u in fs.toposort() if u.op is Ops.INS and getattr(u.arg, "name", None) == "V_WMMA"]
    self.assertEqual(len(vwmma), 16, f"expected 16 V_WMMA INS (one per subtile), got {len(vwmma)}")
    # (b) WM*WN distinct, non-overlapping, 8-aligned, LOW accumulator ranges, none exceeding the 256-VGPR file
    bases = sorted(getattr(ictx, "_accfrag", {}).values())
    self.assertEqual(len(bases), 16, f"expected 16 distinct LOW accumulator ranges, got {bases}")
    for b in bases:
      self.assertEqual(b % 8, 0, f"accumulator base {b} not 8-aligned")
      self.assertGreaterEqual(b, WMMA_ACC_BASE); self.assertLess(b, FRAG_BASE)   # LOW (below the A/B high window)
      self.assertLess(b + 7, 256)                                                # base+7 inside the file
    runs = [set(range(b, b + 8)) for b in bases]
    for i in range(len(runs)):
      for j in range(i + 1, len(runs)):
        self.assertEqual(len(runs[i] & runs[j]), 0, f"accumulator ranges {bases} overlap")
    self.assertEqual(bases, list(range(WMMA_ACC_BASE, WMMA_ACC_BASE + 16 * 8, 8)), "128 contiguous 8-aligned acc VGPRs")
    # B0.M per-row/col RESIDENCY: WM DISTINCT A-row + WN DISTINCT B-col fragments (NOT one reused pair), each packed ONCE
    # and shared across its row/col. WM=WN=4 -> 8 resident 8-VGPR runs in the LOW window [_acc_top, FRAG_BASE), none in
    # the (now free) legacy high window. Distinct, non-overlapping, 8-aligned.
    self.assertEqual(getattr(ictx, "_frag", {}), {}, "multi-tile must NOT use the legacy high A/B window")
    ab = sorted(getattr(ictx, "_abfrag", {}).values())
    self.assertEqual(len(ab), 8, f"expected WM+WN=8 resident A/B fragments (4 A-rows + 4 B-cols), got {ab}")
    self.assertEqual(ab, list(range(_acc_top(ictx), _acc_top(ictx) + 8 * 8, 8)), "8 contiguous 8-aligned resident A/B runs above the accumulators")
    for b in ab:
      self.assertEqual(b % 8, 0); self.assertGreaterEqual(b, _acc_top(ictx)); self.assertLess(b + 7, FRAG_BASE)   # LOW, below the freed high window
    ab_idx = set().union(*[set(range(b, b + 8)) for b in ab])
    acc_idx = set().union(*runs)
    self.assertTrue(ab_idx.isdisjoint(acc_idx), "resident A/B window and LOW accumulator region must not overlap")
    # exactly WM+WN=8 distinct packed fragment sets (one per A-row + one per B-col) -> V_PACK count = (WM+WN)*8, not WM*WN*16
    packs = [u for u in fs.toposort() if u.op is Ops.INS and getattr(u.arg, "name", None) == "V_PACK"]
    self.assertEqual(len(packs), (4 + 4) * 8, f"expected 64 V_PACK (each fragment packed ONCE), got {len(packs)}")
    self.assertEqual(len(set(u.tag for u in packs)), 64, "each pack pinned to a distinct resident VGPR")
    # (c) _vpool excludes EXACTLY the LOW accumulator region AND the resident A/B window (collision -> v_wmma clobbers a live virtual)
    pool = {r.index for r in _vpool(ictx)}
    self.assertEqual(len(pool & acc_idx), 0, "_vpool must exclude the LOW accumulator VGPRs")
    self.assertEqual(len(pool & ab_idx), 0, "_vpool must exclude the resident A/B fragment window")
    self.assertEqual(_acc_top(ictx), WMMA_ACC_BASE + 16 * 8, "reserved LOW region top = base + 128")
    self.assertEqual(min(pool), _acc_top(ictx) + 8 * 8, "virtuals start immediately above the accumulator + resident A/B regions")
    # (e) every physical VGPR the MODEL pins is inside the 256 file (accumulators [8,135], A/B [136,199], pool <=255)
    self.assertLess(max(acc_idx | ab_idx | pool), 256)
    # budget: WM*WN*8 accumulators (128) + (WM+WN)*8 resident A/B (64) = 192 physical VGPRs pinned, < 256
    self.assertEqual(len(acc_idx | ab_idx), 128 + 64)

  def test_4_subtile_end_to_end_assembles(self):
    # 64x64x64 WM=4 (one UPCAST) -> 4 subtiles (32 acc VGPRs): fits the file and lowers all the way to a binary with NO
    # spill (stack_pointer stays unimplemented). Proves the multi-tile model produces a working, non-spilling program.
    fs, ictx, n_wmma = self._isel(_tc_matmul_ast_multitile(1))
    self.assertEqual(n_wmma, 4)
    bases = sorted(getattr(ictx, "_accfrag", {}).values())
    self.assertEqual(bases, [8, 16, 24, 32], f"4 distinct LOW 8-aligned acc ranges, got {bases}")
    prg = to_program(_tc_matmul_ast_multitile(1), self.ren)   # (d) no NotImplementedError (spill) reaching regalloc
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    insts = lin_uop.src
    # (e) total distinct VGPR indices used <= 256
    vidx = set()
    for u in insts:
      if isinstance(u.arg, tuple): continue
      for name, field in u.arg._fields:
        v = getattr(u.arg, name)
        if isinstance(v, Reg):
          for o in range(v.offset, v.offset + v.sz):
            if o >= 256: vidx.add(o - 256)
    self.assertLessEqual(len(vidx), 256); self.assertLess(max(vidx), 256, "no VGPR index escapes the 256 file")
    # (f) assembles clean: 4 in-place v_wmma + a non-empty binary
    mns = [str(u.arg).split("(", 1)[0] for u in insts if not isinstance(u.arg, tuple)]
    self.assertEqual(sum(1 for m in mns if m == "v_wmma_f32_16x16x16_f16"), 4, "one v_wmma per subtile in the rendered list")
    self.assertTrue(any(u.op is Ops.BINARY and len(u.arg) > 0 for u in prg.src), "assemble_linear produced no binary")

  def test_16_subtile_end_to_end_no_spill(self):
    # B0.M per-row/col residency killer check: 64x64x64 WM=WN=4 (16 subtiles, 128 acc + 64 resident A/B VGPRs) lowers all
    # the way to a binary with NO spill. BEFORE residency this SPILLED ("Inc 0: no spills") because all 16 subtiles
    # re-packed A/B into ONE reused 16-VGPR pair (16*16 = 256 packs contending). AFTER: each A-row / B-col is packed ONCE
    # (WM+WN = 8 fragment sets, 64 packs pinned to 64 distinct regs) -> the constraint is satisfiable -> no spill.
    prg = to_program(_tc_matmul_ast_multitile(2), self.ren)   # must NOT raise NotImplementedError("no spills")
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    insts = lin_uop.src
    mns = [str(u.arg).split("(", 1)[0] for u in insts if not isinstance(u.arg, tuple)]
    # one v_wmma per subtile, and each fragment packed ONCE -> (WM+WN)*8 = 64 v_pack (NOT WM*WN*16 = 256)
    self.assertEqual(sum(1 for m in mns if m == "v_wmma_f32_16x16x16_f16"), 16, "16 in-place v_wmma (one per subtile)")
    self.assertEqual(sum(1 for m in mns if m == "v_pack_b32_f16"), (4 + 4) * 8, "64 v_pack: each A-row/B-col packed once")
    # every VGPR index stays inside the 256 file
    vidx = set()
    for u in insts:
      if isinstance(u.arg, tuple): continue
      for name, _field in u.arg._fields:
        v = getattr(u.arg, name)
        if isinstance(v, Reg):
          for o in range(v.offset, v.offset + v.sz):
            if o >= 256: vidx.add(o - 256)
    self.assertLess(max(vidx), 256, "no VGPR index escapes the 256 file")
    self.assertTrue(any(u.op is Ops.BINARY and len(u.arg) > 0 for u in prg.src), "assemble_linear produced no binary")


if __name__ == "__main__":
  unittest.main()
