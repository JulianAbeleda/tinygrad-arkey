import itertools, os, unittest
from dataclasses import replace
from tinygrad import Tensor
from tinygrad.uop.ops import Ops, UOp, graph_rewrite
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import Target, getenv
from tinygrad.renderer.isa import IselContext, Register
from tinygrad.renderer.isa.amd import (
  AMDISARenderer, FRAG_BASE, FRAG_TOP, WMMA_ACC_BASE, _vpool, _acc_top, AMDOps, _wmma_chain_prev, _chain_epilogue_stores, decompose_lds_index, isel_index, isel_store, lower_inst)
from tinygrad.codegen import full_rewrite_to_sink, to_program, to_program_cache
from tinygrad.codegen.late.devectorizer import load_store_folding
from tinygrad.renderer.amd.dsl import Reg

class TestAMDISAWmmaCarrierNormalization(unittest.TestCase):
  def test_scalarized_wmma_lane_stack_recovers_previous_vector(self):
    base = UOp(Ops.WMMA, dtypes.float.vec(8), src=(), arg=())
    carrier = UOp(Ops.STACK, dtypes.float.vec(8), tuple(base.gep((i,)) for i in range(8)))
    self.assertIs(_wmma_chain_prev(carrier), base)
    self.assertIsNone(_wmma_chain_prev(carrier.replace(src=carrier.src[:-1] + (base.gep((0,)),))))


class TestAMDISAEpilogueStoreChaining(unittest.TestCase):
  def test_linear_epilogue_serializes_every_store_once(self):
    # The first store is the chain head; N stores therefore have N-1 predecessor edges.
    stores = []
    for i in range(68):
      off = UOp.const(dtypes.int32, i)
      stores.append(UOp(Ops.INS, dtypes.void, src=(off, UOp.const(dtypes.int32, 0),
        UOp.const(dtypes.float32, 0), UOp.const(dtypes.int32, 4)), arg=AMDOps.GLOBAL_STORE,
        tag=("store_owner", i)))
    sink = UOp(Ops.SINK, dtypes.void, src=tuple(stores))
    ctx = IselContext(sink); ctx._ncruns = 2
    out = _chain_epilogue_stores(ctx, sink)
    out_stores = [u for u in out.toposort() if u.op is Ops.INS and u.arg is AMDOps.GLOBAL_STORE]
    self.assertEqual(len(out_stores), 68)
    edges = []
    for st in out_stores:
      predecessors = [u for u in st.src[0].src if u.op is Ops.INS and u.arg is AMDOps.GLOBAL_STORE]
      expected = 0 if st.tag[1] == 0 else st.tag[1] - 1
      self.assertEqual(len(predecessors), 0 if st.tag[1] == 0 else 1)
      if predecessors: self.assertEqual(predecessors[0].tag[1], expected)
      edges.extend(predecessors)
    self.assertEqual(len(edges), 67)
    self.assertEqual(len({id(u) for u in edges}), 67)
    self.assertEqual({u.tag[1] for u in out_stores}, set(range(68)))
    pred = {u.tag[1]: [s for s in u.src[0].src if s.op is Ops.INS and s.arg is AMDOps.GLOBAL_STORE] for u in out_stores}
    self.assertEqual(sum(not ps for ps in pred.values()), 1)
    self.assertEqual({tag for tag, ps in pred.items() if not ps}, {0})
    self.assertTrue(all(len(pred[tag]) == 1 for tag in range(1, 68)))
    self.assertEqual([pred[tag][0].tag[1] for tag in range(1, 68)], list(range(67)))


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
    self.assertEqual(sum(1 for m in mns if m == "global_load_b128"), 2, "contiguous A fragment uses two b128 loads")
    self.assertEqual(sum(1 for m in mns if m == "v_pack_b32_f16"), 8, "strided B fragment still packs")
    # and it assembled to a non-empty binary
    self.assertTrue(any(u.op is Ops.BINARY and len(u.arg) > 0 for u in prg.src), "assemble_linear produced no binary")

  def test_nested_q4k_contraction_renders_signed_i8_wmma(self):
    from extra.qk.layout import Q4K_WORDS_PER_BLOCK
    from extra.qk.prefill_int8_wmma_spec import (
      describe_q4k_int8_wmma_tiled_prefill, emit_q4k_int8_wmma_tiled_scheduler_tensor)
    n, k, m = 16, 256, 16
    spec = describe_q4k_int8_wmma_tiled_prefill(n, k, m, role="isa_nested",
                                                m_tile=16, n_tile=16, group_tile=8)
    words = Tensor.empty(n * (k // 256) * Q4K_WORDS_PER_BLOCK, dtype=dtypes.uint)
    xq = Tensor.empty(m, k, dtype=dtypes.char)
    xscales = Tensor.empty(m, k // 32, dtype=dtypes.float32)
    linear = emit_q4k_int8_wmma_tiled_scheduler_tensor(words, xq, xscales, spec).schedule_linear()
    asts = [u for u in linear.toposort() if u.op is Ops.SINK]
    # Packed Q4 decode is fused into the contraction; only bounded metadata/Q8 prerequisites remain.
    self.assertEqual(len(asts), 4)
    self.assertEqual(sum(u.op is Ops.REDUCE for u in asts[-1].toposort()), 2)
    prg = to_program(asts[-1], self.ren)
    self.assertEqual(prg.src[0].arg.name,
      "prefill_q4k_q8_1_wmma_tiled_generated_gemm_isa_nested_16_256_16_16x16x8")
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    lines = [str(u.arg) for u in lin_uop.src if not isinstance(u.arg, tuple)]
    wmmas = [line for line in lines if line.startswith("v_wmma_i32_16x16x16_iu8")]
    self.assertEqual(len(wmmas), 1)
    self.assertTrue(wmmas[0].endswith(", 3)"), f"signed A/B flags missing from iu8 WMMA: {wmmas[0]}")


class TestAMDISAIntegerCastGate(unittest.TestCase):
  def test_variable_integer_shifts_use_vgpr_shift_operands(self):
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    value, shift = Tensor.empty(8, dtype=dtypes.uint), Tensor.empty(8, dtype=dtypes.uint)
    lin = ((value >> (shift & 7)) + (value << (shift & 3))).contiguous().schedule_linear()
    ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
    prg = to_program(ast, ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    lines = [str(u.arg) for u in lin_uop.src if not isinstance(u.arg, tuple)]
    right = [line for line in lines if line.startswith("v_lshrrev_b32_e32")]
    left = [line for line in lines if line.startswith("v_lshlrev_b32_e32")]
    self.assertTrue(right and left)
    self.assertTrue(all(", v" in line for line in right + left), (right, left))

  def test_char_to_int_sign_extends(self):
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    a = Tensor.empty(8, dtype=dtypes.char)
    lin = a.cast(dtypes.int).contiguous().schedule_linear()
    ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
    prg = to_program(ast, ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    mns = [str(u.arg).split("(", 1)[0] for u in lin_uop.src if not isinstance(u.arg, tuple)]
    self.assertIn("v_and_b32_e32", mns)
    self.assertIn("v_xor_b32_e32", mns)
    self.assertIn("v_add_nc_u32_e32", mns)

  def test_uint_to_ushort_masks_to_destination_width(self):
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    a = Tensor.empty(8, dtype=dtypes.uint)
    lin = a.cast(dtypes.ushort).contiguous().schedule_linear()
    ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
    prg = to_program(ast, ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    mns = [str(u.arg).split("(", 1)[0] for u in lin_uop.src if not isinstance(u.arg, tuple)]
    self.assertIn("v_and_b32_e32", mns)
    self.assertIn("global_store_b16", mns)

  def test_ushort_to_int_zero_extends(self):
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    a = Tensor.empty(8, dtype=dtypes.ushort)
    lin = a.cast(dtypes.int).contiguous().schedule_linear()
    ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
    prg = to_program(ast, ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    mns = [str(u.arg).split("(", 1)[0] for u in lin_uop.src if not isinstance(u.arg, tuple)]
    self.assertEqual(mns.count("global_load_b64"), 1)
    self.assertEqual(mns.count("v_bfe_u32"), 4)
    self.assertEqual(mns.count("v_and_b32_e32"), 4)
    self.assertEqual(mns.count("global_store_b32"), 4)

  def test_int_to_ushort_masks_to_destination_width(self):
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    a = Tensor.empty(8, dtype=dtypes.int)
    lin = a.cast(dtypes.ushort).contiguous().schedule_linear()
    ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
    prg = to_program(ast, ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    mns = [str(u.arg).split("(", 1)[0] for u in lin_uop.src if not isinstance(u.arg, tuple)]
    self.assertIn("v_and_b32_e32", mns)
    self.assertIn("global_store_b16", mns)

  def test_uchar_to_ushort_uses_byte_load_half_store(self):
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    a = Tensor.empty(8, dtype=dtypes.uchar)
    lin = a.cast(dtypes.ushort).contiguous().schedule_linear()
    ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
    prg = to_program(ast, ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    mns = [str(u.arg).split("(", 1)[0] for u in lin_uop.src if not isinstance(u.arg, tuple)]
    self.assertIn("global_load_u8", mns)
    self.assertIn("global_store_b16", mns)
    self.assertNotIn("global_load_b32", mns)

  def test_uchar_to_float_masks_then_converts(self):
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    a = Tensor.empty(8, dtype=dtypes.uchar)
    lin = a.cast(dtypes.float32).contiguous().schedule_linear()
    ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
    prg = to_program(ast, ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    mns = [str(u.arg).split("(", 1)[0] for u in lin_uop.src if not isinstance(u.arg, tuple)]
    self.assertIn("global_load_u8", mns)
    self.assertIn("v_and_b32_e32", mns)
    self.assertIn("v_cvt_f32_u32_e32", mns)
    self.assertIn("global_store_b32", mns)

  def test_char_to_float_sign_extends_then_converts(self):
    ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
    a = Tensor.empty(8, dtype=dtypes.char)
    lin = a.cast(dtypes.float32).contiguous().schedule_linear()
    ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
    prg = to_program(ast, ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    mns = [str(u.arg).split("(", 1)[0] for u in lin_uop.src if not isinstance(u.arg, tuple)]
    self.assertIn("global_load_u8", mns)
    self.assertIn("v_and_b32_e32", mns)
    self.assertIn("v_xor_b32_e32", mns)
    self.assertIn("v_add_nc_u32_e32", mns)
    self.assertIn("v_cvt_f32_i32_e32", mns)
    self.assertIn("global_store_b32", mns)


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

def _tc_matmul_ast_multitile_transposed_b(m_up:int):
  # Route-shaped fp16 prefill GEMM: A[M,K] @ B[N,K].T. Unlike the plain unit matmul B[K,N], the B fragment is contiguous
  # over K, so b128 can fold both A and B fragments.
  a = Tensor.empty(64, 64, dtype="half"); b = Tensor.empty(64, 64, dtype="half")
  lin = (a @ b.transpose()).schedule_linear()
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
    # A-row fragments are contiguous and default to b128; B-col fragments are strided and still pack once per B-col.
    packs = [u for u in fs.toposort() if u.op is Ops.INS and getattr(u.arg, "name", None) == "V_PACK"]
    b128 = [u for u in fs.toposort() if u.op is Ops.INS and getattr(u.arg, "name", None) == "GLOBAL_LOAD_B128"]
    self.assertEqual(len(b128), 4 * 2, f"expected 8 b128 loads for 4 contiguous A fragments, got {len(b128)}")
    self.assertEqual(len(packs), 4 * 8, f"expected 32 V_PACK for 4 strided B fragments, got {len(packs)}")
    self.assertEqual(len(set(u.tag for u in packs)), 32, "each pack pinned to a distinct resident VGPR")
    # (c) _vpool excludes the LOW accumulator region and resident A/B window, while reclaiming the v1..v7 padding
    # as scalar scratch. The low scratch keeps post-loop epilogues away from high WMMA/load scratch like v201/v202.
    pool = {r.index for r in _vpool(ictx)}
    self.assertEqual(len(pool & acc_idx), 0, "_vpool must exclude the LOW accumulator VGPRs")
    self.assertEqual(len(pool & ab_idx), 0, "_vpool must exclude the resident A/B fragment window")
    self.assertEqual(_acc_top(ictx), WMMA_ACC_BASE + 16 * 8, "reserved LOW region top = base + 128")
    self.assertEqual(set(range(1, WMMA_ACC_BASE)), pool & set(range(1, WMMA_ACC_BASE)), "v1..v7 are available scratch")
    self.assertEqual(min(p for p in pool if p >= WMMA_ACC_BASE), _acc_top(ictx) + 8 * 8,
                     "high virtuals start immediately above the accumulator + resident A/B regions")
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
    # one v_wmma per subtile; contiguous A rows use b128, strided B cols pack once.
    self.assertEqual(sum(1 for m in mns if m == "v_wmma_f32_16x16x16_f16"), 16, "16 in-place v_wmma (one per subtile)")
    self.assertEqual(sum(1 for m in mns if m == "global_load_b128"), 4 * 2, "8 b128: each contiguous A-row loaded once")
    self.assertEqual(sum(1 for m in mns if m == "v_pack_b32_f16"), 4 * 8, "32 v_pack: each strided B-col packed once")
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

  def test_16_subtile_b128_fragment_load_default(self):
    # L3 hand-trace parity: when an operand's 16 half lanes are two contiguous 8-half spans, the default path may load
    # the packed fragment directly with two b128 loads instead of scalar half loads + v_pack. In this AST the A-row
    # fragments are contiguous (4 rows -> 8 b128 loads), while B is column-strided and correctly remains packed.
    prg = to_program(_tc_matmul_ast_multitile(2), self.ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    mns = [str(u.arg).split("(", 1)[0] for u in lin_uop.src if not isinstance(u.arg, tuple)]
    self.assertEqual(sum(1 for m in mns if m == "v_wmma_f32_16x16x16_f16"), 16)
    self.assertEqual(sum(1 for m in mns if m == "global_load_b128"), 8, "4 contiguous A fragments -> 2 b128 loads each")
    self.assertEqual(sum(1 for m in mns if m == "v_pack_b32_f16"), 4 * 8, "strided B fragments still require packing")
    self.assertTrue(any(u.op is Ops.BINARY and len(u.arg) > 0 for u in prg.src), "assemble_linear produced no binary")

  def test_16_subtile_transposed_b_full_b128_fragment_loads(self):
    prg = to_program(_tc_matmul_ast_multitile_transposed_b(2), self.ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    mns = [str(u.arg).split("(", 1)[0] for u in lin_uop.src if not isinstance(u.arg, tuple)]
    self.assertEqual(sum(1 for m in mns if m == "v_wmma_f32_16x16x16_f16"), 16)
    self.assertEqual(sum(1 for m in mns if m == "global_load_b128"), (4 + 4) * 2,
                     "route-shaped A and transposed-B fragments are both contiguous")
    self.assertEqual(sum(1 for m in mns if m == "v_pack_b32_f16"), 0)
    self.assertEqual(sum(1 for m in mns if m == "global_load_u16"), 0)

  def test_targeted_waitcnt_coalesces_scalar_pack_path(self):
    prg = to_program(_tc_matmul_ast_multitile(2), self.ren)
    lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
    insts = self.ren._resolve_labels(self.ren._insert_waitcnt(self.ren._schedule(list(lin_uop.src))))
    mns = [str(u.arg).split("(", 1)[0] for u in insts if not isinstance(u.arg, tuple)]
    self.assertEqual(sum(1 for m in mns if m == "v_pack_b32_f16"), 32)
    self.assertLessEqual(sum(1 for m in mns if m == "s_waitcnt"), 10,
                         "targeted waitcnt must not emit one wait per scalar v_pack")

class TestAMDISALDSB128Lowering(unittest.TestCase):
  def _v(self, i:int):
    return UOp(Ops.INS, dtypes.int32, arg=AMDOps.MOV, tag=(Register(f"v{i}", i),))

  def test_decompose_lds_index_reports_half_and_byte_constants(self):
    ctx = IselContext(UOp.sink())
    lds_dtype = dtypes.half.ptr(64, AddrSpace.LOCAL)
    buf = UOp(Ops.DEFINE_LOCAL, lds_dtype, arg=64)
    dyn = UOp(Ops.SPECIAL, dtypes.int32, arg="lidx0")
    idx = UOp(Ops.INDEX, lds_dtype, src=(buf, UOp(Ops.ADD, dtypes.int32, src=(dyn, UOp.const(dtypes.int32, 7)))))
    order = UOp(Ops.NOOP, dtypes.void)
    desc = decompose_lds_index(ctx, idx, order)
    self.assertIsNotNone(desc)
    self.assertIs(desc.buf, buf)
    self.assertIs(desc.dyn, dyn)
    self.assertEqual(desc.const_half, 7)
    self.assertEqual(desc.const_bytes, 14)
    self.assertEqual(desc.itemsize, 2)
    self.assertEqual(desc.base_bytes, 0)
    self.assertIs(desc.order, order)

  def test_ds_load_b128_lowering_uses_offset0(self):
    addr = self._v(5)
    x = UOp(Ops.INS, dtypes.int32, src=(addr, UOp(Ops.NOOP, dtypes.void), UOp.const(dtypes.int32, 16).rtag()),
            arg=AMDOps.DS_LOAD_B128, tag=(Register("v200", 200),))
    inst, waits = lower_inst(x)
    self.assertEqual(waits, [inst])
    self.assertIn("ds_load_b128(v[200:203], v[5]", str(inst.arg))
    self.assertTrue(str(inst.arg).endswith(", 16)"), f"expected offset0=16 in {inst.arg}")

  def test_global_load_b128_lowering_uses_offset0(self):
    off = self._v(5)
    saddr = self._v(7)
    x = UOp(Ops.INS, dtypes.int32, src=(off, saddr, UOp.const(dtypes.int32, 16).rtag()), arg=AMDOps.GLOBAL_LOAD_B128, tag=(Register("v200", 200),))
    inst, waits = lower_inst(x)
    self.assertEqual(waits, [inst])
    self.assertIn("global_load_b128(v[200:203], v[5]", str(inst.arg))
    self.assertTrue(str(inst.arg).endswith(", 16)"), f"expected offset0=16 in {inst.arg}")

  def test_ds_store_b128_lowering_uses_offset0(self):
    addr, data = self._v(6), tuple(self._v(i) for i in range(220, 224))
    x = UOp(Ops.INS, dtypes.void, src=(addr,) + data + (UOp(Ops.NOOP, dtypes.void), UOp.const(dtypes.int32, 16).rtag()),
            arg=AMDOps.DS_STORE_B128)
    inst, waits = lower_inst(x)
    self.assertEqual(waits, [inst])
    self.assertIn("v[6], v[220:223]", str(inst.arg))
    self.assertTrue(str(inst.arg).endswith(", 16)"), f"expected offset0=16 in {inst.arg}")

  def test_ds_store_b64_lowering_uses_offset0(self):
    addr, data = self._v(6), tuple(self._v(i) for i in range(220, 222))
    x = UOp(Ops.INS, dtypes.void, src=(addr,) + data + (UOp(Ops.NOOP, dtypes.void), UOp.const(dtypes.int32, 16).rtag()),
            arg=AMDOps.DS_STORE_B64)
    inst, waits = lower_inst(x)
    self.assertEqual(waits, [inst])
    self.assertIn("v[6], v[220:221]", str(inst.arg))
    self.assertTrue(str(inst.arg).endswith(", 16)"), f"expected offset0=16 in {inst.arg}")

  def test_lds_store_selects_b128_only_for_fixed_contiguous_packed_vgprs(self):
    addr, order = self._v(7), UOp(Ops.NOOP, dtypes.void)
    a = UOp(Ops.NOOP, dtypes.int32.ptr(), src=(addr, order), arg="lds")
    packed = UOp(Ops.NOOP, dtypes.int32.vec(4), src=tuple(self._v(i) for i in range(224, 228)))
    out = isel_store(None, a, packed, UOp(Ops.STORE, dtypes.void))
    self.assertIs(out.arg, AMDOps.DS_STORE_B128)
    self.assertEqual(out.src[-1].arg, 0, "LDS address VGPR carries the dynamic byte address; b128 immediate stays offset0=0")

  def test_ds_store_b128_keeps_aligned_safe_offset0(self):
    addr, order = self._v(7), UOp(Ops.NOOP, dtypes.void)
    a = UOp(Ops.NOOP, dtypes.int32.ptr(), src=(addr, order, UOp.const(dtypes.int32, 16).rtag()), arg="lds")
    packed = UOp(Ops.NOOP, dtypes.int32.vec(4), src=tuple(self._v(i) for i in range(224, 228)))
    out = isel_store(IselContext(UOp.sink()), a, packed, UOp(Ops.STORE, dtypes.void))
    self.assertIs(out.arg, AMDOps.DS_STORE_B128)
    self.assertIs(out.src[0], addr)
    self.assertEqual(out.src[-1].arg, 16)

  def test_ds_store_b128_materializes_unaligned_offset_candidate(self):
    addr, order = self._v(7), UOp(Ops.NOOP, dtypes.void)
    a = UOp(Ops.NOOP, dtypes.int32.ptr(), src=(addr, order, UOp.const(dtypes.int32, 14).rtag()), arg="lds")
    packed = UOp(Ops.NOOP, dtypes.int32.vec(4), src=tuple(self._v(i) for i in range(224, 228)))
    out = isel_store(IselContext(UOp.sink()), a, packed, UOp(Ops.STORE, dtypes.void))
    self.assertIs(out.arg, AMDOps.DS_STORE_B128)
    self.assertIs(out.src[0].arg, AMDOps.V_IADD)
    self.assertEqual(out.src[0].src[1].arg, 14)
    self.assertEqual(out.src[-1].arg, 0)

  def test_lds_store_sorts_constrained_vpack_tuple_for_b128_span(self):
    addr, order = self._v(7), UOp(Ops.NOOP, dtypes.void)
    a = UOp(Ops.NOOP, dtypes.int32.ptr(), src=(addr, order), arg="lds")
    packs = tuple(
      UOp(Ops.INS, dtypes.int32, src=(self._v(10 + 2*i), self._v(11 + 2*i)), arg=AMDOps.V_PACK, tag=(Register(f"v{r}", r),))
      for i, r in enumerate((234, 232, 235, 233)))
    packed = UOp(Ops.NOOP, dtypes.int32.vec(4), src=packs)
    out = isel_store(None, a, packed, UOp(Ops.STORE, dtypes.void))
    self.assertIs(out.arg, AMDOps.DS_STORE_B128)
    self.assertEqual([u.tag[0].cons[0].index for u in out.src[1:5]], [232, 233, 234, 235])

  def test_lds_store_selects_b128_for_global_load_b128_operand(self):
    addr, order = self._v(31), UOp(Ops.NOOP, dtypes.void)
    a = UOp(Ops.NOOP, dtypes.int32.ptr(), src=(addr, order), arg="lds")
    packed = UOp(Ops.INS, dtypes.int32, src=(self._v(16), self._v(17), UOp(Ops.CONST, dtypes.int32, arg=0).rtag()),
                 arg=AMDOps.GLOBAL_LOAD_B128, tag=(Register("v220", 220),))
    out = isel_store(None, a, packed, UOp(Ops.STORE, dtypes.void))
    self.assertIs(out.arg, AMDOps.DS_STORE_B128)
    self.assertEqual(out.src[-1].arg, 0, "Wide source from GLOBAL_LOAD_B128 should keep the packed offset0=0")

  def test_lds_store_half_vec4_selects_scalar_ds_store(self):
    # half.vec(4) LDS store lowers to the scalar DS_STORE path (the removed PREFILL_LDS_PACK_WITHLOCAL_B64 probe
    # that once produced DS_STORE_B64 here was a dead, never-routed diagnostic; deleted in the Phase-2 flag-collapse).
    addr, order = self._v(31), UOp(Ops.NOOP, dtypes.void)
    a = UOp(Ops.NOOP, dtypes.int32.ptr(), src=(addr, order), arg="lds")
    vals = UOp(Ops.NOOP, dtypes.half.vec(4), src=tuple(UOp(Ops.INS, dtypes.half, arg=AMDOps.MOV, tag=(Register(f"v{i}", i),)) for i in range(10, 14)))
    out = isel_store(IselContext(UOp.sink()), a, vals, UOp(Ops.STORE, dtypes.void))
    self.assertIs(out.arg, AMDOps.DS_STORE)

  def test_lds_store_rejects_global_load_b128_without_fixed_register_span(self):
    addr, order = self._v(31), UOp(Ops.NOOP, dtypes.void)
    a = UOp(Ops.NOOP, dtypes.int32.ptr(), src=(addr, order), arg="lds")
    packed = UOp(Ops.INS, dtypes.int32, src=(self._v(16), self._v(17), UOp(Ops.CONST, dtypes.int32, arg=0).rtag()),
                 arg=AMDOps.GLOBAL_LOAD_B128, tag=())
    out = isel_store(None, a, packed, UOp(Ops.STORE, dtypes.void))
    self.assertIs(out.arg, AMDOps.DS_STORE)

  def test_lds_store_rejects_noncontiguous_packed_vgpr_sources(self):
    addr, order = self._v(11), UOp(Ops.NOOP, dtypes.void)
    a = UOp(Ops.NOOP, dtypes.int32.ptr(), src=(addr, order), arg="lds")
    packed = UOp(Ops.NOOP, dtypes.int32.vec(4), src=(self._v(224), self._v(225), self._v(227), self._v(228)))
    out = isel_store(None, a, packed, UOp(Ops.STORE, dtypes.void))
    self.assertIs(out.arg, AMDOps.DS_STORE)

  def test_ds_store_b128_lowering_with_single_packed_ins_operand(self):
    addr = self._v(6)
    data = UOp(Ops.INS, dtypes.int32, src=(self._v(15), self._v(16), UOp(Ops.CONST, dtypes.int32, arg=0).rtag()), arg=AMDOps.GLOBAL_LOAD_B128,
               tag=(Register("v200", 200),))
    x = UOp(Ops.INS, dtypes.void, src=(addr, data, UOp(Ops.NOOP, dtypes.void), UOp(Ops.CONST, dtypes.int32, arg=16).rtag()), arg=AMDOps.DS_STORE_B128)
    inst, waits = lower_inst(x)
    self.assertEqual(waits, [inst])
    self.assertIn("v[6], v[200:203]", str(inst.arg))
    self.assertTrue(str(inst.arg).endswith(", 16)"), f"expected offset0=16 in {inst.arg}")

  def test_gated_ds_store_b128_lowering_masks_exec(self):
    gate, addr, data = self._v(4), self._v(6), tuple(self._v(i) for i in range(238, 242))
    x = UOp(Ops.INS, dtypes.void, src=(gate, addr) + data + (UOp(Ops.NOOP, dtypes.void), UOp.const(dtypes.int32, 0).rtag()),
            arg=AMDOps.GATED_STORE_B128)
    inst, waits = lower_inst(x)
    self.assertEqual(inst, waits[-1])
    self.assertEqual(len(waits), 4)
    self.assertIn("v[6], v[238:241]", str(waits[2].arg))
    self.assertIn("s_and_saveexec_b32", str(waits[1].arg))


if __name__ == "__main__":
  unittest.main()
