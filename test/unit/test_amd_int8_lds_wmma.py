import unittest
from dataclasses import replace

from tinygrad import Tensor
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
from tinygrad.dtype import AddrSpace, dtypes
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops, UOp
from extra.qk.kernel_vocabulary import KernelCandidateContext, KernelLDSWindow, KernelTileGeometry
from tinygrad.codegen.late.devectorizer import split_load_store
from tinygrad.renderer.isa import IselContext, Register
from tinygrad.renderer.isa.amd import AMDISARenderer, AMDOps, _frag_b128_loads, _wmma_operand_regs, isel_store


class TestAMDInt8LDSWMMA(unittest.TestCase):
  def setUp(self):
    self.renderer = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))

  @staticmethod
  def _v(i:int, dtype=dtypes.int32):
    return UOp(Ops.INS, dtype, arg=AMDOps.MOV, tag=(Register(f"v{i}", i),))

  def test_aligned_local_char16_store_survives_devectorization(self):
    buf = UOp.placeholder((1024,), dtypes.char, 0, addrspace=AddrSpace.LOCAL)
    idx = UOp.range(16, 0) * 16
    val = UOp(Ops.STACK, dtypes.char.vec(16), tuple(UOp.const(dtypes.char, i) for i in range(16)))
    store = buf.index(idx, dtype=dtypes.char.vec(16)).store(val)
    self.assertIsNone(split_load_store(self.renderer, store, store.src[0]))

  def test_char16_local_store_selects_one_b128(self):
    ctx, order = IselContext(UOp.sink()), UOp(Ops.NOOP, dtypes.void)
    addr = self._v(7)
    carrier = UOp(Ops.NOOP, dtypes.char.vec(16), tuple(self._v(20+i, dtypes.char) for i in range(16)))
    out = isel_store(ctx, UOp(Ops.NOOP, dtypes.char.ptr(), (addr, order), arg="lds"), carrier, UOp(Ops.STORE, dtypes.void))
    self.assertIs(out.arg, AMDOps.DS_STORE_B128)
    self.assertEqual(len(out.src[1:5]), 4)

  def test_byte_lds_store_uses_b8_for_scalar_and_fails_closed_for_bad_vectors(self):
    ctx, order, addr = IselContext(UOp.sink()), UOp(Ops.NOOP, dtypes.void), self._v(7)
    char16 = UOp(Ops.NOOP, dtypes.char.vec(16), tuple(self._v(20+i, dtypes.char) for i in range(16)))
    with self.assertRaisesRegex(NotImplementedError, "unaligned byte LDS"):
      isel_store(ctx, UOp(Ops.NOOP, dtypes.char.ptr(), (addr, order, UOp.const(dtypes.int32, 1).rtag()), arg="lds"),
                 char16, UOp(Ops.STORE, dtypes.void))
    scalar = isel_store(ctx, UOp(Ops.NOOP, dtypes.char.ptr(), (addr, order), arg="lds"), self._v(20, dtypes.char), UOp(Ops.STORE, dtypes.void))
    self.assertIs(scalar.arg, AMDOps.DS_STORE)
    self.assertEqual(scalar.src[3].arg, 1)
    bad = UOp(Ops.NOOP, dtypes.char.vec(4), tuple(self._v(20+i, dtypes.char) for i in range(4)))
    with self.assertRaisesRegex(NotImplementedError, "unsupported or unaligned byte LDS"):
      isel_store(ctx, UOp(Ops.NOOP, dtypes.char.ptr(), (addr, order), arg="lds"), bad, UOp(Ops.STORE, dtypes.void))

  def test_aligned_char16_fragment_load_is_one_b128_four_vgprs(self):
    ctx = IselContext(UOp.sink())
    buf = UOp(Ops.DEFINE_LOCAL, dtypes.char.ptr(1024, AddrSpace.LOCAL), arg=1024)
    dyn = UOp(Ops.SPECIAL, dtypes.int32, arg="lidx0") * 16
    elems = tuple(buf.index(dyn + i).load() for i in range(16))
    loads = _frag_b128_loads(ctx, elems, 200, role="A")
    self.assertIsNotNone(loads)
    self.assertEqual(len(loads), 4)
    self.assertEqual(sum(x.arg is AMDOps.DS_LOAD_B128 for x in loads), 1)

    hbuf = UOp(Ops.DEFINE_LOCAL, dtypes.half.ptr(1024, AddrSpace.LOCAL), arg=1024)
    helems = tuple(hbuf.index(dyn + i).load() for i in range(16))
    hloads = _frag_b128_loads(ctx, helems, 208, role="A")
    self.assertEqual(len(hloads), 8)
    self.assertEqual(sum(x.arg is AMDOps.DS_LOAD_B128 for x in hloads), 2)

  def test_operand_width_comes_from_carrier_and_c_contract_stays_eight(self):
    char = UOp(Ops.NOOP, dtypes.char.vec(16), tuple(UOp.const(dtypes.char, 0) for _ in range(16)))
    half = UOp(Ops.NOOP, dtypes.half.vec(16), tuple(UOp.const(dtypes.half, 0) for _ in range(16)))
    self.assertEqual(_wmma_operand_regs(char), 4)
    self.assertEqual(_wmma_operand_regs(half), 8)

  def test_stage1_int8_candidate_compiles_end_to_end(self):
    geometry = KernelTileGeometry((128, 128, 32), (4, 2), 256, 32,
      (KernelLDSWindow("A", 0, 4096, 32), KernelLDSWindow("B", 4096, 8192, 32)))
    context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1", "b" * 64, geometry,
                                     KernelStage1PipelinePlan(2, 8192))
    a, b = Tensor.empty(128, 256, dtype=dtypes.char), Tensor.empty(256, 128, dtype=dtypes.char)
    sink = next(u for u in a.matmul(b, dtype=dtypes.int).schedule_linear().toposort() if u.op is Ops.SINK)
    sink = sink.replace(arg=replace(sink.arg, opts_to_apply=(Opt(OptOps.TC, 0, (3, 0, 1)), Opt(OptOps.UNROLL, 0, 0)),
                                    candidate_context=context))
    to_program_cache.clear()
    prg = to_program(sink, self.renderer)
    source = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
    self.assertIn("ds_store_b128", source)
    self.assertIn("ds_store_b8", source)
    self.assertIn("v_wmma_i32_16x16x16_iu8", source)


if __name__ == "__main__":
  unittest.main()
