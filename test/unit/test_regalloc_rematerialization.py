from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.dtype import dtypes
from tinygrad.renderer.isa import ISARenderer, Register
from tinygrad.uop.ops import Ops, UOp


class RematerializingRenderer(ISARenderer):
  def is_rematerializable(self, u:UOp) -> bool: return u.op is Ops.INS and u.arg == "const"


def _def(v:Register, arg:str) -> UOp:
  return UOp(Ops.INS, dtypes.int32, arg=arg, tag=(v,))


def _use(*src:UOp) -> UOp:
  return UOp(Ops.INS, dtypes.int32, src=src, arg="use")


def test_renderer_rematerializable_value_avoids_stack_at_ordinary_use():
  physical = (Register("r0", 0),)
  const, temporary = Register("const", 10, physical), Register("temporary", 11, physical)
  cdef, tdef = _def(const, "const"), _def(temporary, "compute")

  ctx = LinearScanRegallocContext([cdef, tdef, _use(tdef), _use(cdef)], RematerializingRenderer("TEST"))

  assert not ctx.spills and ctx.stack_size == 0
  assert (3, const) in ctx.remats
  assert not ctx.remat_before


def test_non_rematerializable_value_keeps_existing_spill_behavior():
  physical = (Register("r0", 0),)
  value, temporary = Register("value", 10, physical), Register("temporary", 11, physical)
  vdef, tdef = _def(value, "compute"), _def(temporary, "compute")

  ctx = LinearScanRegallocContext([vdef, tdef, _use(tdef), _use(vdef)], RematerializingRenderer("TEST"))

  assert value in ctx.spills and ctx.stack_size == dtypes.int32.itemsize
  assert not ctx.remats


def test_renderer_rematerialization_is_safe_inside_loop_and_at_backedge():
  physical = (Register("r0", 0), Register("r1", 1))
  const = Register("const", 10, (physical[0],))
  temporary = Register("temporary", 11, (physical[0],))
  late_temporary = Register("late_temporary", 13, (physical[0],))
  range_reg = Register("range", 12, (physical[1],))
  cdef = _def(const, "const")
  rng = UOp(Ops.RANGE, dtypes.int32, src=(UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 4)), tag=(range_reg,))
  tdef = _def(temporary, "compute")
  late_tdef = _def(late_temporary, "compute")
  end = UOp(Ops.END, dtypes.void, src=(rng,))

  ctx = LinearScanRegallocContext([cdef, rng, tdef, _use(tdef), _use(cdef), late_tdef, _use(late_tdef), end],
                                   RematerializingRenderer("TEST"))

  assert not ctx.spills and ctx.stack_size == 0
  assert (4, const) in ctx.remats
  assert (7, const) in ctx.remats
  assert ctx.remat_before[7] == [const]
