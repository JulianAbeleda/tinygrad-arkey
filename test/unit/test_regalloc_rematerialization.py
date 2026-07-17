from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target
from tinygrad.renderer.isa import ISARenderer, Register
from tinygrad.renderer.isa.amd import AMDISARenderer, AMDOps
from tinygrad.uop.ops import Ops, UOp


class RematerializingRenderer(ISARenderer):
  def is_rematerializable(self, u:UOp) -> bool: return u.op is Ops.INS and u.arg == "const"


def _def(v:Register, arg:str) -> UOp:
  return UOp(Ops.INS, dtypes.int32, arg=arg, tag=(v,))


def _use(*src:UOp) -> UOp:
  return UOp(Ops.INS, dtypes.int32, src=src, arg="use")


def _amd_def(v:Register, op:AMDOps, *src:UOp, dtype=dtypes.int32) -> UOp:
  return UOp(Ops.INS, dtype, src=src, arg=op, tag=(v,))


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


def test_amd_pure_compare_rematerializes_with_dependencies_at_ordinary_use():
  physical = tuple(Register(f"r{i}", i) for i in range(4))
  inputs = tuple(Register(f"input{i}", 10+i, physical) for i in range(3))
  predicate = Register("predicate", 13, physical)
  temporaries = tuple(Register(f"temporary{i}", 20+i, physical) for i in range(4))
  idefs = tuple(_amd_def(v, AMDOps.V_CONST, UOp.const(dtypes.int32, i).rtag()) for i,v in enumerate(inputs))
  compare = _amd_def(predicate, AMDOps.V_CMPLT_I, *idefs)
  tdefs = tuple(_def(v, "compute") for v in temporaries)
  use = _use(compare)

  ctx = LinearScanRegallocContext([*idefs, compare, *tdefs, _use(*tdefs), use],
                                   AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))

  assert not ctx.spills and ctx.stack_size == 0
  assert (9, predicate) in ctx.remats
  assert all((9, v) in ctx.remats for v in inputs)
  remat, before = ctx.remat(predicate, 9)
  assert before[-1] is remat and remat.arg is AMDOps.V_CMPLT_I
  assert tuple(before[:-1]) == remat.src


def test_amd_pure_compare_rematerializes_at_loop_backedge():
  physical = tuple(Register(f"r{i}", i) for i in range(5))
  inputs = tuple(Register(f"input{i}", 10+i, (physical[i],)) for i in range(3))
  predicate = Register("predicate", 13, (physical[3],))
  temporary = Register("temporary", 14, (physical[3],))
  range_reg = Register("range", 15, (physical[4],))
  idefs = tuple(_amd_def(v, AMDOps.V_CONST, UOp.const(dtypes.int32, i).rtag()) for i,v in enumerate(inputs))
  compare = _amd_def(predicate, AMDOps.V_CMPNE_I, *idefs)
  rng = UOp(Ops.RANGE, dtypes.int32, src=(UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 4)), tag=(range_reg,))
  tdef = _def(temporary, "compute")
  end = UOp(Ops.END, dtypes.void, src=(rng,))

  ctx = LinearScanRegallocContext([*idefs, compare, rng, _use(compare), tdef, _use(tdef), end],
                                   AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))

  assert not ctx.spills and ctx.stack_size == 0
  assert (8, predicate) in ctx.remats
  assert ctx.remat_before[8] == [predicate]
