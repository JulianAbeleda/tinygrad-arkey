from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.dtype import dtypes
from tinygrad.renderer.isa import ISARenderer, Register
from tinygrad.uop.ops import Ops, UOp


def test_broad_values_do_not_occupy_every_constrained_candidate():
  physical = tuple(Register(f"r{i}", i) for i in range(8))
  broad = tuple(Register(f"b{i}", 10+i, physical) for i in range(5))
  constrained = Register("half", 20, physical[:4])
  result = Register("result", 21, physical)

  broad_defs = tuple(UOp(Ops.INS, dtypes.int32, arg="broad", tag=(v,)) for v in broad)
  half_def = UOp(Ops.INS, dtypes.half, arg="half", tag=(constrained,))
  consume_half = UOp(Ops.INS, dtypes.int32, (half_def,), "consume_half", tag=(result,))
  consume_all = UOp(Ops.INS, dtypes.int32, broad_defs+(consume_half,), "consume_all")

  ctx = LinearScanRegallocContext([*broad_defs, half_def, consume_half, consume_all], ISARenderer("TEST"))

  assert not ctx.spills
  assert {ctx.reals[i][u.reg].index for i,u in enumerate(broad_defs)} >= {4, 5, 6, 7}
