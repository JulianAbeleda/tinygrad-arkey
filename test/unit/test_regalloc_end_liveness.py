from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.dtype import dtypes
from tinygrad.renderer.isa import ISARenderer, Register
from tinygrad.uop.ops import Ops, UOp


class DummyRenderer(ISARenderer): pass


def test_end_keeps_only_range_machine_operand_live():
  physical = tuple(Register(f"r{i}", i) for i in range(3))
  body_reg, range_reg = Register("body", 10, physical), Register("range", 11, physical)
  body = UOp(Ops.INS, dtypes.int32, arg="body", tag=(body_reg,))
  rng = UOp(Ops.RANGE, dtypes.int32,
            src=(UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 4)), tag=(range_reg,))
  store = UOp(Ops.INS, dtypes.void, src=(body,), arg="store")
  release = UOp(Ops.AFTER, dtypes.int32, src=(body, store))
  end = UOp(Ops.END, dtypes.void, src=(release, rng))

  ctx = LinearScanRegallocContext([rng, body, store, release, end], DummyRenderer("TEST"))

  assert store in release.src and release in end.src  # release and loop ordering remain structural dependencies
  assert ctx.live_range[body_reg] == [1, 2]  # the real store consumer, not END
  assert ctx.live_range[range_reg] == [0, 4]  # the backedge counter remains live through END
