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


def test_unselected_op_is_named_instead_of_desyncing_the_rewrite_index():
  # regalloc_rewrite pairs each visited uop with uops[next(ctx.idx)] BY POSITION.  An op with no ISA selection rule
  # is never visited, so every later index silently shifts by one and surfaces as a KeyError on an unrelated vreg.
  # Report the unselected op itself: a missing lowering rule is a backend defect, not a register-allocation one.
  import pytest
  a = UOp(Ops.INS, dtypes.half, arg="a", tag=(Register("r0", 0),))
  b = UOp(Ops.INS, dtypes.half, arg="b", tag=(Register("r1", 1),))
  unselected = UOp(Ops.MUL, dtypes.half, (a, b))
  with pytest.raises(RuntimeError, match="without an ISA selection rule"):
    LinearScanRegallocContext([a, b, unselected], DummyRenderer("TEST"))


def test_fully_selected_program_passes_the_positional_integrity_check():
  a = UOp(Ops.INS, dtypes.half, arg="a", tag=(Register("r0", 0),))
  LinearScanRegallocContext([a, UOp(Ops.SINK, dtypes.void, (a,))], DummyRenderer("TEST"))
