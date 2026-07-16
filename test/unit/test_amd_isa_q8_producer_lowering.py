import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.helpers import Target
from tinygrad.renderer.isa import IselContext, Register
from tinygrad.renderer.isa.amd import AMDISARenderer, AMDOps, isel_cast, isel_matcher, lower_inst
from tinygrad.uop.ops import Ops, UOp

from extra.qk.q4k_q8_activation_producer import produce_physical_ds4_q8_1


def _renderer(): return AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))


def _mnemonics(prg):
  linear = next(u for u in prg.src if u.op is Ops.LINEAR)
  insts = [u.arg for u in linear.src if not isinstance(u.arg, tuple)]
  assert insts and all(inst.to_bytes() for inst in insts)
  return [str(inst).split("(", 1)[0] for inst in insts]


def test_float32_reciprocal_selects_and_encodes_gfx1100_v_rcp_f32():
  ctx = IselContext(UOp.sink())
  source = UOp(Ops.INS, dtypes.float32, arg=AMDOps.V_CONST, tag=(ctx.vreg((Register("v1", 1),)),))
  selected = isel_matcher.rewrite(UOp(Ops.RECIPROCAL, dtypes.float32, (source,)), ctx)
  assert selected is not None and selected.arg is AMDOps.V_RCP

  physical = selected.replace(src=(source.replace(tag=(Register("v1", 1),)),), tag=(Register("v2", 2),))
  encoded = lower_inst(physical)
  assert str(encoded.arg).startswith("v_rcp_f32_e32")
  assert encoded.arg.to_bytes()


def test_float32_to_signed_int8_selects_i32_conversion_and_encodes():
  ctx = IselContext(UOp.sink())
  source = UOp(Ops.INS, dtypes.float32, arg=AMDOps.V_CONST, tag=(ctx.vreg((Register("v1", 1),)),))
  selected = isel_cast(ctx, UOp(Ops.CAST, dtypes.int8, (source,)))
  assert selected.arg is AMDOps.V_CVT_F2I and selected.dtype is dtypes.int8

  physical = selected.replace(src=(source.replace(tag=(Register("v1", 1),)),), tag=(Register("v2", 2),))
  encoded = lower_inst(physical)
  assert str(encoded.arg).startswith("v_cvt_i32_f32_e32")
  assert encoded.arg.to_bytes()


def test_reciprocal_and_signed_int8_cast_survive_regalloc_without_spills():
  to_program_cache.clear()
  source = Tensor.empty(32, dtype=dtypes.float32)
  sink = next(u for u in source.reciprocal().cast(dtypes.int8).schedule_linear().toposort() if u.op is Ops.SINK)
  mnemonics = _mnemonics(to_program(sink, _renderer()))
  # The scheduler may vectorize this elementwise kernel; every lane must keep both native operations.
  assert mnemonics.count("v_rcp_f32_e32") >= 1
  assert mnemonics.count("v_cvt_i32_f32_e32") == mnemonics.count("v_rcp_f32_e32")


@pytest.mark.parametrize("shape", ((1, 128), (256, 256)))
def test_physical_ds4_static_to_program(shape):
  to_program_cache.clear()
  source = Tensor.empty(*shape, dtype=dtypes.float32, device="CPU")
  output = produce_physical_ds4_q8_1(source)
  sink = next(u for u in output.values.schedule_linear().toposort() if u.op is Ops.SINK)
  mnemonics = _mnemonics(to_program(sink, _renderer()))
  assert "v_rcp_f32_e32" in mnemonics
  assert "v_cvt_i32_f32_e32" in mnemonics
  assert "v_trunc_f32_e32" in mnemonics
