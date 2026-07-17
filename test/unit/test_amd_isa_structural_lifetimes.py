from types import SimpleNamespace

from tinygrad import dtypes
from tinygrad.codegen import line_rewrite
from tinygrad.codegen.late.regalloc import pressure_schedule
from tinygrad.renderer.isa import Register
from tinygrad.renderer.isa import amd
from tinygrad.renderer.isa.amd import AMDOps, lower_inst, pre_regalloc_matcher
from tinygrad.uop.ops import Ops, UOp


def _vreg(name:str, index:int) -> Register:
  return Register(name, index)


def test_wide_fragment_release_order_survives_pre_regalloc_cleanup():
  """A later wide producer starts only after the prior fragment's FP32 update."""
  addr0 = UOp(Ops.INS, dtypes.int32, arg=AMDOps.MOV, tag=(_vreg("v4", 4),))
  load0 = UOp(Ops.INS, dtypes.int32, (addr0, UOp(Ops.NOOP, dtypes.void), UOp.const(dtypes.int32, 0).rtag()),
              AMDOps.DS_LOAD_B128, tag=(_vreg("v200", 200),))
  wmma = UOp(Ops.INS, dtypes.int32, (load0,), AMDOps.V_WMMA_I8, tag=(_vreg("v8", 8),))
  updates = tuple(UOp(Ops.INS, dtypes.float32,
                      (UOp(Ops.INS, dtypes.float32, (wmma,), AMDOps.V_CVT_I2F, tag=(_vreg(f"v{32+i}", 32+i),)),),
                      AMDOps.V_ADD, tag=(_vreg(f"v{48+i}", 48+i),)) for i in range(8))

  # The unrelated integer dependency models an old address/order carrier.  It
  # must be discarded rather than becoming part of the retained boundary.
  stale = UOp(Ops.INS, dtypes.int32, (addr0,), AMDOps.V_IADD, tag=(_vreg("v7", 7),))
  addr1 = UOp(Ops.INS, dtypes.int32, arg=AMDOps.MOV, tag=(_vreg("v5", 5),))
  load1 = UOp(Ops.INS, dtypes.int32,
              (addr1, UOp(Ops.NOOP, dtypes.void), UOp.const(dtypes.int32, 16).rtag(), stale) + updates,
              AMDOps.DS_LOAD_B128, tag=(_vreg("v200", 200),))
  linear = pressure_schedule(list(UOp.sink(load1).toposort()))
  cleaned = line_rewrite(linear, pre_regalloc_matcher)
  selected = next(x for x in cleaned if x.op is Ops.INS and x.arg is AMDOps.DS_LOAD_B128 and x.src[2].arg == 16)

  assert len(selected.src) == 3 and selected.src[0].op is Ops.AFTER
  assert selected.src[0].src[0] is addr1
  assert set(selected.src[0].src[1:]) == set(updates)
  assert stale not in selected.backward_slice_with_self
  assert max(cleaned.index(x) for x in updates) < cleaned.index(selected)

  # The canonical operand shape remains directly lowerable as one b128 load;
  # the AFTER is a zero-code register alias, not an extra ISA operand.
  inst, waits = lower_inst(selected)
  assert waits == [inst]
  assert "ds_load_b128(v[200:203], v[5]" in str(inst.arg)


def test_progressive_c_marked_carriers_serialize_all_lane_drains(monkeypatch):
  symbolic0 = UOp(Ops.WMMA, dtypes.float32.vec(8), src=())
  symbolic1 = UOp(Ops.WMMA, dtypes.float32.vec(8), src=(symbolic0,))
  operands0 = tuple(UOp(Ops.INS, dtypes.int32, arg=AMDOps.MOV, tag=(_vreg(f"v{64+i}", 64+i),)) for i in range(24))
  operands1 = tuple(UOp(Ops.INS, dtypes.int32, arg=AMDOps.MOV, tag=(_vreg(f"v{128+i}", 128+i),)) for i in range(24))
  def marked(root, operands):
    machine = UOp(Ops.INS, dtypes.float32, operands, AMDOps.V_WMMA, tag=(_vreg("v8", 8),))
    marker = UOp(Ops.NOOP, dtypes.void, arg=("selected_wmma_root", root))
    carrier = UOp(Ops.NOOP, dtypes.float32.vec(8), src=(machine,) + tuple(
      UOp(Ops.INS, dtypes.float32, (machine,), AMDOps.MOV, tag=(_vreg(f"v{8+i}", 8+i),)) for i in range(1, 8)) + (marker,))
    drains = tuple(UOp(Ops.INS, dtypes.float32, (carrier,), AMDOps.V_CVT_I2F,
                       tag=(_vreg(f"v{96+i+(16 if root is symbolic1 else 0)}", 96+i+(16 if root is symbolic1 else 0)),)) for i in range(8))
    return machine, carrier, drains
  machine0, carrier0, drains0 = marked(symbolic0, operands0)
  machine1, carrier1, drains1 = marked(symbolic1, operands1)
  monkeypatch.setattr(amd, "_progressive_c_assignment", lambda ctx: ({symbolic0:0, symbolic1:0}, 1))
  serialized = amd._serialize_progressive_c_drains(SimpleNamespace(), UOp.sink(*drains0, *drains1))
  assert serialized is not None
  selected = [u for u in serialized.toposort() if u.op is Ops.INS and u.arg is AMDOps.V_WMMA]
  assert len(selected) == 2
  second = next(u for u in selected if set(drains0).issubset(u.src))
  assert second.src[:24] == operands1 and second.src[24:] == drains0
  linear = pressure_schedule(list(serialized.toposort()))
  assert max(linear.index(x) for x in drains0) < linear.index(second)
  cleaned = line_rewrite(linear, pre_regalloc_matcher)
  cleaned_second = next(u for u in cleaned if u is second)
  assert cleaned_second.src[:24] == operands1 and cleaned_second.src[24:] == drains0
  assert "v_wmma_f32_16x16x16_f16" in str(lower_inst(cleaned_second).arg)
