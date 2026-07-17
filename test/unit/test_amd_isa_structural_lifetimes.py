from tinygrad import dtypes
from tinygrad.codegen import line_rewrite
from tinygrad.codegen.late.regalloc import pressure_schedule
from tinygrad.renderer.isa import Register
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
