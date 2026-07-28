import os

from tinygrad import dtypes
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.late.reg_store import _devec_distinct_reg_store
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target, getenv
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from extra.qk.reg_store_devec import _devec_reg_store


def _target(reg, indices, *, load=True):
  ptrs = [reg.index(UOp.const(dtypes.weakint, i)) for i in indices]
  lanes = [p.load() if load else p for p in ptrs]
  return UOp(Ops.STACK, dtypes.float.vec(len(lanes)), tuple(lanes)), ptrs


def _value(values):
  return UOp.const(dtypes.float.vec(len(values)), tuple(float(x) for x in values))


def _stores(group):
  assert group is not None and group.op is Ops.GROUP
  return list(group.src)


def test_distinct_register_stack_with_load_wrappers_preserves_lane_order():
  reg = UOp.placeholder((8,), dtypes.float, 9800, addrspace=AddrSpace.REG)
  target, ptrs = _target(reg, (2, 0, 1), load=True)
  out = _devec_reg_store(target, _value((10, 20, 30)))
  stores = _stores(out)
  assert [s.src[0] for s in stores] == ptrs
  assert [s.src[1].arg for s in stores] == [10.0, 20.0, 30.0]
  assert all(s.src[1].dtype == dtypes.float for s in stores)


def test_direct_and_load_wrapped_register_targets_are_both_supported():
  reg = UOp.placeholder((4,), dtypes.float, 9801, addrspace=AddrSpace.REG)
  p0 = reg.index(UOp.const(dtypes.weakint, 0))
  p1 = reg.index(UOp.const(dtypes.weakint, 1))
  target = UOp(Ops.STACK, dtypes.float.vec(2), (p0, p1.load()))
  out = _devec_reg_store(target, _value((1, 2)))
  assert [s.src[0] for s in _stores(out)] == [p0, p1]


def test_duplicate_register_pointers_are_the_extra_pass_residual_contract():
  reg = UOp.placeholder((2,), dtypes.float, 9802, addrspace=AddrSpace.REG)
  target, ptrs = _target(reg, (0, 0), load=True)
  extra = _stores(_devec_reg_store(target, _value((3, 4))))
  assert _devec_distinct_reg_store(target, _value((3, 4))) is None
  assert [s.src[0] for s in extra] == ptrs
  assert [s.src[1].arg for s in extra] == [3.0, 4.0]


def test_non_register_or_malformed_targets_fail_closed_without_partial_rewrite():
  reg = UOp.placeholder((2,), dtypes.float, 9803, addrspace=AddrSpace.REG)
  glob = UOp.placeholder((2,), dtypes.float, 9804, addrspace=AddrSpace.GLOBAL)
  good = reg.index(UOp.const(dtypes.weakint, 0)).load()
  bad_global = glob.index(UOp.const(dtypes.weakint, 0)).load()
  target = UOp(Ops.STACK, dtypes.float.vec(2), (good, bad_global))
  assert _devec_reg_store(target, _value((1, 2))) is None
  assert _devec_reg_store(UOp(Ops.STACK, dtypes.float.vec(1), (UOp.const(dtypes.float, 1.0),)),
                           _value((1,))) is None


def test_value_width_mismatch_fails_closed():
  reg = UOp.placeholder((2,), dtypes.float, 9805, addrspace=AddrSpace.REG)
  target, _ = _target(reg, (0, 1), load=True)
  assert _devec_reg_store(target, UOp.const(dtypes.float, 7.0)) is None


def _sum4_ast():
  src = UOp.param(1, dtypes.float.ptr(4))
  out = UOp.param(0, dtypes.float.ptr(1))
  i = UOp.range(4, 0)
  red = src.index(i).load().reduce(i, arg=Ops.ADD)
  return out.index(UOp.const(dtypes.int, 0), ptr=True).store(red).sink(arg=KernelInfo(opts_to_apply=()))


def _run_pipeline(device, gate, monkeypatch):
  calls = []
  import tinygrad.codegen as codegen
  original = codegen.cg_extras.reg_store_devec_pm
  monkeypatch.setattr(codegen.cg_extras, "reg_store_devec_pm", lambda: calls.append(True) or codegen.PatternMatcher([]))
  previous = os.environ.get("COALESCED_LOAD_LOWERING")
  if gate is None: os.environ.pop("COALESCED_LOAD_LOWERING", None)
  else: os.environ["COALESCED_LOAD_LOWERING"] = gate
  getenv.cache_clear()
  try:
    full_rewrite_to_sink(_sum4_ast(), AMDISARenderer(Target.parse(device)), optimize=True)
  finally:
    codegen.cg_extras.reg_store_devec_pm = original
    if previous is None: os.environ.pop("COALESCED_LOAD_LOWERING", None)
    else: os.environ["COALESCED_LOAD_LOWERING"] = previous
    getenv.cache_clear()
  return calls


def test_pipeline_dispatch_is_amd_and_gate_scoped(monkeypatch):
  assert _run_pipeline("AMD:ISA:gfx1100", "1", monkeypatch) == [True]
  assert _run_pipeline("AMD:ISA:gfx1100", None, monkeypatch) == []
  assert _run_pipeline("CPU", "1", monkeypatch) == []
