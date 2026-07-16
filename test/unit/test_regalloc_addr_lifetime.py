import pytest

from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer.isa import ISARenderer, Register
from tinygrad.uop.ops import Ops, UOp


class DummyRenderer(ISARenderer): pass


@pytest.fixture(autouse=True)
def _clear_getenv_cache():
  getenv.cache_clear()
  yield
  getenv.cache_clear()


def _loop_with_users(*user_names:str):
  physical = (Register("r0", 0), Register("r1", 1))
  addr_reg, range_reg = Register("addr", 10, physical), Register("range", 11, physical)
  rng = UOp(Ops.RANGE, dtypes.int32, src=(UOp.const(dtypes.int32, 0), UOp.const(dtypes.int32, 4)), tag=(range_reg,))
  addr = UOp(Ops.INS, dtypes.int32, arg="V_OFFSET", tag=(addr_reg,))
  users = [UOp(Ops.INS, dtypes.int32, src=(addr, UOp.const(dtypes.int32, 1)), arg=name) for name in user_names]
  end = UOp(Ops.END, dtypes.void, src=(rng,))
  return addr_reg, [addr, rng, *users, end]


def test_supported_address_def_rematerializes_at_loop_backedge(monkeypatch):
  monkeypatch.setenv("REGALLOC_ADDR_REMAT", "1")
  addr, uops = _loop_with_users("V_IADD")
  ctx = LinearScanRegallocContext(uops, DummyRenderer("TEST"))
  assert ctx.live_range[addr] == [0, 2, 3]
  assert (3, addr) in ctx.remats
  assert ctx.remat_before[3] == [addr]


def test_mixed_non_address_consumer_keeps_loop_extension(monkeypatch):
  monkeypatch.setenv("REGALLOC_ADDR_REMAT", "1")
  addr, uops = _loop_with_users("V_IADD", "V_OR")
  ctx = LinearScanRegallocContext(uops, DummyRenderer("TEST"))
  assert ctx.live_range[addr] == [0, 2, 3, 4]


def test_removed_legacy_knob_cannot_shorten_lifetime(monkeypatch):
  monkeypatch.setenv("REGALLOC_NO_LOOP_EXTEND_ADDR", "1")
  monkeypatch.setenv("REGALLOC_ADDR_REMAT", "0")
  addr, uops = _loop_with_users("V_IADD")
  ctx = LinearScanRegallocContext(uops, DummyRenderer("TEST"))
  assert ctx.live_range[addr] == [0, 2, 3]
