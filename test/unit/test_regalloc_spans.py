import pytest

from tinygrad.codegen.late.regalloc import LinearScanRegallocContext
from tinygrad.dtype import dtypes
from tinygrad.renderer.isa import ISARenderer, Register, RegisterSpan
from tinygrad.uop.ops import Ops, UOp


class DummyRenderer(ISARenderer):
  pass


def pool(n:int, start:int=0) -> tuple[Register, ...]:
  return tuple(Register(f"r{i}", i) for i in range(start, start+n))


def vreg(n:int, regs:tuple[Register, ...], count:int=1, alignment:int=1) -> Register:
  return Register(f"v{n}", n, regs, RegisterSpan(count, alignment))


def define(v:Register) -> UOp:
  return UOp(Ops.INS, dtypes.int32, arg="def", tag=(v,))


def use(*src:UOp) -> UOp:
  return UOp(Ops.INS, dtypes.void, src=src, arg="use")


def alloc(uops:list[UOp]) -> LinearScanRegallocContext:
  return LinearScanRegallocContext(uops, DummyRenderer("TEST"))


@pytest.mark.parametrize("count", [2, 4])
def test_allocates_consecutive_register_span(count:int):
  regs = pool(count + 2)
  v = vreg(0, regs, count)
  d = define(v)
  ctx = alloc([d, use(d)])
  base = ctx.reals[0][v]
  assert [base.index+i for i in range(count)] == list(range(base.index, base.index+count))
  assert base.index + count <= len(regs)


def test_span_alignment():
  regs = pool(6, 1)
  v = vreg(0, regs, 2, 4)
  ctx = alloc([define(v)])
  assert ctx.reals[0][v].index == 4


def test_span_does_not_overwrite_overlapping_live_scalar():
  regs = pool(4)
  scalar, wide = vreg(0, regs), vreg(1, regs, 2)
  sd, wd = define(scalar), define(wide)
  ctx = alloc([sd, wd, use(sd, wd)])
  assert ctx.reals[0][scalar].index not in {ctx.reals[1][wide].index, ctx.reals[1][wide].index+1}
  assert not ctx.spills


def test_released_span_is_reused_atomically():
  regs = pool(2)
  a, b = vreg(0, regs, 2), vreg(1, regs, 2)
  ad, bd = define(a), define(b)
  ctx = alloc([ad, use(ad), bd, use(bd)])
  assert ctx.reals[0][a] == ctx.reals[2][b]
  assert not ctx.spills


def test_constrained_pool_without_aligned_span_fails():
  regs = pool(3, 1)
  v = vreg(0, regs, 2, 4)
  with pytest.raises(RuntimeError, match="no 2-register span aligned to 4"):
    alloc([define(v)])


def test_span_pressure_spills_whole_owner_before_reuse():
  regs = pool(2)
  scalar, wide = vreg(0, regs), vreg(1, regs, 2)
  sd, wd = define(scalar), define(wide)
  ctx = alloc([sd, wd, use(wd), use(sd)])
  assert scalar in ctx.spills
  assert ctx.reals[1][wide].index == 0
  assert ctx.reals[2][wide].index == 0


def test_evicted_span_fails_closed_instead_of_partially_spilling():
  regs = pool(2)
  wide, scalar = vreg(0, regs, 2), vreg(1, regs)
  wd, sd = define(wide), define(scalar)
  with pytest.raises(RuntimeError, match="spilling a multi-register span is unsupported"):
    alloc([wd, sd, use(sd), use(wd)])


def test_scalar_allocation_is_unchanged():
  regs = pool(2)
  a, b = vreg(0, regs), vreg(1, regs)
  ad, bd = define(a), define(b)
  ctx = alloc([ad, bd, use(ad, bd)])
  assert (ctx.reals[0][a], ctx.reals[1][b]) == regs
  assert a.span == b.span == RegisterSpan(1)


@pytest.mark.parametrize("count,alignment", [(0, 1), (1, 0), (True, 1), (1, False), (1.5, 1), (1, 1.5)])
def test_invalid_span_shape(count:int, alignment:int):
  with pytest.raises(ValueError): RegisterSpan(count, alignment)
