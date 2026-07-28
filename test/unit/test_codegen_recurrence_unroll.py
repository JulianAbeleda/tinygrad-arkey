from types import SimpleNamespace
import importlib.util

import pytest

from tinygrad import dtypes
from tinygrad.codegen.late.recurrence import unroll_recurrence
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import AxisType, Ops, UOp


def test_recurrence_owner_is_core_not_extra():
  assert importlib.util.find_spec("tinygrad.codegen.late.recurrence") is not None
  assert importlib.util.find_spec("extra.llm_research.codegen_recurrence_unroll") is None


def _recurrence(end=8, *, inner=False, reinit=False, multi_end=False):
  zero = UOp.const(dtypes.weakint, 0)
  rng = UOp.range(end, 9101, AxisType.REDUCE)
  acc = UOp.placeholder((1,), dtypes.float, 9100, addrspace=AddrSpace.REG)
  carry = acc.after(rng)
  value = carry.index(zero).load() + 1.0

  reset_reg = None
  reset = None
  if reinit:
    reset_reg = UOp.placeholder((1,), dtypes.float, 9102, addrspace=AddrSpace.REG)
    reset = reset_reg.after(rng).index(zero).store(0.0)
    value = value + reset_reg.after(reset).index(zero).load()

  update = acc.index(zero).store(value)
  state = UOp.group(reset, update) if reset is not None else update
  if inner:
    inner_rng = UOp.range(3, 9103, AxisType.LOOP)
    update = acc.index(zero).store(value + inner_rng.cast(dtypes.float))
    state = (UOp.group(reset, update) if reset is not None else update).end(inner_rng)
  if multi_end:
    sibling = UOp.range(2, 9104, AxisType.LOOP)
    ended = UOp(Ops.END, src=(state, rng, sibling))
  else:
    ended = state.end(rng)
  sink = UOp.sink(ended, acc.after(ended).index(zero).load())
  return sink, acc, reset_reg


def test_recurrence_unroll_identity_and_fail_closed_cases():
  sink, _, _ = _recurrence()
  assert unroll_recurrence(sink, 1) is sink
  assert unroll_recurrence(sink, 3) is sink  # factor does not divide the eight-iteration range

  no_carry = UOp.sink(UOp.const(dtypes.float, 1.0))
  assert unroll_recurrence(no_carry, 2) is no_carry

  symbolic, _, _ = _recurrence(UOp.variable("recurrence_extent", 1, 8))
  assert unroll_recurrence(symbolic, 2) is symbolic

  multi_end, _, _ = _recurrence(multi_end=True)
  assert unroll_recurrence(multi_end, 2) is multi_end


def test_recurrence_unroll_rethreads_the_canonical_carry():
  sink, acc, _ = _recurrence()
  rewritten = unroll_recurrence(sink, 2)
  assert rewritten is not sink

  ranges = [u for u in rewritten.toposort() if u.op is Ops.RANGE]
  ends = [u for u in rewritten.toposort() if u.op is Ops.END]
  stores = [u for u in rewritten.toposort() if u.op is Ops.STORE and acc in u.src[0].toposort()]
  assert len(ranges) == 1 and ranges[0].arg[-1] is AxisType.REDUCE and int(ranges[0].vmax) + 1 == 4
  assert len(ends) == 1 and ends[0].src[1] is ranges[0]
  assert len(stores) == 2
  assert stores[0] in stores[1].src[1].toposort(), "copy 1 must consume copy 0's carried state"


def test_recurrence_unroll_duplicates_nested_ranges_and_reinit_registers():
  sink, acc, reset_reg = _recurrence(inner=True, reinit=True)
  rewritten = unroll_recurrence(sink, 2)
  ranges = [u for u in rewritten.toposort() if u.op is Ops.RANGE]
  regs = [u for u in rewritten.toposort() if u.op is Ops.DEFINE_REG]

  outer = [u for u in ranges if u.arg[-1] is AxisType.REDUCE]
  inners = [u for u in ranges if u.arg[-1] is AxisType.LOOP]
  assert len(outer) == 1 and int(outer[0].vmax) + 1 == 4
  assert len(inners) == 2 and all(int(u.vmax) + 1 == 3 for u in inners) and inners[0] is not inners[1]
  assert acc in regs and reset_reg not in regs
  private_resets = [u for u in regs if u is not acc]
  assert len(private_resets) == 2 and private_resets[0].arg != private_resets[1].arg


def test_full_rewrite_dispatches_recurrence_unroll_for_amd(monkeypatch):
  import tinygrad.codegen as codegen

  class Dispatched(Exception): pass

  sink = UOp.sink(UOp.const(dtypes.float, 1.0))
  monkeypatch.setattr(codegen, "graph_rewrite", lambda ast, *args, **kwargs: ast)
  monkeypatch.setattr(codegen, "getenv", lambda name, *args: 2 if name == "SCHED_UNROLL" else 0)
  monkeypatch.setattr(codegen, "unroll_recurrence", lambda ast, factor: (_ for _ in ()).throw(Dispatched((ast, factor))))

  with pytest.raises(Dispatched) as exc:
    codegen._full_rewrite_to_sink(sink, SimpleNamespace(target=SimpleNamespace(device="AMD")))
  assert exc.value.args == ((sink, 2),)
