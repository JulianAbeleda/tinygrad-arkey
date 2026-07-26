"""LR-051 regression test.

tinygrad/uop/trace.py records LOWERING_GATES_NOT_IN_CACHE_KEY: PREFILL_SOFTMAX_REDUCE_FUSE, UNSAFE_DISABLE_MASK and
REGALLOC_ADDR_REMAT change generated code (inside do_to_program's lowering pipeline) but were absent from
tinygrad.codegen.to_program's cache key. Flipping one of those gates in-process therefore returned the program
lowered under the OTHER setting -- latent only because this repo's A/B measurements use one subprocess per arm.

LR-051 fixes this by deriving the cache key's gate suffix from `tinygrad.codegen.plan.PLAN_GATES` -- the single,
already-existing inventory `OptimizationPlan.from_env` reads -- instead of a second hand-picked list of getenv(...)
calls (the hand-maintained list is exactly how the three gates went missing in the first place).

This test proves the fix directly: it counts real calls into `do_to_program` (the actual lowering entry point) while
flipping each of the three gates, using a real (CPU-only, PythonRenderer) AST and renderer. Before the fix, the
second call for each gate would have been served from the first call's cache entry -- `do_to_program` would only
have been called once per gate, not twice.
"""
import os
import pytest

import tinygrad.codegen as cg
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.codegen.plan import PLAN_GATES
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv
from tinygrad.renderer import Target
from tinygrad.runtime.ops_python import PythonRenderer
from tinygrad.uop.ops import KernelInfo, UOp
from tinygrad.uop.trace import LOWERING_GATES_NOT_IN_CACHE_KEY


def _sink(value: int) -> UOp:
  out = UOp.param(0, dtypes.float.ptr(1))
  return out.index(UOp.const(dtypes.int, 0), ptr=True).store(UOp.const(dtypes.float, value)).sink(arg=KernelInfo())


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
  to_program_cache.clear()
  getenv.cache_clear()
  yield
  to_program_cache.clear()
  getenv.cache_clear()


def _spy_do_to_program(monkeypatch) -> list[str | None]:
  """Wrap the REAL do_to_program with a counter, recording the gate value active at each real lowering call."""
  calls: list[str | None] = []
  real = cg.do_to_program
  def spy(ast, renderer):
    calls.append(os.environ.get("PREFILL_SOFTMAX_REDUCE_FUSE"))
    return real(ast, renderer)
  monkeypatch.setattr(cg, "do_to_program", spy)
  return calls


@pytest.mark.parametrize("gate_name", list(LOWERING_GATES_NOT_IN_CACHE_KEY))
def test_flipping_a_not_in_cache_key_gate_forces_a_real_relowering(monkeypatch, gate_name):
  """Each of the three gates trace.py flags as missing from the OLD key must now be part of the key: flipping it
  in-process must trigger a fresh `do_to_program` call, not a stale cache hit from the other setting."""
  assert gate_name in {name for name, _ in PLAN_GATES}, f"{gate_name} must be in PLAN_GATES for the key to cover it"
  calls = _spy_do_to_program(monkeypatch)
  renderer = PythonRenderer(Target("PYTHON"))
  ast = _sink(1)

  monkeypatch.setenv(gate_name, "1")
  getenv.cache_clear()
  to_program(ast, renderer)

  monkeypatch.setenv(gate_name, "0")
  getenv.cache_clear()
  to_program(ast, renderer)

  assert len(calls) == 2, f"second call under a different {gate_name} value was served from the first's cache entry"


def test_returning_to_a_previously_seen_gate_value_is_still_a_real_cache_hit(monkeypatch):
  """The fix must not turn the cache into a no-op: re-using an already-seen gate value must still hit the cache."""
  calls = _spy_do_to_program(monkeypatch)
  renderer = PythonRenderer(Target("PYTHON"))
  ast = _sink(2)

  monkeypatch.setenv("PREFILL_SOFTMAX_REDUCE_FUSE", "1")
  getenv.cache_clear()
  first = to_program(ast, renderer)

  monkeypatch.setenv("PREFILL_SOFTMAX_REDUCE_FUSE", "0")
  getenv.cache_clear()
  to_program(ast, renderer)

  monkeypatch.setenv("PREFILL_SOFTMAX_REDUCE_FUSE", "1")
  getenv.cache_clear()
  third = to_program(ast, renderer)

  assert len(calls) == 2                      # only two DISTINCT gate values were ever seen
  assert third is first                       # the third call, back at the first setting, is a genuine cache hit


def test_cache_key_gate_suffix_is_derived_from_plan_gates_not_a_hand_picked_list():
  """The key must be built from PLAN_GATES (so any gate later added to the plan's inventory is automatically part of
  the cache key too), not from a second hand-maintained list of getenv(...) names."""
  import inspect
  src = inspect.getsource(cg.to_program)
  assert "PLAN_GATES" in src
  for name in LOWERING_GATES_NOT_IN_CACHE_KEY:
    assert f'"{name}"' not in src, f"{name} must not be individually hand-picked in to_program's key construction"
