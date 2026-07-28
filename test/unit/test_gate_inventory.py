"""LR-019: the lowering gate inventory must match how the gates are really read.

What remains of the old test_optimization_plan.py. The OptimizationPlan machinery it covered was deleted in the
post-refactor cut -- nothing outside that file ever constructed one. These tests stayed because they gate live
behaviour: PLAN_GATES and observed_gate_values() build the `to_program` cache key, and if that inventory drifts
from the tree, the key starts describing a compile that did not happen.
"""
import os
import pytest   # noqa: F401  (kept for parametrised additions)

from tinygrad.codegen.plan import PLAN_GATES

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_observed_values_are_what_a_pass_sees_not_what_environ_says():
  """The defect this inventory exists to prevent: a gate flipped after first read must NOT change the observed
  value, because it does not change what the pass does either -- getenv froze it."""
  from tinygrad.helpers import getenv
  from tinygrad.codegen.plan import observed_gate_value
  os.environ.pop("UNSAFE_DISABLE_MASK", None)
  before = observed_gate_value("UNSAFE_DISABLE_MASK")
  assert before == getenv("UNSAFE_DISABLE_MASK", 0)
  os.environ["UNSAFE_DISABLE_MASK"] = "1"
  try:
    assert observed_gate_value("UNSAFE_DISABLE_MASK") == before, \
      "observed_gate_value tracked os.environ instead of the frozen value the pass reads"
  finally:
    os.environ.pop("UNSAFE_DISABLE_MASK", None)


def test_gate_set_matches_the_declared_inventory():
  from tinygrad.codegen.plan import GATE_READERS
  assert {n for n, _ in PLAN_GATES} == set(GATE_READERS), "PLAN_GATES and GATE_READERS must cover the same gates"


def test_gate_readers_match_the_real_call_sites():
  """GATE_READERS records HOW each gate is read, and observed_gate_values() depends on that being exact.

  It has to be exact rather than approximately right because getenv is @functools.cache'd and lru_cache keys on the
  argument tuple: getenv("X"), getenv("X", 0) and getenv("X", "0") are three separate entries frozen at three
  different times. Reading a gate "like the pass does" means matching its ARITY, not just its default.
  """
  import pathlib, re
  from tinygrad.codegen.plan import GATE_READERS
  # Scans extra/qk as well as tinygrad/: those builders construct ASTs that feed to_program, so a gate read there
  # is as load-bearing as one read in codegen. An earlier version scanned tinygrad/ only, which let
  # DECODE_FAST_EXP2 be recorded as having no reader while extra/qk/flash_common.py:15 read it.
  roots = [pathlib.Path(ROOT) / "tinygrad", pathlib.Path(ROOT) / "extra" / "qk"]
  for name, reader in GATE_READERS.items():
    getenv_defaults, ctxvar = set(), False
    for p in [f for r in roots for f in r.rglob("*.py")]:
      if p.name in ("plan.py", "trace.py"): continue
      text = p.read_text()
      for m in re.finditer(r'getenv\(\s*"' + re.escape(name) + r'"\s*(?:,\s*([^)]*))?\)', text):
        getenv_defaults.add((m.group(1) or "").strip())
      if re.search(r'ContextVar\(\s*"' + re.escape(name) + r'"', text): ctxvar = True

    if reader[0] == "contextvar":
      assert ctxvar, f"{name} declared contextvar but no ContextVar({name!r}) exists"
    elif reader[0] == "none":
      assert not getenv_defaults and not ctxvar, f"{name} declared unread but has readers {getenv_defaults}"
    else:
      expected = "" if len(reader) == 1 else str(reader[1])
      assert getenv_defaults == {expected}, \
        f"{name}: declared getenv arity {reader!r} but call sites use defaults {sorted(getenv_defaults)!r}"


def test_the_cache_key_no_longer_moves_without_the_program_moving():
  """Regression test for the defect LR-051 introduced: the to_program key was built from a LIVE os.environ read
  while the passes read frozen getenv, so flipping a gate created a second cache entry for a byte-identical
  program."""
  from tinygrad import Tensor, Device
  from tinygrad.codegen import to_program, to_program_cache
  from tinygrad.uop.ops import Ops
  os.environ.pop("UNSAFE_DISABLE_MASK", None)
  lin = ((Tensor.rand(32, 32, device="CPU") + 1.0) * 2.0).sum().schedule_linear()
  ast = [u.src[0] for u in lin.src if u.op is Ops.CALL and u.src[0].op is Ops.SINK][0]
  ren = Device["CPU"].renderer
  p1 = to_program(ast, ren); n1 = len(to_program_cache)
  os.environ["UNSAFE_DISABLE_MASK"] = "1"
  try:
    p2 = to_program(ast, ren)
    assert p1.key == p2.key, "same frozen gates must give the same program"
    assert len(to_program_cache) == n1, "a second cache entry was created for an identical program"
  finally:
    os.environ.pop("UNSAFE_DISABLE_MASK", None)


# ---------------------------------------------------- LR-019b: the inventory must not fall behind the tree ----

# Gates read inside codegen/renderer that are deliberately NOT in PLAN_GATES, each with the reason it cannot
# change a compiled program. This list is the whole escape hatch: anything not here and not in PLAN_GATES fails.
_NOT_IN_KEY_DEBUG = {          # emit logs/telemetry only; do not alter the graph or the emitted source
  "COALESCED_LOAD_DEBUG", "DEBUG_LINEARIZE", "NOSKIP", "SCHED_LIST_REPORT", "SCHED_MODULO_PROBE", "SQTT_EVENT",
  "REGALLOC_DEBUG", "REGALLOC_DEBUG_DETAIL", "REGALLOC_DEBUG_END_DETAIL", "REGALLOC_DEBUG_LOOP_LIVE",
  "REGALLOC_DEBUG_NOSPILL", "REGALLOC_DEBUG_PRESSURE", "REGALLOC_DEBUG_REMAT", "REGALLOC_DEBUG_REMAT_LIMIT",
  "REGALLOC_DEBUG_SPILLS", "REGALLOC_DEBUG_WINDOW", "REGALLOC_DEBUG_WINDOW_CENTER", "SCHED_UNROLL_DEBUG",
}
_NOT_IN_KEY_OTHER = {
  "LOWER_DISK_CACHE",          # selects the caching mechanism itself; cannot be part of its own key
}


_NOT_IN_KEY_DEBUG = {          # emit logs/telemetry only; do not alter the graph or the emitted source
  "COALESCED_LOAD_DEBUG", "DEBUG_LINEARIZE", "NOSKIP", "SCHED_LIST_REPORT", "SCHED_MODULO_PROBE", "SQTT_EVENT",
  "REGALLOC_DEBUG", "REGALLOC_DEBUG_DETAIL", "REGALLOC_DEBUG_END_DETAIL", "REGALLOC_DEBUG_LOOP_LIVE",
  "REGALLOC_DEBUG_NOSPILL", "REGALLOC_DEBUG_PRESSURE", "REGALLOC_DEBUG_REMAT", "REGALLOC_DEBUG_REMAT_LIMIT",
  "REGALLOC_DEBUG_SPILLS", "REGALLOC_DEBUG_WINDOW", "REGALLOC_DEBUG_WINDOW_CENTER", "SCHED_UNROLL_DEBUG",
}
_NOT_IN_KEY_OTHER = {
  "LOWER_DISK_CACHE",          # selects the caching mechanism itself; cannot be part of its own key
}

def test_no_codegen_gate_is_missing_from_the_inventory():
  """Every getenv gate in codegen/renderer is in PLAN_GATES or explicitly excused.

  This is the test that was missing when LOWERING_GATES_NOT_IN_CACHE_KEY got relabelled "historical". Deriving
  the cache key from PLAN_GATES removes one hand-maintained list only if PLAN_GATES is itself complete; without
  this, it silently swaps one incomplete list for another. Scoped to codegen/renderer because that is
  do_to_program's territory -- scheduler-stage gates run before it and are already captured by `ast.key`, which
  is the first element of the key.
  """
  import pathlib, re
  plan = {n for n, _ in PLAN_GATES}
  found: dict[str, str] = {}
  for d in ("tinygrad/codegen", "tinygrad/renderer"):
    for p in (pathlib.Path(ROOT) / d).rglob("*.py"):
      if p.name == "plan.py": continue
      for m in re.finditer(r'getenv\(\s*"([A-Z][A-Z0-9_]*)"', p.read_text()):
        found.setdefault(m.group(1), str(p.relative_to(ROOT)))
  missing = {g: f for g, f in found.items() if g not in plan and g not in _NOT_IN_KEY_DEBUG and g not in _NOT_IN_KEY_OTHER}
  assert missing == {}, (
    "gates read during codegen but absent from PLAN_GATES and not excused: "
    + ", ".join(f"{g} ({f})" for g, f in sorted(missing.items()))
    + ". Add to PLAN_GATES + GATE_READERS, or excuse it with a reason it cannot change a compiled program.")


def test_the_excuse_lists_are_not_a_dumping_ground():
  """Every excused gate must still exist. A stale name here is a slot where a real gate could hide."""
  import pathlib, re
  src = "\n".join(p.read_text() for d in ("tinygrad/codegen", "tinygrad/renderer")
                  for p in (pathlib.Path(ROOT) / d).rglob("*.py"))
  live = set(re.findall(r'getenv\(\s*"([A-Z][A-Z0-9_]*)"', src))
  stale = (_NOT_IN_KEY_DEBUG | _NOT_IN_KEY_OTHER) - live
  assert stale == set(), f"excused gates that no longer exist: {sorted(stale)}"
