"""LR-011 acceptance: pass invariants attribute a bad graph to the pass that produced it.

The value here is attribution. These violations already fail eventually -- as a confusing AttributeError, or a crash
several passes downstream. Failing at the pass that introduced the state is what turns an afternoon into five minutes.
"""
import os, subprocess, sys, textwrap
import pytest

from tinygrad.uop import invariants
from tinygrad.uop.invariants import PassInvariantError, check_graph, check_pass

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _child(body: str, env_extra: dict):
  env = {**os.environ, "PYTHONPATH": ROOT, **env_extra}
  return subprocess.run([sys.executable, "-c", textwrap.dedent(body)], cwd=ROOT, env=env, capture_output=True, text=True)

# ------------------------------------------------------------------------------ disabled by default ----
def test_disabled_by_default():
  assert invariants.ENABLED is False or os.environ.get("LOWER_CHECK") not in (None, "0")

def test_real_lowering_passes_all_invariants():
  """A check that fires on correct code is worse than no check -- it gets switched off and stays off."""
  r = _child("""
    from tinygrad import Tensor
    ((Tensor.rand(64,64)+1.0)*2.0).sum().schedule_linear()
    (Tensor.rand(64,64) @ Tensor.rand(64,64)).sum().schedule_linear()
    Tensor.rand(16,64).softmax(axis=-1).sum().schedule_linear()
    print("CLEAN")
  """, {"LOWER_CHECK": "1"})
  assert r.returncode == 0 and "CLEAN" in r.stdout, r.stderr[-1500:]

def test_checks_do_not_alter_generated_code():
  """Enabling the checks must not change lowering output."""
  prog = """
    import hashlib
    from tinygrad import Tensor
    lin = ((Tensor.rand(64,64)+1.0)*2.0).sum().schedule_linear()
    print(hashlib.sha256(lin.key).hexdigest())
  """
  off = _child(prog, {})
  on = _child(prog, {"LOWER_CHECK": "1"})
  assert off.returncode == on.returncode == 0
  assert off.stdout.strip() == on.stdout.strip()

# ------------------------------------------------------------------------------------- attribution ----
def test_invalid_graph_is_attributed_to_the_pass_that_produced_it():
  """Acceptance: an invalid hand-constructed graph fails AT the pass that introduced the invalid state."""
  class FakeNode:
    op, dtype, arg = "NOT_AN_OPS_MEMBER", None, None
  class FakeSink:
    def toposort(self): return [FakeNode()]
  invariants.reset(reread_env=False)
  prev, invariants.ENABLED = invariants.ENABLED, True
  try:
    with pytest.raises(PassInvariantError) as e:
      check_pass("the guilty pass", FakeSink())
    assert e.value.pass_name == "the guilty pass"          # names the culprit, not a downstream victim
    assert "NOT_AN_OPS_MEMBER" in str(e.value)
  finally:
    invariants.ENABLED = prev

def test_hinted_contiguous_is_only_a_violation_at_the_codegen_stage():
  """The hazard is stage-specific: hints on CONTIGUOUS are correct during rangeify and fatal at codegen.

  rangeify_codegen's first rule assigns ctx.opts, and that matcher runs with LocalAddBufferContext at
  schedule/rangeify.py:920 but a bare itertools.count at codegen/__init__.py:147, which rejects attribute
  assignment.
  """
  from tinygrad.uop import Ops
  class Hinted:
    op, dtype, arg = Ops.CONTIGUOUS, "f32", ("opt_a", "opt_b")
  class Sink:
    def toposort(self): return [Hinted()]
  assert check_graph(Sink(), stage="rangeify") == []           # legitimate here
  bad = check_graph(Sink(), stage="codegen")
  assert bad and "itertools.count" in bad[0]                    # and fatal here, with the reason attached

def test_itertools_count_really_rejects_the_assignment():
  """Pins the fact the invariant is built on, so the check cannot outlive its own justification."""
  import itertools
  with pytest.raises(AttributeError):
    itertools.count(0).opts = ("x",)

# ------------------------------------------------------------------------------------------- units ----
def test_unwalkable_graph_is_reported_not_raised_through():
  class Exploding:
    def toposort(self): raise RuntimeError("boom")
  out = check_graph(Exploding())
  assert len(out) == 1 and "not walkable" in out[0] and "boom" in out[0]

def test_check_pass_is_a_noop_when_disabled():
  class FakeNode:
    op, dtype, arg = "NOT_AN_OPS_MEMBER", None, None
  class Sink:
    def toposort(self): return [FakeNode()]
  prev, invariants.ENABLED = invariants.ENABLED, False
  try:
    check_pass("whatever", Sink())    # must not raise
  finally:
    invariants.ENABLED = prev

def test_violations_are_capped_so_one_bad_pass_cannot_flood():
  from tinygrad.uop import Ops
  class Bad:
    op, dtype, arg = "NOT_AN_OPS_MEMBER", None, None
  class Sink:
    def toposort(self): return [Bad() for _ in range(50)]
  assert len(check_graph(Sink())) <= 4
