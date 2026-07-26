"""LR-031 acceptance: an optimizer decision that can be inspected, replayed, and applied exactly once."""
import json, os, subprocess, sys, textwrap
import pytest

from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.plan import (OptimizationPlan, PlanReapplied, ResourceBudget, TargetCapabilities, PLAN_GATES,
                                   PLAN_SCHEMA)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _plan(**kw) -> OptimizationPlan:
  return OptimizationPlan(opts=(Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.LOCAL, 1, 16)),
                          axis_types=("LOOP", "REDUCE"), coalescing_width=4, staging_policy="double_buffer",
                          reduction_mode="warp", budget=ResourceBudget(shared_bytes=32768, max_threads=256,
                                                                       max_upcast=4, max_unroll=8), **kw)

# --------------------------------------------------------------------------- applied exactly once ----
def test_applying_a_plan_twice_is_rejected():
  """Acceptance: applying twice is rejected or proven idempotent. UPCAST/UNROLL/LOCAL are not idempotent --
  applying twice silently doubles a factor, which surfaces as a resource regression phases later."""
  p = _plan()
  once = p.mark_applied()
  assert once.applied and not p.applied          # the original is untouched; plans are frozen
  with pytest.raises(PlanReapplied) as e:
    once.mark_applied()
  assert once.plan_id in str(e.value)            # the error names which plan

def test_is_idempotent_is_honest():
  """Only an empty plan is safe to reapply. This must not optimistically claim otherwise."""
  assert OptimizationPlan().is_idempotent()
  assert not _plan().is_idempotent()

def test_applied_does_not_change_plan_identity():
  """Replaying a plan must produce the same identity, so `applied` is excluded from the digest."""
  p = _plan()
  assert p.plan_id == p.mark_applied().plan_id

# ------------------------------------------------------------------------ serialize and replay ----
def test_round_trips_through_json():
  p = _plan()
  back = OptimizationPlan.from_json(json.loads(json.dumps(p.to_json())))
  assert back.plan_id == p.plan_id
  assert back.opts == p.opts and back.budget == p.budget and back.capabilities == p.capabilities

def test_plan_replays_identically_in_a_clean_subprocess(tmp_path):
  """Acceptance: a plan can be serialized and replayed in a clean subprocess.

  The subprocess shares no memory and no environment overrides, so an identical plan_id means the plan really did
  carry the whole decision rather than depending on ambient state.
  """
  p = _plan()
  path = tmp_path / "plan.json"
  path.write_text(json.dumps(p.to_json()))
  r = subprocess.run([sys.executable, "-c", textwrap.dedent(f"""
    import json
    from tinygrad.codegen.plan import OptimizationPlan
    plan = OptimizationPlan.from_json(json.load(open({str(path)!r})))
    print(plan.plan_id)
  """)], cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT}, capture_output=True, text=True)
  assert r.returncode == 0, r.stderr[-1500:]
  assert r.stdout.strip() == p.plan_id

def test_unknown_schema_is_refused():
  bad = _plan().to_json(); bad["schema"] = "something.else.v9"
  with pytest.raises(ValueError, match="schema"):
    OptimizationPlan.from_json(bad)

# ------------------------------------------------------------------- env is read once, up front ----
def test_gates_are_captured_once_into_the_plan():
  """Acceptance: environment variables are parsed once into a plan and not read deep inside transformations."""
  os.environ["WARP_REDUCE_LOWERING"] = "1"
  try:
    p = OptimizationPlan.from_env()
    assert p.gate("WARP_REDUCE_LOWERING") == "1"
    os.environ["WARP_REDUCE_LOWERING"] = "0"      # the world changes underneath...
    assert p.gate("WARP_REDUCE_LOWERING") == "1"  # ...the plan does not. That is the whole point.
  finally:
    os.environ.pop("WARP_REDUCE_LOWERING", None)

def test_unset_gates_record_their_real_default():
  p = OptimizationPlan.from_env()
  assert p.gate("PREFILL_SOFTMAX_REDUCE_FUSE") == "1"   # default-ON, and it must say so
  assert p.gate("NOOPT") == "0"
  assert p.gate("NOT_A_GATE") is None

def test_gate_set_matches_the_declared_inventory():
  p = OptimizationPlan.from_env()
  assert {k for k, _ in p.gates} == {name for name, _ in PLAN_GATES}

def test_a_gate_change_changes_the_plan_id():
  """Two plans that would lower differently must not share an identity -- this is what the to_program cache key
  fails to do for PREFILL_SOFTMAX_REDUCE_FUSE, UNSAFE_DISABLE_MASK and REGALLOC_ADDR_REMAT."""
  before = OptimizationPlan.from_env()
  os.environ["PREFILL_SOFTMAX_REDUCE_FUSE"] = "0"
  try:
    after = OptimizationPlan.from_env()
    assert before.plan_id != after.plan_id
  finally:
    os.environ.pop("PREFILL_SOFTMAX_REDUCE_FUSE", None)

# --------------------------------------------------------------------------- target capabilities ----
def test_capabilities_come_from_the_real_renderer():
  from tinygrad import Device
  caps = TargetCapabilities.from_renderer(Device["CPU"].renderer)
  assert caps.target and isinstance(caps.has_local, bool) and caps.shared_max > 0

def test_capabilities_survive_a_round_trip():
  from tinygrad import Device
  caps = TargetCapabilities.from_renderer(Device["CPU"].renderer)
  assert TargetCapabilities.from_json(caps.to_json()) == caps

def test_plan_is_frozen():
  p = _plan()
  with pytest.raises(Exception):
    p.coalescing_width = 8            # type: ignore[misc]

def test_schema_is_declared():
  assert _plan().to_json()["schema"] == PLAN_SCHEMA
