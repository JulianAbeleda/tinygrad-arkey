"""LR-031 acceptance: an optimizer decision that can be inspected, replayed, and applied exactly once."""
import json, os, subprocess, sys, textwrap
import pytest

from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.plan import (OptimizationPlan, PlanReapplied, PlanRejected, ResourceBudget, TargetCapabilities,
                                   PLAN_GATES, PLAN_SCHEMA)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _plan(**kw) -> OptimizationPlan:
  base = dict(opts=(Opt(OptOps.UPCAST, 0, 4), Opt(OptOps.LOCAL, 1, 16)),
              axis_types=("LOOP", "REDUCE"), coalescing_width=4, staging_policy="double_buffer",
              reduction_mode="warp", budget=ResourceBudget(shared_bytes=32768, max_threads=256,
                                                            max_upcast=4, max_unroll=8),
              capabilities=TargetCapabilities(has_threads=True))
  base.update(kw)
  return OptimizationPlan(**base)

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

# --------------------------------------------------------------------------- LR-051: named fields from real gates ----
def test_reduction_mode_is_derived_from_warp_reduce_lowering_gate():
  os.environ.pop("WARP_REDUCE_LOWERING", None)
  try:
    assert OptimizationPlan.from_env().reduction_mode == "default"
    os.environ["WARP_REDUCE_LOWERING"] = "1"
    assert OptimizationPlan.from_env().reduction_mode == "warp"
  finally:
    os.environ.pop("WARP_REDUCE_LOWERING", None)

def test_coalescing_width_is_derived_from_coalesced_load_lowering_gate():
  os.environ.pop("COALESCED_LOAD_LOWERING", None)
  try:
    assert OptimizationPlan.from_env().coalescing_width == 0
    os.environ["COALESCED_LOAD_LOWERING"] = "1"
    assert OptimizationPlan.from_env().coalescing_width == 4
  finally:
    os.environ.pop("COALESCED_LOAD_LOWERING", None)

def test_vectorization_policy_is_derived_from_v_dot2_lowering_gate():
  os.environ.pop("V_DOT2_LOWERING", None)
  try:
    assert OptimizationPlan.from_env().vectorization_policy == "default"
    os.environ["V_DOT2_LOWERING"] = "1"
    assert OptimizationPlan.from_env().vectorization_policy == "dot2"
  finally:
    os.environ.pop("V_DOT2_LOWERING", None)

def test_fields_with_no_real_flag_are_left_unset_by_from_env():
  """staging_policy stays at its structural default and score_split_mode/quant_load_transform/rope_load_transform
  stay None: no env var in this codebase currently drives any of them (see the comments on OptimizationPlan)."""
  p = OptimizationPlan.from_env()
  assert p.score_split_mode is None
  assert p.quant_load_transform is None
  assert p.rope_load_transform is None
  assert p.staging_policy == "default"

def test_from_env_kwarg_overrides_a_derived_field():
  os.environ["WARP_REDUCE_LOWERING"] = "1"
  try:
    p = OptimizationPlan.from_env(reduction_mode="explicit_override")
    assert p.reduction_mode == "explicit_override"
  finally:
    os.environ.pop("WARP_REDUCE_LOWERING", None)

def test_new_fields_round_trip_through_json():
  p = _plan(score_split_mode="live_split", quant_load_transform=True, rope_load_transform=False,
            vectorization_policy="dot2")
  back = OptimizationPlan.from_json(json.loads(json.dumps(p.to_json())))
  assert back.plan_id == p.plan_id
  assert (back.score_split_mode, back.quant_load_transform, back.rope_load_transform, back.vectorization_policy) == \
         ("live_split", True, False, "dot2")

# --------------------------------------------------------------------------------------- LR-051: validate() ----
def test_validate_accepts_a_consistent_plan():
  _plan().validate()   # must not raise

def test_validate_rejects_a_coalescing_width_the_target_cannot_support():
  """Acceptance: a rejected plan fails before launching a GPU kernel -- validate() is the choke point, called before
  anything downstream (a pass, a compile, a launch) is allowed to consume the plan."""
  p = _plan(capabilities=TargetCapabilities(supports_float4=False))
  with pytest.raises(PlanRejected, match="float4"):
    p.validate()

def test_validate_rejects_coalescing_without_local_memory():
  p = _plan(capabilities=TargetCapabilities(has_local=False))
  with pytest.raises(PlanRejected, match="has_local"):
    p.validate()

def test_validate_rejects_warp_reduction_without_thread_support():
  p = _plan(coalescing_width=0, reduction_mode="warp", capabilities=TargetCapabilities(has_threads=False))
  with pytest.raises(PlanRejected, match="has_threads"):
    p.validate()

def test_validate_accepts_warp_reduction_when_target_has_threads():
  p = _plan(coalescing_width=0, reduction_mode="warp", capabilities=TargetCapabilities(has_threads=True))
  p.validate()   # must not raise

def test_validate_rejects_coalescing_width_exceeding_the_upcast_budget():
  p = _plan(budget=ResourceBudget(max_upcast=2))
  with pytest.raises(PlanRejected, match="max_upcast"):
    p.validate()

def test_a_rejected_plan_fails_before_any_kernel_launch():
  """Direct acceptance-criterion test: build a plan that validate() rejects, and prove the launch step is never
  reached when validate() is called first, as callers are expected to do."""
  launched = []
  def launch_kernel(plan: OptimizationPlan):
    launched.append(plan)   # would be an actual GPU dispatch in a real caller
  p = _plan(capabilities=TargetCapabilities(supports_float4=False))
  with pytest.raises(PlanRejected):
    p.validate()
    launch_kernel(p)   # unreachable
  assert launched == []


# --------------------------------------------------------------------------- LR-019: readers, not just values ----
# The plan is NOT the single reader of these variables today. These tests hold the line that the module at least
# describes the real readers accurately, since the to_program cache key is built from that description.

def test_gate_readers_match_the_real_call_sites():
  """GATE_READERS records HOW each gate is read, and observed_gate_values() depends on that being exact.

  It has to be exact rather than approximately right because getenv is @functools.cache'd and lru_cache keys on the
  argument tuple: getenv("X"), getenv("X", 0) and getenv("X", "0") are three separate entries frozen at three
  different times. Reading a gate "like the pass does" means matching its ARITY, not just its default.
  """
  import pathlib, re
  from tinygrad.codegen.plan import GATE_READERS
  root = pathlib.Path(ROOT) / "tinygrad"
  for name, reader in GATE_READERS.items():
    getenv_defaults, ctxvar = set(), False
    for p in root.rglob("*.py"):
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


def test_observed_values_are_what_a_pass_sees_not_what_environ_says():
  """The bug this exists to prevent, stated as a test: a gate flipped after first read must NOT change the observed
  value, because it does not change what the pass does either."""
  from tinygrad.helpers import getenv
  from tinygrad.codegen.plan import observed_gate_value
  os.environ.pop("UNSAFE_DISABLE_MASK", None)
  before = observed_gate_value("UNSAFE_DISABLE_MASK")
  assert before == getenv("UNSAFE_DISABLE_MASK", 0)
  os.environ["UNSAFE_DISABLE_MASK"] = "1"
  try:
    assert observed_gate_value("UNSAFE_DISABLE_MASK") == before, \
      "observed_gate_value tracked os.environ instead of the frozen value the pass reads"
    assert OptimizationPlan.from_env().gate("UNSAFE_DISABLE_MASK") == "1", \
      "from_env is documented to read live env; if that changed, the cache-key comment needs revisiting"
  finally:
    os.environ.pop("UNSAFE_DISABLE_MASK", None)


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


def test_lowering_gates_are_in_the_cache_key():
  """trace.py's LOWERING_GATES_NOT_IN_CACHE_KEY is historical; its name is now the opposite of the truth."""
  from tinygrad.uop.trace import LOWERING_GATES_NOT_IN_CACHE_KEY
  assert set(LOWERING_GATES_NOT_IN_CACHE_KEY) <= {n for n, _ in PLAN_GATES}


def test_decode_fast_exp2_has_no_reader_and_says_so():
  """Kept in the inventory rather than deleted: the record that someone believed this gated lowering is worth more
  than the line it costs. If a reader ever appears, GATE_READERS must be updated deliberately."""
  from tinygrad.codegen.plan import GATE_READERS, observed_gate_value
  assert GATE_READERS["DECODE_FAST_EXP2"] == ("none",)
  assert observed_gate_value("DECODE_FAST_EXP2") is None
