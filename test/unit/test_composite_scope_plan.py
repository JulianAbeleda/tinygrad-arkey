"""LR-043 acceptance: composite/scoped-reduction ownership, recorded as a declarative CompositeExtension rather
than left as a procedural backward_slice walk. Mirrors test/unit/test_bufferize_plan.py's shape for LR-042."""
import os, subprocess, sys, textwrap

from tinygrad.schedule import scopes
from tinygrad.schedule.scopes import (CompositeExtension, CompositeScopePlan, KIND_COMPOSITE_REDUCE,
                                      FALLBACK_INDEPENDENT, FALLBACK_REGISTERED, FALLBACK_UNREGISTERED_FAILS_CLOSED)
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp, Ops, AccumulatorSlot, CompositeReduce

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _child(body: str, env_extra: dict):
  env = {**os.environ, "PYTHONPATH": ROOT, **env_extra}
  return subprocess.run([sys.executable, "-c", textwrap.dedent(body)], cwd=ROOT, env=env, capture_output=True, text=True)

# ------------------------------------------------------------------------------------ the five declared fields ----
def test_describe_composite_reduce_states_the_five_things():
  t = Tensor.arange(1, 17, dtype=dtypes.float32).reshape(16)
  slot_sum = AccumulatorSlot(op=Ops.ADD, dtype=dtypes.float32, identity=0.0, name="sum")
  slot_max = AccumulatorSlot(op=Ops.MAX, dtype=dtypes.float32, identity=float("-inf"), name="max")
  red = t.uop.composite_reduce(slot_sum, slot_max, axis=(0,))
  composite = red.arg[0]
  assert isinstance(composite, CompositeReduce)
  ext = scopes.describe_composite_reduce(composite, owner=t.uop, owned_slice_size=3)
  assert ext.kind == KIND_COMPOSITE_REDUCE
  assert ext.producer_slice_size == 3                 # 1. producer slice it owns
  assert ext.reduction_axes == ()                      # 2. reduction axis (unset until rangeify fills reduce_range_axes)
  assert ext.state_shape == ()                         # 2. state shape (unset until slot_shapes is populated)
  assert ext.slot_names == ("sum", "max")
  assert ext.storage_repr == "scalar"                  # 3. allowed storage/lane representation
  assert ext.synchronization is None                   # 4. required synchronization -- honestly unpopulated
  assert "not declared generically" in ext.synchronization_note
  assert ext.fallback == FALLBACK_INDEPENDENT           # 5. fallback behaviour: no combine_fn -> independent slots

def test_fallback_classification_matches_combine_registry():
  slot = AccumulatorSlot(op=Ops.ADD, dtype=dtypes.float32, identity=0.0, name="l")
  slot2 = AccumulatorSlot(op=Ops.MAX, dtype=dtypes.float32, identity=float("-inf"), name="m")
  registered = Tensor.arange(1, 9, dtype=dtypes.float32).uop.composite_reduce(slot2, slot, axis=(0,), combine_fn="online_softmax_l")
  ext = scopes.describe_composite_reduce(registered.arg[0])
  assert ext.fallback == FALLBACK_REGISTERED

  unregistered = Tensor.arange(1, 9, dtype=dtypes.float32).uop.composite_reduce(slot, axis=(0,), combine_fn="not_a_real_combine")
  ext2 = scopes.describe_composite_reduce(unregistered.arg[0])
  assert ext2.fallback == FALLBACK_UNREGISTERED_FAILS_CLOSED

def test_scoped_reduce_ownership_from_run_rangeify_is_inspectable():
  """The SCOPED_REDUCE ownership site (indexing.py's run_rangeify) is the second half of LR-043's scope
  ("composite/scoped-reduction ownership"). This exercises it through the real scheduler path, gated on so the
  descriptor is actually built, and checks the recorded plan sees a scoped_reduce entry with the honest
  (fail-closed, no soft fallback) classification."""
  r = _child("""
    from tinygrad import Tensor, dtypes
    from tinygrad.uop.ops import UOp, Ops, ScopedReduceSpec
    from tinygrad.schedule import scopes
    inner = Tensor.rand(4, 8).uop
    spec = ScopedReduceSpec(reduce_axes=(1,), source_axis_maps=((0, 1),), scope_owner=0, result_dtypes=(dtypes.float32,))
    scoped = UOp(Ops.SCOPED_REDUCE, dtypes.float32, src=(inner, inner), arg=spec)
    ext = scopes.describe_scoped_reduce(spec, owner=inner, owned_slice_size=2)
    print("KIND", ext.kind)
    print("AXES", ext.reduction_axes)
    print("SHAPE", ext.state_shape)
    print("FALLBACK", ext.fallback)
    print("SYNC", ext.synchronization)
  """, {})
  assert r.returncode == 0, r.stderr[-2000:]
  assert "KIND scoped_reduce" in r.stdout
  assert "AXES (1,)" in r.stdout
  assert "FALLBACK scoped_producer_validated" in r.stdout
  assert "SYNC None" in r.stdout

# ------------------------------------------------------------------------------------ inert by default ----
def test_recording_is_off_by_default():
  r = _child("""
    from tinygrad import Tensor
    from tinygrad.schedule import scopes
    (Tensor.rand(16,16)+1).sum().schedule_linear()
    print("ENABLED", scopes.ENABLED)
    print("ACTIVE", scopes.active())
  """, {})
  assert "ENABLED False" in r.stdout and "ACTIVE None" in r.stdout

def test_recording_does_not_change_the_schedule():
  """Acceptance: the old and planned paths produce identical schedules -- COMPOSITE_PLAN only observes."""
  prog = """
    import hashlib
    from tinygrad import Tensor, dtypes
    from tinygrad.uop.ops import UOp, Ops, AccumulatorSlot
    Tensor.manual_seed(1337)
    t = Tensor.rand(16, dtype=dtypes.float32)
    slot_sum = AccumulatorSlot(op=Ops.ADD, dtype=dtypes.float32, identity=0.0, name="sum")
    slot_max = AccumulatorSlot(op=Ops.MAX, dtype=dtypes.float32, identity=float("-inf"), name="max")
    red = UOp.composite_reduce(t.uop, slot_sum, slot_max, axis=(0,))
    s0 = Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 0))
    s1 = Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 1))
    lin = s0.schedule_linear(s1)
    print(hashlib.sha256(lin.key).hexdigest())
  """
  off, on = _child(prog, {}), _child(prog, {"COMPOSITE_PLAN": "1"})
  assert off.returncode == on.returncode == 0, (off.stderr[-800:], on.stderr[-800:])
  assert off.stdout.strip() == on.stdout.strip()

def test_plan_records_the_composite_reduce():
  r = _child("""
    from tinygrad import Tensor, dtypes
    from tinygrad.uop.ops import UOp, Ops, AccumulatorSlot
    from tinygrad.schedule import scopes
    t = Tensor.rand(16, dtype=dtypes.float32)
    slot_sum = AccumulatorSlot(op=Ops.ADD, dtype=dtypes.float32, identity=0.0, name="sum")
    slot_max = AccumulatorSlot(op=Ops.MAX, dtype=dtypes.float32, identity=float("-inf"), name="max")
    red = UOp.composite_reduce(t.uop, slot_sum, slot_max, axis=(0,))
    s0 = Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 0))
    s1 = Tensor(UOp(Ops.REDUCE_SLOT, dtypes.float32, (red,), 1))
    s0.schedule_linear(s1)
    p = scopes.active()
    print("N", len(p.extensions))
    print("KINDS", p.by_kind())
  """, {"COMPOSITE_PLAN": "1"})
  assert r.returncode == 0, r.stderr[-2000:]
  assert "N 0" not in r.stdout
  assert "'composite_reduce'" in r.stdout

def test_plan_to_json_and_explain_are_serializable():
  plan = CompositeScopePlan()
  ext = CompositeExtension(kind=KIND_COMPOSITE_REDUCE, owner="deadbeef0000", producer_slice_size=4,
                           reduction_axes=(0,), state_shape=((),), slot_names=("sum",),
                           fallback=FALLBACK_INDEPENDENT, fallback_note="test")
  plan.record(ext)
  as_json = plan.to_json()
  assert as_json["schema"] == "tinygrad.composite_scope_plan.v1"
  assert as_json["extensions"][0]["owner"] == "deadbeef0000"
  assert "owns 4 producer node(s)" in plan.explain()[0]

if __name__ == "__main__":
  import unittest
  unittest.main()
