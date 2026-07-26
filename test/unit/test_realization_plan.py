"""LR-030 acceptance: the materialization decision, recorded as a result rather than lost in control flow."""
import os, subprocess, sys, textwrap
import pytest

from tinygrad.schedule import plan
from tinygrad.schedule.plan import (RealizationDecision, RealizationPlan, FORCED_ALWAYS_RUN, FORCED_COST,
                                    FORCED_RECOMPUTE_HOSTILE, INLINED)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _child(body: str, env_extra: dict):
  env = {**os.environ, "PYTHONPATH": ROOT, **env_extra}
  return subprocess.run([sys.executable, "-c", textwrap.dedent(body)], cwd=ROOT, env=env, capture_output=True,
                        text=True)

# ------------------------------------------------------------------------------------ inert by default ----
def test_recording_is_off_by_default():
  r = _child("""
    from tinygrad import Tensor
    from tinygrad.schedule import plan
    (Tensor.rand(16,16)+1).sum().schedule_linear()
    print("ENABLED", plan.ENABLED)
    print("ACTIVE", plan.active())
  """, {})
  assert "ENABLED False" in r.stdout and "ACTIVE None" in r.stdout

def test_recording_does_not_change_the_schedule():
  """Acceptance: the old and planned paths produce identical schedules."""
  prog = """
    import hashlib
    from tinygrad import Tensor
    Tensor.manual_seed(1337)
    lin = ((Tensor.rand(64,64)+1.0)*2.0).sum().schedule_linear()
    print(hashlib.sha256(lin.key).hexdigest())
  """
  off, on = _child(prog, {}), _child(prog, {"REALIZE_PLAN": "1"})
  assert off.returncode == on.returncode == 0, (off.stderr[-800:], on.stderr[-800:])
  assert off.stdout.strip() == on.stdout.strip()

# --------------------------------------------------------- the case that motivated the refactor ----
def test_plan_explains_the_producer_reduce_case():
  """Acceptance: the plan explains the producer/reduce case that motivated this refactor.

  Gumbel-max sampling: a wide producer carrying transcendentals feeds an argmax, which lowers to a
  low-parallelism reduce. remove_bufferize's own comment records 128 independent outputs at REDUCE trip 1187.
  The plan must reconstruct that reasoning from the recorded decision, not from the comment.
  """
  r = _child("""
    from tinygrad import Tensor
    from tinygrad.schedule import plan
    logits = Tensor.rand(1, 151936)
    ((logits - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1)).schedule_linear()
    p = plan.active()
    hostile = [d for d in p.decisions if d.reason == "recompute_hostile_low_parallelism"]
    assert hostile, "the cost gate did not fire on the motivating case"
    d = hostile[0]
    print("HOSTILE_OPS", ",".join(d.hostile_ops))
    print("PARALLELISM", d.consumer_parallelism)
    print("TRIP", d.consumer_trip)
    print("EXPLAIN", d.explain())
  """, {"REALIZE_PLAN": "1"})
  assert r.returncode == 0, r.stderr[-1500:]
  assert "LOG2" in r.stdout                       # the transcendental the gate keys on
  assert "PARALLELISM 128" in r.stdout            # matches the measured pathological case
  assert "TRIP 1187" in r.stdout
  assert "too few to hide a serialized recompute" in r.stdout

def test_the_four_questions_are_answerable():
  """The scope names four things the plan must state; each has an accessor."""
  r = _child("""
    from tinygrad import Tensor
    from tinygrad.schedule import plan
    logits = Tensor.rand(1, 151936)
    ((logits - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1)).schedule_linear()
    p = plan.active()
    print("MATERIALIZED", len(p.materialized_producers()))
    print("FORCED_CAUSES", sorted(p.forced()))
    print("DECISIONS", len(p.decisions))
  """, {"REALIZE_PLAN": "1"})
  assert r.returncode == 0, r.stderr[-1500:]
  assert "recompute_hostile_low_parallelism" in r.stdout
  assert int(r.stdout.split("MATERIALIZED")[1].split()[0]) > 0

# ------------------------------------------------------------------------------------------- units ----
def _d(producer, materialized, reason, *, parallelism=None, trip=None, hostile_ops=()):
  """Mirrors plan.record()'s signature so tests read the way call sites do."""
  return RealizationDecision(producer, materialized, reason, consumer_parallelism=parallelism,
                             consumer_trip=trip, hostile_ops=hostile_ops)

def test_split_decisions_finds_producers_whose_fate_differs_by_consumer():
  """The refactor exists because one producer can be worth inlining into a wide consumer and ruinous in a narrow
  one. A plan that could not surface that would be missing the point."""
  p = RealizationPlan()
  p.record(_d("wide_producer", False, INLINED, parallelism=131072, trip=4096))
  p.record(_d("wide_producer", True, FORCED_RECOMPUTE_HOSTILE, parallelism=128, trip=1187, hostile_ops=("LOG2",)))
  p.record(_d("boring", True, FORCED_ALWAYS_RUN))
  split = p.split_decisions()
  assert list(split) == ["wide_producer"] and len(split["wide_producer"]) == 2

def test_forced_groups_by_cause():
  p = RealizationPlan()
  p.record(_d("a", True, FORCED_ALWAYS_RUN)); p.record(_d("b", True, FORCED_COST))
  p.record(_d("c", False, INLINED))
  assert p.forced() == {FORCED_ALWAYS_RUN: ["a"], FORCED_COST: ["b"]}

def test_inlining_consumers_are_the_ones_that_retained_ownership():
  p = RealizationPlan()
  p.record(_d("a", True, FORCED_ALWAYS_RUN)); p.record(_d("b", False, INLINED, parallelism=99999, trip=8))
  assert [d.producer for d in p.inlining_consumers()] == ["b"]

def test_explanations_name_the_actual_cause():
  assert "never removable" in _d("x", True, FORCED_ALWAYS_RUN).explain()
  assert "cost model" in _d("x", True, FORCED_COST).explain()
  assert "fusion" in _d("x", False, INLINED, parallelism=131072, trip=2).explain()

def test_plan_serializes():
  p = RealizationPlan()
  p.record(_d("a", True, FORCED_RECOMPUTE_HOSTILE, parallelism=128, trip=1187, hostile_ops=("LOG2", "EXP2")))
  j = p.to_json()
  assert j["schema"] == "tinygrad.realization_plan.v1"
  assert j["decisions"][0]["hostile_ops"] == ("LOG2", "EXP2") or j["decisions"][0]["hostile_ops"] == ["LOG2", "EXP2"]
