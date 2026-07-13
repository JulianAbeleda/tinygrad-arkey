"""C5 synthetic (fake-runtime) matrix for the attn_qo progressive-correctness ladder.

Every test drives :func:`run_progressive_correctness` through its injectable
seams so NOTHING touches a real GPU: the isolated executor is replaced by a fake
``runner`` that never invokes the (child-only) builder and never dispatches, and
the CPU reference is exercised directly.  No real ``Device`` is constructed.

The matrix proves: the CPU reference is deterministic, nonconstant, and equals
``A @ B^T``; audit mode prints the bounded plan and never dispatches; a faulting
stage STOPS the ladder (no larger stage runs); a passing ladder advances to the
exact shape; and correctness is identity-joined to the DISTINCT candidate
identity (a mismatched returned identity fails closed; direct != lds).
"""
from __future__ import annotations

import pickle

import numpy as np

from extra.qk.prefill.guarded_execution import GuardPolicy
from extra.qk.prefill import attn_qo_progressive_correctness_20260713 as pc
from extra.qk.prefill.attn_qo_progressive_correctness_20260713 import (EXACT_SHAPE, ProgressiveCorrectnessRecord,
                                                                       attn_qo_reference_inputs,
                                                                       default_stage_ladder,
                                                                       run_progressive_correctness,
                                                                       stage_candidate_identity)
from tinygrad.runtime.execution_bridge_contracts import dispatch_state


# --- a fake isolated executor (the child-only builder is NEVER invoked) -------

class FakeExecutor:
  """Stands in for run_isolated_guarded_execution: records calls, dispatches nothing."""
  def __init__(self, *, fail_shapes=(), echo_identity=True, forced_identity=None, forced_state=None):
    self.fail_shapes = set(tuple(s) for s in fail_shapes)
    self.echo_identity = echo_identity
    self.forced_identity = forced_identity
    self.forced_state = forced_state
    self.calls = []

  def __call__(self, *, builder, request, health_probe=None, timeout_seconds=None,
               terminate_grace_seconds=0.25, health_timeout_seconds=30.0, persist=None):
    shape = tuple(request.identity["shape"])
    self.calls.append(shape)
    # The builder is a picklable child-only spec; a fake executor MUST NOT call it
    # (calling it would trigger a real compile + real device construction).
    assert builder is not None
    failed = shape in self.fail_shapes
    ident = dict(request.identity) if self.echo_identity else {}
    if self.forced_identity is not None: ident["canonical_identity"] = self.forced_identity
    state = self.forced_state or (dispatch_state("failed") if failed else dispatch_state("completed"))

    class _Out:
      dispatch_state = state
      passed = not failed
      errors = () if not failed else ("synthetic numerical fault",)
      identity = ident
      def to_dict(self_inner):
        return {"dispatch_state": state, "passed": not failed, "identity": ident,
                "guarded": {"numerics_passed": not failed}}
    out = _Out()
    if not out.passed and persist is not None: persist(out.to_dict())
    return out


def _fake_factory(candidate, shape):
  # Returns the REAL picklable builder spec (module-level build fn + scalars); the
  # fake executor never calls it, so no compile/device happens.
  return pc._stage_builder(candidate, shape, compile_target=None, runtime_device="AMD")


# --- deterministic CPU reference ----------------------------------------------

def test_reference_is_deterministic_nonconstant_and_equals_a_bt():
  a1, b1, ref1 = attn_qo_reference_inputs((128, 256, 64), seed=7)
  a2, b2, ref2 = attn_qo_reference_inputs((128, 256, 64), seed=7)
  assert np.array_equal(a1, a2) and np.array_equal(b1, b2) and np.array_equal(ref1, ref2)  # reproducible
  assert a1.dtype == np.float16 and b1.dtype == np.float16 and ref1.dtype == np.float32
  assert a1.shape == (128, 64) and b1.shape == (256, 64) and ref1.shape == (128, 256)
  assert np.ptp(a1) > 0 and np.ptp(b1) > 0  # NONCONSTANT (guarded lifecycle rejects constant inputs)
  expected = a1.astype(np.float32) @ b1.astype(np.float32).T  # C = A @ B^T, fp32 accumulate
  assert np.array_equal(ref1, expected)


def test_reference_differs_across_seeds_and_shapes():
  _, _, r_a = attn_qo_reference_inputs((128, 128, 32), seed=1)
  _, _, r_b = attn_qo_reference_inputs((128, 128, 32), seed=2)
  assert not np.array_equal(r_a, r_b)
  a1, _, _ = attn_qo_reference_inputs((128, 128, 32), seed=1)
  a2, _, _ = attn_qo_reference_inputs((256, 128, 32), seed=1)
  assert not np.array_equal(a1[:128], a2[:128])  # shape is mixed into the seed


# --- audit mode: prints the plan, NEVER dispatches ----------------------------

def test_audit_mode_prints_plan_and_never_dispatches():
  printed: list[str] = []
  def _no_dispatch(**kwargs): raise AssertionError("audit mode must never invoke the executor")
  out = run_progressive_correctness(candidate="lds", execute=False, runner=_no_dispatch,
                                    builder_factory=_fake_factory, printer=printed.append)
  assert isinstance(out, ProgressiveCorrectnessRecord)
  assert out.mode == "audit" and out.passed is False and out.reached_exact is False
  assert all(s.dispatch_state == dispatch_state("not_attempted") and s.executed is False for s in out.stages)
  assert out.stages[-1].plan.shape == EXACT_SHAPE
  assert printed and "NO DISPATCH" in printed[0] and "STOP on first fault" in printed[0]
  # buffer bytes are the analytic fp16 sizes; first lds stage is the small one.
  first = out.stages[0].plan
  assert first.buffers_bytes["a"] == first.shape[0] * first.shape[2] * 2


def test_audit_direct_ladder_is_exact_only():
  # direct_l2 is admission-locked to the exact shape, so its ladder is one stage.
  out = run_progressive_correctness(candidate="direct_l2", execute=False, printer=None)
  assert len(out.stages) == 1 and out.stages[0].plan.shape == EXACT_SHAPE
  assert out.stages[0].plan.argument_order == ("output", "a", "b")


# --- execute (fake executor): a passing ladder advances to exact --------------

def test_passing_ladder_advances_to_exact():
  fake = FakeExecutor()
  out = run_progressive_correctness(candidate="lds", execute=True, runner=fake, builder_factory=_fake_factory)
  assert out.mode == "execute" and out.passed is True and out.reached_exact is True
  assert out.stopped_at is None
  assert fake.calls == list(default_stage_ladder("lds"))  # every stage ran, in order
  assert all(s.passed and s.executed for s in out.stages)
  assert out.stages[-1].plan.shape == EXACT_SHAPE


# --- execute: a faulting stage STOPS the ladder -------------------------------

def test_faulting_stage_stops_the_ladder():
  ladder = default_stage_ladder("lds")
  fake = FakeExecutor(fail_shapes=[ladder[0]])  # fail the FIRST (smallest) stage
  persisted: list[dict] = []
  out = run_progressive_correctness(candidate="lds", execute=True, runner=fake, builder_factory=_fake_factory,
                                    persist=persisted.append)
  assert out.passed is False and out.reached_exact is False
  assert out.stopped_at == ladder[0]
  assert fake.calls == [ladder[0]]  # larger stages were NOT attempted
  assert len(out.stages) == 1 and out.stages[0].passed is False
  assert out.stages[0].dispatch_state == dispatch_state("failed")
  # persisted twice: the executor persists the stage fault, the harness the record.
  assert all(row["passed"] is False for row in persisted)
  assert any(row.get("schema") == pc.SCHEMA and row.get("candidate") == "lds" for row in persisted)


def test_midladder_fault_stops_before_exact():
  ladder = default_stage_ladder("lds")
  fake = FakeExecutor(fail_shapes=[ladder[1]])  # first passes, second faults
  out = run_progressive_correctness(candidate="lds", execute=True, runner=fake, builder_factory=_fake_factory)
  assert fake.calls == [ladder[0], ladder[1]] and EXACT_SHAPE not in fake.calls
  assert out.stopped_at == ladder[1] and out.reached_exact is False and out.passed is False


# --- identity-join (P0-3): bound to the DISTINCT candidate identity -----------

def test_identity_join_binds_the_correct_candidate_identity():
  fake = FakeExecutor()
  out = run_progressive_correctness(candidate="lds", execute=True, runner=fake, builder_factory=_fake_factory)
  assert out.canonical_identity == stage_candidate_identity("lds", EXACT_SHAPE)
  for stage in out.stages:
    assert stage.plan.canonical_identity == stage_candidate_identity("lds", stage.plan.shape)
    assert stage.result["identity"]["canonical_identity"] == stage.plan.canonical_identity


def test_direct_and_lds_identities_are_distinct():
  assert stage_candidate_identity("direct_l2", EXACT_SHAPE) != stage_candidate_identity("lds", EXACT_SHAPE)
  # each shape also gets its own distinct identity
  ids = {s: stage_candidate_identity("lds", s) for s in default_stage_ladder("lds")}
  assert len(set(ids.values())) == len(ids)


def test_mismatched_returned_identity_fails_closed():
  # The executor returns a WRONG canonical identity: the join must fail the stage
  # closed rather than crediting correctness to the wrong candidate.
  fake = FakeExecutor(forced_identity="0" * 64)
  out = run_progressive_correctness(candidate="lds", execute=True, runner=fake, builder_factory=_fake_factory)
  assert out.passed is False and out.reached_exact is False
  assert out.stopped_at == default_stage_ladder("lds")[0]
  assert any("identity join failed" in e for e in out.stages[0].errors)


# --- spawn-safety: the child-only builder spec is picklable -------------------

def test_stage_builder_spec_is_picklable_for_spawn():
  # The production execute=True path hands the executor a picklable spec (a UOp/
  # runtime is never pickled: build_attn_qo_stage_bundle recompiles in-child).
  spec = pc._stage_builder("lds", EXACT_SHAPE, compile_target=None, runtime_device="AMD")
  restored = pickle.loads(pickle.dumps(spec))  # no real GPU touched: only the descriptor is serialized
  assert restored.build is pc.build_attn_qo_stage_bundle
  assert restored.kwargs["transport"] == "lds" and tuple(restored.kwargs["argument_order"]) == ("output", "a", "b")


def test_ladders_end_at_exact_and_reject_bad_ladders():
  import pytest
  for cand in ("direct_l2", "lds"):
    assert default_stage_ladder(cand)[-1] == EXACT_SHAPE
  with pytest.raises(ValueError):
    run_progressive_correctness(candidate="lds", execute=False, stages=[(128, 4096, 4096)], printer=None)
  with pytest.raises(ValueError):
    run_progressive_correctness(candidate="nope", execute=False, printer=None)


def test_harness_parent_constructs_no_runtime():
  # The parent orchestration references no live Device/runtime/Buffer; runtime
  # construction is delegated to the child-only build fn.
  import inspect
  src = inspect.getsource(run_progressive_correctness) + inspect.getsource(pc._stage_plan)
  for forbidden in ("Device[", ".runtime(", "get_runtime", "prepare_executable", "Buffer("):
    assert forbidden not in src, f"progressive-correctness parent path must not reference {forbidden!r}"
