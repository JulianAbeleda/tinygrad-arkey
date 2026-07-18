import pytest

from extra.qk.q4k_q8_mmq_search import (AggregatePolicy, SearchPolicy, enumerate_descriptors,
                                       TIMING_SCOPE_SCHEMA, evaluate_aggregate_policy, replay_descriptors, run_search)


def scope(role="ffn", shape=(16, 16, 256), kind="full_role"):
  return {"schema": TIMING_SCOPE_SCHEMA, "scope": kind, "role": role, "shape": list(shape)}


def test_enumerates_stable_generated_axes():
  rows = enumerate_descriptors({"tile_m": (8, 16), "tile_n": (8, 16)})
  assert len(rows) == 4
  assert [row.axes for row in rows] == [
    {"tile_m": 8, "tile_n": 8}, {"tile_m": 8, "tile_n": 16},
    {"tile_m": 16, "tile_n": 8}, {"tile_m": 16, "tile_n": 16},
  ]
  assert enumerate_descriptors({"empty": ()}) == ()


def test_rejects_resources_and_runs_correctness_before_timing_in_one_session():
  events = []

  class Session:
    def prepare(self, descriptor): events.append(("prepare", descriptor.axes)); return descriptor
    def check_correctness(self, prepared): events.append("correctness"); return {"passed": prepared.axes["ok"]}
    def evidence_gate(self, prepared, correctness):
      return {"timing_allowed": correctness["passed"], "promotion_eligible": correctness["passed"], "blockers": []}
    def measure(self, prepared, **kwargs):
      events.append("candidate_timing"); return {"min_ms": 2.0, "timing_scope": scope()}
    def measure_direct_packed(self, **kwargs):
      events.append("direct_timing"); return {"min_ms": 3.0, "timing_scope": scope()}

  report = run_search(
    axes={"ok": (True, False), "resources": ({"lds": 8}, {"lds": 99})},
    session_factory=Session,
    policy=SearchPolicy(warmups=0, rounds=1, resource_limits={"lds": 16}),
  )
  assert sum(row["status"] == "measured" for row in report["candidates"]) == 1
  assert any(row["status"] == "rejected" for row in report["candidates"])
  assert events.index("correctness") < events.index("candidate_timing") < events.index("direct_timing")
  assert report["winner_evidence"]["speedup_vs_direct_packed"] == 1.5
  assert report["full_role_performance_qualified"] is True
  assert report["default_route"] == "direct_packed"
  assert report["production_dispatch_changed"] is False


def test_correctness_failure_never_times():
  calls = []

  class Session:
    def prepare(self, descriptor): return descriptor
    def check_correctness(self, prepared): return {"passed": False}
    def measure(self, **kwargs): calls.append("candidate")
    def measure_direct_packed(self, **kwargs): calls.append("direct")

  report = run_search(axes={"x": (1,)}, session_factory=Session)
  assert report["candidates"][0]["status"] == "correctness_failed"
  assert calls == []


def test_replay_recovers_verified_descriptor_identity_without_reenumerating_axes():
  class Session:
    def prepare(self, descriptor): return descriptor
    def check_correctness(self, prepared): return {"passed": False}

  report = run_search(axes={"tile": (8, 16)}, session_factory=Session)
  replayed = replay_descriptors(report)
  assert [descriptor.canonical() for descriptor in replayed] == [row["descriptor"] for row in report["candidates"]]

  tampered = dict(report)
  tampered["candidates"] = list(report["candidates"])[::-1]
  with pytest.raises(ValueError, match="digest mismatch"):
    replay_descriptors(tampered)


def test_enumeration_rejects_duplicate_generated_identity():
  with pytest.raises(ValueError, match="duplicate descriptor identities"):
    enumerate_descriptors({"tile": (8, 8)})


def test_aggregate_policy_includes_q8_costs_and_requires_every_role():
  def row(ms):
    return {"status": "measured", "session_id": "s1", "correctness": {"passed": True},
            "evidence_gate": {"timing_allowed": True}, "min_ms": ms, "timing_scope": scope("aggregate", (16, 16, 256))}
  policy = AggregatePolicy(required_roles=("q", "kv"), preparation_ms={"q": 1},
    packing_ms={"q": 2, "kv": 2}, reduction_ms={"kv": 3},
    direct_preparation_ms={"q": 1, "kv": 1})
  result = evaluate_aggregate_policy(candidate_rows={"c": {"q": row(2), "kv": row(4)}},
    direct_packed_rows={"q": {"session_id": "s1", "min_ms": 10, "timing_scope": scope("aggregate", (16, 16, 256))},
                        "kv": {"session_id": "s1", "min_ms": 10, "timing_scope": scope("aggregate", (16, 16, 256))}},
    policy=policy, session_id="s1")
  assert result["status"] == "PASS"
  assert result["winner"]["aggregate_ms"] == 14.0
  assert result["winner"]["direct_packed_ms"] == 22.0

  incomplete = evaluate_aggregate_policy(candidate_rows={"c": {"q": row(2)}},
    direct_packed_rows={"q": {"session_id": "s1", "min_ms": 10, "timing_scope": scope("aggregate", (16, 16, 256))}},
    policy=policy, session_id="s1")
  assert incomplete["status"] == "NO_AGGREGATE_WINNER"
  assert "kv: missing evidence" in incomplete["candidates"]["c"]["blockers"]


def test_single_epoch_candidate_cannot_rank_against_full_role_direct_timing():
  class Session:
    def prepare(self, descriptor): return descriptor
    def check_correctness(self, prepared): return {"passed": True}
    def evidence_gate(self, prepared, correctness):
      return {"timing_allowed": True, "promotion_eligible": False, "blockers": []}
    def measure(self, prepared, **kwargs):
      return {"min_ms": .277502, "timing_scope": scope("ffn_gate_up", (512, 17408, 256), "bounded")}
    def measure_direct_packed(self, **kwargs):
      return {"min_ms": 9.351988, "timing_scope": scope("ffn_gate_up", (512, 17408, 5120))}

  report = run_search(axes={"candidate": ("five_buffer",)}, session_factory=Session)
  assert report["status"] == "NO_PASSING_CANDIDATE"
  assert report["winner"] is None and report["full_role_performance_qualified"] is False
  assert report["candidates"][0]["status"] == "timing_blocked"
  assert "identical completed work" in report["candidates"][0]["blocker"]


def test_slower_comparable_candidate_is_measured_but_never_a_winner():
  class Session:
    def prepare(self, descriptor): return descriptor
    def check_correctness(self, prepared): return {"passed": True}
    def evidence_gate(self, prepared, correctness):
      return {"timing_allowed": True, "promotion_eligible": False, "blockers": []}
    def measure(self, prepared, **kwargs):
      return {"min_ms": 27.076011, "timing_scope": scope("ffn_gate_up", (512, 17408, 5120))}
    def measure_direct_packed(self, **kwargs):
      return {"min_ms": 9.351988, "timing_scope": scope("ffn_gate_up", (512, 17408, 5120))}

  report = run_search(axes={"candidate": ("five_buffer",)}, session_factory=Session)
  assert report["status"] == "NO_PASSING_CANDIDATE"
  assert report["candidates"][0]["status"] == "measured"
  assert report["candidates"][0]["performance_status"] == "LOSS"
  assert report["winner"] is None


def test_aggregate_policy_rejects_scope_mismatch_and_comparable_loss():
  def row(ms, shape):
    return {"status": "measured", "correctness": {"passed": True},
            "evidence_gate": {"timing_allowed": True}, "min_ms": ms,
            "timing_scope": scope("ffn_gate_up", shape)}
  policy = AggregatePolicy(required_roles=("ffn_gate_up",))
  mismatch = evaluate_aggregate_policy(
    candidate_rows={"c": {"ffn_gate_up": row(.277502, (512, 17408, 256))}},
    direct_packed_rows={"ffn_gate_up": row(9.351988, (512, 17408, 5120))}, policy=policy)
  assert mismatch["status"] == "NO_AGGREGATE_WINNER"
  assert mismatch["candidates"]["c"]["status"] == "BLOCKED"
  assert "identical completed work" in mismatch["candidates"]["c"]["blockers"][0]

  loss = evaluate_aggregate_policy(
    candidate_rows={"c": {"ffn_gate_up": row(27.076011, (512, 17408, 5120))}},
    direct_packed_rows={"ffn_gate_up": row(9.351988, (512, 17408, 5120))}, policy=policy)
  assert loss["status"] == "NO_AGGREGATE_WINNER"
  assert loss["candidates"]["c"]["status"] == "LOSS"
  assert loss["winner"] is None
