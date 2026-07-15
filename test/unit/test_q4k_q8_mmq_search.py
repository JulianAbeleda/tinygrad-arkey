import pytest

from extra.qk.q4k_q8_mmq_search import SearchPolicy, enumerate_descriptors, replay_descriptors, run_search


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
      events.append("candidate_timing"); return {"min_ms": 2.0}
    def measure_direct_packed(self, **kwargs):
      events.append("direct_timing"); return {"min_ms": 3.0}

  report = run_search(
    axes={"ok": (True, False), "resources": ({"lds": 8}, {"lds": 99})},
    session_factory=Session,
    policy=SearchPolicy(warmups=0, rounds=1, resource_limits={"lds": 16}),
  )
  assert sum(row["status"] == "measured" for row in report["candidates"]) == 1
  assert any(row["status"] == "rejected" for row in report["candidates"])
  assert events.index("correctness") < events.index("candidate_timing") < events.index("direct_timing")
  assert report["winner_evidence"]["speedup_vs_direct_packed"] == 1.5
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
