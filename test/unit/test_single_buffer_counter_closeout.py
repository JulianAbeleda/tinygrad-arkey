from extra.qk.prefill.single_buffer_counter_closeout import GROUPS, build_report
from extra.qk.prefill.pure_single_buffer_evaluation_gate import canonical_candidate_hash
from test.unit.test_pure_single_buffer_evaluation_gate import _payload


def test_counter_closeout_joins_binary_and_fails_closed_on_unavailable_derived_metrics():
  payload = _payload(); identity = canonical_candidate_hash(payload); binary = "b" * 64; commit = "c" * 40
  resource = {"canonical_identity": identity, "passed": True, "program": {"binary_sha256": binary},
              "git": {"revision": commit, "dirty": False}}
  def collect(_candidate, counters, repetitions, **_kwargs):
    return {"samples": [{"status": "live", "binary_sha256": binary,
                          "counters": {name: 1 for name in counters}} for _ in range(repetitions)]}
  report = build_report(payload, identity, resource, collector=collect,
                        command=["fake-child"], git_state={"revision": commit, "dirty": False})
  assert all(row["status"] == "live" for row in report["groups"])
  assert report["status"] == "blocked"
  assert {row["category"] for row in report["unavailable"]} == {"wmma", "occupancy"}


def test_counter_closeout_rejects_binary_mismatch_and_does_not_run_on_commit_mismatch():
  payload = _payload(); identity = canonical_candidate_hash(payload); called = []
  resource = {"canonical_identity": identity, "passed": True, "program": {"binary_sha256": "b" * 64},
              "git": {"revision": "c" * 40}}
  report = build_report(payload, identity, resource, collector=lambda *a, **k: called.append(1),
                        git_state={"revision": "d" * 40, "dirty": False})
  assert not called and report["groups"] == [] and "source commit" in report["blockers"][0]
