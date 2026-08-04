"""Hermetic CPU-only tests for the Phase 4 full-token dependency capture tooling."""
import json

from extra.llm_research.decode.full_token_dag_capture import (
  SCHEMA, FullTokenDagError, RecordingDepsTracker, attach_summary, build_synthetic_dag,
  compute_dag_metrics, restrict_dag, run_synthetic, to_sim_nodes, unknown_dep_nodes,
  validate_schema,
)
from extra.llm_research.decode.dag_critical_path_sim import compute_tails, list_schedule


def _node_ids(nodes):
  return sorted(n["id"] for n in nodes)


def test_restriction_reproduces_known_per_group_critical_paths():
  dag = build_synthetic_dag()
  g0 = restrict_dag(dag, 0)
  g1 = restrict_dag(dag, 1)
  # Per-group restriction keeps only intra-group edges: {0->1 RAW, 0->2 WAW}
  # and {3->6 RAW}, never the cross-group edges.
  assert {(e["from"], e["to"]) for e in g0["edges"]} == {(0, 1), (0, 2)}
  assert {(e["from"], e["to"]) for e in g1["edges"]} == {(3, 6)}
  assert _node_ids(g0["nodes"]) == [0, 1, 2]
  assert _node_ids(g1["nodes"]) == [3, 4, 5, 6]
  m0 = compute_dag_metrics(g0)
  m1 = compute_dag_metrics(g1)
  assert m0["serialized_span_us"] == 45.0 and m0["critical_path_us"] == 30.0
  assert m1["serialized_span_us"] == 36.0 and m1["critical_path_us"] == 18.0
  # Restriction is idempotent: restricting a restricted group changes nothing.
  assert restrict_dag(g0, 0)["edges"] == g0["edges"]


def test_cross_group_edge_retention_raw_war_waw():
  dag = build_synthetic_dag()
  cross = [e for e in dag["edges"] if e["crosses_group"]]
  assert {(e["from"], e["to"], e["kind"]) for e in cross} == {
    (0, 5, "WAW"), (1, 3, "RAW"), (1, 4, "RAW"), (1, 5, "WAR"), (1, 6, "RAW")}
  assert {e["kind"] for e in cross} == {"RAW", "WAR", "WAW"}
  intra = [e for e in dag["edges"] if not e["crosses_group"]]
  assert {(e["from"], e["to"]) for e in intra} == {(0, 1), (0, 2), (3, 6)}
  # Cross-group edges span the 32/64-style boundary: endpoints in different groups.
  by_id = {n["id"]: n["group_id"] for n in dag["nodes"]}
  for e in cross:
    assert by_id[e["from"]] != by_id[e["to"]]


def test_full_token_2_3_queue_schedule_correctness_and_determinism():
  dag = build_synthetic_dag()
  m = compute_dag_metrics(dag)
  assert m["node_count"] == 7
  assert m["serialized_span_us"] == 81.0
  assert m["critical_path_us"] == 48.0
  assert m["schedule_2q_us"] == 56.0
  assert m["schedule_3q_us"] == 48.0
  # Deterministic ties: same inputs, same schedule, twice.
  sim_nodes = to_sim_nodes(dag)
  tails = compute_tails(sim_nodes)
  span1, start1, end1 = list_schedule(sim_nodes, tails, 2)
  span2, start2, end2 = list_schedule(sim_nodes, tails, 2)
  assert (span1, start1, end1) == (span2, start2, end2)
  assert span1 == 56.0


def test_unknown_labeling_when_deps_missing():
  dag = build_synthetic_dag()
  dag["nodes"][1]["metadata"] = {"deps_status": "UNKNOWN"}
  assert unknown_dep_nodes(dag) == [1]
  s = attach_summary(dag)["summary"]
  assert s["unknown_dep_node_count"] == 1
  assert s["independence_assumption"] is True
  # Absent dependency information labels ALL nodes UNKNOWN, never independent.
  stripped = build_synthetic_dag()
  stripped.pop("edges")
  assert sorted(unknown_dep_nodes(stripped)) == sorted(n["id"] for n in stripped["nodes"])
  report = attach_summary(stripped)["summary"]
  assert report["unknown_dep_node_count"] == len(stripped["nodes"])
  assert report["independence_assumption"] is True


def test_schema_validity_of_emitted_json(tmp_path):
  out = tmp_path / "dag.json"
  dag = run_synthetic(str(out))
  validate_schema(dag)  # summary attached, all required keys present
  assert dag["schema"] == SCHEMA
  parsed = json.loads(out.read_text())
  validate_schema(parsed)
  assert parsed["summary"]["node_count"] == len(parsed["nodes"])
  assert parsed["summary"]["cross_group_edge_count"] == 5
  assert set(parsed["summary"]["schedules"]) == {"queues_2", "queues_3"}
  # Malformed DAGs are rejected with a clear error, not silently accepted.
  bad = build_synthetic_dag()
  bad["nodes"][0]["id"] = bad["nodes"][1]["id"]
  try:
    validate_schema(bad)
  except FullTokenDagError:
    pass
  else:
    raise AssertionError("duplicate node ids must fail schema validation")
  bad2 = build_synthetic_dag()
  bad2["edges"][0]["kind"] = "RMW"
  try:
    validate_schema(bad2)
  except FullTokenDagError:
    pass
  else:
    raise AssertionError("unknown edge kind must fail schema validation")


class _StubBuf:
  """Duck-typed Buffer stand-in with the DepsTracker-relevant attributes."""

  def __init__(self, base, offset, nbytes):
    self.base, self.offset, self.nbytes = base, offset, nbytes


def test_recording_deps_tracker_labels_raw_war_waw():
  tr = RecordingDepsTracker()
  a, b = object(), object()
  # n0 writes A[0:8] (base a)
  tr.access_resources([_StubBuf(a, 0, 8)], [0], 0)
  # n1 reads A[0:8]: RAW from n0
  waits = tr.access_resources([_StubBuf(a, 0, 8)], [], 1)
  assert waits == [0]
  # n2 writes A[0:8]: WAW from n0 and WAR from n1
  waits = tr.access_resources([_StubBuf(a, 0, 8)], [0], 2)
  assert sorted(waits) == [0, 1]
  # n3 reads B (base b): no dependency at all
  waits = tr.access_resources([_StubBuf(b, 0, 8)], [], 3)
  assert waits == []
  kinds = {(f, t): k for f, t, k in tr.edges}
  assert kinds[(0, 1)] == "RAW"
  assert kinds[(0, 2)] == "WAW"
  assert kinds[(1, 2)] == "WAR"


def test_recording_deps_tracker_uses_suballocated_ranges():
  tr = RecordingDepsTracker()
  base = object()
  # Writes to disjoint ranges of one base buffer do not create edges.
  tr.access_resources([_StubBuf(base, 0, 8)], [0], 0)
  tr.access_resources([_StubBuf(base, 16, 8)], [0], 1)
  # A later write overlapping the first write's range does.
  tr.access_resources([_StubBuf(base, 4, 8)], [0], 2)
  assert [(f, t) for f, t, _ in tr.edges] == [(0, 2)]
