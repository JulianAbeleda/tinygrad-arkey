"""Hermetic CPU-only tests for the B3.4 offline selective-unalias candidate search.

The tool loads a B3.1 aligned capture report and predicts which manifest
logical buffers, if held out of the shared planner arena, would remove which
PLANNER_ALIAS edges, with added memory cost and an edge-count critical-path
delta proxy. These tests pin the removal model on synthetic fixtures (the
same two-chain arena-reuse fixture route_b3_dag_attribution.py uses), the
empty/trivial reports, and one integration-style run over the anchored real
capture that skips when that file is absent.
"""
import json
import pathlib

import pytest

from extra.llm_research.decode.route_b3_dag_attribution import (
  build_attribution_fixture, build_partial_overlap_fixture, compute_attribution_report,
)
from extra.llm_research.decode.route_b3_4_candidate_search import (
  SCHEMA, align_up, search_candidates,
)

CAPTURE = pathlib.Path(
  "/home/ubuntu/tinygrad-arkey/docs/task_workflow/output/"
  "nv-decode-overlap-b3-2-aligned-capture-manifest-20260804.json")

ROW_KEYS = ("held_buffer_ids", "added_memory_bytes", "removed_planner_edges_total",
            "removed_by_kind", "removed_cross_group", "cp_delta_proxy", "pareto_rank")


def _minimal_report(nodes, edges):
  """Hand-built B3.1-shaped report for trivial/empty cases (no manifest)."""
  node_list = [{"id": i, "name": "n%d" % i, "duration_us": 0.0, "group_id": 0,
                "metadata": {}} for i in range(nodes)]
  dag_edges = [{"from": e["from"], "to": e["to"], "kind": e["kind"],
                "crosses_group": e.get("crosses_group", False)} for e in edges]
  return {
    "schema": "tinygrad.route_b3.dag_attribution.v1",
    "arms": {"logical": {"nodes": node_list,
                         "edges": [e for e in dag_edges if e["kind"] in ("RAW", "WAR", "WAW")]},
             "physical": {"nodes": node_list, "edges": dag_edges}},
    "attributed_edges": [{"from": e["from"], "to": e["to"], "kind": e["kind"],
                          "source": e["source"], "arena": e["arena"],
                          "range": list(e["range"]),
                          "crosses_group": e.get("crosses_group", False),
                          "logical_buffer_ids": e.get("logical_buffer_ids", [])}
                         for e in edges],
    "manifest": {},
    "alignment": {"aligned": True},
  }


def _by_held(ledger):
  return {tuple(r["held_buffer_ids"]): r for r in ledger["rows"]}


def test_holding_chain_buffer_removes_exactly_the_predicted_alias_edges():
  """Two logically independent chains chained by arena reuse (attribution fixture).

  Holding 'A' (4096B at arena offset 0) must remove exactly the three planner
  edges whose alias range [0, 4096) sits inside A's placement: 0->3 WAW,
  1->3 WAR, 0->4 RAW, at a memory cost of 4096 bytes.
  """
  calls, manifest = build_attribution_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  ledger = search_candidates(report, {
    "cp_mode": "exact", "top_n_by_bytes": 0, "chain": False, "chain_full_cover": False,
    "greedy": True, "min_five_pct": False, "baseline": True,
  }, include_edge_pairs=True)
  rows = _by_held(ledger)

  row = rows[("A",)]
  assert row["removed_planner_edges_total"] == 3
  assert row["removed_by_kind"] == {"RAW": 1, "WAR": 1, "WAW": 1}
  assert row["removed_cross_group"] == 3
  assert row["added_memory_bytes"] == align_up(4096)
  assert row["removed_edge_pairs"] == [[0, 3], [0, 4], [1, 3]]

  row_x = rows[("X",)]
  assert row_x["removed_planner_edges_total"] == 3
  assert row_x["added_memory_bytes"] == align_up(512)
  assert row_x["removed_edge_pairs"] == [[1, 4], [1, 5], [2, 4]]

  full = rows[("A", "C", "D")]
  assert full["removed_planner_edges_total"] == 8
  assert full["removed_by_kind"] == {"RAW": 3, "WAR": 2, "WAW": 3}
  assert full["added_memory_bytes"] == align_up(4096) + 2 * align_up(512)
  assert full["cp_delta_proxy"] == 0.0

  baseline = rows[()]
  assert baseline["removed_planner_edges_total"] == 0
  assert baseline["added_memory_bytes"] == 0
  assert baseline["cp_delta_proxy"] == 2.0
  assert baseline["pareto_rank"] == 0


def test_physical_critical_path_rerouting_is_kept_in_delta_proxy():
  """Removing 0->3/1->3/0->4 via A does not shorten the chain: 2->4 reroutes it."""
  calls, manifest = build_attribution_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  ledger = search_candidates(report, {
    "cp_mode": "exact", "top_n_by_bytes": 0, "chain": False, "chain_full_cover": False,
    "greedy": False, "min_five_pct": False, "baseline": True,
  })
  row = _by_held(ledger)[("A",)]
  assert ledger["proxies"]["logical_cp_edges"] == 4
  assert ledger["proxies"]["physical_cp_edges"] == 6
  assert row["cp_delta_proxy"] == 2.0


def test_empty_report_yields_baseline_only():
  report = _minimal_report(nodes=2, edges=[])
  ledger = search_candidates(report, {"cp_mode": "exact"})
  assert ledger["schema"] == SCHEMA
  assert ledger["source"]["planner_edge_count"] == 0
  assert len(ledger["rows"]) == 1
  assert ledger["rows"][0]["held_buffer_ids"] == []
  assert ledger["rows"][0]["removed_planner_edges_total"] == 0
  assert ledger["rows"][0]["added_memory_bytes"] == 0
  assert ledger["rows"][0]["cp_delta_proxy"] == 0.0
  assert ledger["rows"][0]["pareto_rank"] == 0
  assert ledger["top_candidates"] == []
  assert ledger["recommended"] is None


def test_semantic_only_report_has_no_planner_edges_to_remove():
  report = _minimal_report(nodes=2, edges=[{
    "from": 0, "to": 1, "kind": "RAW", "source": "SEMANTIC",
    "arena": "a", "range": [0, 4], "crosses_group": False,
  }])
  ledger = search_candidates(report, {"cp_mode": "exact"})
  assert ledger["source"]["planner_edge_count"] == 0
  assert ledger["proxies"]["logical_cp_edges"] == 1
  assert ledger["proxies"]["physical_cp_edges"] == 1
  assert len(ledger["rows"]) == 1
  assert ledger["rows"][0]["cp_delta_proxy"] == 0.0
  assert ledger["recommended"] is None


def test_partial_overlap_fixture_hold_P_removes_two_alias_edges():
  """P and Q share [1024, 2048): holding P removes the two alias edges at 2048B."""
  calls, manifest = build_partial_overlap_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  ledger = search_candidates(report, {
    "cp_mode": "exact", "top_n_by_bytes": 0, "chain": False, "chain_full_cover": False,
    "greedy": False, "min_five_pct": False, "baseline": True,
  }, include_edge_pairs=True)
  row = _by_held(ledger)[("P",)]
  assert row["removed_planner_edges_total"] == 2
  assert row["removed_by_kind"] == {"RAW": 1, "WAR": 0, "WAW": 1}
  assert row["removed_cross_group"] == 0
  assert row["added_memory_bytes"] == align_up(2048)
  assert row["removed_edge_pairs"] == [[0, 1], [0, 2]]


@pytest.mark.skipif(not CAPTURE.exists(),
                    reason="anchored real B3.2 capture manifest not present")
def test_real_capture_ledger_schema_and_nonzero_candidates():
  """Integration-style: real capture loads, search emits a valid ledger.

  Uses the fast chain cp proxy so the suite stays quick; the CLI full run
  defaults to the exact edge-count longest-path proxy.
  """
  with open(CAPTURE, "r", encoding="utf-8") as f:
    report = json.load(f)
  ledger = search_candidates(report, {
    "cp_mode": "chain", "top_n_by_bytes": 32, "greedy_step_cap": 8,
  })
  assert ledger["schema"] == SCHEMA
  assert ledger["source"]["planner_edge_count"] > 0
  assert ledger["proxies"]["physical_cp_edges"] > 0
  assert ledger["rows"]
  assert ledger["top_candidates"]
  assert ledger["pareto_frontier"]
  for key in ROW_KEYS:
    assert key in ledger["rows"][0]
  for r in ledger["rows"][:50]:
    assert sum(r["removed_by_kind"].values()) == r["removed_planner_edges_total"]
    assert r["added_memory_bytes"] >= 0
  rec = ledger["recommended"]
  assert rec is not None
  assert rec["removed_planner_edges_total"] > 0
  json.dumps(ledger)  # must be JSON-serializable
