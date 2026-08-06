"""Hermetic CPU-only tests for the NV in-graph co-schedule candidate scan.

Synthetic DAGs only: no GPU, no /tmp, no capture dependence.  Covers
dependency-independent pair detection, exact critical-path recovery math,
greedy ranking without double counting, per-population aggregation, the +50 us
promotion gate verdict, and fail-closed validation.
"""
import importlib.util, pathlib

import pytest

PATH = pathlib.Path(__file__).resolve().parents[2] / "extra" / "llm_research" / "decode" / "nv_co_schedule_candidate_scan.py"
SPEC = importlib.util.spec_from_file_location("nv_co_schedule_candidate_scan", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)

scan = MOD.scan
classify = MOD.classify


def _dag(specs: list[tuple[int, str, float, int]], edges: list[tuple[int, int]]) -> dict:
  nodes = [{"id": i, "name": name, "duration_us": d, "group_id": g, "metadata": None}
           for i, name, d, g in specs]
  return {"nodes": nodes, "edges": [{"from": a, "to": b} for a, b in edges]}


def test_independent_pair_detection_uses_full_reachability():
  # 0 (support) and 1 (host) are mutually unreachable; 2 depends on both, so
  # the pair (2, 1) must not appear even though they share an edge.
  dag = _dag([
    (0, "E_a", 4.0, 0), (1, "q4k_gemv", 10.0, 0), (2, "r_b", 2.0, 0),
  ], [(0, 2), (1, 2)])
  out = scan(dag)
  assert out["candidate_pairs"]["pair_count"] == 1
  row = out["pairs"][0]
  assert row["support_id"] == 0 and row["host_id"] == 1
  assert row["hideable_us"] == 4.0
  # Node 0 is off the critical path: hiding it recovers nothing.
  assert row["delta_cp_full_hide_us"] == 0.0
  assert row["recovery_us"] == 0.0


def test_exact_pair_recovery_and_full_hide_ceiling():
  # S0 (support, 10) and host (4) are independent, both feed the join; the
  # support branch is the critical path, so partial hiding recovers 4.
  dag = _dag([
    (0, "E_a", 10.0, 0), (1, "q4k_gemv", 4.0, 0), (2, "r_b", 3.0, 0),
  ], [(0, 2), (1, 2)])
  out = scan(dag)
  row = out["pairs"][0]
  assert row["hideable_us"] == 4.0
  assert row["fully_hideable"] is False
  assert row["recovery_us"] == 4.0
  assert row["delta_cp_full_hide_us"] == 6.0
  assert out["baseline"]["critical_path_us"] == 13.0
  assert out["ceiling"]["co_schedule_ceiling_us"] == 4.0
  # Fusing every support away (0 and 2 both disappear) leaves only host 1's
  # branch: 4 us, i.e. 9 us of critical-path recovery.
  assert out["ceiling"]["fusion_only_ceiling_us"] == 9.0


def test_greedy_ranking_does_not_double_count_parallel_branches():
  # Branch A (support 0, 10; host 1, 4) and branch B (support 2, 6; host 3, 2)
  # join at node 6; nodes 4 and 5 are also support.  Branch A is the critical
  # path; hiding 0 behind 1 recovers 4, after which no other pair recovers
  # anything.  Containment (10) exceeds the greedy total (4): the greedy must
  # not book the parallel-branch overcount.
  dag = _dag([
    (0, "E_a", 10.0, 0), (1, "q4k_gemv", 4.0, 0), (2, "r_b", 6.0, 0), (3, "q6k_gemv", 2.0, 0),
    (4, "E_c", 1.0, 0), (5, "r_d", 1.0, 0), (6, "r_join", 1.0, 0),
  ], [(0, 4), (1, 4), (2, 5), (3, 5), (4, 6), (5, 6)])
  out = scan(dag, floor_us=1.0)
  assert out["baseline"]["critical_path_us"] == 12.0
  assert out["candidate_pairs"]["best_partner_containment_us"] == 10.0
  assert out["greedy"]["selected_count"] == 1
  assert out["greedy"]["recovery_us"] == 4.0
  assert out["per_population"]["q4k"]["greedy_recovery_us"] == 4.0
  assert out["per_population"]["q6k"]["greedy_recovery_us"] == 0.0
  assert out["per_population"]["q4k"]["pair_count"] == 3
  assert out["per_population"]["q6k"]["pair_count"] == 3
  assert out["per_population"]["flash"]["pair_count"] == 0
  assert out["per_population"]["flash"]["ceiling_us"] == 0.0


def test_greedy_ranking_chain_recovery_recomputes_critical_path():
  # S0 (10) and S2 (6) sit serially on a 17 us critical path.  The greedy hides
  # each behind an independent 4 us host, and the second selection must
  # recompute against the graph left by the first (6 + 4 = 8, not 6 + 6).
  dag = _dag([
    (0, "E_a", 10.0, 0), (1, "q4k_gemv", 4.0, 0), (2, "r_b", 6.0, 0), (3, "q4k_gemv", 2.0, 0),
    (4, "E_c", 1.0, 0), (5, "r_join", 1.0, 0),
  ], [(0, 2), (2, 5), (1, 4), (3, 4), (4, 5)])
  out = scan(dag, floor_us=1.0)
  assert out["baseline"]["critical_path_us"] == 17.0
  assert out["greedy"]["selected_count"] == 2
  assert out["greedy"]["recovery_us"] == 8.0


def test_gate_verdict_can_clear_50_us_in_principle():
  # S0 (90) on a 152 us critical path hides behind an independent q4k host (80)
  # on a short branch: exact recovery 70, which must clear the promotion gate.
  # Proves the gate is not vacuously closed on arbitrary DAGs.
  dag = _dag([
    (0, "q4k_q0", 30.0, 0), (1, "E_s0", 90.0, 0), (2, "q4k_q1", 30.0, 0), (3, "r_tail", 1.0, 0),
    (4, "q4k_q2", 80.0, 0), (5, "r_c", 1.0, 0), (6, "r_end", 1.0, 0),
  ], [(0, 1), (1, 2), (2, 3), (3, 6), (4, 5), (5, 6)])
  out = scan(dag, floor_us=1.0)
  assert out["baseline"]["critical_path_us"] == 152.0
  # 70 from hiding S0 behind Q2, then 1 more from hiding r_c behind Q0.
  assert out["greedy"]["recovery_us"] == pytest.approx(71.0, abs=1e-3)
  assert out["verdict"] == "GPU_ELIGIBLE"
  assert out["gate_us"] == 50.0


def test_same_group_default_excludes_cross_graph_launches():
  specs = [(0, "E_a", 4.0, 0), (1, "q4k_gemv", 10.0, 1), (2, "r_b", 3.0, 1), (3, "q4k_gemv", 9.0, 1)]
  dag = _dag(specs, [])
  same = scan(dag)
  assert same["candidate_pairs"]["same_group_only"] is True
  assert same["candidate_pairs"]["pair_count"] == 2  # only r_b (group 1)
  assert all(r["same_group"] for r in same["pairs"])
  cross = scan(dag, same_group_only=False)
  assert cross["candidate_pairs"]["pair_count"] == 4
  assert any(not r["same_group"] for r in cross["pairs"])


def test_metadata_quant_hosts_and_support_contract():
  # Quant nodes carry semantic metadata; E_/r_ with metadata must not be
  # silently reclassified, and a support node is exactly E_/r_ without it.
  with pytest.raises(ValueError):
    classify({"id": 0, "name": "E_meta", "metadata": {"semantic": [{"role": "x"}]}})
  assert classify({"id": 0, "name": "E_2_8_16_4", "metadata": None}) == "support"
  assert classify({"id": 0, "name": "q4k_gemv", "metadata": {"semantic": []}}) == "host"
  with pytest.raises(ValueError):
    classify({"id": 0, "name": "mystery_1", "metadata": None})


def test_fail_closed_on_malformed_input():
  dag = _dag([(0, "E_a", 1.0, 0), (1, "q4k_gemv", 2.0, 0)], [(0, 1)])
  bad = dict(dag)
  bad["nodes"][0]["duration_us"] = -1.0
  with pytest.raises(ValueError): scan(bad)
  bad = dict(dag)
  bad["nodes"][0].pop("duration_us")
  with pytest.raises(ValueError): scan(bad)
  bad = dict(dag)
  bad["nodes"][0]["id"] = 5
  with pytest.raises(ValueError): scan(bad)
  bad = dict(dag)
  bad["edges"] = [{"from": 0, "to": 0}]
  with pytest.raises(ValueError): scan(bad)
  bad = dict(dag)
  bad["edges"] = [{"from": 1, "to": 0}]
  with pytest.raises(ValueError): scan(bad)
  bad = dict(dag)
  bad["edges"] = [{"from": 0, "to": 9}]
  with pytest.raises(ValueError): scan(bad)
  bad = dict(dag)
  bad["nodes"].append({"id": 2, "name": "x_1", "duration_us": 1.0, "group_id": 0, "metadata": None})
  with pytest.raises(ValueError): scan(bad)
  bad = dict(dag)
  bad["nodes"] = [{"id": 0, "name": "E_a", "duration_us": 1.0, "group_id": 0, "metadata": None}]
  bad["edges"] = []
  with pytest.raises(ValueError): scan(bad)
