"""Hermetic CPU-only tests for Route B3.2 CUPTI/nsys node-duration attachment.

Covers the duration-weighted critical-path math on the build_attribution_fixture
two-chains case (logical CP 35 us vs physical CP 63 us), the G-B3-D scale
classification thresholds, fail-closed mismatch behavior, and (skipif-missing)
an integration check that loads the real anchored capture and a real CUPTI
trace when both are present.
"""
import json
import os
import pathlib
import sqlite3

import pytest

from extra.llm_research.decode.cuda_duration_attach import (
  HISTORICAL_CUDA_WALL_MS, ROUTE_TAX_US, NV_GAP_US,
  attach_trace, classify_scale, compute_report,
  run_synthetic,
)
from extra.llm_research.decode.route_b3_dag_attribution import (
  build_attribution_fixture, compute_attribution_report,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
CAPTURE = REPO / "docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-capture-manifest-20260804.json"


def _fixture_capture() -> dict:
  calls, manifest = build_attribution_fixture()
  return compute_attribution_report(calls, calls, manifest)


def _attach_from_durations(capture: dict) -> dict:
  """Attach fixture durations directly (no trace file needed)."""
  durs: dict[int, float] = {}
  for arm_name in ("logical", "physical"):
    for n in capture["arms"][arm_name]["nodes"]:
      durs[n["id"]] = float(n.get("duration_us", 0.0))
  return {"duration_by_call": durs, "groups": [], "matched_calls": len(durs),
          "total_groups": 2, "aligned_groups": 2, "synthetic": True}


def test_synthetic_duration_weighted_cp_both_chains():
  """The two independent chains fixture: logical CP 35 us vs physical CP 63 us."""
  out = run_synthetic()
  wt = out["whole_token"]
  assert wt["logical_merged_cp_us"] == 35.0
  assert wt["physical_merged_cp_us"] == 63.0
  assert wt["planner_delta_cp_us"] == 28.0
  assert out["schema"] == "tinygrad.route_b3.duration_weighted.v1"
  # The merged CPs match the anchored route_b3 report summary semantics.
  assert out["physical"]["merged"]["edge_count"] > out["logical"]["merged"]["edge_count"]
  assert "verdict" in out and "g_b3_d" in out["verdict"]


def test_fixture_report_via_compute_report_matches_route_b3():
  """compute_report reproduces the route_b3 fixture CPs through the same path
  the anchored capture uses (arms are full_token_dag.v1 objects)."""
  capture = _fixture_capture()
  attach = _attach_from_durations(capture)
  report = compute_report(capture, attach, {"capture_path": "x", "capture_sha256": "x"},
                          HISTORICAL_CUDA_WALL_MS, "historical B0.2 anchor",
                          {"source": "synthetic"})
  assert report["whole_token"]["logical_merged_cp_us"] == 35.0
  assert report["whole_token"]["physical_merged_cp_us"] == 63.0
  assert report["whole_token"]["planner_delta_cp_us"] == 28.0


def test_scale_classification_thresholds():
  """G-B3-D thresholds: <5% NOT_MECHANISM_SCALE; then route tax and parity steps."""
  wall = HISTORICAL_CUDA_WALL_MS * 1000.0

  def cls(delta_us: float) -> str:
    return classify_scale(delta_us, wall)["scale_classification"]

  # Below the 5% bar.
  assert cls(0.04 * wall) == "NOT_MECHANISM_SCALE"
  # Above 5% but below route tax.
  assert cls(0.05 * wall) == "MECHANISM_SCALE_ONLY"
  assert cls(ROUTE_TAX_US - 0.001) == "MECHANISM_SCALE_ONLY"
  # At/above route tax, below route tax + NV gap.
  assert cls(ROUTE_TAX_US) == "ROUTE_TAX_SCALE"
  assert cls(ROUTE_TAX_US + NV_GAP_US - 0.001) == "ROUTE_TAX_SCALE"
  # At/above route tax + NV gap.
  assert cls(ROUTE_TAX_US + NV_GAP_US) == "PARITY_SCALE_THEORETICAL"


def test_scale_percentage_uses_cuda_wall_anchor():
  scale = classify_scale(ROUTE_TAX_US, HISTORICAL_CUDA_WALL_MS * 1000.0)
  # 705.1 us / 6331.9 us = 11.136% -> ROUTE_TAX_SCALE.
  assert scale["scale_classification"] == "ROUTE_TAX_SCALE"
  assert round(scale["planner_delta_pct_of_cuda_wall"], 3) == round(
    100.0 * ROUTE_TAX_US / (HISTORICAL_CUDA_WALL_MS * 1000.0), 3)


def _tiny_trace(path: str, group_names: dict[int, list[str]]) -> None:
  """Write a minimal CUPTI-shaped sqlite trace for one group with two replays.

  The tool requires >= 3 steady-state replays, so write 5 replay clusters with
  identical ordered (name, grid, block) signatures.
  """
  con = sqlite3.connect(path)
  cur = con.cursor()
  cur.execute("create table StringIds (id integer primary key, value text)")
  cur.execute("create table CUPTI_ACTIVITY_KIND_KERNEL ("
              "graphNodeId integer, shortName integer, gridX integer, gridY integer, gridZ integer,"
              "blockX integer, blockY integer, blockZ integer, start integer, end integer, graphId integer)")
  name_ids: dict[str, int] = {}
  for gid, names in group_names.items():
    graph_id = 100 + gid  # real CUPTI decode graphIds are non-zero
    base = 4294967296 * (gid + 1)  # per-group node-id base, same across replays
    for replay in range(5):
      replay_start = replay * 1000000  # 1 ms gap so replays form separate clusters
      for pos, name in enumerate(names):
        if name not in name_ids:
          name_ids[name] = len(name_ids) + 1
          cur.execute("insert into StringIds (id, value) values (?, ?)", (name_ids[name], name))
        cur.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?)", (
          base + pos, name_ids[name], 1, 1, 1, 1, 1, 1,
          replay_start + pos * 1000, replay_start + pos * 1000 + 500, graph_id))
  con.commit()
  con.close()


def test_attach_trace_fails_closed_on_name_mismatch(tmp_path):
  """A trace whose ordered names differ from the DAG must not attach durations."""
  capture = _fixture_capture()
  groups: dict[int, list[str]] = {}
  for n in sorted(capture["arms"]["physical"]["nodes"], key=lambda x: x["id"]):
    groups.setdefault(n["group_id"], []).append(str(n["name"]))
  bad = {gid: list(names) for gid, names in groups.items()}
  # Corrupt position 1 of group 0 only.
  bad[0][1] = "SOME_OTHER_KERNEL"
  trace_path = str(tmp_path / "bad.sqlite")
  _tiny_trace(trace_path, bad)
  attach = attach_trace(capture, trace_path)
  # Group 0 is unaligned; no durations are attached for it.
  g0_row = next(g for g in attach["groups"] if g["group_id"] == 0)
  assert g0_row["aligned"] is False
  assert g0_row["mismatched_positions"] >= 1
  # Durations for the other (aligned) group are attached; group 0 nodes are UNKNOWN.
  assert attach["matched_calls"] == 5  # only group 1 (5 members) matched


def test_attach_trace_aligned_attaches_medians(tmp_path):
  """An aligned trace attaches median durations to every node."""
  capture = _fixture_capture()
  groups: dict[int, list[str]] = {}
  for n in sorted(capture["arms"]["physical"]["nodes"], key=lambda x: x["id"]):
    groups.setdefault(n["group_id"], []).append(str(n["name"]))
  trace_path = str(tmp_path / "good.sqlite")
  _tiny_trace(trace_path, groups)
  attach = attach_trace(capture, trace_path)
  assert attach["aligned_groups"] == 2
  assert attach["matched_calls"] == 8
  # Median of 5 identical 500 ns rows = 0.5 us.
  assert all(abs(v - 0.5) < 1e-9 for v in attach["duration_by_call"].values())


def test_attach_trace_single_replay_evidence(tmp_path):
  """min_replays=1 accepts a one-replay trace and labels it single_replay."""
  capture = _fixture_capture()
  groups: dict[int, list[str]] = {}
  for n in sorted(capture["arms"]["physical"]["nodes"], key=lambda x: x["id"]):
    groups.setdefault(n["group_id"], []).append(str(n["name"]))
  con = sqlite3.connect(str(tmp_path / "one.sqlite"))
  cur = con.cursor()
  cur.execute("create table StringIds (id integer primary key, value text)")
  cur.execute("create table CUPTI_ACTIVITY_KIND_KERNEL ("
              "graphNodeId integer, shortName integer, gridX integer, gridY integer, gridZ integer,"
              "blockX integer, blockY integer, blockZ integer, start integer, end integer, graphId integer)")
  name_ids: dict[str, int] = {}
  for gid, names in groups.items():
    graph_id = 100 + gid
    base = 4294967296 * (gid + 1)
    for pos, name in enumerate(names):
      if name not in name_ids:
        name_ids[name] = len(name_ids) + 1
        cur.execute("insert into StringIds (id, value) values (?, ?)", (name_ids[name], name))
      cur.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?,?,?,?,?,?,?)", (
        base + pos, name_ids[name], 1, 1, 1, 1, 1, 1,
        pos * 1000, pos * 1000 + 500, graph_id))
  con.commit()
  con.close()
  # Strict default: one replay is not enough.
  strict = attach_trace(capture, str(tmp_path / "one.sqlite"))
  assert strict["aligned_groups"] == 0
  # Relaxed: single-replay evidence attaches and is labelled.
  relaxed = attach_trace(capture, str(tmp_path / "one.sqlite"), min_replays=1)
  assert relaxed["aligned_groups"] == 2
  assert all(g["single_replay"] for g in relaxed["groups"])
  assert all(abs(v - 0.5) < 1e-9 for v in relaxed["duration_by_call"].values())


@pytest.mark.skipif(not CAPTURE.exists(), reason="anchored capture not present")
def test_integration_real_capture_and_trace_if_present():
  """Load the real anchored capture + a real CUPTI trace when both exist."""
  candidates = [
    REPO / "docs/task_workflow/output/nv-decode-overlap-b3-2-cupti-20260804.sqlite",
    pathlib.Path("/tmp/b0_cuda_trace.sqlite"),
  ]
  trace_path = next((p for p in candidates if p.exists()), None)
  if trace_path is None:
    pytest.skip("no real CUPTI trace artifact present")
  with open(CAPTURE, encoding="utf-8") as f:
    capture = json.load(f)
  attach = attach_trace(capture, str(trace_path))
  assert attach["matched_calls"] == 1021
  assert attach["aligned_groups"] == 6
  report = compute_report(capture, attach, {"capture_path": str(CAPTURE), "capture_sha256": "x"},
                          HISTORICAL_CUDA_WALL_MS, "historical B0.2 anchor",
                          {"path": str(trace_path), "sha256": "x"})
  assert report["whole_token"]["logical_merged_cp_us"] > 0
  assert report["whole_token"]["physical_merged_cp_us"] > 0
  assert report["verdict"]["g_b3_d"] in (
    "SEMANTIC_CHAIN", "PLANNER_NOT_ROOT_CAUSE", "PLANNER_EFFECT_NOT_SCALE",
    "PLANNER_CANDIDATE", "ATTRIBUTION_CONFOUNDED")
