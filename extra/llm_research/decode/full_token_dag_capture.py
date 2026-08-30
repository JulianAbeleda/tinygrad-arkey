#!/usr/bin/env python3
"""Phase 4 pre-split full-token dependency capture and validation (CPU tooling).

This is measurement tooling ONLY. It adds standalone files and never edits
tinygrad runtime files. It must not be run on a GPU in this phase; --capture
is written for the live NV decode path but is NOT run here (--validate and
--synthetic are the CPU-only modes).

Modes
-----
--validate [--profile-jsonl X] [--full-token-dag Y] [--e2-rows Z]
           [--tolerance-us T] [--out W]
  Reproduces the E2 per-group rows from the existing per-group capture
  (default /tmp/replay_overlap_graph.jsonl, the session artifact of
  replay_overlap_probe.py) and demonstrates that merging those per-group
  captures is NOT a legal full-token DAG: per-group captures carry no
  cross-group edges, so cross-group dependency status is UNKNOWN, never
  independent. When a full-token DAG JSON (schema below) is supplied, it is
  restricted back to each existing group and checked against the E2 per-group
  rows (node count, serialized node-sum, critical path) within a declared
  tolerance, and the corrected full-token critical path and 2/3-queue list
  schedules are emitted. Schedule logic is imported from
  dag_critical_path_sim.py (compute_tails / list_schedule via compute_metrics).

--synthetic [--out W]
  Hermetic self-test: builds a known 7-node 2-group DAG with cross-group
  RAW/WAR/WAW edges, verifies the full-token schedule and that restriction to
  each group reproduces the per-group edges, and writes the DAG JSON.

--capture --depth N [--out W] [--profile-jsonl X]
  LIVE, GPU-REQUIRED, DO NOT RUN in this phase. Runs the decode model path
  under PROFILE=1 and wraps the JIT lower/split seam (see below) so that on
  the first token the pre-split linear is walked with ONE range-aware
  DepsTracker across all calls, retaining every RAW/WAR/WAW edge that crosses
  a current 32/64/128/256/468 boundary. Durations are attached from the
  HCQ_GRAPH_PROFILE_JSON replay lines when they are present.

Pre-split capture hook (standalone seam, no runtime edits)
----------------------------------------------------------
The decode harness captures each token's linear via TinyJit; jit_lower()
substitutes input uops with PARAMs, memory-plans, compiles, and then calls
graph_split_rewrite() which inserts the five graph-group execution barriers.
This script monkeypatches TWO module attributes of tinygrad.engine.jit:

  1. jit_lower: records the concrete (held_bufs, input_uops) so PARAMs in the
     post-substitution linear can be resolved to real Buffer objects exactly
     as GraphRunner.__init__ does (resolve_params + unwrap_multi +
     ensure_allocated).
  2. graph_split_rewrite: forwards to the original with a recording
     GraphAdmissionObserver (the same admission logic HCQGraph uses, so the
     batch boundaries are the same 32/64/128/256/468 the runtime builds) and
     then builds the full-token DAG from the pre-split linear: per call,
     bufs + outs with ranges via DepsTracker.access_resources (mirroring
     jit.py's _access_resources usage), one tracker across ALL calls.

The wrap lives entirely in capture_full_token_dag(); no tinygrad file is
modified, and the original functions are restored afterwards. If the seam
ever stops firing (e.g. jit_lower stops reaching graph_split_rewrite), the
fallback documented here is to wrap tinygrad.engine.realize.run_linear on the
first replay and rebuild the call order from the graph CALL batches; that
fallback is NOT implemented because the primary seam is clean.

DAG JSON schema (tinygrad.full_token_dag.v1)
--------------------------------------------
{
  "schema": "tinygrad.full_token_dag.v1",
  "nodes": [{"id": int, "name": str, "duration_us": float, "group_id": int|str,
             "metadata": {optional, "deps_status" may be "UNKNOWN"}}],
  "edges": [{"from": int, "to": int, "kind": "RAW"|"WAR"|"WAW",
             "crosses_group": bool}],
  "summary": {"node_count", "edge_count", "cross_group_edge_count",
              "per_group": {id: {"n", "serialized_us", "critical_path_us"}},
              "critical_path_us", "serialized_us",
              "schedules": {"queues_2": {"span_us", "saving_us"},
                            "queues_3": {"span_us", "saving_us"}},
              "unknown_dep_node_count", "independence_assumption"}
}

Missing dependency information is labeled UNKNOWN in every report; it is
never claimed independent. The reused sim functions mechanically treat absent
deps as ready, so when UNKNOWN nodes exist the schedule arithmetic is reported
as an independence ASSUMPTION, not a measurement.

Usage:
  python3 extra/llm_research/decode/full_token_dag_capture.py --validate
  python3 extra/llm_research/decode/full_token_dag_capture.py --synthetic
"""
from __future__ import annotations

import argparse, contextlib, contextvars, json, os, pathlib, sys
from typing import Any, Callable, Iterator

try:
  from extra.llm_research.decode import dag_critical_path_sim as _sim
except ImportError:  # run directly as a script: script dir is on sys.path
  import dag_critical_path_sim as _sim  # type: ignore

SCHEMA = "tinygrad.full_token_dag.v1"
KINDS = ("RAW", "WAR", "WAW")
UNKNOWN = "UNKNOWN"
DEFAULT_PROFILE_JSONL = "/tmp/replay_overlap_graph.jsonl"
DEFAULT_OUT = "/tmp/full_token_dag_capture_out.json"

# E2 record table (docs/.../nv-decode-parity-e2-dag-critical-path-measurement-
# record-20260803.md section 2.1), 1-based group labels, rounded to 0.1 us.
# Used ONLY as the reference baseline when no capture JSONL or summary JSON is
# supplied; any capture always takes precedence (never hardcode sizes as fact
# where they can be read from a capture).
E2_REFERENCE_ROWS = [
  {"label": "1", "n": 32, "serialized_us": 179.6, "critical_path_us": 153.6},
  {"label": "2", "n": 64, "serialized_us": 350.9, "critical_path_us": 288.3},
  {"label": "3", "n": 128, "serialized_us": 666.9, "critical_path_us": 592.3},
  {"label": "4", "n": 256, "serialized_us": 1340.7, "critical_path_us": 1183.3},
  {"label": "5", "n": 468, "serialized_us": 2828.1, "critical_path_us": 2539.9},
]


class FullTokenDagError(ValueError):
  pass


def load_json(path: str) -> Any:
  with open(path, encoding="utf-8") as f:
    return json.load(f)


def validate_schema(dag: dict) -> None:
  """Validate the full-token DAG JSON schema; raise FullTokenDagError on any violation."""
  if not isinstance(dag, dict):
    raise FullTokenDagError("dag must be a JSON object")
  if dag.get("schema") != SCHEMA:
    raise FullTokenDagError("dag schema must be %r, got %r" % (SCHEMA, dag.get("schema")))
  nodes = dag.get("nodes")
  if not isinstance(nodes, list) or not nodes:
    raise FullTokenDagError("dag.nodes must be a non-empty list")
  ids: set[int] = set()
  for n in nodes:
    if not isinstance(n, dict) or "id" not in n:
      raise FullTokenDagError("each node must be an object with an id")
    if not isinstance(n["id"], int) or isinstance(n["id"], bool):
      raise FullTokenDagError("node id must be an int")
    if n["id"] in ids:
      raise FullTokenDagError("duplicate node id %r" % (n["id"],))
    ids.add(n["id"])
    if "name" in n and not isinstance(n["name"], str):
      raise FullTokenDagError("node name must be a string")
    if "duration_us" in n and not isinstance(n["duration_us"], (int, float)):
      raise FullTokenDagError("node duration_us must be numeric")
    if "group_id" in n and not isinstance(n["group_id"], (int, str)):
      raise FullTokenDagError("node group_id must be an int or string")
  edges = dag.get("edges")
  if edges is not None:
    if not isinstance(edges, list):
      raise FullTokenDagError("dag.edges must be a list when present")
    for e in edges:
      if not isinstance(e, dict) or e.get("from") not in ids or e.get("to") not in ids:
        raise FullTokenDagError("edge endpoints must reference existing node ids: %r" % (e,))
      if e.get("kind") not in KINDS:
        raise FullTokenDagError("edge kind must be one of %r, got %r" % (KINDS, e.get("kind")))
      if not isinstance(e.get("crosses_group"), bool):
        raise FullTokenDagError("edge crosses_group must be a bool")
      if "spans" in e:
        if not isinstance(e["spans"], list) or any(
            not isinstance(span, list) or len(span) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) for v in span)
            or span[0] < 0 or span[1] < span[0] for span in e["spans"]):
          raise FullTokenDagError("edge spans must be a list of non-decreasing [lo, hi) integer pairs")
  summary = dag.get("summary")
  if summary is not None:
    if not isinstance(summary, dict):
      raise FullTokenDagError("summary must be an object when present")
    for key in ("node_count", "cross_group_edge_count", "critical_path_us"):
      if key not in summary:
        raise FullTokenDagError("summary missing required key %r" % key)
    if summary.get("node_count") != len(nodes):
      raise FullTokenDagError("summary.node_count does not match nodes list")
    if summary.get("cross_group_edge_count") != (
        0 if edges is None else sum(1 for e in edges if e.get("crosses_group"))):
      raise FullTokenDagError("summary.cross_group_edge_count does not match edges")
    per_group = summary.get("per_group")
    if not isinstance(per_group, dict):
      raise FullTokenDagError("summary.per_group must be an object")
    for gid, row in per_group.items():
      for key in ("n", "serialized_us", "critical_path_us"):
        if key not in row:
          raise FullTokenDagError("per_group row %r missing %r" % (gid, key))
    schedules = summary.get("schedules")
    if not isinstance(schedules, dict) or "queues_2" not in schedules or "queues_3" not in schedules:
      raise FullTokenDagError("summary.schedules must contain queues_2 and queues_3")
    for q in ("queues_2", "queues_3"):
      for key in ("span_us", "saving_us"):
        if key not in schedules[q]:
          raise FullTokenDagError("schedules.%s missing %r" % (q, key))


def to_sim_nodes(dag: dict) -> list[dict]:
  """Convert the DAG JSON to the node list dag_critical_path_sim consumes."""
  ids = {n["id"]: i for i, n in enumerate(dag["nodes"])}
  by_to: dict[int, list[int]] = {}
  for e in dag.get("edges") or []:
    by_to.setdefault(e["to"], []).append(e["from"])
  nodes = []
  for i, n in enumerate(dag["nodes"]):
    deps = [ids[d] for d in by_to.get(n["id"], []) if d in ids]
    nodes.append({"id": i, "name": str(n.get("name", "node-%d" % n["id"])),
                  "duration": float(n.get("duration_us", 0.0) or 0.0), "deps": deps})
  return nodes


def compute_dag_metrics(dag: dict) -> dict:
  """Full-token metrics via dag_critical_path_sim.compute_metrics (reused)."""
  return _sim.compute_metrics(to_sim_nodes(dag))


def restrict_dag(dag: dict, group_id: Any) -> dict:
  """Restrict a full-token DAG to one group: nodes and edges inside the group."""
  nodes = [n for n in dag["nodes"] if n.get("group_id") == group_id]
  ids = {n["id"] for n in nodes}
  edges = [e for e in (dag.get("edges") or []) if e.get("from") in ids and e.get("to") in ids]
  return {"nodes": nodes, "edges": edges}


def ordered_group_ids(dag: dict) -> list[Any]:
  """Group ids in first-appearance order of nodes."""
  seen: list[Any] = []
  for n in dag["nodes"]:
    g = n.get("group_id")
    if g not in seen:
      seen.append(g)
  return seen


def unknown_dep_nodes(dag: dict) -> list[int]:
  """Node ids whose dependency information is UNKNOWN (never assumed independent)."""
  if not isinstance(dag.get("edges"), list):
    return [n["id"] for n in dag["nodes"]]
  return [n["id"] for n in dag["nodes"]
          if isinstance(n.get("metadata"), dict) and n.get("metadata", {}).get("deps_status") == UNKNOWN]


def _savings(serialized_us: float, span_us: float) -> float:
  return round(serialized_us - span_us, 3)


def attach_summary(dag: dict) -> dict:
  """Compute and attach the summary block required by the schema."""
  validate_schema(dag)
  metrics = compute_dag_metrics(dag)
  serialized = metrics["serialized_span_us"]
  per_group: dict[str, dict] = {}
  for gid in ordered_group_ids(dag):
    rg = restrict_dag(dag, gid)
    m = compute_dag_metrics(rg)
    per_group[str(gid)] = {"n": m["node_count"], "serialized_us": round(m["serialized_span_us"], 3),
                           "critical_path_us": round(m["critical_path_us"], 3)}
  edges = dag.get("edges") or []
  unknown = unknown_dep_nodes(dag)
  out = dict(dag)
  out["summary"] = {
    "node_count": metrics["node_count"],
    "edge_count": len(edges),
    "cross_group_edge_count": sum(1 for e in edges if e.get("crosses_group")),
    "per_group": per_group,
    "critical_path_us": round(metrics["critical_path_us"], 3),
    "serialized_us": round(serialized, 3),
    "schedules": {
      "queues_2": {"span_us": round(metrics["schedule_2q_us"], 3),
                   "saving_us": _savings(serialized, metrics["schedule_2q_us"])},
      "queues_3": {"span_us": round(metrics["schedule_3q_us"], 3),
                   "saving_us": _savings(serialized, metrics["schedule_3q_us"])},
    },
    "unknown_dep_node_count": len(unknown),
    "independence_assumption": len(unknown) > 0,
  }
  return out


def emit_dag(dag: dict, path: str | None) -> None:
  validate_schema(dag)
  text = json.dumps(dag, indent=2, sort_keys=True) + "\n"
  if path is None:
    sys.stdout.write(text)
  else:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      f.write(text)


# ---------------------------------------------------------------------------
# E2 per-group reference loading and comparison
# ---------------------------------------------------------------------------

def load_profile_records(path: str) -> list[dict]:
  return _sim.load_records(path)


def profile_rows(records: list[dict]) -> list[dict]:
  """One row per JSONL line: n, serialized, critical path, and raw deps."""
  rows = []
  for idx, rec in enumerate(records):
    nodes, _ = _sim.build_nodes(rec)
    m = _sim.compute_metrics(nodes)
    rows.append({"group_id": str(idx), "label": str(idx + 1), "n": m["node_count"],
                 "serialized_us": m["serialized_span_us"], "critical_path_us": m["critical_path_us"],
                 "entries": len(nodes), "deps": [n["deps"] for n in nodes]})
  return rows


def load_e2_rows(path: str | None) -> list[dict] | None:
  """Load E2 reference rows from a summary or sim JSON artifact if present."""
  if path is None or not os.path.exists(path):
    return None
  data = load_json(path)
  if isinstance(data, dict) and isinstance(data.get("rows"), list):
    return [{"label": str(r.get("group", i)), "n": r["kernels"],
             "serialized_us": float(r["node_sum_us"]), "critical_path_us": None}
            for i, r in enumerate(data["rows"])]
  if isinstance(data, dict) and isinstance(data.get("groups"), dict):
    return [{"label": str(i + 1), "n": g["node_count"],
             "serialized_us": g["serialized_span_us"], "critical_path_us": g["critical_path_us"]}
            for i, (_, g) in enumerate(sorted(data["groups"].items()))]
  raise FullTokenDagError("unsupported --e2-rows JSON shape (want probe summary 'rows' or sim 'groups')")


def compare_rows(reference: list[dict], actual: list[dict], tolerance_us: float) -> list[dict]:
  """Compare restricted-DAG rows against E2 rows by position (group order)."""
  diffs = []
  for i, (ref, act) in enumerate(zip(reference, actual)):
    issues = []
    if act.get("n") != ref["n"]:
      issues.append("node count %r != %r" % (act.get("n"), ref["n"]))
    sd = abs(act.get("serialized_us", 0.0) - ref["serialized_us"])
    if sd > tolerance_us:
      issues.append("serialized diff %.3f us > tolerance %.3f" % (sd, tolerance_us))
    cd = None
    if ref.get("critical_path_us") is not None and act.get("critical_path_us") is not None:
      cd = abs(act["critical_path_us"] - ref["critical_path_us"])
      if cd > tolerance_us:
        issues.append("critical-path diff %.3f us > tolerance %.3f" % (cd, tolerance_us))
    diffs.append({"group": str(i), "ok": not issues, "issues": issues,
                  "serialized_diff_us": round(sd, 3),
                  "critical_path_diff_us": None if cd is None else round(cd, 3)})
  return diffs


def compare_restricted_edges(restricted: dict, line_deps: list[list[int]]) -> tuple[set, set]:
  """Map restricted DAG edges to per-group entry positions and compare to E2 deps."""
  node_ids = sorted(n["id"] for n in restricted["nodes"])
  pos = {nid: i for i, nid in enumerate(node_ids)}
  dag_edges = set()
  for e in restricted.get("edges") or []:
    if e.get("from") in pos and e.get("to") in pos:
      dag_edges.add((pos[e["from"]], pos[e["to"]]))
  e2_edges = set()
  for to_i, deps in enumerate(line_deps):
    for d in deps:
      if isinstance(d, int) and 0 <= d < len(line_deps):
        e2_edges.add((d, to_i))
  return dag_edges, e2_edges


def _reference_source(profile_jsonl: str | None, e2_rows_path: str | None) -> tuple[list[dict], str, str]:
  """Pick the E2 reference rows: capture JSONL, then summary JSON, then the record table."""
  if profile_jsonl is not None and os.path.exists(profile_jsonl):
    rows = profile_rows(load_profile_records(profile_jsonl))
    return rows, "profile JSONL", profile_jsonl
  rows = load_e2_rows(e2_rows_path)
  if rows is not None:
    return rows, "summary JSON", e2_rows_path or ""
  return [dict(r) for r in E2_REFERENCE_ROWS], "E2 record table", "embedded reference"


def merged_hypothesis(records: list[dict]) -> dict | None:
  """Merged no-cross-edge hypothesis for the per-group capture (sim union, no hardcoding)."""
  if not records:
    return None
  union, dropped = _sim.union_nodes(records)
  m = _sim.compute_metrics(union)
  serial = m["serialized_span_us"]
  out = {"cross_group_edges": dropped, "serialized_us": serial,
         "critical_path_us": m["critical_path_us"],
         "schedule_2q_us": m["schedule_2q_us"], "schedule_3q_us": m["schedule_3q_us"]}
  out["schedule_2q_pct"] = round(100 * (serial - out["schedule_2q_us"]) / serial if serial else 0.0, 1)
  out["schedule_3q_pct"] = round(100 * (serial - out["schedule_3q_us"]) / serial if serial else 0.0, 1)
  return out


# ---------------------------------------------------------------------------
# Synthetic self-test
# ---------------------------------------------------------------------------

def build_synthetic_dag() -> dict:
  """Known 7-node 2-group DAG with cross-group RAW/WAR/WAW edges (see module tests)."""
  nodes = [
    {"id": 0, "name": "a_w0", "duration_us": 10.0, "group_id": 0, "metadata": None},
    {"id": 1, "name": "a_r1", "duration_us": 20.0, "group_id": 0, "metadata": None},
    {"id": 2, "name": "a_w2", "duration_us": 15.0, "group_id": 0, "metadata": None},
    {"id": 3, "name": "b_r3", "duration_us": 8.0, "group_id": 1, "metadata": None},
    {"id": 4, "name": "b_r4", "duration_us": 12.0, "group_id": 1, "metadata": None},
    {"id": 5, "name": "b_w5", "duration_us": 6.0, "group_id": 1, "metadata": None},
    {"id": 6, "name": "b_r6", "duration_us": 10.0, "group_id": 1, "metadata": None},
  ]
  edges = [
    {"from": 0, "to": 1, "kind": "RAW", "crosses_group": False},
    {"from": 0, "to": 2, "kind": "WAW", "crosses_group": False},
    {"from": 0, "to": 5, "kind": "WAW", "crosses_group": True},
    {"from": 1, "to": 3, "kind": "RAW", "crosses_group": True},
    {"from": 1, "to": 4, "kind": "RAW", "crosses_group": True},
    {"from": 1, "to": 5, "kind": "WAR", "crosses_group": True},
    {"from": 1, "to": 6, "kind": "RAW", "crosses_group": True},
    {"from": 3, "to": 6, "kind": "RAW", "crosses_group": False},
  ]
  return {"schema": SCHEMA, "nodes": nodes, "edges": edges}


SYNTHETIC_EXPECTED = {
  "node_count": 7, "edge_count": 8, "cross_group_edge_count": 5,
  "serialized_us": 81.0, "critical_path_us": 48.0,
  "schedule_2q_us": 56.0, "schedule_2q_saving_us": 25.0,
  "schedule_3q_us": 48.0, "schedule_3q_saving_us": 33.0,
  "per_group": {
    "0": {"n": 3, "serialized_us": 45.0, "critical_path_us": 30.0},
    "1": {"n": 4, "serialized_us": 36.0, "critical_path_us": 18.0},
  },
  "cross_kinds": {"RAW": 3, "WAR": 1, "WAW": 1},
}


def run_synthetic(out: str | None = None) -> dict:
  """Build, validate, schedule, and assert the synthetic DAG; return the emitted DAG."""
  dag = attach_summary(build_synthetic_dag())
  exp = SYNTHETIC_EXPECTED
  s = dag["summary"]
  checks = []

  def check(name: str, ok: bool, detail: str) -> None:
    checks.append((name, ok, detail))
    if not ok:
      raise FullTokenDagError("synthetic self-test failed: %s (%s)" % (name, detail))

  check("node_count", s["node_count"] == exp["node_count"], "got %r" % s["node_count"])
  check("edge_count", s["edge_count"] == exp["edge_count"], "got %r" % s["edge_count"])
  check("cross_group_edge_count", s["cross_group_edge_count"] == exp["cross_group_edge_count"],
        "got %r" % s["cross_group_edge_count"])
  check("serialized_us", s["serialized_us"] == exp["serialized_us"], "got %r" % s["serialized_us"])
  check("critical_path_us", s["critical_path_us"] == exp["critical_path_us"],
        "got %r" % s["critical_path_us"])
  check("schedule_2q_us", s["schedules"]["queues_2"]["span_us"] == exp["schedule_2q_us"],
        "got %r" % s["schedules"]["queues_2"]["span_us"])
  check("schedule_2q_saving_us", s["schedules"]["queues_2"]["saving_us"] == exp["schedule_2q_saving_us"],
        "got %r" % s["schedules"]["queues_2"]["saving_us"])
  check("schedule_3q_us", s["schedules"]["queues_3"]["span_us"] == exp["schedule_3q_us"],
        "got %r" % s["schedules"]["queues_3"]["span_us"])
  check("schedule_3q_saving_us", s["schedules"]["queues_3"]["saving_us"] == exp["schedule_3q_saving_us"],
        "got %r" % s["schedules"]["queues_3"]["saving_us"])
  for gid, row in exp["per_group"].items():
    got = s["per_group"].get(gid)
    check("per_group_%s" % gid, got == row, "got %r" % got)
  cross_kinds = {}
  for e in dag["edges"]:
    if e["crosses_group"]:
      cross_kinds[e["kind"]] = cross_kinds.get(e["kind"], 0) + 1
  check("cross_kinds", cross_kinds == exp["cross_kinds"], "got %r" % cross_kinds)
  for gid in ("0", "1"):
    rg = restrict_dag(dag, int(gid))
    check("restriction_group_%s_edge_kinds" % gid,
          all(not e["crosses_group"] for e in rg["edges"]),
          "restricted edges still marked crosses_group")
  if out is not None:
    emit_dag(dag, out)
  return dag


# ---------------------------------------------------------------------------
# Live pre-split capture (GPU-only; do not run in this phase)
# ---------------------------------------------------------------------------

_CAPTURE_STATE: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
  "full_token_dag_capture_state", default=None)


class RecordingDepsTracker:
  """Range-aware DepsTracker mirroring jit.py usage, with edge kinds labeled.

  State mutation is delegated to tinygrad.engine.jit.DepsTracker (the canonical
  edge builder); this class only records which kind (RAW/WAR/WAW) each returned
  wait edge was, by scanning the same write/read range maps the engine scans.
  """

  def __init__(self):
    from tinygrad.engine.jit import DepsTracker
    self._tracker = DepsTracker()
    self.edges: list[tuple[int, int, str]] = []
    self.span_edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    self.span_skipped: list[tuple[int, int, Any, Any]] = []

  def _span(self, lo: Any, hi: Any) -> tuple[int, int] | None:
    if all(isinstance(v, int) and not isinstance(v, bool) for v in (lo, hi)) and lo <= hi:
      return int(lo), int(hi)
    self.span_skipped.append((lo, hi, type(lo).__name__, type(hi).__name__))
    return None

  def access_resources(self, bufs: list[Any], write: list[int], new_dependency: int) -> list[Any]:
    kinds: dict[tuple[int, int], str] = {}
    spans: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i, buf in enumerate(bufs):
      key = id(buf.base)
      s, e = buf.offset, buf.offset + buf.nbytes
      if i in write:
        for st, en, dep in self._tracker.w_dependency_map[key]:
          if st < e and s < en:
            kinds.setdefault((id(dep), id(new_dependency)), "WAW")
            if (span := self._span(max(st, s), min(en, e))) is not None:
              spans.setdefault((id(dep), id(new_dependency)), []).append(span)
        for st, en, dep in self._tracker.r_dependency_map[key]:
          if st < e and s < en:
            kinds.setdefault((id(dep), id(new_dependency)), "WAR")
            if (span := self._span(max(st, s), min(en, e))) is not None:
              spans.setdefault((id(dep), id(new_dependency)), []).append(span)
      else:
        for st, en, dep in self._tracker.w_dependency_map[key]:
          if st < e and s < en:
            kinds.setdefault((id(dep), id(new_dependency)), "RAW")
            if (span := self._span(max(st, s), min(en, e))) is not None:
              spans.setdefault((id(dep), id(new_dependency)), []).append(span)
    wait_nodes = self._tracker.access_resources(bufs, write, new_dependency)
    for dep in wait_nodes:
      edge_key = (id(dep), id(new_dependency))
      self.edges.append((int(dep), int(new_dependency), kinds.get(edge_key, UNKNOWN)))
      self.span_edges[(int(dep), int(new_dependency))] = spans.get(edge_key, [])
    return wait_nodes


class _RecordingObserver:
  """GraphAdmissionObserver that records batch assignments and forwards to any existing observer."""

  def __init__(self, forward: Callable[[Any], None] | None):
    self.forward = forward
    self.records: list[Any] = []

  def __call__(self, event: Any) -> None:
    if self.forward is not None:
      self.forward(event)
    from tinygrad.engine.jit import GraphAdmissionObservation
    if isinstance(event, GraphAdmissionObservation):
      self.records.append(event)

  def bind_call(self, call_index: int, call: Any) -> None:
    if hasattr(self.forward, "bind_call"):
      self.forward.bind_call(call_index, call)


def _group_map(records: list[Any]) -> dict[int, Any]:
  """call_index -> group id: graph members use batch_index; direct calls get own ids."""
  m: dict[int, Any] = {}
  for r in records:
    if r.assignment == "graph":
      m[r.call_index] = r.batch_index
    elif r.assignment == "direct":
      m[r.call_index] = "direct-%d" % r.direct_call_index
    else:
      m[r.call_index] = None
  return m


def _call_name(call: Any) -> str:
  from tinygrad.uop.ops import Ops
  ast = call.src[0]
  if ast.op is Ops.PROGRAM:
    return str(ast.arg.name)
  if ast.op is Ops.SLICE:
    return "view"
  if ast.op is Ops.COPY:
    return "copy"
  if ast.op is Ops.CUSTOM_FUNCTION:
    return "custom-%s" % ast.arg
  return str(ast.op)


def _call_metadata(call: Any) -> list[dict] | None:
  md = getattr(call, "arg", None)
  md = getattr(md, "metadata", None)
  if not md:
    return None
  out = []
  for m in md:
    row = {}
    for field in ("name", "caller", "backward"):
      if hasattr(m, field):
        row[field] = getattr(m, field)
    if row:
      out.append(row)
  return out or None


def _admission_metadata(records: list[Any]) -> dict[int, list[dict]]:
  """Serialize semantic metadata already observed at graph admission.

  The lowered CALL does not retain metadata in ``call.arg`` on every route,
  while ``GraphAdmissionObservation`` does.  This is attribution-only: the
  observer is already present and does not alter graph construction.
  """
  out: dict[int, list[dict]] = {}
  for record in records:
    rows = []
    for item in getattr(record, "metadata", ()):
      row = {field:getattr(item, field) for field in ("name", "caller", "backward") if hasattr(item, field)}
      for field in ("phase", "tensor_name", "module_path", "role", "logical_m", "logical_n", "logical_k",
                    "source_quant_storage", "source_layout", "module_representation", "input_dtype", "output_dtype",
                    "accumulator_dtype"):
        if hasattr(item, field): row[field] = getattr(item, field)
      if row: rows.append(row)
    if rows: out[int(record.call_index)] = rows
  return out


def _build_full_token_dag(linear: Any, input_uops: tuple[Any, ...], group_map: dict[int, Any],
                          admission_records: list[Any] | None = None) -> dict:
  """Walk the pre-split linear with ONE range-aware DepsTracker across all calls."""
  from tinygrad.engine.realize import (get_call_arg_uops, get_call_outs_ins, unwrap_multi, resolve_params)
  from tinygrad.uop.ops import Ops
  tracker = RecordingDepsTracker()
  nodes: list[dict] = []
  unknown_nodes: list[int] = []
  observed_metadata = _admission_metadata(admission_records or [])
  for call_index, call in enumerate(linear.src):
    node_id = call_index
    group_id = group_map.get(call_index)
    dep_unknown = False
    try:
      arg_uops = resolve_params(call, input_uops)
      outs, _ins = get_call_outs_ins(call)
      all_bufs: list[Any] = []
      write_idx: list[int] = []
      for bufs, _device_vars in unwrap_multi(call, arg_uops):
        start = len(all_bufs)
        all_bufs.extend(b.ensure_allocated() for b in bufs)
        write_idx.extend(start + i for i in outs)
      tracker.access_resources(all_bufs, write_idx, node_id)
    except Exception:
      dep_unknown = True
      unknown_nodes.append(node_id)
    metadata: dict[str, Any] = {}
    if group_id is None:
      group_id = "unassigned-%d" % call_index
      metadata["group_status"] = "unassigned"
    if dep_unknown:
      metadata["deps_status"] = UNKNOWN
    meta = observed_metadata.get(call_index) or _call_metadata(call)
    if meta is not None:
      metadata["semantic"] = meta
    nodes.append({"id": node_id, "name": _call_name(call), "duration_us": 0.0,
                  "group_id": group_id, "metadata": metadata or None})
  edges = []
  for dep, new, kind in tracker.edges:
    dep_group = next((n["group_id"] for n in nodes if n["id"] == dep), None)
    new_group = next((n["group_id"] for n in nodes if n["id"] == new), None)
    edge_row = {"from": dep, "to": new, "kind": kind,
                "crosses_group": dep_group is not None and new_group is not None and dep_group != new_group}
    edge_spans = tracker.span_edges.get((dep, new))
    if edge_spans:
      edge_row["spans"] = sorted([list(span) for span in set(edge_spans)])
    edges.append(edge_row)
  span_skip_samples = []
  seen_skips: set[tuple[Any, Any, str, str]] = set()
  for lo, hi, lot, hit in tracker.span_skipped:
    key = (repr(lo), repr(hi), lot, hit)
    if key in seen_skips or len(span_skip_samples) >= 20: continue
    seen_skips.add(key)
    span_skip_samples.append({"lo": repr(lo), "hi": repr(hi), "lo_type": lot, "hi_type": hit})
  return {"schema": SCHEMA, "nodes": nodes, "edges": edges,
          "span_skip_summary": {"count": len(tracker.span_skipped), "samples": span_skip_samples}}


def _select_dag(dags: list[dict]) -> dict:
  """Pick the decode-token DAG by its semantic 36-layer flash population."""
  if not dags:
    raise FullTokenDagError("no pre-split linear was captured; the jit_lower/graph_split_rewrite seam did not fire")
  # Retain the historical signature and recognize the current composed decode
  # graph after completion/cache-sink/node-elimination promotions.
  decode_signatures = ([32, 64, 128, 228], [32, 64, 128, 256, 468])

  def sizes(dag: dict) -> list[int]:
    counts: dict[Any, int] = {}
    order: list[Any] = []
    for n in dag["nodes"]:
      g = n.get("group_id")
      if g not in counts:
        counts[g] = 0
        order.append(g)
      counts[g] += 1
    return [counts[g] for g in order]

  def flash_count(dag: dict) -> int:
    return sum(str(n.get("name", "")).startswith("flash_block_tiled_xlane_score_pv_tile_whole_cache")
               for n in dag.get("nodes", []))

  semantic = [dag for dag in dags if flash_count(dag) == 36]
  if len(semantic) == 1:
    return semantic[0]
  if len(semantic) > 1:
    for signature in decode_signatures:
      for dag in semantic:
        if sizes(dag) == signature:
          return dag
    return min(semantic, key=lambda d: len(d.get("nodes", [])))

  for signature in decode_signatures:
    for dag in dags:
      if sizes(dag) == signature:
        return dag
  raise FullTokenDagError("no semantic decode DAG (36 flash-score nodes); candidates=%r" %
    [{"sizes":sizes(d), "nodes":len(d.get("nodes", [])), "flash":flash_count(d)} for d in dags])


def _apply_profile_durations(dag: dict, jsonl_path: str) -> dict:
  """Attach durations from HCQ_GRAPH_PROFILE_JSON lines whose entry counts match group sizes."""
  if not jsonl_path or not os.path.exists(jsonl_path):
    return dag
  records = load_profile_records(jsonl_path)
  if not records:
    return dag
  groups = ordered_group_ids(dag)
  sizes = []
  for gid in groups:
    sizes.append(sum(1 for n in dag["nodes"] if n.get("group_id") == gid))
  run_len = len(groups)
  matches = [i for i in range(0, len(records) - run_len + 1)
             if [len(r.get("entries") or []) for r in records[i:i + run_len]] == sizes]
  if not matches:
    return dag
  start = matches[-1]  # last matching replay cycle = most measured
  nodes = list(dag["nodes"])
  for gi, gid in enumerate(groups):
    members = sorted([n for n in nodes if n.get("group_id") == gid], key=lambda n: n["id"])
    entries = records[start + gi].get("entries") or []
    for i, n in enumerate(members):
      if i < len(entries):
        try:
          n["duration_us"] = round(float(entries[i].get("duration", 0) or 0), 3)
        except (TypeError, ValueError):
          n["duration_us"] = 0.0
  return dag


@contextlib.contextmanager
def capture_full_token_dag(harness: Callable[[], Any], dag_path: str | None = None) -> Iterator[dict]:
  """Install the pre-split capture seam, run the harness, restore, yield the DAG.

  The seam wraps tinygrad.engine.jit.jit_lower (to record the concrete
  input_uops needed to resolve PARAMs) and tinygrad.engine.jit.graph_split_rewrite
  (to record the graph-group boundaries via its admission observer and to build
  the full-token DAG with one range-aware DepsTracker across all calls). No
  tinygrad runtime file is modified.
  """
  from tinygrad.engine import jit as tjit
  state: dict = {"input_uops": None, "dags": [], "captures": 0}
  token = _CAPTURE_STATE.set(state)
  orig_lower, orig_split = tjit.jit_lower, tjit.graph_split_rewrite

  def wrapped_jit_lower(linear: Any, held_bufs: set[Any], input_uops: list[Any]) -> Any:
    state["input_uops"] = tuple(input_uops)
    return orig_lower(linear, held_bufs, input_uops)

  def wrapped_graph_split(linear: Any, max_batch_size: int = 0, observer: Callable[[Any], None] | None = None) -> Any:
    rec = _RecordingObserver(observer)
    result = orig_split(linear, max_batch_size=max_batch_size, observer=rec)
    group_map = _group_map(rec.records)
    state["dags"].append(_build_full_token_dag(linear, state["input_uops"] or (), group_map, rec.records))
    state["captures"] += 1
    return result

  tjit.jit_lower = wrapped_jit_lower
  tjit.graph_split_rewrite = wrapped_graph_split
  try:
    harness()
    dag = _select_dag(state["dags"])
    if dag_path is not None:
      emit_dag(attach_summary(dag), dag_path)
    yield dag
  finally:
    tjit.jit_lower = orig_lower
    tjit.graph_split_rewrite = orig_split
    _CAPTURE_STATE.reset(token)


def _decode_harness(depth: int, model_path: str) -> None:
  """Replicate replay_overlap_probe's drive: capture token then replay tokens."""
  import pathlib
  import tinygrad.llm.model as tgm
  from tinygrad.llm.model import Transformer
  from tinygrad.device import Device
  from tinygrad.helpers import Context
  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  model, _kv = Transformer.from_gguf(model_path, 4608)
  prompt = [1] * depth
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  pathlib.Path(os.environ.get("HCQ_GRAPH_PROFILE_JSON", "")).unlink(missing_ok=True)
  with Context(DEBUG=0):
    next(gen)  # capture token: jit_lower + graph_split_rewrite fire here
  with Context(DEBUG=0):
    next(gen)  # replay 1
  with Context(DEBUG=0):
    next(gen)  # replay 2 (measured)
  Device["NV"].synchronize()
  with Context(DEBUG=0):
    next(gen)  # flush: collects the measured replay's timestamps into the JSONL


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def format_validate_report(report: dict) -> str:
  lines = ["== Phase 4 full-token dependency capture: --validate =="]
  lines.append("E2 per-group reference source: %s" % report["reference_source"])
  lines.append("declared tolerance: %.1f us" % report["tolerance_us"])
  lines.append("")
  lines.append("per-group rows (restriction baseline):")
  for row in report["per_group_rows"]:
    lines.append("  group %s (n=%d): serialized %.3f us, critical path %.3f us"
                 % (row["group"], row["n"], row["serialized_us"], row["critical_path_us"]))
  merged = report.get("merged_not_legal", {})
  if merged:
    lines.append("")
    lines.append("merged per-group capture is NOT a legal full-token DAG:")
    lines.append("  cross_group_edges recorded: %s" % merged.get("cross_group_edges", 0))
    lines.append("  cross-group dependency status: %s (never assumed independent)" % UNKNOWN)
    if merged.get("schedule_2q_us") is not None:
      lines.append("  merged 2-queue %.1f us / %.1f%% and 3-queue %.1f us / %.1f%% savings are hypotheses"
                   % (merged["schedule_2q_us"], merged.get("schedule_2q_pct", 0.0),
                      merged["schedule_3q_us"], merged.get("schedule_3q_pct", 0.0)))
    else:
      lines.append("  merged 2/3-queue savings: not computable from the supplied reference")
    lines.append("  (%s)" % merged["reason"])
  full = report.get("full_token_dag")
  if full:
    lines.append("")
    lines.append("full-token DAG restriction validation:")
    for cmp in full["group_checks"]:
      status = "OK" if cmp["ok"] else "FAIL"
      lines.append("  group %s (n=%d): %s  serialized diff %.3f us, critical-path diff %s"
                   % (cmp["group"], cmp["n"], status, cmp["serialized_diff_us"],
                      "n/a" if cmp["critical_path_diff_us"] is None else "%.3f us" % cmp["critical_path_diff_us"]))
      for issue in cmp["issues"]:
        lines.append("    - %s" % issue)
    edge = full["edge_checks"]
    if edge.get("ran"):
      lines.append("  restricted edge sets vs E2 deps: %d/%d matched, %d dag-only, %d e2-only"
                   % (edge["matched"], edge["expected"], edge["dag_only"], edge["e2_only"]))
    else:
      lines.append("  restricted edge comparison: skipped (no profile JSONL supplied)")
    lines.append("")
    lines.append("corrected full-token schedules (dag_critical_path_sim reuse):")
    sched = full["schedules"]
    lines.append("  serialized: %.3f us" % sched["serialized_us"])
    lines.append("  critical path: %.3f us (saving %.3f us / %.1f%%)"
                 % (sched["critical_path_us"], sched["saving_us"], sched["saving_pct"]))
    for q in ("queues_2", "queues_3"):
      lines.append("  %s: %.3f us (saving %.3f us / %.1f%%)"
                   % (q, sched[q]["span_us"], sched[q]["saving_us"], sched[q]["saving_pct"]))
    lines.append("  cross_group_edge_count: %d" % full["cross_group_edge_count"])
    lines.append("  UNKNOWN-dep nodes: %d (independence_assumption=%s)"
                 % (full["unknown_dep_node_count"], full["independence_assumption"]))
    if full["unknown_dep_node_count"]:
      lines.append("  WARNING: %d nodes have %s dependency status; the schedules above treat them as"
                   " independent for arithmetic only, which is NOT a measurement."
                   % (full["unknown_dep_node_count"], UNKNOWN))
  return "\n".join(lines) + "\n"


def run_validation(profile_jsonl: str | None, full_token_dag_path: str | None,
                   e2_rows_path: str | None, tolerance_us: float, out: str | None) -> dict:
  reference, source, source_path = _reference_source(profile_jsonl, e2_rows_path)
  report: dict[str, Any] = {"schema": "tinygrad.full_token_dag.validate.v1",
                            "reference_source": "%s (%s)" % (source, source_path),
                            "tolerance_us": tolerance_us,
                            "per_group_rows": []}
  for i, row in enumerate(reference):
    report["per_group_rows"].append({"group": str(i), "label": row.get("label", str(i)),
                                     "n": row["n"], "serialized_us": row["serialized_us"],
                                     "critical_path_us": row.get("critical_path_us")})
  records = load_profile_records(profile_jsonl) if profile_jsonl and os.path.exists(profile_jsonl) else []
  merged = merged_hypothesis(records)
  report["merged_not_legal"] = merged if merged is not None else {
    "cross_group_edges": 0, "serialized_us": None, "critical_path_us": None,
    "schedule_2q_us": None, "schedule_3q_us": None,
    "reason": "no per-group capture supplied; merged schedule hypothesis not computable"}
  report["merged_not_legal"]["reason"] = (
    "per-group HCQGraph captures lack cross-group edges; cross_group_edges=0 means "
    "'not recorded', never 'proven absent'")
  if full_token_dag_path is not None:
    dag = load_json(full_token_dag_path)
    validate_schema(dag)
    dag = attach_summary(dag)
    s = dag["summary"]
    checks = []
    for i, gid in enumerate(ordered_group_ids(dag)):
      rg = restrict_dag(dag, gid)
      m = compute_dag_metrics(rg)
      ref = reference[i] if i < len(reference) else None
      cmp = {"group": str(gid), "n": m["node_count"], "ok": True, "issues": [],
             "serialized_us": m["serialized_span_us"],
             "critical_path_us": m["critical_path_us"],
             "serialized_diff_us": None, "critical_path_diff_us": None}
      if ref is not None:
        sd = abs(m["serialized_span_us"] - ref["serialized_us"])
        cmp["serialized_diff_us"] = round(sd, 3)
        if cmp["n"] != ref["n"]:
          cmp["ok"] = False
          cmp["issues"].append("node count %d != E2 %d" % (cmp["n"], ref["n"]))
        if sd > tolerance_us:
          cmp["ok"] = False
          cmp["issues"].append("serialized diff %.3f us > %.3f" % (sd, tolerance_us))
        if ref.get("critical_path_us") is not None:
          cd = abs(m["critical_path_us"] - ref["critical_path_us"])
          cmp["critical_path_diff_us"] = round(cd, 3)
          if cd > tolerance_us:
            cmp["ok"] = False
            cmp["issues"].append("critical-path diff %.3f us > %.3f" % (cd, tolerance_us))
      checks.append(cmp)
    edge_checks = {"ran": False, "matched": 0, "expected": 0, "dag_only": 0, "e2_only": 0}
    if profile_jsonl is not None and os.path.exists(profile_jsonl):
      edge_checks["ran"] = True
      rows = profile_rows(load_profile_records(profile_jsonl))
      for i, gid in enumerate(ordered_group_ids(dag)):
        if i >= len(rows):
          break
        dag_edges, e2_edges = compare_restricted_edges(restrict_dag(dag, gid), rows[i]["deps"])
        edge_checks["expected"] += len(e2_edges)
        edge_checks["matched"] += len(dag_edges & e2_edges)
        edge_checks["dag_only"] += len(dag_edges - e2_edges)
        edge_checks["e2_only"] += len(e2_edges - dag_edges)
    serial = s["serialized_us"]
    schedules = {"serialized_us": serial, "critical_path_us": s["critical_path_us"],
                 "saving_us": round(serial - s["critical_path_us"], 3),
                 "saving_pct": round(100 * (serial - s["critical_path_us"]) / serial if serial else 0.0, 1),
                 "queues_2": {k: s["schedules"]["queues_2"][k] for k in ("span_us", "saving_us")},
                 "queues_3": {k: s["schedules"]["queues_3"][k] for k in ("span_us", "saving_us")}}
    for q in ("queues_2", "queues_3"):
      schedules[q]["saving_pct"] = round(100 * schedules[q]["saving_us"] / serial if serial else 0.0, 1)
    report["full_token_dag"] = {
      "path": full_token_dag_path, "group_checks": checks, "edge_checks": edge_checks,
      "schedules": schedules, "cross_group_edge_count": s["cross_group_edge_count"],
      "unknown_dep_node_count": s["unknown_dep_node_count"],
      "independence_assumption": s["independence_assumption"],
      "per_group_summary": s["per_group"],
    }
    report["pass"] = all(c["ok"] for c in checks) and edge_checks["dag_only"] == 0 and edge_checks["e2_only"] == 0
  else:
    report["pass"] = True
    report["full_token_dag"] = None
    report["note"] = ("no --full-token-dag supplied; restriction logic demonstrated on the per-group "
                      "capture only. Provide the pre-split full-token DAG JSON to validate cross-group edges.")
  if out is not None:
    with open(out, "w", encoding="utf-8") as f:
      json.dump(report, f, indent=2, sort_keys=True)
      f.write("\n")
  return report


def main_capture(args: argparse.Namespace) -> int:
  os.environ["PROFILE"] = "1"
  os.environ["HCQ_GRAPH_PROFILE_JSON"] = args.profile_jsonl
  dag = None
  with capture_full_token_dag(lambda: _decode_harness(args.depth, args.model), args.out) as dag:
    pass
  assert dag is not None
  dag = _apply_profile_durations(dag, args.profile_jsonl)
  dag = attach_summary(dag)
  emit_dag(dag, args.out)
  sys.stdout.write("== Phase 4 full-token DAG capture (live, GPU) ==\n")
  sys.stdout.write("captured %d nodes, %d edges, %d cross-group edges from the pre-split linear\n"
                   % (len(dag["nodes"]), len(dag["edges"]),
                      sum(1 for e in dag["edges"] if e["crosses_group"])))
  sys.stdout.write("wrote %s\n" % args.out)
  return 0


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(
    description="Phase 4 pre-split full-token dependency capture/validation (CPU tooling).")
  mode = ap.add_mutually_exclusive_group(required=True)
  mode.add_argument("--validate", action="store_true", help="validate full-token DAG restriction against E2 per-group rows")
  mode.add_argument("--synthetic", action="store_true", help="hermetic synthetic-DAG self-test")
  mode.add_argument("--capture", action="store_true", help="live GPU capture (do NOT run in this phase)")
  ap.add_argument("--profile-jsonl", default=DEFAULT_PROFILE_JSONL,
                  help="per-group HCQ_GRAPH_PROFILE_JSON capture (default %s)" % DEFAULT_PROFILE_JSONL)
  ap.add_argument("--full-token-dag", default=None, help="pre-split full-token DAG JSON to validate")
  ap.add_argument("--e2-rows", default=None, help="E2 summary/sim JSON with per-group rows")
  ap.add_argument("--tolerance-us", type=float, default=1.0, help="declared per-group tolerance (default 1.0)")
  ap.add_argument("--out", default=None, help="output JSON path")
  ap.add_argument("--depth", type=int, default=512, help="decode depth for --capture")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", help="GGUF model for --capture")
  args = ap.parse_args(argv)
  try:
    if args.validate:
      report = run_validation(args.profile_jsonl, args.full_token_dag, args.e2_rows,
                              args.tolerance_us, args.out)
      sys.stdout.write(format_validate_report(report))
      return 0 if report.get("pass") else 1
    if args.synthetic:
      dag = run_synthetic(args.out)
      s = dag["summary"]
      sys.stdout.write("== Phase 4 synthetic DAG self-test: PASS ==\n")
      sys.stdout.write("nodes=%d edges=%d cross_group=%d serialized=%.1f cp=%.1f 2q=%.1f (save %.1f) "
                       "3q=%.1f (save %.1f) us\n"
                       % (s["node_count"], s["edge_count"], s["cross_group_edge_count"],
                          s["serialized_us"], s["critical_path_us"],
                          s["schedules"]["queues_2"]["span_us"], s["schedules"]["queues_2"]["saving_us"],
                          s["schedules"]["queues_3"]["span_us"], s["schedules"]["queues_3"]["saving_us"]))
      sys.stdout.write("per-group restriction: %s\n" % json.dumps(s["per_group"], sort_keys=True))
      if args.out:
        sys.stdout.write("wrote %s\n" % args.out)
      return 0
    if args.capture:
      return main_capture(args)
  except FullTokenDagError as exc:
    sys.stderr.write("full_token_dag_capture: %s\n" % exc)
    return 1
  return 1


if __name__ == "__main__":
  sys.exit(main())
