#!/usr/bin/env python3
"""Route B3.2-G: NV DEBUG=2 prime-trace duration attachment + HEAD DAG emission.

Attaches kernel durations from the DEBUG=2 prime trace (lines
`*** NV <idx> <name> ... tm <us>/...`) to the anchored NV decode capture (a
tinygrad.route_b3.dag_attribution.v1 report whose physical arm carries
post-split kernel names and group ids), then emits the HEAD duration-bearing
kernel-named DAG that replaces the lost /tmp/ln_20260805.json.

Matching mirrors cuda_route_aligned_census.align_capture / cuda_duration_attach:
within each graph group, trace rows are aligned to DAG nodes positionally
(ordered kernel identity = name + occurrence within the group). The trace
token is the last contiguous ordered-name match of the capture's physical
nodes (the captured decode token). Any signature mismatch fails closed: the
group is reported UNKNOWN and its nodes get NO attached duration (never
zeroed). The emitted DAG records a canonical schema digest and node names are
the real q4k/q6k/flash/E_/r_ post-split kernel names, never pre-split names.

Modes:
  --attach <--capture <json> --trace <log> [--dag-out <json>]>  attach + emit (CPU)
  --synthetic                                                    hermetic fixture
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, re, subprocess, sys, tempfile
from typing import Any

from extra.llm_research.decode.full_token_dag_capture import compute_dag_metrics

SCHEMA = "tinygrad.nv_duration_attach.v1"
DAG_SCHEMA = "tinygrad.nv_dag_duration_head.v1"
CAPTURE_SCHEMA = "tinygrad.route_b3.dag_attribution.v1"
UNKNOWN = "UNKNOWN"

DEFAULT_CAPTURE = "docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-capture-manifest-20260804.json"
DEFAULT_TRACE = "/tmp/tg_debug_probe_20260812.log"
DEFAULT_DAG_OUT = "docs/task_workflow/evidence/nv-dag-duration-head-20260812.json"
DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

TM_RE = re.compile(r"^\s*\*\*\* NV\s+(\d+)\s+(\S+)\s+.*\btm\s+([\d.]+)(us|ms|s)\s*/")
_UNIT_US = {"us": 1.0, "ms": 1e3, "s": 1e6}


class NVDurationAttachError(ValueError):
  pass


def _atomic_json(path: str, payload: dict) -> None:
  p = pathlib.Path(path)
  p.parent.mkdir(parents=True, exist_ok=True)
  fd, temporary = tempfile.mkstemp(prefix=".%s." % p.name, suffix=".tmp", dir=p.parent)
  try:
    with os.fdopen(fd, "w") as f:
      json.dump(payload, f, indent=2, sort_keys=True)
      f.write("\n")
      f.flush()
      os.fsync(f.fileno())
    os.replace(temporary, p)
  except BaseException:
    try: os.unlink(temporary)
    except FileNotFoundError: pass
    raise


def _sha256_file(path: str) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def _git_commit() -> str | None:
  try:
    return subprocess.check_output(
      ["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
      text=True).strip()
  except Exception:
    return None


def _canonical_digest(doc: dict) -> str:
  """sha256 over the canonical document, excluding the recorded digest itself."""
  body = {k: v for k, v in doc.items() if k != "schema_digest"}
  return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Trace parsing and per-group matching (fail-closed)
# ---------------------------------------------------------------------------

def parse_trace(trace_path: str) -> list[dict]:
  """Parse `*** NV <idx> <name> ... tm <dur><unit>/...` rows into launch order."""
  rows: list[dict] = []
  with open(trace_path, encoding="utf-8", errors="replace") as f:
    for lineno, line in enumerate(f, 1):
      if not line.startswith("*** NV"):
        continue
      m = TM_RE.match(line)
      if m is None:
        raise NVDurationAttachError("trace line %d: cannot parse NV row %r" % (lineno, line[:120]))
      idx, name, value, unit = m.group(1), m.group(2), m.group(3), m.group(4)
      rows.append({"idx": int(idx), "name": name, "duration_us": float(value) * _UNIT_US[unit]})
  if not rows:
    raise NVDurationAttachError("trace has no NV rows: %s" % trace_path)
  idxs = [r["idx"] for r in rows]
  if any(a >= b for a, b in zip(idxs, idxs[1:])):
    raise NVDurationAttachError("trace NV launch indexes are not strictly increasing")
  return rows


def select_token_window(rows: list[dict], names: list[str]) -> tuple[int, int] | None:
  """Last contiguous window with the most matching ordered names.

  An exact match is preferred (and is the expected steady-state case); when
  names drift the window with the most matches is still selected so per-group
  fail-closed reporting can attribute the aligned groups and mark only the
  mismatching ones UNKNOWN. Returns None when no position matches at all.
  """
  n = len(names)
  if n == 0 or len(rows) < n:
    return None
  best: tuple[int, int] | None = None
  best_score = -1
  for start in range(len(rows) - n, -1, -1):
    score = sum(1 for i in range(n) if rows[start + i]["name"] == names[i])
    if score > best_score:
      best_score = score
      best = (start, start + n)
  return best if best_score > 0 else None


def attach_durations(capture: dict, rows: list[dict],
                     window: tuple[int, int] | None = None) -> dict:
  """Attach trace durations to physical-arm nodes by group + ordered identity.

  Nodes are aligned to trace rows positionally within each graph group (groups
  are consecutive program-order ranges). Every position must match name; any
  mismatch fails closed: the group is marked UNKNOWN and gets no durations.
  """
  physical = capture["arms"]["physical"]
  nodes = sorted(physical["nodes"], key=lambda x: x["id"])
  names = [str(n["name"]) for n in nodes]
  counts: dict[Any, int] = {}
  order: list[Any] = []
  for n in nodes:
    g = n["group_id"]
    if g not in counts:
      counts[g] = 0
      order.append(g)
    counts[g] += 1
  group_sizes = [(g, counts[g]) for g in order]
  if window is None:
    window = select_token_window(rows, names)
  if window is None:
    return {"duration_by_call": {}, "groups": [
        {"group_id": gid, "size": size, "aligned": False,
         "reason": "no trace token matches the capture's ordered kernel names"}
        for gid, size in group_sizes],
      "matched_calls": 0, "total_groups": len(group_sizes), "aligned_groups": 0,
      "window": {"start": None, "end": None, "rows": 0, "trace_rows": len(rows)}}
  start, end = window
  token_rows = rows[start:end]
  duration_by_call: dict[int, float] = {}
  group_rows: list[dict] = []
  offset = 0
  for gid, _size in group_sizes:
    members = [n for n in nodes if n["group_id"] == gid]
    size = len(members)
    slice_rows = token_rows[offset:offset + size]
    mismatches: list[dict] = []
    if len(slice_rows) != size:
      mismatches = [{"position": p, "node_id": members[p]["id"],
                     "dag_name": members[p]["name"], "trace_name": None}
                    for p in range(size)]
    else:
      for p, node in enumerate(members):
        tr = slice_rows[p]["name"]
        if str(node["name"]) != tr:
          mismatches.append({"position": p, "node_id": node["id"],
                             "dag_name": node["name"], "trace_name": tr})
    if mismatches:
      group_rows.append({"group_id": gid, "size": size, "aligned": False,
                         "mismatched_positions": len(mismatches),
                         "examples": mismatches[:4]})
    else:
      group_rows.append({"group_id": gid, "size": size, "aligned": True})
      for p, node in enumerate(members):
        duration_by_call[node["id"]] = slice_rows[p]["duration_us"]
    offset += size
  return {"duration_by_call": duration_by_call, "groups": group_rows,
          "matched_calls": len(duration_by_call), "total_groups": len(group_sizes),
          "aligned_groups": sum(1 for g in group_rows if g["aligned"]),
          "window": {"start": start, "end": end, "rows": end - start,
                     "trace_rows": len(rows)}}


# ---------------------------------------------------------------------------
# HEAD duration-bearing kernel-named DAG emission
# ---------------------------------------------------------------------------

def build_duration_dag(capture: dict, attach: dict,
                       capture_path: str | None = None, capture_sha256: str | None = None,
                       trace_path: str | None = None, trace_sha256: str | None = None,
                       commit: str | None = None) -> dict:
  """Build the scan-consumable DAG: physical nodes (post-split names, attached
  durations) + physical edges, with the canonical schema digest recorded."""
  physical = capture["arms"]["physical"]
  durs = attach["duration_by_call"]
  nodes = [{"id": n["id"], "name": str(n["name"]), "group_id": n["group_id"],
            "duration_us": (round(durs[n["id"]], 3) if n["id"] in durs else None)}
           for n in sorted(physical["nodes"], key=lambda x: x["id"])]
  edges = [{"from": e["from"], "to": e["to"], "kind": e["kind"],
            "crosses_group": bool(e.get("crosses_group"))}
           for e in (physical.get("edges") or [])]
  unknown_node_count = sum(1 for n in nodes if n["duration_us"] is None)
  fully_aligned = attach["total_groups"] > 0 and attach["aligned_groups"] == attach["total_groups"]
  doc: dict[str, Any] = {
    "schema": DAG_SCHEMA,
    "capture": {"path": capture_path, "sha256": capture_sha256, "schema": CAPTURE_SCHEMA},
    "trace": {"path": trace_path, "sha256": trace_sha256, "window": attach["window"]},
    "commit": commit,
    "duration_attach": {"aligned": fully_aligned, "aligned_groups": attach["aligned_groups"],
                        "total_groups": attach["total_groups"],
                        "matched_calls": attach["matched_calls"], "groups": attach["groups"]},
    "node_count": len(nodes),
    "edge_count": len(edges),
    "cross_group_edge_count": sum(1 for e in edges if e["crosses_group"]),
    "unknown_node_count": unknown_node_count,
    "nodes": nodes,
    "edges": edges,
  }
  if fully_aligned:
    doc["summary"] = compute_dag_metrics({"nodes": nodes, "edges": edges})
  doc["schema_digest"] = _canonical_digest(doc)
  return doc


# ---------------------------------------------------------------------------
# Synthetic mode (hermetic fixture)
# ---------------------------------------------------------------------------

def run_synthetic() -> dict:
  """Two-chain fixture: attach synthetic durations and emit the DAG in memory."""
  from extra.llm_research.decode.route_b3_dag_attribution import (
    build_attribution_fixture, compute_attribution_report,
  )
  calls, manifest = build_attribution_fixture()
  capture = compute_attribution_report(calls, calls, manifest)
  rows = [{"idx": i + 1, "name": c.name, "duration_us": c.duration_us}
          for i, c in enumerate(calls)]
  attach = attach_durations(capture, rows)
  assert attach["aligned_groups"] == attach["total_groups"] == 2, attach
  for c in calls:
    assert abs(attach["duration_by_call"][c.index] - c.duration_us) < 1e-9
  dag = build_duration_dag(capture, attach, capture_path="synthetic",
                           capture_sha256="synthetic", trace_path="synthetic",
                           trace_sha256="synthetic", commit="synthetic")
  assert dag["schema"] == DAG_SCHEMA
  assert dag["node_count"] == 8 and dag["edge_count"] == len(capture["arms"]["physical"]["edges"])
  return dag


def run_attach(args: argparse.Namespace) -> int:
  with open(args.capture, encoding="utf-8") as f:
    capture = json.load(f)
  if capture.get("schema") != CAPTURE_SCHEMA:
    sys.stderr.write("capture schema must be %r, got %r\n" % (CAPTURE_SCHEMA, capture.get("schema")))
    return 1
  rows = parse_trace(args.trace)
  window = None
  if args.window_start is not None or args.window_end is not None:
    start = args.window_start if args.window_start is not None else 1
    end = args.window_end if args.window_end is not None else len(rows) + 1
    if not (1 <= start < end <= len(rows) + 1):
      sys.stderr.write("invalid --window-start/--window-end for %d trace rows\n" % len(rows))
      return 1
    window = (start - 1, end - 1)
  attach = attach_durations(capture, rows, window=window)
  dag = build_duration_dag(capture, attach,
                           capture_path=args.capture, capture_sha256=_sha256_file(args.capture),
                           trace_path=args.trace, trace_sha256=_sha256_file(args.trace),
                           commit=_git_commit())
  _atomic_json(args.dag_out, dag)
  sys.stdout.write("\n".join([
    "matched calls: %d / %d (aligned groups %d/%d, trace window %s)" % (
      attach["matched_calls"], dag["node_count"], attach["aligned_groups"],
      attach["total_groups"], attach["window"]),
    "per-group alignment:",
  ]))
  for g in attach["groups"]:
    sys.stdout.write("  group %s size %-3d aligned=%-5s %s\n" % (
      g["group_id"], g["size"], g["aligned"],
      "examples=%s" % g["examples"] if not g["aligned"] else ""))
  sys.stdout.write("DAG: %d nodes / %d edges / %d cross-group / digest %s\n" % (
    dag["node_count"], dag["edge_count"], dag["cross_group_edge_count"],
    dag["schema_digest"][:12]))
  sys.stdout.write("wrote %s\n" % args.dag_out)
  if not dag["duration_attach"]["aligned"]:
    sys.stdout.write("FAIL_CLOSED: %d group(s) UNKNOWN, durations never zeroed\n" % (
      dag["duration_attach"]["total_groups"] - dag["duration_attach"]["aligned_groups"]))
    return 1
  return 0


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  mode = ap.add_mutually_exclusive_group(required=True)
  mode.add_argument("--attach", action="store_true", help="attach trace durations to capture + emit DAG (CPU only)")
  mode.add_argument("--synthetic", action="store_true", help="hermetic fixture DAG + durations")
  ap.add_argument("--capture", default=DEFAULT_CAPTURE)
  ap.add_argument("--trace", default=DEFAULT_TRACE)
  ap.add_argument("--dag-out", default=DEFAULT_DAG_OUT)
  ap.add_argument("--window-start", type=int, default=None)
  ap.add_argument("--window-end", type=int, default=None)
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  args = ap.parse_args()
  if args.synthetic:
    dag = run_synthetic()
    print(json.dumps({"schema": dag["schema"], "node_count": dag["node_count"],
                      "edge_count": dag["edge_count"], "schema_digest": dag["schema_digest"]},
                     indent=2, sort_keys=True))
    return 0
  if args.attach:
    return run_attach(args)
  ap.error("no mode selected (--attach | --synthetic)")
  return 2


if __name__ == "__main__":
  sys.exit(main())
