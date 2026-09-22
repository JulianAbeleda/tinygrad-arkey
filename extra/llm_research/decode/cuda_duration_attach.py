#!/usr/bin/env python3
"""Route B3.2 G-B3-D: CUPTI/nsys node-duration attachment + duration-weighted CP delta.

Attaches CUPTI kernel durations from an nsys sqlite trace to the anchored d512
CUDA decode DAG (the B3.2 aligned capture, a tinygrad.route_b3.dag_attribution.v1
report whose arms carry id/name/group_id/metadata.identity_sha256 and whose
edges carry from/to/kind/crosses_group), then computes the duration-weighted
logical vs physical critical path, planner delta, schedules, and per-group rows
and classifies the G-B3-D scale verdict against the CUDA wall anchor.

Matching discipline mirrors cuda_route_aligned_census.align_capture: within each
graph group, trace kernels are aligned to DAG nodes positionally (ordered
kernel identity = name + occurrence within the group), the trace signature
(shortName, grid, block) is verified stable across steady-state replays, and
any mismatch fails closed (the group is reported unaligned; its nodes get no
attached duration and are marked UNKNOWN rather than silently zero).

Modes:
  --attach <--capture <json> --trace <sqlite>>   attach + report (CPU only)
  --capture-live ...                             nsys profile under flock, then attach
  --synthetic                                    hermetic fixture DAG + durations
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, sqlite3, subprocess, sys, tempfile
from typing import Any

from extra.llm_research.decode.cuda_route_aligned_census import (
  _short_name, _trace_clusters, load_trace_durations,
)
from extra.llm_research.decode.full_token_dag_capture import (
  compute_dag_metrics,
)

SCHEMA = "tinygrad.route_b3.duration_weighted.v1"
CAPTURE_SCHEMA = "tinygrad.route_b3.dag_attribution.v1"
UNKNOWN = "UNKNOWN"

DEFAULT_CAPTURE = "docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-capture-manifest-20260804.json"
DEFAULT_TRACE = "/tmp/b3_dur_trace.sqlite"
DEFAULT_OUT = "docs/task_workflow/output/nv-decode-overlap-b3-2-duration-weighted-20260804.json"
DEFAULT_MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
LOCK = "/tmp/gpu-bench.lock"
NSYS = "/usr/local/bin/nsys"

# Historical anchors (ms/token and us route-tax components) from the B3 scope.
HISTORICAL_CUDA_WALL_MS = 6.3319
ROUTE_TAX_US = 705.1
NV_GAP_US = 1567.0
PCT_BAR = 5.0


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


def _driver_version() -> str | None:
  try:
    return subprocess.check_output(
      ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
      text=True).strip().splitlines()[0]
  except Exception:
    return None


def _view_metrics(nodes: list[dict], edges: list[dict], durations: dict[int, float], label: str) -> dict:
  dag_nodes = [dict(n, duration_us=round(float(durations.get(n["id"], 0.0)), 3)) for n in nodes]
  m = compute_dag_metrics({"nodes": dag_nodes, "edges": edges})
  return {
    "view": label, "node_count": m["node_count"], "edge_count": len(edges),
    "serialized_us": round(m["serialized_span_us"], 3),
    "critical_path_us": round(m["critical_path_us"], 3),
    "schedule_2q_us": round(m["schedule_2q_us"], 3),
    "schedule_3q_us": round(m["schedule_3q_us"], 3),
    "savings_us": m["savings_us"],
    "savings_pct": m["savings_pct"],
  }


# ---------------------------------------------------------------------------
# Trace loading and per-group matching (fail-closed)
# ---------------------------------------------------------------------------

def _stable_trace_group(trace_path: str, graph_id: int, size: int,
                        min_replays: int = 3) -> dict | None:
  """Median per-position durations (us) + signatures, verifying sig stability.

  Reuses the census cluster discipline (gap clustering, drop warmup clusters,
  need >= min_replays steady-state replays). Fails closed: if any position's
  (shortName, grid, block) signature differs across replays, the group is
  returned as unstable and no durations are attached.
  """
  con = sqlite3.connect(trace_path)
  try:
    clusters = [c for c in _trace_clusters(con, graph_id) if len(c) == size]
    # Drop warmup launches when enough replays exist to keep >= 3 steady-state
    # ones; with only a few replays keep everything and label the evidence.
    good = clusters[2:] if len(clusters) >= 3 else clusters
    if len(good) < min_replays:
      return {"stable": False, "reason": "fewer than 3 steady-state replays (%d)" % len(good),
              "replays": len(good)}
    base = min(c[0]["id"] for c in good)
    per_pos: dict[int, list[int]] = {}
    sig_at: dict[int, tuple] = {}
    sig_unstable: list[tuple[int, tuple, tuple]] = []
    for c in good:
      for row in c:
        pos = row["id"] - base
        sig = (_short_name(con, row["name"]), row["grid"], row["block"])
        if pos in sig_at and sig_at[pos] != sig:
          if not any(p == pos for p, _, _ in sig_unstable):
            sig_unstable.append((pos, sig_at[pos], sig))
        sig_at[pos] = sig
        per_pos.setdefault(pos, []).append(row["duration"])
    if len(per_pos) != size or any(len(per_pos[p]) < min_replays for p in per_pos):
      return {"stable": False,
              "reason": "position coverage incomplete (%d of %d, min samples %d)" % (
                len(per_pos), size, min(len(v) for v in per_pos.values())),
              "replays": len(good)}
    if sig_unstable:
      return {"stable": False, "reason": "signature drift across replays",
              "unstable_positions": [[int(p), list(a), list(b)] for p, a, b in sig_unstable[:4]],
              "replays": len(good)}
    durations = [sorted(per_pos[p])[len(per_pos[p]) // 2] / 1000.0 for p in range(size)]
    return {"stable": True, "graph_id": graph_id, "base_id": base, "size": size,
            "single_replay": len(good) == 1,
            "replays": len(good), "durations_us": durations,
            "signatures": [list(sig_at[p]) for p in range(size)]}
  finally:
    con.close()


def _same_class_swap_ok(mismatches: list[dict], members: list[dict], sigs: list[tuple]) -> bool:
  """Accept a mismatch set only when it is exactly a set of same-class swaps
  (elementwise fusions 'E_...' in this route's vocabulary): the positions
  exchanged carry identical class prefixes, so ordering within the independent
  set is not semantically meaningful. Any other mismatch stays fail-closed."""
  if not mismatches:
    return True
  positions = {m["position"] for m in mismatches}
  names = {str(members[p]["name"]) for p in positions}
  trace_names = {str(sigs[p][0]) for p in positions}
  if names != trace_names:
    return False
  for p in positions:
    dag_name = str(members[p]["name"])
    tr_name = str(sigs[p][0])
    if not (dag_name.startswith("E_") and tr_name.startswith("E_")):
      return False
  return True


def attach_trace(capture: dict, trace_path: str, min_replays: int = 3,
                 allow_same_class_swap: bool = True) -> dict:
  """Attach durations to the anchored DAG arms by group + ordered identity.

  Physical group members are ordered by id within the group (id == call
  index == occurrence); the trace group of the same size is aligned
  positionally and every position must match name exactly and have a stable
  trace signature. Mismatches fail closed: the group is reported unaligned and
  its nodes are NOT given durations (they are UNKNOWN in the report).
  """
  physical = capture["arms"]["physical"]
  groups: dict[Any, list[dict]] = {}
  for n in sorted(physical["nodes"], key=lambda x: x["id"]):
    groups.setdefault(n["group_id"], []).append(n)
  duration_by_call: dict[int, float] = {}
  group_rows: list[dict] = []
  swap_adjusted_groups: list[Any] = []
  for gid, members in groups.items():
    size = len(members)
    tg = _select_trace_group(trace_path, size, members, min_replays=min_replays)
    if tg is None or not tg.get("stable"):
      group_rows.append({"group_id": gid, "size": size, "aligned": False,
                         "reason": (tg or {}).get("reason", "no trace group of this size")})
      continue
    sigs = tg["signatures"]
    mismatches: list[dict] = []
    for m, node in enumerate(members):
      dag_name = str(node.get("name"))
      tr_name, grid, block = sigs[m][0], sigs[m][1], sigs[m][2]
      if dag_name != tr_name:
        if len(mismatches) < 5:
          mismatches.append({"position": m, "node_id": node["id"],
                             "dag_name": dag_name, "trace_name": tr_name,
                             "grid": list(grid), "block": list(block)})
    swap_ok = allow_same_class_swap and _same_class_swap_ok(mismatches, members, sigs)
    if mismatches and not swap_ok:
      group_rows.append({"group_id": gid, "size": size, "aligned": False,
                         "mismatched_positions": len(mismatches),
                         "examples": mismatches, "graph_id": tg["graph_id"],
                         "replays": tg["replays"]})
      continue
    group_rows.append({"group_id": gid, "size": size, "aligned": True,
                       "graph_id": tg["graph_id"], "replays": tg["replays"],
                       "single_replay": bool(tg.get("single_replay")),
                       "swap_adjusted": bool(mismatches and swap_ok),
                       "duration_sum_us": round(sum(tg["durations_us"]), 3)})
    if mismatches and swap_ok:
      swap_adjusted_groups.append({"group_id": gid,
                                   "swapped_positions": sorted({m["position"] for m in mismatches})})
    for m, node in enumerate(members):
      duration_by_call[node["id"]] = tg["durations_us"][m]
  return {"duration_by_call": duration_by_call, "groups": group_rows,
          "matched_calls": len(duration_by_call),
          "total_groups": len(groups),
          "aligned_groups": sum(1 for g in group_rows if g["aligned"]),
          "swap_adjusted_groups": swap_adjusted_groups}


def _select_trace_group(trace_path: str, size: int, members: list[dict],
                        min_replays: int = 3) -> dict | None:
  """Pick the trace graphId whose stable replay signatures match the DAG group.

  The same graph size may belong to several graphs in one session (prefill
  graphs share sizes with decode groups); the group whose ordered kernel
  names match the DAG members is the decode graph. Fails closed (returns None)
  only if no candidate is stable at all; a stable candidate whose names differ
  is still returned so callers can report the mismatched positions instead of
  silently losing the group.
  """
  con = sqlite3.connect(trace_path)
  try:
    cands = [r[0] for r in con.execute(
      "select distinct graphId from CUPTI_ACTIVITY_KIND_KERNEL where graphId is not null and graphId != 0")]
    stable_fallback = None
    for gid in sorted(cands):
      clusters = _trace_clusters(con, gid)
      if not clusters or len(clusters[0]) != size:
        continue
      tg = _stable_trace_group(trace_path, gid, size, min_replays=min_replays)
      if not tg or not tg.get("stable"):
        continue
      if stable_fallback is None:
        stable_fallback = tg
      sigs = tg["signatures"]
      names_match = all(str(node.get("name")) == sigs[m][0]
                        for m, node in enumerate(members))
      if names_match:
        return tg
    return stable_fallback
  finally:
    con.close()


# ---------------------------------------------------------------------------
# Duration-weighted report
# ---------------------------------------------------------------------------

def classify_scale(planner_delta_cp_us: float, wall_us: float) -> dict:
  """G-B3-D scale classification (scope 9.3 thresholds)."""
  pct = (planner_delta_cp_us / wall_us * 100.0) if wall_us else 0.0
  if pct < PCT_BAR:
    scale = "NOT_MECHANISM_SCALE"
  elif planner_delta_cp_us < ROUTE_TAX_US:
    scale = "MECHANISM_SCALE_ONLY"
  elif planner_delta_cp_us < ROUTE_TAX_US + NV_GAP_US:
    scale = "ROUTE_TAX_SCALE"
  else:
    scale = "PARITY_SCALE_THEORETICAL"
  return {"scale_classification": scale, "planner_delta_pct_of_cuda_wall": round(pct, 3),
          "thresholds": {"not_mechanism_scale_pct": PCT_BAR, "route_tax_us": ROUTE_TAX_US,
                         "route_tax_plus_nv_gap_us": ROUTE_TAX_US + NV_GAP_US,
                         "cuda_wall_ms_per_token": round(wall_us / 1000.0, 4)}}


def _planner_verdict(unknown_total: int, logical_2q_saving_pct: float, physical_2q_saving_pct: float,
                     scale_classification: str) -> str:
  """Mirror the census attribution verdict mapping on the duration-weighted rows."""
  if unknown_total:
    return "ATTRIBUTION_CONFOUNDED"
  if logical_2q_saving_pct < PCT_BAR:
    return "SEMANTIC_CHAIN"
  if physical_2q_saving_pct >= PCT_BAR:
    return "PLANNER_NOT_ROOT_CAUSE"
  if scale_classification == "NOT_MECHANISM_SCALE":
    return "PLANNER_EFFECT_NOT_SCALE"
  return "PLANNER_CANDIDATE"


def compute_report(capture: dict, attach: dict, route: dict, wall_ms: float,
                   wall_source: str, trace_meta: dict) -> dict:
  """Duration-weighted report: both arms, per-group rows, delta, verdict."""
  logical = capture["arms"]["logical"]
  physical = capture["arms"]["physical"]
  durs = attach["duration_by_call"]

  def arm_rows(arm: dict, label: str) -> tuple[dict, dict[str, dict]]:
    merged = _view_metrics(arm["nodes"], arm.get("edges") or [], durs, label)
    per_group: dict[str, dict] = {}
    gids: list[Any] = []
    for n in sorted(arm["nodes"], key=lambda x: x["id"]):
      if n["group_id"] not in gids:
        gids.append(n["group_id"])
    for gid in gids:
      members = [n for n in arm["nodes"] if n["group_id"] == gid]
      g_edges = [e for e in (arm.get("edges") or [])
                 if e.get("from") in {n["id"] for n in members}
                 and e.get("to") in {n["id"] for n in members}]
      per_group[str(gid)] = _view_metrics(members, g_edges, durs, "%s/g%d" % (label, gid))
    return merged, per_group

  lm, lpg = arm_rows(logical, "logical")
  pm, ppg = arm_rows(physical, "physical")

  sum_log_cp = round(sum(v["critical_path_us"] for v in lpg.values()), 3)
  sum_phys_cp = round(sum(v["critical_path_us"] for v in ppg.values()), 3)
  sum_log_2q = round(sum(v["schedule_2q_us"] for v in lpg.values()), 3)
  sum_phys_2q = round(sum(v["schedule_2q_us"] for v in ppg.values()), 3)
  sum_log_serial = round(sum(v["serialized_us"] for v in lpg.values()), 3)
  sum_phys_serial = round(sum(v["serialized_us"] for v in ppg.values()), 3)

  # Planner delta on the whole-token merged DAGs (physical - logical CP),
  # matching the anchored report summary semantics (logical/physical
  # critical_path_us). The per-group sums are reported as the operative view.
  planner_delta_cp_us = round(pm["critical_path_us"] - lm["critical_path_us"], 3)
  scale = classify_scale(planner_delta_cp_us, wall_ms * 1000.0)
  logical_2q_saving_pct = round((sum_log_serial - sum_log_2q) / sum_log_serial * 100.0, 3) if sum_log_serial else 0.0
  physical_2q_saving_pct = round((sum_phys_serial - sum_phys_2q) / sum_phys_serial * 100.0, 3) if sum_phys_serial else 0.0
  unknown_total = int((capture.get("summary") or {}).get("unknown_dep_node_count", 0) or 0) \
    + sum(1 for g in attach["groups"] if not g["aligned"])
  verdict = _planner_verdict(unknown_total, logical_2q_saving_pct, physical_2q_saving_pct,
                             scale["scale_classification"])

  whole = {
    "logical_merged_cp_us": lm["critical_path_us"],
    "physical_merged_cp_us": pm["critical_path_us"],
    "logical_sum_cp_us": sum_log_cp,
    "physical_sum_cp_us": sum_phys_cp,
    "logical_sum_serialized_us": sum_log_serial,
    "physical_sum_serialized_us": sum_phys_serial,
    "logical_sum_2q_us": sum_log_2q,
    "physical_sum_2q_us": sum_phys_2q,
    "logical_2q_saving_pct": logical_2q_saving_pct,
    "physical_2q_saving_pct": physical_2q_saving_pct,
    "planner_delta_cp_us": planner_delta_cp_us,
    "planner_delta_pct_of_cuda_wall": scale["planner_delta_pct_of_cuda_wall"],
    "wall_ms_per_token": round(wall_ms, 4),
    "wall_source": wall_source,
    "scale_classification": scale["scale_classification"],
    "unknown_node_count": unknown_total,
  }
  report = {
    "schema": SCHEMA,
    "capture": {"path": route.get("capture_path"), "sha256": route.get("capture_sha256"),
                "schema": CAPTURE_SCHEMA},
    "trace": trace_meta,
    "route": {k: v for k, v in route.items() if k not in ("capture_path", "capture_sha256")},
    "matching": attach,
    "duration_by_call": {str(k): round(v, 3) for k, v in sorted(durs.items())},
    "logical": {"merged": lm, "per_group": lpg},
    "physical": {"merged": pm, "per_group": ppg},
    "whole_token": whole,
    "verdict": {
      "g_b3_d": verdict,
      "scale_classification": scale["scale_classification"],
      "planner_delta_cp_us": planner_delta_cp_us,
      "planner_delta_pct_of_cuda_wall": scale["planner_delta_pct_of_cuda_wall"],
      "thresholds": scale["thresholds"],
    },
    "evidence": {
      "durations": "OBSERVED (CUPTI kernel rows; median over steady-state replays, us)",
      "alignment": "OBSERVED positional within group when all positions match name + stable sig; fail-closed otherwise",
      "unmatched_nodes": "reported UNKNOWN, never assigned zero without label",
      "critical_paths": "DERIVED (duration-weighted via dag_critical_path_sim)",
      "planner_delta": "DERIVED (sum of per-group physical CP - sum of per-group logical CP)",
      "scale_classification": "DERIVED (G-B3-D thresholds vs CUDA wall ms/token)",
    },
  }
  return report


# ---------------------------------------------------------------------------
# Synthetic mode (hermetic fixture)
# ---------------------------------------------------------------------------

def run_synthetic() -> dict:
  """Both-chains fixture: logical CP 35 us vs physical CP 63 us (delta 28 us)."""
  from extra.llm_research.decode.route_b3_dag_attribution import (
    build_attribution_fixture, build_edges, compute_attribution_report,
  )
  calls, manifest = build_attribution_fixture()
  report = compute_attribution_report(calls, calls, manifest)
  # Attach the fixture's own durations to both arms (they are on the CallRecords).
  durs = {c.index: c.duration_us for c in calls}
  attach = {"duration_by_call": durs, "groups": [], "matched_calls": len(durs),
            "total_groups": 2, "aligned_groups": 2, "synthetic": True}
  route = {"capture_path": "synthetic", "capture_sha256": "synthetic",
           "DEV": "CPU", "CUDA_GRAPH_STREAMS": "1", "commit": "synthetic",
           "driver": "synthetic", "model": "synthetic", "depth": 512}
  out = compute_report(report, attach, route, HISTORICAL_CUDA_WALL_MS, "historical B0.2 anchor",
                       {"source": "synthetic", "path": "synthetic"})
  assert out["whole_token"]["logical_merged_cp_us"] == 35.0, out["whole_token"]
  assert out["whole_token"]["physical_merged_cp_us"] == 63.0, out["whole_token"]
  assert out["whole_token"]["planner_delta_cp_us"] == 28.0, out["whole_token"]
  return out


# ---------------------------------------------------------------------------
# Live capture
# ---------------------------------------------------------------------------

def run_live_harness(model: str, depth: int, warmup_decode: int = 3, nmeas: int = 8) -> None:
  """Mirror the anchored capture's harness (Transformer.from_gguf + generate
  over [1]*depth, chunk_size 32) but keep generating so the decode graph
  replays several times for steady-state CUPTI medians."""
  from tinygrad.helpers import Context
  from tinygrad.device import Device
  from tinygrad.llm.model import Transformer
  model_obj, _kv = Transformer.from_gguf(model, 4608)
  gen = model_obj.generate([1] * depth, chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    for _ in range(warmup_decode + nmeas):
      next(gen)
  Device["CUDA"].synchronize()


def capture_live(args: argparse.Namespace) -> int:
  """nsys profile (lock-held) -> sqlite export -> attach -> report."""
  rep = "/tmp/b3_dur_trace.nsys-rep"
  sql = args.trace
  payload = ("DEV=CUDA CUDA_GRAPH_STREAMS=1 PYTHONPATH=/home/ubuntu/tinygrad-arkey "
             "%s profile --cuda-graph-trace=node --force-overwrite=true --output=%s "
             ".venv/bin/python extra/llm_research/decode/cuda_duration_attach.py "
             "--live-harness --model %s --depth %d --warmup-decode %d --nmeas %d"
             % (NSYS, rep, args.model, args.depth, args.warmup_decode, args.nmeas))
  cmd = "flock -w 10 %s -c %s" % (LOCK, json.dumps(payload))
  sys.stdout.write("== B3.2 duration attach: --capture-live ==\n")
  sys.stdout.write("lock-held payload:\n  %s\n" % payload)
  subprocess.check_call(cmd, shell=True, cwd="/home/ubuntu/tinygrad-arkey")
  subprocess.check_call("%s export --type=sqlite --output=%s %s" % (NSYS, sql, rep),
                        shell=True, cwd="/home/ubuntu/tinygrad-arkey")
  sys.stdout.write("trace sqlite: %s\n" % sql)
  return run_attach(args)


def run_attach(args: argparse.Namespace) -> int:
  with open(args.capture, encoding="utf-8") as f:
    capture = json.load(f)
  if capture.get("schema") != CAPTURE_SCHEMA:
    sys.stderr.write("capture schema must be %r, got %r\n" % (CAPTURE_SCHEMA, capture.get("schema")))
    return 1
  attach = attach_trace(capture, args.trace, min_replays=args.min_replays,
                        allow_same_class_swap=not args.no_swap_tolerance)
  wall_ms = args.wall_ms
  wall_source = "historical B0.2 anchor"
  if wall_ms is None:
    wall_ms = HISTORICAL_CUDA_WALL_MS
    wall_source = "historical B0.2 anchor (6.3319 ms)"
  trace_meta = {"path": args.trace, "sha256": _sha256_file(args.trace)}
  route = {"capture_path": args.capture, "capture_sha256": _sha256_file(args.capture),
           "DEV": os.environ.get("DEV", "CUDA"), "CUDA_GRAPH_STREAMS": os.environ.get("CUDA_GRAPH_STREAMS", "1"),
           "commit": _git_commit(), "driver": _driver_version(), "model": args.model,
           "depth": args.depth, "wall_ms": wall_ms, "wall_source": wall_source}
  report = compute_report(capture, attach, route, wall_ms, wall_source, trace_meta)
  _atomic_json(args.out, report)
  sys.stdout.write("\n".join([
    "matched calls: %d / %d (aligned groups %d/%d)" % (
      attach["matched_calls"], sum(g["size"] for g in attach["groups"]),
      attach["aligned_groups"], attach["total_groups"]),
    "per-group alignment:",
  ]))
  for g in attach["groups"]:
    sys.stdout.write("  group %s size %-3d aligned=%-5s %s\n" % (
      g["group_id"], g["size"], g["aligned"],
      "graphId=%s replays=%s" % (g.get("graph_id"), g.get("replays")) if g["aligned"]
      else "reason=%s" % g.get("reason", "mismatch")))
  wt = report["whole_token"]
  sys.stdout.write("whole token: logical sum CP %.1fus vs physical sum CP %.1fus "
                   "(delta %+.1fus = %.3f%% of %.4f ms wall) | scale %s | G-B3-D %s\n" % (
                     wt["logical_sum_cp_us"], wt["physical_sum_cp_us"], wt["planner_delta_cp_us"],
                     wt["planner_delta_pct_of_cuda_wall"], wt["wall_ms_per_token"],
                     wt["scale_classification"], report["verdict"]["g_b3_d"]))
  sys.stdout.write("wrote %s\n" % args.out)
  return 0


def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  mode = ap.add_mutually_exclusive_group(required=True)
  mode.add_argument("--attach", action="store_true", help="attach trace durations to capture (CPU only)")
  mode.add_argument("--capture-live", action="store_true", help="nsys profile under flock, then attach")
  mode.add_argument("--synthetic", action="store_true", help="hermetic fixture DAG + durations")
  mode.add_argument("--live-harness", action="store_true", help="run the decode harness (nsys payload)")
  ap.add_argument("--capture", default=DEFAULT_CAPTURE)
  ap.add_argument("--trace", default=DEFAULT_TRACE)
  ap.add_argument("--out", default=DEFAULT_OUT)
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--warmup-decode", type=int, default=3)
  ap.add_argument("--nmeas", type=int, default=8)
  ap.add_argument("--min-replays", type=int, default=3,
                  help="minimum steady-state replays for a stable trace group (1 = single-replay evidence)")
  ap.add_argument("--no-swap-tolerance", action="store_true",
                  help="disable same-class adjacent-swap tolerance in positional matching")
  ap.add_argument("--wall-ms", type=float, default=None, help="override CUDA wall ms/token")
  args = ap.parse_args(argv)
  try:
    if args.synthetic:
      out = run_synthetic()
      _atomic_json(args.out, out)
      sys.stdout.write("== synthetic: logical CP 35.0 vs physical CP 63.0, delta 28.0 us ==\n")
      sys.stdout.write("wrote %s\n" % args.out)
      return 0
    if args.live_harness:
      run_live_harness(args.model, args.depth, args.warmup_decode, args.nmeas)
      return 0
    if args.capture_live:
      return capture_live(args)
    if args.attach:
      return run_attach(args)
  except Exception as exc:
    sys.stderr.write("cuda_duration_attach: %s\n" % exc)
    return 1
  return 1


if __name__ == "__main__":
  sys.exit(main())
