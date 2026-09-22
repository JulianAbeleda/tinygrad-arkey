#!/usr/bin/env python3
"""Build a compact, CPU-only ledger from an nsys CUDA node-trace SQLite file.

The report separates node-sum, interval union, overlap, exposed work, and
inter-launch gaps.  It is intentionally descriptive: profiled durations are
never presented as unprofiled token wall time.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, shlex, sqlite3, statistics, sys
from collections import Counter, defaultdict
from typing import Iterable

SCHEMA = "tinygrad.cuda_graph_timeline_ledger.v1"


def classify(name: str) -> str:
  n = name.lower()
  if "mul_mat_vec_q" in n: return "mmq"
  if "quantize_q8_1" in n: return "quantize_q8_1"
  if "flash_attn_ext_vec" in n: return "flash_score"
  if "flash_attn_combine" in n: return "flash_combine"
  if "rms_norm" in n: return "rms_norm"
  if "rope" in n: return "rope"
  if "set_rows" in n: return "kv_set_rows"
  if "get_rows" in n: return "get_rows"
  if "bin_bcast" in n: return "elementwise"
  return "other"


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
  out: list[list[int]] = []
  for start, end in sorted(intervals):
    if end < start: raise ValueError("interval end precedes start")
    if not out or start > out[-1][1]: out.append([start, end])
    elif end > out[-1][1]: out[-1][1] = end
  return [(x[0], x[1]) for x in out]


def interval_ns(intervals: Iterable[tuple[int, int]]) -> int:
  return sum(end-start for start, end in merge_intervals(intervals))


def intersection_ns(a: Iterable[tuple[int, int]], b: Iterable[tuple[int, int]]) -> int:
  aa, bb = merge_intervals(a), merge_intervals(b)
  i = j = total = 0
  while i < len(aa) and j < len(bb):
    total += max(0, min(aa[i][1], bb[j][1]) - max(aa[i][0], bb[j][0]))
    if aa[i][1] <= bb[j][1]: i += 1
    else: j += 1
  return total


def _median(values: list[float]) -> float | None:
  return statistics.median(values) if values else None


def _us(ns: float | int) -> float:
  return round(float(ns) / 1000.0, 3)


def _sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
  return h.hexdigest()


def _name_map(con: sqlite3.Connection) -> dict[int, str]:
  return {int(i): str(value) for i, value in con.execute("select id, value from StringIds")}


def _columns(con: sqlite3.Connection) -> set[str]:
  return {str(row[1]) for row in con.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}


def _load_rows(con: sqlite3.Connection, graph_id: int) -> list[dict]:
  cols = _columns(con)
  required = {"start", "end", "shortName", "graphNodeId", "graphId"}
  if not required <= cols: raise ValueError("trace is missing CUPTI kernel columns: %s" % sorted(required-cols))
  optional = [x for x in ("demangledName", "mangledName", "streamId", "registersPerThread", "gridX", "gridY", "gridZ",
                           "blockX", "blockY", "blockZ", "staticSharedMemory",
                           "dynamicSharedMemory", "localMemoryPerThread") if x in cols]
  selected = ["start", "end", "shortName", "graphNodeId"] + optional
  names = _name_map(con)
  out = []
  for raw in con.execute("select %s from CUPTI_ACTIVITY_KIND_KERNEL where graphId=? order by start" % ",".join(selected), (graph_id,)):
    row = dict(zip(selected, raw))
    row["name"] = names.get(int(row["shortName"]), str(row["shortName"]))
    row["variant"] = names.get(int(row["demangledName"]), str(row["demangledName"])) if row.get("demangledName") is not None else row["name"]
    row["mangled_variant"] = names.get(int(row["mangledName"]), str(row["mangledName"])) if row.get("mangledName") is not None else None
    row["class"] = classify(row["name"])
    out.append(row)
  return out


def _split_replays(rows: list[dict]) -> tuple[list[list[dict]], dict]:
  """Split launches without relying on an arbitrary time-gap threshold.

  A CUDA graph node occurs once per graph launch.  Encountering the same node
  id again therefore starts the next replay.  Every retained replay must have
  the same complete node-id set; partial profiler fragments fail closed.
  """
  if not rows: return [], {"raw_fragments": 0, "complete_replays": 0, "discarded_fragments": 0, "discarded_sizes": []}
  launches, cur, seen = [], [], set()
  for row in rows:
    node = int(row["graphNodeId"])
    if node in seen:
      launches.append(cur)
      cur, seen = [], set()
    cur.append(row); seen.add(node)
  if cur: launches.append(cur)
  expected = Counter(frozenset(int(r["graphNodeId"]) for r in replay) for replay in launches)
  authority = expected.most_common(1)[0][0]
  complete = [r for r in launches if frozenset(int(x["graphNodeId"]) for x in r) == authority and len(r) == len(authority)]
  if not complete: raise ValueError("no complete graph replays found")
  for prior, current in zip(complete, complete[1:]):
    if all("start" in x and "end" in x for x in prior+current) and min(int(x["start"]) for x in current) < max(int(x["end"]) for x in prior):
      raise ValueError("complete graph replay ranges overlap")
  discarded = [r for r in launches if r not in complete]
  return complete, {"raw_fragments": len(launches), "complete_replays": len(complete),
                    "discarded_fragments": len(discarded), "discarded_sizes": [len(r) for r in discarded],
                    "authority_node_count": len(authority)}


def split_replays(rows: list[dict]) -> list[list[dict]]:
  return _split_replays(rows)[0]


def _replay_metrics(rows: list[dict], anchor_class: str) -> dict:
  starts, ends = [int(r["start"]) for r in rows], [int(r["end"]) for r in rows]
  all_intervals = [(int(r["start"]), int(r["end"])) for r in rows]
  by_class: dict[str, list[tuple[int, int]]] = defaultdict(list)
  sums: Counter[str] = Counter()
  counts: Counter[str] = Counter()
  for r in rows:
    interval = (int(r["start"]), int(r["end"]))
    by_class[str(r["class"])].append(interval)
    sums[str(r["class"])] += interval[1]-interval[0]
    counts[str(r["class"])] += 1
  anchor = by_class.get(anchor_class, [])
  non_anchor = [i for cls, intervals in by_class.items() if cls != anchor_class for i in intervals]
  classes = {}
  for cls in sorted(by_class):
    union = interval_ns(by_class[cls])
    hidden = intersection_ns(by_class[cls], anchor) if cls != anchor_class else 0
    classes[cls] = {"nodes": counts[cls], "node_sum_us": _us(sums[cls]), "union_us": _us(union),
                    "hidden_behind_%s_us" % anchor_class: _us(hidden),
                    "exposed_vs_%s_us" % anchor_class: _us(union-hidden)}
  span = max(ends)-min(starts)
  union = interval_ns(all_intervals)
  node_sum = sum(end-start for start, end in all_intervals)
  non_anchor_union = interval_ns(non_anchor)
  non_anchor_hidden = intersection_ns(non_anchor, anchor)
  return {"start_ns": min(starts), "end_ns": max(ends), "nodes": len(rows),
          "streams": len({r.get("streamId") for r in rows if r.get("streamId") is not None}),
          "node_sum_us": _us(node_sum), "span_us": _us(span), "kernel_union_us": _us(union),
          "overlap_mass_us": _us(node_sum-union), "internal_gap_us": _us(span-union),
          "non_anchor": {"union_us": _us(non_anchor_union),
                         "hidden_behind_%s_us" % anchor_class: _us(non_anchor_hidden),
                         "exposed_vs_%s_us" % anchor_class: _us(non_anchor_union-non_anchor_hidden)},
          "classes": classes}


def _shape_census(replays: list[list[dict]], warmup: int) -> list[dict]:
  rows = [r for replay in replays[warmup:] for r in replay]
  replay_count = len(replays[warmup:])
  grouped: dict[tuple, list[dict]] = defaultdict(list)
  for r in rows:
    key = (r["class"], r["name"], r["variant"], r["mangled_variant"], tuple(r.get(x) for x in ("gridX", "gridY", "gridZ")),
           tuple(r.get(x) for x in ("blockX", "blockY", "blockZ")), r.get("registersPerThread"),
           r.get("staticSharedMemory"), r.get("dynamicSharedMemory"), r.get("localMemoryPerThread"))
    grouped[key].append(r)
  out = []
  for key, rs in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
    cls, name, variant, mangled_variant, grid, block, regs, static_smem, dynamic_smem, local_mem = key
    count = len(rs) / replay_count
    if not count.is_integer(): raise ValueError("unstable per-replay shape count for %s" % (key,))
    out.append({"class": cls, "name": name, "variant": variant, "mangled_variant": mangled_variant,
                "count_per_replay": int(count), "grid": list(grid), "block": list(block),
                "registers_per_thread": regs, "static_smem_bytes": static_smem,
                "dynamic_smem_bytes": dynamic_smem, "local_memory_per_thread": local_mem,
                "median_duration_us": _us(statistics.median([int(r["end"])-int(r["start"]) for r in rs]))})
  return out


def _percentile(values: list[float], percentile: float) -> float | None:
  if not values: return None
  ordered = sorted(values)
  pos = (len(ordered)-1) * percentile
  lo, hi = int(pos), min(int(pos)+1, len(ordered)-1)
  return ordered[lo] + (ordered[hi]-ordered[lo]) * (pos-lo)


def analyze(trace: str, graph_id: int, warmup: int = 2, anchor_class: str = "mmq",
            inter_replay_gap_cap_us: float = 1000.0, provenance: dict | None = None) -> dict:
  path = pathlib.Path(trace)
  con = sqlite3.connect(str(path))
  try: replays, replay_split = _split_replays(_load_rows(con, graph_id))
  finally: con.close()
  if len(replays) <= warmup: raise ValueError("need more complete replays than warmup=%d" % warmup)
  steady = replays[warmup:]
  metrics = [_replay_metrics(r, anchor_class) for r in steady]
  gaps_us = [(int(b[0]["start"])-max(int(x["end"]) for x in a))/1000.0 for a, b in zip(steady, steady[1:])]
  bounded = [x for x in gaps_us if x <= inter_replay_gap_cap_us]
  class_names = sorted({c for m in metrics for c in m["classes"]})
  class_medians = {}
  for cls in class_names:
    keys = ("nodes", "node_sum_us", "union_us", "hidden_behind_%s_us" % anchor_class,
            "exposed_vs_%s_us" % anchor_class)
    class_medians[cls] = {k: round(float(statistics.median([m["classes"][cls][k] for m in metrics if cls in m["classes"]])), 3) for k in keys}
  scalar_keys = ("nodes", "streams", "node_sum_us", "span_us", "kernel_union_us", "overlap_mass_us", "internal_gap_us")
  median = {k: round(float(statistics.median([m[k] for m in metrics])), 3) for k in scalar_keys}
  median["span_discount_vs_node_sum_pct"] = round(100.0*(median["node_sum_us"]-median["span_us"])/median["node_sum_us"], 3)
  non_anchor_keys = ("union_us", "hidden_behind_%s_us" % anchor_class, "exposed_vs_%s_us" % anchor_class)
  non_anchor = {k: round(float(statistics.median([m["non_anchor"][k] for m in metrics])), 3) for k in non_anchor_keys}
  tool_path = pathlib.Path(__file__).resolve()
  return {"schema": SCHEMA, "evidence": "OBSERVED_PROFILED_CUPTI_INTERVALS",
          "warning": "Profiled intervals are not an unprofiled token-wall decomposition.",
          "definitions": {"overlap_mass_us": "node sum minus all-kernel interval union; not recoverable wall savings",
                          "span_discount_vs_node_sum_pct": "(node sum - graph span) / node sum; overlap mass net of internal gaps",
                          "streams": "distinct CUPTI kernel-row streamId values; does not prove graph-internal scheduling structure",
                          "class_exposure": "per-class exposure is non-additive; use non_anchor_aggregate for the equation",
                          "median_equation": "terms are medianed independently and can differ by sub-us rounding/non-additivity"},
          "tool": {"path": str(tool_path), "sha256": _sha256(tool_path)},
          "source": {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}, "graph_id": graph_id,
          "provenance": provenance or {},
          "anchor_class": anchor_class, "warmup_replays_dropped": warmup,
          "complete_replays": len(replays), "steady_replays": len(steady),
          "replay_split": replay_split,
          "median": median, "non_anchor_aggregate": non_anchor, "classes": class_medians,
          "inter_replay_gap": {"total_count": len(gaps_us), "all_median_us": round(float(_median(gaps_us) or 0), 3),
                               "all_p05_us": round(float(_percentile(gaps_us, 0.05) or 0), 3),
                               "all_p95_us": round(float(_percentile(gaps_us, 0.95) or 0), 3),
                               "bounded_cap_us": inter_replay_gap_cap_us,
                               "bounded_count": len(bounded),
                               "excluded_count": len(gaps_us)-len(bounded),
                               "bounded_median_us": round(float(_median(bounded) or 0), 3)},
          "shape_census": _shape_census(replays, warmup)}


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--trace", required=True)
  p.add_argument("--graph-id", required=True, type=int)
  p.add_argument("--warmup", type=int, default=2)
  p.add_argument("--anchor-class", default="mmq")
  p.add_argument("--inter-replay-gap-cap-us", type=float, default=1000.0)
  p.add_argument("--capture-nsys-version")
  p.add_argument("--capture-command")
  p.add_argument("--export-command")
  p.add_argument("--output")
  args = p.parse_args()
  report = analyze(args.trace, args.graph_id, args.warmup, args.anchor_class, args.inter_replay_gap_cap_us,
                   {"capture_nsys_version": args.capture_nsys_version, "capture_command": args.capture_command,
                    "export_command": args.export_command,
                    "reproduction_command": " ".join(shlex.quote(x) for x in sys.argv)})
  payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
  if args.output:
    out = pathlib.Path(args.output); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(payload)
  else: print(payload, end="")


if __name__ == "__main__": main()
