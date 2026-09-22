#!/usr/bin/env python3
"""Build the d512 tinygrad decode node ledger from an nsys CUPTI sqlite trace.

tinygrad captures the decode token as several CUDA graphs (the flash-decode
graph is partitioned into 32/64/128/256/512-node graphs plus the 29-node
vocab-feedback graph at d512).  This ledger concatenates the matching replay
of every graph into one per-token timeline, classifies each kernel with the
same role vocabulary used by nv_fusion_population_ledger, and computes the
exact same metrics as cuda_graph_timeline_ledger so the result is a direct
per-class subtraction against llama's `nv-llama-d512-node-ledger`.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, shlex, sqlite3, statistics, sys
from collections import defaultdict

from extra.llm_research.decode.cuda_graph_timeline_ledger import (
  _load_rows, _replay_metrics, _shape_census, _split_replays, _percentile, _sha256,
)
from extra.llm_research.decode.nv_fusion_population_ledger import (
  classify as population_classify,
)

SCHEMA = "tinygrad.cuda_graph_timeline_ledger.v1"
DEFAULT_GRAPH_IDS = (2, 5, 8, 11, 14, 17)
ANCHOR_CLASS = "gemv"


def tg_classify(name: str) -> str:
  """Map a tinygrad rendered program name to a ledger class.

  The quantized GEMV cores are the anchor (llama's mmq + its folded-in
  quantize_q8_1).  Every non-anchor population mirrors the llama ledger
  vocabulary: flash score/combine, norms, rope/kv, residual cast, vocab aux.
  """
  if name.startswith("q6k_vocab_scalar_reduce"):
    return "vocab_aux"
  if "r_32_32_4_2_8" in name:
    return "reduce_output"
  population, role, _exact = population_classify(name)
  if population == "quant_core":
    return "gemv"
  if population == "flash":
    return "flash_score" if role == "flash_score" else "flash_combine"
  if population == "norms":
    return "norms"
  if population == "rope_kv":
    return "rope_kv"
  if population == "residual_cast_contiguous":
    return "residual_cast"
  if population == "vocab_feedback":
    return "vocab_aux"
  return "other"


def _load_graph_rows(trace: str, graph_ids: tuple[int, ...]) -> tuple[dict[int, list[dict]], dict]:
  con = sqlite3.connect(trace)
  try:
    per_graph: dict[int, list[dict]] = {}
    meta = {}
    for gid in graph_ids:
      rows = _load_rows(con, gid)
      for r in rows:
        r["class"] = tg_classify(r["name"])
      per_graph[gid] = rows
      meta[str(gid)] = {"rows": len(rows),
                        "classes": sorted({r["class"] for r in rows})}
  finally:
    con.close()
  return per_graph, meta


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ValueError(message)


def analyze(trace: str, graph_ids: tuple[int, ...], warmup: int = 2,
            anchor_class: str = ANCHOR_CLASS, inter_replay_gap_cap_us: float = 5000.0,
            provenance: dict | None = None) -> dict:
  per_graph, graph_meta = _load_graph_rows(trace, graph_ids)
  replays_by_graph: dict[int, list[list[dict]]] = {}
  replay_split: dict = {}
  for gid in graph_ids:
    replays, split = _split_replays(per_graph[gid])
    replays_by_graph[gid] = replays
    replay_split[str(gid)] = split
  replay_counts = {gid: len(replays_by_graph[gid]) for gid in graph_ids}
  _require(len(set(replay_counts.values())) == 1,
           "decode graphs must share a replay count, got %s" % replay_counts)
  total_replays = replay_counts[graph_ids[0]]
  _require(total_replays > warmup,
           "need more complete replays than warmup=%d" % warmup)

  # One steady token = the same replay index of every graph, concatenated in
  # capture order and sorted by CUPTI start (they are sequential, never overlapping).
  tokens: list[list[dict]] = []
  for idx in range(total_replays):
    rows = [r for gid in graph_ids for r in replays_by_graph[gid][idx]]
    rows.sort(key=lambda r: int(r["start"]))
    tokens.append(rows)

  steady = tokens[warmup:]
  metrics = [_replay_metrics(tok, anchor_class) for tok in steady]
  gaps_us = [(int(b[0]["start"]) - max(int(x["end"]) for x in a)) / 1000.0
             for a, b in zip(steady, steady[1:])]
  bounded = [x for x in gaps_us if x <= inter_replay_gap_cap_us]

  class_names = sorted({c for m in metrics for c in m["classes"]})
  class_medians: dict = {}
  for cls in class_names:
    keys = ("nodes", "node_sum_us", "union_us", "hidden_behind_%s_us" % anchor_class,
            "exposed_vs_%s_us" % anchor_class)
    class_medians[cls] = {
      k: round(float(statistics.median([m["classes"][cls][k] for m in metrics if cls in m["classes"]])), 3)
      for k in keys}

  scalar_keys = ("nodes", "streams", "node_sum_us", "span_us", "kernel_union_us",
                 "overlap_mass_us", "internal_gap_us")
  median = {k: round(float(statistics.median([m[k] for m in metrics])), 3) for k in scalar_keys}
  median["span_discount_vs_node_sum_pct"] = round(
    100.0 * (median["node_sum_us"] - median["span_us"]) / median["node_sum_us"], 3)
  non_anchor_keys = ("union_us", "hidden_behind_%s_us" % anchor_class,
                     "exposed_vs_%s_us" % anchor_class)
  non_anchor = {k: round(float(statistics.median([m["non_anchor"][k] for m in metrics])), 3)
                for k in non_anchor_keys}

  tool_path = pathlib.Path(__file__).resolve()
  source = pathlib.Path(trace)
  report = {
    "schema": SCHEMA,
    "evidence": "OBSERVED_PROFILED_CUPTI_INTERVALS",
    "warning": "Profiled intervals are not an unprofiled token-wall decomposition.",
    "definitions": {
      "overlap_mass_us": "node sum minus all-kernel interval union; not recoverable wall savings",
      "span_discount_vs_node_sum_pct": "(node sum - graph span) / node sum; overlap mass net of internal gaps",
      "streams": "distinct CUPTI kernel-row streamId values; does not prove graph-internal scheduling structure",
      "class_exposure": "per-class exposure is non-additive; use non_anchor_aggregate for the equation",
      "median_equation": "terms are medianed independently and can differ by sub-us rounding/non-additivity",
      "multi_graph": "the decode token is concatenated across the graph list; graphs are sequential so exposure sums",
    },
    "tool": {"path": str(tool_path), "sha256": _sha256(tool_path)},
    "source": {"path": str(source), "size_bytes": source.stat().st_size, "sha256": _sha256(source)},
    "graph_ids": list(graph_ids),
    "graphs": graph_meta,
    "replay_counts": replay_counts,
    "provenance": provenance or {},
    "anchor_class": anchor_class,
    "warmup_replays_dropped": warmup,
    "complete_replays": total_replays,
    "steady_replays": len(steady),
    "replay_split": replay_split,
    "median": median,
    "non_anchor_aggregate": non_anchor,
    "classes": class_medians,
    "inter_replay_gap": {
      "total_count": len(gaps_us),
      "all_median_us": round(float(statistics.median(gaps_us)) if gaps_us else 0, 3),
      "all_p05_us": round(float(_percentile(gaps_us, 0.05) or 0), 3),
      "all_p95_us": round(float(_percentile(gaps_us, 0.95) or 0), 3),
      "bounded_cap_us": inter_replay_gap_cap_us,
      "bounded_count": len(bounded),
      "excluded_count": len(gaps_us) - len(bounded),
      "bounded_median_us": round(float(statistics.median(bounded)) if bounded else 0, 3),
    },
    "shape_census": _shape_census(steady, 0),
  }
  return report


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--trace", required=True)
  ap.add_argument("--graph-ids", default=",".join(map(str, DEFAULT_GRAPH_IDS)),
                  help="comma-separated CUPTI graphIds of one decode token")
  ap.add_argument("--warmup", type=int, default=2)
  ap.add_argument("--anchor-class", default=ANCHOR_CLASS)
  ap.add_argument("--inter-replay-gap-cap-us", type=float, default=5000.0)
  ap.add_argument("--output", required=True)
  ap.add_argument("--capture-nsys-version")
  ap.add_argument("--capture-command")
  ap.add_argument("--export-command")
  args = ap.parse_args()
  graph_ids = tuple(int(x) for x in args.graph_ids.split(",") if x)
  report = analyze(args.trace, graph_ids, args.warmup, args.anchor_class,
                   args.inter_replay_gap_cap_us,
                   {"capture_nsys_version": args.capture_nsys_version,
                    "capture_command": args.capture_command,
                    "export_command": args.export_command,
                    "reproduction_command": " ".join(shlex.quote(x) for x in sys.argv)})
  out = pathlib.Path(args.output)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  m = report["median"]
  print("nodes=%d node_sum=%.1fus span=%.1fus union=%.1fus overlap_mass=%.1fus (%.2f%%)" % (
    m["nodes"], m["node_sum_us"], m["span_us"], m["kernel_union_us"],
    m["overlap_mass_us"], m["span_discount_vs_node_sum_pct"]))
  print("non_anchor exposed=%.1fus hidden=%.1fus union=%.1fus" % (
    report["non_anchor_aggregate"]["exposed_vs_%s_us" % args.anchor_class],
    report["non_anchor_aggregate"]["hidden_behind_%s_us" % args.anchor_class],
    report["non_anchor_aggregate"]["union_us"]))
  for cls in report["classes"]:
    c = report["classes"][cls]
    print("  %-16s n=%4d sum=%8.1fus union=%8.1fus exposed=%8.1fus hidden=%8.1fus" % (
      cls, c["nodes"], c["node_sum_us"], c["union_us"],
      c["exposed_vs_%s_us" % args.anchor_class],
      c["hidden_behind_%s_us" % args.anchor_class]))
  print("wrote %s" % out)


if __name__ == "__main__":
  sys.exit(main())
