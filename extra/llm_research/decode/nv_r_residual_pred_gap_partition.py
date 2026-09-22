#!/usr/bin/env python3
"""Offline predecessor-gap partition for the R-residual adjudication.

For each retained production R kernel the probe measured cache_state ~= 0 on
the native HCQ path.  This tool tests the complementary hypothesis that the
residual is *upstream dependency wait* by reading the retained full-token
production capture and, for every R-kernel node, computing

    pred_gap = node.start_us - max(predecessor.end_us)

The production schedule chains QMDs by reusing the producer's release as the
consumer's start timestamp, so a chained consumer records ``start ==
producer end`` and ``pred_gap ~= 0``.  A large positive pred_gap would instead
mean the residual is visible idle wait before the command starts.  The tool
also emits the production command-interval (``duration_us``) distribution per
symbol and, for the 36-call rows, per occurrence ordinal as a layer proxy.

Measurement tooling only; no production code path is touched.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
R_SYMBOLS = [
  "q4k_warp_coop_q8_dp4a_partial_4096_4096",
  "q4k_g3_lanemap_gemv_4096_4096",
  "q4k_g3_lanemap_gemv_epi_resadd_4096_4096",
  "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128",
  "q4k_warp_coop_q8_dp4a_partial_1024_4096",
  "q4k_g3_lanemap_gemv_1024_4096",
]


def _qs(vals: list[float]) -> dict:
  s = sorted(vals)
  n = len(s)
  return {
    "count": n,
    "mean": round(statistics.mean(s), 4),
    "median": round(statistics.median(s), 4),
    "min": round(s[0], 4),
    "max": round(s[-1], 4),
    "p10": round(s[max(0, int(0.10 * (n - 1)))], 4),
    "p90": round(s[min(n - 1, int(0.90 * (n - 1)))], 4),
  }


def main() -> int:
  capture_path = pathlib.Path(sys.argv[1])
  out_dir = pathlib.Path(sys.argv[2])
  out_dir.mkdir(parents=True, exist_ok=True)

  d = json.loads(capture_path.read_text())
  nodes, edges = d["nodes"], d["edges"]
  by_id = {n["id"]: n for n in nodes}
  assert len(by_id) == len(nodes), "non-contiguous node ids"

  pred_end: dict[int, float] = {}
  pred_count: dict[int, int] = {}
  cross_pred_count: dict[int, int] = {}
  for e in edges:
    src, dst = e["from"], e["to"]
    if by_id[src]["group_id"] == by_id[dst]["group_id"]:
      if dst not in pred_end or by_id[src]["end_us"] > pred_end[dst]:
        pred_end[dst] = by_id[src]["end_us"]
    else:
      cross_pred_count[dst] = cross_pred_count.get(dst, 0) + 1
    pred_count[dst] = pred_count.get(dst, 0) + 1

  rows = []
  for sym in R_SYMBOLS:
    hits = [n for n in nodes if n["name"] == sym]
    hits.sort(key=lambda n: n["start_us"])
    gaps = []
    for n in hits:
      if n["id"] in pred_end:
        gaps.append(round(n["start_us"] - pred_end[n["id"]], 4))
    p_vals = [round(n["duration_us"], 4) for n in hits]
    ordinal_p = [round(n["duration_us"], 4) for n in hits]
    rows.append({
      "symbol": sym,
      "occurrences": len(hits),
      "same_group_pred_measured": len(gaps),
      "cross_group_pred_nodes": sum(1 for n in hits if n["id"] in cross_pred_count),
      "pred_gap_us": _qs(gaps),
      "pred_count_median": statistics.median([pred_count.get(n["id"], 0) for n in hits]),
      "command_interval_p_us": _qs(p_vals),
      "ordinal_p_us": ordinal_p,
    })

  # Same-group edges only: fraction whose consumer starts before the
  # predecessor end (chain/timestamp-reuse, negative or zero gap).
  same_group_edges = [e for e in edges
                      if by_id[e["from"]]["group_id"] == by_id[e["to"]]["group_id"]]
  same_group_gaps = [by_id[e["to"]]["start_us"] - by_id[e["from"]]["end_us"]
                     for e in same_group_edges]
  chain_share = sum(1 for g in same_group_gaps if g <= 0.001) / len(same_group_gaps)

  payload = {
    "schema": "tinygrad.nv_r_residual_pred_gap_partition.v1",
    "capture": str(capture_path),
    "capture_sha256": "see nv-third-party-theory-audit sha256.txt",
    "nodes": len(nodes),
    "edges": len(edges),
    "same_group_edges": len(same_group_edges),
    "cross_group_edges": len(edges) - len(same_group_edges),
    "method": "pred_gap = start_us - max(same-group pred end_us); "
              "cross-group (replay-boundary) predecessors excluded because their "
              "end_us sits in a prior capture window separated by host replay gaps",
    "same_group_edges_nonpositive_gap_share": round(chain_share, 4),
    "rows": rows,
  }
  out = out_dir / "nv-r-residual-pred-gap-partition.json"
  out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"out": str(out), "rows": len(rows),
                    "same_group_edges_nonpositive_gap_share": round(chain_share, 4)}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
