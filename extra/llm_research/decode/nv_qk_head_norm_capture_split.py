#!/usr/bin/env python3
"""Split retained q/k head-norm durations into body vs predecessor idle.

Measurement/audit tooling only. It reads the two SHA-pinned retained captures
and, for the tinygrad ``reduce_output_rmsnorm_{32,8}_128`` nodes and the llama
``q_norm`` / ``k_norm`` nodes, reports three quantities in the capture's own
clock domain:

  duration_us   = end_us - start_us            (kernel residence / body)
  dep_gap_us    = start_us - max(data-pred end) (wait after the semantic
                                                  producer is ready)
  launch_gap_us = start_us - raw-pred end       (gap after the immediately
                                                  preceding stream kernel)

``dep_gap`` and ``launch_gap`` are negative when the profiler observes the
consumer starting before its predecessor ends (PDL/overlap). No new GPU work
is launched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]

TINYGRAD_CAPTURE = ROOT / "docs/task_workflow/evidence/nv-third-party-theory-audit-20260822/probe2-tinygrad-capture.json"
LLAMA_PDL0_DAG = ROOT / "docs/task_workflow/evidence/nv-third-party-theory-audit-20260822/probe2-llama-pdl0-dag.json"


def _median(values: list[float]) -> float | None:
  return statistics.median(values) if values else None


def _summary(tag: str, rows: list[dict]) -> dict:
  def vals(key: str) -> list[float]:
    return [r[key] for r in rows if key in r]
  return {
    "tag": tag,
    "n": len(rows),
    "duration_us_median": _median(vals("duration_us")),
    "duration_us_min": min(vals("duration_us")) if vals("duration_us") else None,
    "duration_us_max": max(vals("duration_us")) if vals("duration_us") else None,
    "dep_gap_us_median": _median(vals("dep_gap_us")),
    "launch_gap_us_median": _median(vals("launch_gap_us")),
    "samples": rows,
  }


def _tinygrad_rows() -> dict[str, list[dict]]:
  data = json.loads(TINYGRAD_CAPTURE.read_text())
  nodes = {n["id"]: n for n in data["nodes"]}
  incoming: dict[int, list[int]] = {n["id"]: [] for n in data["nodes"]}
  for edge in data["edges"]:
    if edge.get("kind") == "RAW":
      incoming[edge["to"]].append(edge["from"])

  out: dict[str, list[dict]] = {"q_norm": [], "k_norm": []}
  target = {
    "reduce_output_rmsnorm_32_128": "q_norm",
    "reduce_output_rmsnorm_8_128": "k_norm",
  }
  for node in data["nodes"]:
    role = target.get(node.get("name"))
    if role is None:
      continue
    preds = incoming[node["id"]]
    dep_gap = None
    if preds:
      dep_gap = round(node["start_us"] - max(nodes[p]["end_us"] for p in preds), 3)
    # The capture has no stream id. Approximate the immediately preceding
    # stream kernel as the latest end in the same graph group at or before
    # this node's start. This separates "data is ready" (dep_gap) from the
    # actual queue gap after the kernel physically ahead of it.
    launch_gap = None
    same_group = [n for n in data["nodes"]
                  if n["group_id"] == node["group_id"] and n["end_us"] <= node["start_us"]]
    if same_group:
      launch_gap = round(node["start_us"] - max(n["end_us"] for n in same_group), 3)
    out[role].append({
      "id": node["id"],
      "layer": node["id"],
      "name": node["name"],
      "duration_us": round(node["end_us"] - node["start_us"], 3),
      "start_us": round(node["start_us"], 3),
      "end_us": round(node["end_us"], 3),
      "dep_gap_us": dep_gap,
      "launch_gap_us": launch_gap,
    })
  return out


def _llama_rows() -> dict[str, list[dict]]:
  data = json.loads(LLAMA_PDL0_DAG.read_text())
  nodes = data["nodes"]
  incoming_data: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
  for edge in data.get("data_edges", []):
    incoming_data[edge["to"]].append(edge["from"])
  raw_pred: dict[int, int | None] = {i: None for i in range(len(nodes))}
  for edge in data.get("raw_edges", []):
    raw_pred[edge["to"]] = edge["from"]

  out: dict[str, list[dict]] = {"q_norm": [], "k_norm": []}
  for index, node in enumerate(nodes):
    role = node.get("role")
    if role not in ("q_norm", "k_norm"):
      continue
    dep_gap = None
    preds = incoming_data.get(index) or []
    if preds:
      dep_gap = round(node["start_us"] - max(nodes[p]["end_us"] for p in preds), 3)
    launch_gap = None
    rp = raw_pred.get(index)
    if rp is not None:
      launch_gap = round(node["start_us"] - nodes[rp]["end_us"], 3)
    out[role].append({
      "index": index,
      "layer": node.get("layer"),
      "name": node.get("name"),
      "duration_us": node["duration_us"],
      "start_us": node["start_us"],
      "end_us": node["end_us"],
      "dep_gap_us": dep_gap,
      "launch_gap_us": launch_gap,
    })
  return out


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out-json", type=pathlib.Path, required=True)
  args = ap.parse_args()

  tg = _tinygrad_rows()
  ll = _llama_rows()
  result = {
    "schema": "tinygrad.nv_qk_head_norm_capture_split.v1",
    "sources": {
      "tinygrad_capture": str(TINYGRAD_CAPTURE),
      "llama_pdl0_dag": str(LLAMA_PDL0_DAG),
    },
    "tinygrad": {k: _summary(f"tinygrad_{k}", v) for k, v in tg.items()},
    "llama": {k: _summary(f"llama_{k}", v) for k, v in ll.items()},
  }
  args.out_json.parent.mkdir(parents=True, exist_ok=True)
  args.out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  for side in ("tinygrad", "llama"):
    for role in ("q_norm", "k_norm"):
      s = result[side][role]
      print(f"{side:8} {role:6} n={s['n']:2} "
            f"dur_med={s['duration_us_median']:.3f}us "
            f"dep_gap_med={s['dep_gap_us_median']} "
            f"launch_gap_med={s['launch_gap_us_median']}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
