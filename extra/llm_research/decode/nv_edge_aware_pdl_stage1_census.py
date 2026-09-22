#!/usr/bin/env python3
"""Stage 1 typed edge-aware split-phase census prediction.

This is GPU-free measurement tooling.  It consumes the canonical tinygrad DAG
capture and the Phase D schedule replay, then applies the closed scheduler
rule that the production graph will use:

- RAW forward edges only;
- same graph group, same queue, consecutive QMD pair, zero encoded waits;
- exactly one RAW producer for the consumer;
- exactly one RAW consumer for the producer;
- span-level alias safety when the capture carries spans (otherwise the edge
  remains alias-unverified and cannot be called safe).

The production construction census written by ``NV_SPLIT_PHASE_CENSUS_JSON``
must reconcile against this prediction before any endpoint run.
"""
from __future__ import annotations

import argparse, collections, json, pathlib, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import nv_pdl_phase_a_census as phase_a
import nv_pdl_phase_d_static_coverage as phase_d

REASONS = ("cross_group", "queue_split", "adjacency", "encoded_wait",
           "multi_producer_fallback", "multi_consumer_fallback", "alias_rejected",
           "candidate_armed")


def _git_head() -> str:
  return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
  return max(a[0], b[0]) < min(a[1], b[1])


def raw_producer_map(edges: list[dict]) -> dict[int, list[int]]:
  producers: dict[int, set[int]] = collections.defaultdict(set)
  for edge in edges:
    if edge.get("kind") == "RAW": producers[int(edge["to"])].add(int(edge["from"]))
  return {consumer: sorted(ids) for consumer, ids in producers.items()}


def raw_consumer_map(edges: list[dict]) -> dict[int, list[int]]:
  consumers: dict[int, set[int]] = collections.defaultdict(set)
  for edge in edges:
    if edge.get("kind") == "RAW": consumers[int(edge["from"])].add(int(edge["to"]))
  return {producer: sorted(ids) for producer, ids in consumers.items()}


def classify(dag: dict, broad: dict) -> dict:
  raw_edges = [e for e in dag.get("edges") or [] if e.get("kind") == "RAW"]
  structural = {(int(row["from"]), int(row["to"])): row for row in broad["static_edges"]["rows"]}
  raw_producers = raw_producer_map(dag.get("edges") or [])
  raw_consumers = raw_consumer_map(dag.get("edges") or [])
  non_raw_edges = [e for e in dag.get("edges") or [] if e.get("kind") != "RAW"]

  rows = []
  for edge in raw_edges:
    producer, consumer = int(edge["from"]), int(edge["to"])
    structural_row = structural.get((producer, consumer))
    structural_reason = structural_row["reason"] if structural_row is not None else "unknown"
    if structural_reason == "cross_group": reason = "cross_group"
    elif structural_reason != "armed": reason = structural_reason
    elif len(raw_producers.get(consumer, [])) > 1: reason = "multi_producer_fallback"
    elif len(raw_consumers.get(producer, [])) > 1: reason = "multi_consumer_fallback"
    else:
      reason = "candidate_armed"
      spans = edge.get("spans")
      if spans is None:
        alias_status = "unverified"
      else:
        alias_status = "safe"
        for other in non_raw_edges:
          if int(other["to"]) != consumer or other.get("spans") is None: continue
          if any(_spans_overlap(tuple(s), tuple(o)) for s in spans for o in other["spans"]):
            reason = "alias_rejected"
            alias_status = "rejected"
            break
      if alias_status == "unverified": reason = "candidate_armed_unverified"
    rows.append({
      "from": producer, "to": consumer, "reason": reason,
      "structural_reason": structural_reason,
      "alias_status": "unverified" if reason == "candidate_armed_unverified" else ("safe" if reason == "candidate_armed" else "rejected" if reason == "alias_rejected" else "n/a"),
      "spans": edge.get("spans"),
    })

  by_reason = collections.Counter(row["reason"] for row in rows)
  return {
    "raw_total": len(raw_edges),
    "by_reason": {reason: by_reason.get(reason, 0) for reason in REASONS}
                 | {"candidate_armed_unverified": by_reason.get("candidate_armed_unverified", 0)},
    "alias_safe_total": by_reason.get("candidate_armed", 0),
    "span_unverified_total": by_reason.get("candidate_armed_unverified", 0),
    "rows": rows,
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--control", required=True, type=pathlib.Path)
  ap.add_argument("--phase-d", required=True, type=pathlib.Path)
  ap.add_argument("--out", required=True, type=pathlib.Path)
  ap.add_argument("--queues", default="1,2")
  args = ap.parse_args()

  queue_modes = tuple(int(v) for v in args.queues.split(","))
  if any(v not in (1, 2) for v in queue_modes):
    raise SystemExit("--queues must be a comma-separated list of 1 and 2")

  dag = phase_a.load_tinygrad(args.control)
  artifact = json.loads(args.phase_d.read_text(encoding="utf-8"))
  unique_names = tuple(sorted({str(n["name"]) for n in dag["nodes"]}))
  broad_env = {"producers": unique_names, "consumers": unique_names, "trigger_position": "end"}
  broad = {q: phase_d.broad_census(dag, q, broad_env) for q in queue_modes}

  checks = {}
  for q, census in broad.items():
    expected = artifact["broad_census"][str(q)]
    checks[str(q)] = {
      "armed_pair_count": census["armed_pair_count"] == expected["armed_pair_count"],
      "armed_with_forward_static_edge": census["armed_with_forward_static_edge"] == expected["armed_with_forward_static_edge"],
      "static_edge_reasons": census["static_edges"]["by_reason"] == expected["static_edges"]["by_reason"],
    }
    if not all(checks[str(q)].values()):
      raise SystemExit(f"Phase D replay mismatch on {q} queues: {checks[str(q)]}")

  doc = {
    "schema": "tinygrad.nv_edge_aware_pdl_stage1_census.v1",
    "commit": _git_head(),
    "inputs": {"control": str(args.control), "phase_d": str(args.phase_d), "queues": list(queue_modes)},
    "method": {
      "arm_rule": "RAW, same-group same-queue consecutive QMD pair, zero encoded waits, one RAW producer, one RAW consumer, alias-safe spans",
      "phase_d_replay": True,
      "span_policy": "spans required for alias-safe; absent spans remain unverified",
    },
    "reconciliation": checks,
    "censuses": {str(q): classify(dag, broad[q]) for q in queue_modes},
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps({str(q): {k: v for k, v in doc["censuses"][str(q)].items() if k != "rows"}
                    for q in queue_modes}, indent=2))
  return 0


if __name__ == "__main__":
  sys.exit(main())
