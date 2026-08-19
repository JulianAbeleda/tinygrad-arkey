#!/usr/bin/env python3
"""Queue-count scheduling sweep over a recorded decode dependency DAG.

Answers the narrow question: does adding a third (or fourth, or Nth) compute
queue move the decode wall? It computes the dependency-critical-path lower
bound and a longest-tail list schedule for 1..N queues. The schedule ignores
cross-queue wait overhead, so it is an upper bound on the benefit of more
queues; real hardware can only be equal or worse.

Input is the duration-attached DAG produced by the route attribution tooling
(schema tinygrad.nv_dag_duration_head.v1). Output is a deterministic sweep
record.
"""
from __future__ import annotations

import argparse, heapq, json
from typing import Any


def load_dag(path: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  with open(path, encoding="utf-8") as f:
    payload = json.load(f)
  if "nodes" not in payload or "edges" not in payload:
    raise ValueError("dag must contain nodes and edges")
  return payload["nodes"], payload["edges"]


def sweep(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], max_queues: int) -> dict[str, Any]:
  n = len(nodes)
  duration = [float(node["duration_us"]) for node in nodes]
  preds: list[set[int]] = [set() for _ in range(n)]
  succs: list[set[int]] = [set() for _ in range(n)]
  for edge in edges:
    a, b = int(edge["from"]), int(edge["to"])
    if a < 0 or b < 0 or a >= n or b >= n:
      raise ValueError(f"edge out of range: {edge}")
    preds[b].add(a)
    succs[a].add(b)

  # Topological order is implicit in call order, but derive it defensively.
  indeg = [len(p) for p in preds]
  frontier = [i for i in range(n) if indeg[i] == 0]
  order: list[int] = []
  while frontier:
    i = frontier.pop(0)
    order.append(i)
    for j in succs[i]:
      indeg[j] -= 1
      if indeg[j] == 0:
        frontier.append(j)
  if len(order) != n:
    raise ValueError("dag contains a cycle")

  longest_to = [0.0] * n
  for i in order:
    longest_to[i] = duration[i] + max((longest_to[p] for p in preds[i]), default=0.0)
  critical_path = max(longest_to)

  tail = [0.0] * n
  for i in reversed(order):
    tail[i] = duration[i] + max((tail[j] for j in succs[i]), default=0.0)

  def list_schedule(queue_count: int) -> float:
    remaining = [len(p) for p in preds]
    ready = [(-tail[i], i) for i in range(n) if remaining[i] == 0]
    heapq.heapify(ready)
    busy = [0.0] * queue_count
    end = [0.0] * n
    while ready:
      _, i = heapq.heappop(ready)
      base = max((end[p] for p in preds[i]), default=0.0)
      queue = min(range(queue_count), key=lambda q: (max(busy[q], base), q))
      start = max(busy[queue], base)
      end[i] = start + duration[i]
      busy[queue] = end[i]
      for j in succs[i]:
        remaining[j] -= 1
        if remaining[j] == 0:
          heapq.heappush(ready, (-tail[j], j))
    return max(busy, default=0.0)

  serialized = sum(duration)
  rows = []
  for q in range(1, max_queues + 1):
    span = list_schedule(q)
    rows.append({
      "queues": q,
      "span_us": round(span, 3),
      "saving_vs_serial_us": round(serialized - span, 3),
      "slack_vs_critical_path_us": round(span - critical_path, 3),
    })

  return {
    "schema": "tinygrad.nv_queue_count_sweep.v1",
    "node_count": n,
    "serialized_us": round(serialized, 3),
    "critical_path_us": round(critical_path, 3),
    "total_schedule_slack_us": round(serialized - critical_path, 3),
    "schedule": rows,
    "method": "longest-tail list schedule; cross-queue wait overhead omitted (upper bound on queue-count benefit)",
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--dag", required=True)
  ap.add_argument("--max-queues", type=int, default=8)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  nodes, edges = load_dag(args.dag)
  result = sweep(nodes, edges, args.max_queues)
  with open(args.out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
  print(json.dumps(result, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
