#!/usr/bin/env python3
"""Derive a closed NV two-queue support allowlist from a captured full-token DAG.

This is CPU-only.  It mirrors the bounded HCQ policy: named admitted programs are
assigned to the least-populated compute queue; all other programs remain on queue
zero; every captured RAW/WAR/WAW edge and each queue's own order is respected.
The output is an input to a *separate* GPU correctness/wall gate, never a timing
credit.  Quantized projections and flash are excluded structurally: a candidate
must have no semantic identity and must pass an explicit name/duration whitelist.
"""
from __future__ import annotations
import argparse, json, pathlib


def load(path:str) -> dict:
  with open(path, encoding="utf-8") as f: return json.load(f)


def schedule(nodes:list[dict], deps:list[list[int]], edges:list[dict], admitted:set[str]):
  """CPU model of the opt-in HCQ name-pinned two-queue picker."""
  end, q_end, q_count, queues = [0.0] * len(nodes), [0.0, 0.0], [0, 0], []
  for node in nodes:
    i = node["id"]
    q = min(range(2), key=lambda x:q_count[x]) if node["name"] in admitted else 0
    start = max([q_end[q], *[end[d] for d in deps[i]]])
    end[i] = start + float(node["duration_us"])
    q_end[q], q_count[q] = end[i], q_count[q] + 1
    queues.append(q)
  cross = sum(queues[e["from"]] != queues[e["to"]] for e in edges)
  return {"span_us":max(q_end), "queue_end_us":q_end, "queue_nodes":q_count,
          "cross_queue_edges":cross}


def derive(dag:dict, max_duration_us:float=6.0) -> dict:
  nodes, edges = dag["nodes"], dag["edges"]
  if [n["id"] for n in nodes] != list(range(len(nodes))): raise ValueError("node IDs must be dense program order")
  deps = [[] for _ in nodes]
  for edge in edges: deps[edge["to"]].append(edge["from"])
  # Candidate eligibility is a proof obligation, not a broad kernel-class
  # heuristic: the full captured node has measured duration, no semantic MMQ
  # identity, and a small support-program spelling.  The optimizer below then
  # rejects any candidate which does not shorten the dependency-aware schedule.
  candidates = sorted({n["name"] for n in nodes if n.get("metadata") is None and n["name"].startswith(("E_", "r_"))
                      and float(n["duration_us"]) <= max_duration_us})
  baseline = schedule(nodes, deps, edges, set())
  chosen, current, steps = set(), baseline, []
  while True:
    options = []
    for name in candidates:
      if name in chosen: continue
      row = schedule(nodes, deps, edges, chosen | {name})
      options.append((current["span_us"] - row["span_us"], name, row))
    gain, name, row = max(options, default=(0.0, "", current))
    if gain <= 0: break
    chosen.add(name); current = row
    steps.append({"name":name, "incremental_saving_us":gain, "span_us":row["span_us"]})
  instances = {name:sum(n["name"] == name for n in nodes) for name in sorted(chosen)}
  return {"schema":"tinygrad.nv_support_overlap_allowlist.v1", "candidate_contract":{
    "metadata":"none", "name_prefixes":["E_", "r_"], "max_duration_us":max_duration_us,
    "excludes":"all semantic/MMQ/GEMV and flash nodes"}, "baseline":baseline,
    "selected":sorted(chosen), "selected_instances":instances, "steps":steps,
    "candidate":current, "predicted_saving_us":baseline["span_us"]-current["span_us"],
    "warning":"CPU duration schedule only; cross-queue waits and DRAM contention require native token evidence."}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dag", required=True); ap.add_argument("--out", required=True)
  ap.add_argument("--max-duration-us", type=float, default=6.0)
  args = ap.parse_args()
  out = derive(load(args.dag), args.max_duration_us)
  path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
  print(json.dumps(out, indent=2))

if __name__ == "__main__": main()
