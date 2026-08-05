#!/usr/bin/env python3
"""CPU-only evaluation of occurrence-pinned NV decode queue cuts."""
from __future__ import annotations
import argparse, hashlib, json, pathlib


def load(path:str) -> dict:
  with open(path, encoding="utf-8") as f: return json.load(f)


def program_identity(name:str) -> str:
  stem, sep, suffix = name.rpartition("_")
  return stem if sep and len(suffix) == 64 and all(c in "0123456789abcdef" for c in suffix) else name


def attention_cuts(dag:dict) -> dict[str, set[int]]:
  """Return the two coherent pre-flash branches from each attention block.

  The K branch stops before the shared merge.  The Q branch includes its
  independent reduction tail.  Generated names are checked so an unrelated
  graph layout fails closed instead of silently selecting offsets.
  """
  nodes = dag["nodes"]
  flashes = [n["id"] for n in nodes if n["name"].startswith("flash_block_tiled_")]
  q, k = set(), set()
  for f in flashes:
    if f < 8: raise ValueError(f"flash node {f} has no complete attention prefix")
    qids, kids = [f-x for x in (7, 5, 3, 1)], [f-x for x in (8, 6, 4)]
    qprefix = ("E_2_8_16_4_", "r_8_16_8_", "E_8_2_16_4_", ("r_8_8_16_2_4_", "E_8_8_16_2_"))
    kprefix = ("E_4_2_8_16_4_", "r_2_8_4_4_16_", "E_2_8_16_4_4_")
    if any(not nodes[i]["name"].startswith(p) for i, p in zip(qids, qprefix)): raise ValueError(f"Q branch mismatch before flash {f}")
    if any(not nodes[i]["name"].startswith(p) for i, p in zip(kids, kprefix)): raise ValueError(f"K branch mismatch before flash {f}")
    q.update(qids); k.update(kids)
  return {"attention_q": q, "attention_k": k}


def schedule(dag:dict, auxiliary:set[int], wait_cost_us:float=0.0) -> dict:
  nodes, edges = dag["nodes"], dag["edges"]
  deps:list[list[int]] = [[] for _ in nodes]
  for edge in edges: deps[edge["to"]].append(edge["from"])
  end, queue_end = [0.0] * len(nodes), [0.0, 0.0]
  # queue_access[q][other] mirrors HCQGraph's monotonically cached signal value.
  queue_access = [[0, 0], [0, 0]]
  waits:list[dict] = []
  for node in nodes:
    i, q = node["id"], int(node["id"] in auxiliary)
    cross = [d for d in deps[i] if int(d in auxiliary) != q]
    start = max([queue_end[q], *[end[d] for d in deps[i]]])
    if cross and (value := max(cross)+1) > queue_access[q][1-q]:
      queue_access[q][1-q] = value
      start += wait_cost_us
      waits.append({"node": i, "queue": q, "waits_for": value-1})
    end[i] = start + float(node["duration_us"])
    queue_end[q] = end[i]
  return {"span_us": max(queue_end), "queue_end_us": queue_end, "aux_nodes": len(auxiliary),
          "wait_events": waits, "wait_count": len(waits)}


def analyze(dag:dict, wait_cost_us:float=0.363) -> dict:
  nodes = dag["nodes"]
  if [n["id"] for n in nodes] != list(range(len(nodes))): raise ValueError("node IDs must be dense program order")
  if any(float(n.get("duration_us", 0)) <= 0 for n in nodes): raise ValueError("every node needs a positive measured duration")
  baseline = schedule(dag, set(), wait_cost_us)
  rows = {}
  for name, cut in attention_cuts(dag).items():
    raw, costed = schedule(dag, cut, 0.0), schedule(dag, cut, wait_cost_us)
    policy_graphs = []
    for gid in dict.fromkeys(n["group_id"] for n in nodes):
      members = [n for n in nodes if n["group_id"] == gid]
      start = members[0]["id"]
      selected = [{"index":n["id"]-start, "identity":program_identity(n["name"])} for n in members if n["id"] in cut]
      # Include the first post-cut consumer in the identity.  Later diagnostic
      # sampler/logit-return suffixes may legitimately differ from production.
      prefix_count = min(len(members), max((x["index"] for x in selected), default=-1) + 2)
      policy_graphs.append({"prefix_count":prefix_count,
        "prefix_name_digest":hashlib.sha256("\n".join(program_identity(n["name"]) for n in members[:prefix_count]).encode()).hexdigest(),
        "selected":selected})
    rows[name] = {"indices": sorted(cut), "index_spec": ",".join(map(str, sorted(cut))),
      "cut_policy":{"schema":"tinygrad.nv_multi_queue_cut_policy.v1", "graphs":policy_graphs},
      "raw": raw, "costed": costed, "raw_saving_us": baseline["span_us"]-raw["span_us"],
      "costed_saving_us": baseline["span_us"]-costed["span_us"],
      "syncs_per_attention_block": costed["wait_count"] / 36}
  return {"schema":"tinygrad.nv_dependency_closed_cut.v1", "wait_cost_us":wait_cost_us,
          "capture_identity":{"node_count":len(nodes), "name_digest":hashlib.sha256("\n".join(n["name"] for n in nodes).encode()).hexdigest()},
          "baseline":baseline, "candidates":rows,
          "verdict":"GPU_ELIGIBLE" if max(x["costed_saving_us"] for x in rows.values()) >= 50 else "CPU_NO_GO"}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--dag", required=True); ap.add_argument("--out")
  ap.add_argument("--wait-cost-us", type=float, default=0.363)
  args = ap.parse_args(); result = analyze(load(args.dag), args.wait_cost_us)
  text = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.out:
    path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
  print(text, end="")


if __name__ == "__main__": main()
