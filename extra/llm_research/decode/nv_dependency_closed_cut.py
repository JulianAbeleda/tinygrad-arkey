#!/usr/bin/env python3
"""CPU-only evaluation of occurrence-pinned NV decode queue cuts."""
from __future__ import annotations
import argparse, hashlib, json, pathlib


def load(path:str) -> dict:
  with open(path, encoding="utf-8") as f: return json.load(f)


def program_identity(name:str) -> str:
  stem, sep, suffix = name.rpartition("_")
  return stem if sep and len(suffix) == 64 and all(c in "0123456789abcdef" for c in suffix) else name


# Pre-split (2026-08-04/05) attention blocks were self-contained 9-node runs:
# the fixed offsets below name the Q/K branches that precede each flash.  The
# 2026-08-12 post-split DAG names the same roles with q4k/q6k/reduce_output/
# rmsnorm families, so a layout detector selects the matching selector and any
# unrelated graph fails closed.
LEGACY_Q_OFFSETS = ((7, "E_2_8_16_4_"), (5, "r_8_16_8_"), (3, "E_8_2_16_4_"),
                    (1, ("r_8_8_16_2_4_", "E_8_8_16_2_")))
LEGACY_K_OFFSETS = ((8, "E_4_2_8_16_4_"), (6, "r_2_8_4_4_16_"), (4, "E_2_8_16_4_4_"))
POST_SPLIT_Q_PREFIXES = ("q4k_g3_lanemap_gemv_4096_4096",
                         "q4k_warp_coop_q8_dp4a_partial_4096_4096",
                         "reduce_output_rmsnorm_32_128", "E_16_32_4_2_")
POST_SPLIT_K_PREFIXES = ("q4k_g3_lanemap_gemv_1024_4096",
                         "q4k_warp_coop_q8_dp4a_partial_1024_4096",
                         "q6k_gen_partial_1024_4096", "q6k_q8_dp4a_1024_4096",
                         "reduce_output_rmsnorm_8_128", "r_8_8_16_2_4_", "E_8_8_16_2_")


def attention_cuts(dag:dict) -> dict[str, set[int]]:
  """Return the two coherent pre-flash branches from each attention block.

  The K branch stops before the shared merge.  The Q branch includes its
  independent reduction tail.  Both the pre-split 9-node block layout and the
  post-split q4k/q6k/rmsnorm family layout are recognized by name; any other
  graph layout fails closed instead of silently selecting offsets.
  """
  nodes = dag["nodes"]
  flashes = [n["id"] for n in nodes if n["name"].startswith("flash_block_tiled_")]
  if not flashes:
    raise ValueError("no flash_block_tiled_ nodes; attention cuts undefined")
  if _matches_legacy_layout(nodes, flashes[0]):
    return _legacy_attention_cuts(nodes, flashes)
  return _post_split_attention_cuts(nodes, flashes)


def _matches_legacy_layout(nodes:list[dict], flash:int) -> bool:
  if flash < 8: return False
  return all(nodes[flash-x]["name"].startswith(p) for x, p in LEGACY_Q_OFFSETS + LEGACY_K_OFFSETS)


def _legacy_attention_cuts(nodes:list[dict], flashes:list[int]) -> dict[str, set[int]]:
  q, k = set(), set()
  for f in flashes:
    if not _matches_legacy_layout(nodes, f):
      raise ValueError(f"Q/K branch mismatch before flash {f}: layout is not the pre-split 9-node block")
    q.update(f-x for x, _ in LEGACY_Q_OFFSETS)
    k.update(f-x for x, _ in LEGACY_K_OFFSETS)
  return {"attention_q": q, "attention_k": k}


def _post_split_attention_cuts(nodes:list[dict], flashes:list[int]) -> dict[str, set[int]]:
  """Select Q/K family nodes from each program-order flash window.

  The post-split DAG interleaves attention blocks with FFN work, so the Q/K
  roles are matched by family prefix inside the window between consecutive
  flash_block_tiled_ nodes (the first flash's window starts at program order
  0).  A flash whose window lacks either family fails closed.
  """
  q, k, prev = set(), set(), -1
  for f in flashes:
    window = nodes[prev+1:f]
    qids = [n["id"] for n in window if n["name"].startswith(POST_SPLIT_Q_PREFIXES)]
    kids = [n["id"] for n in window if n["name"].startswith(POST_SPLIT_K_PREFIXES)]
    if not qids or not kids:
      raise ValueError(f"flash {f}: window [{prev+1},{f}) lacks Q/K family nodes; graph layout not recognized")
    q.update(qids); k.update(kids)
    prev = f
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
