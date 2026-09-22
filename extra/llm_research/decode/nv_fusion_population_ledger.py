#!/usr/bin/env python3
"""CPU-only exhaustive population ledger for the NV decode fusion/dataflow workstream.

Loads a duration-bearing DAG in the `nv_dependency_closed_cut` schema (nodes with
id/name/duration_us/group_id, edges with from/to) and classifies every node into
the disjoint fusion/dataflow populations of the d512 semantic partition.  The
identity rules mirror the exact 731-node native role partition; unknown programs
fall back to prefix heuristics and are flagged ``exact=False`` rather than being
silently assigned.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, statistics
from collections import Counter, defaultdict

SCHEMA = "tinygrad.nv_fusion_population_ledger.v1"

POP_FLASH = "flash"
POP_NORMS = "norms"
POP_RESIDUAL = "residual_cast_contiguous"
POP_VOCAB = "vocab_feedback"
POP_ROPE_KV = "rope_kv"
POP_QUANT = "quant_core"
POP_Q8PACK = "llama_q8_pack"
POP_OTHER = "other"

SUPPORT = frozenset((POP_NORMS, POP_RESIDUAL, POP_ROPE_KV, POP_VOCAB, POP_Q8PACK, POP_OTHER))
ANCHOR = frozenset((POP_QUANT, POP_FLASH))

HASH64 = re.compile(r"_([0-9a-f]{64})$")


def clean(name: str) -> str:
  return HASH64.sub("", name)


# Exact identity rules, longest prefix first.  Short ambiguous stems ("E", "E_2")
# are equality-only; everything else matches by prefix.
_RULES = [
  ("flash_block_tiled_", POP_FLASH, "flash_score"),
  ("flash_fused_", POP_FLASH, "flash_combine"),
  ("q4k_g3_lanemap_gemv_", POP_QUANT, "quant_core"),
  ("q6k_gen_", POP_QUANT, "quant_core"),
  ("q4k_", POP_QUANT, "quant_core"),
  ("quantize_q8", POP_Q8PACK, "llama_q8_pack"),
  ("E_16_4_2_8_16_2_4_4", POP_VOCAB, "vocab_sampler"),
  ("E_1187_32_4", POP_VOCAB, "vocab_sampler"),
  ("r_32_32_4_32_4", POP_VOCAB, "vocab_sampler"),
  ("r_32_4_1187", POP_VOCAB, "vocab_sampler"),
  ("r_128_16_8_1187", POP_VOCAB, "vocab_sampler"),
  ("r_16_8", POP_VOCAB, "vocab_sampler"),
  ("E_2_8_16_4_4", POP_NORMS, "q_norm_epilogue"),
  ("E_2_8_16_4", POP_NORMS, "k_norm_epilogue"),
  ("E_4_2_8_16_4", POP_NORMS, "q_norm_epilogue"),
  ("E_8_2_16_4", POP_NORMS, "k_norm_epilogue"),
  ("E_128_32_3", POP_RESIDUAL, "ffn_activation_cast"),
  ("E_16_32_4_2", POP_ROPE_KV, "rope_q"),
  ("E_8_8_16_2", POP_ROPE_KV, "kv_store_k_rope_cast"),
  ("r_8_8_16_2_4", POP_ROPE_KV, "kv_store_k_rope_cast_with_q6_partial_reduce"),
  ("r_16_256", POP_NORMS, "rmsnorm_reduce"),
  ("r_2_8_4_4_16", POP_NORMS, "q_norm_reduce"),
  ("r_8_16_8", POP_NORMS, "k_norm_reduce"),
  ("E_2", POP_VOCAB, "token_feedback"),
  ("E", POP_VOCAB, "token_feedback"),
]

# E_32_32_4 is shared by norm epilogues and residual/cast/contiguous roles; the
# 16-hex hash prefix disambiguates.  Hashes marked heuristic were verified by
# structural position on the current 875-node redirect-on authority DAG.
_E32 = [
  ("f14a5cc0d0ed4c90", POP_NORMS, "rmsnorm_epilogue", True),
  ("c6fef3561a9fbeaf", POP_NORMS, "final_rmsnorm_epilogue", True),
  ("0a5eb0ac56c097a0", POP_RESIDUAL, "attention_cast", True),
  ("02a9738c0547f555", POP_RESIDUAL, "attention_residual_add_or_ffn_down_cast", True),
  ("fab82d40f922cf5f", POP_RESIDUAL, "ffn_residual_add_or_block_output_contiguous", True),
  ("86a23e1a5cd1cbd6", POP_RESIDUAL, "block_output_contiguous", False),
  ("81c96a8e654e707f", POP_RESIDUAL, "ffn_residual_add", False),
]


def classify(name: str) -> tuple[str, str, bool]:
  """Return (population, role, exact) for one rendered program name."""
  match = HASH64.search(name)
  if match is not None:
    stem, h = name[:match.start()], match.group(1)
    if stem == "E_32_32_4":
      for hprefix, pop, role, exact in _E32:
        if h.startswith(hprefix): return pop, role, exact
      return POP_RESIDUAL, "elementwise_ambiguous", False
    identity = stem
  else:
    identity = name
  for prefix, pop, role in sorted(_RULES, key=lambda r: -len(r[0])):
    if identity == prefix or (len(prefix) >= 6 and identity.startswith(prefix)):
      return pop, role, True
  return POP_OTHER, "unclassified", False


def _require(condition: bool, message: str) -> None:
  if not condition: raise ValueError(message)


def load(path: str | pathlib.Path) -> dict:
  """Load and validate a duration-bearing DAG; fail closed on malformed input."""
  data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
  nodes, edges = data.get("nodes"), data.get("edges")
  _require(isinstance(nodes, list) and len(nodes) > 0, f"{path}: missing or empty 'nodes'")
  _require(isinstance(edges, list), f"{path}: missing 'edges'")
  ids = [n.get("id") for n in nodes]
  _require(ids == list(range(len(nodes))), f"{path}: node ids must be dense program order 0..{len(nodes)-1}")
  for node in nodes:
    _require(isinstance(node.get("name"), str) and node["name"], f"{path}: node {node.get('id')} needs a non-empty name")
    duration = node.get("duration_us")
    _require(isinstance(duration, (int, float)) and duration > 0, f"{path}: node {node.get('id')} needs a positive duration_us")
    _require("group_id" in node, f"{path}: node {node.get('id')} needs a group_id")
  for edge in edges:
    src, dst = edge.get("from"), edge.get("to")
    _require(isinstance(src, int) and isinstance(dst, int) and 0 <= src < len(nodes) and 0 <= dst < len(nodes),
             f"{path}: edge {src}->{dst} out of range")
  return {"nodes": nodes, "edges": edges}


def analyze(dag: dict) -> dict:
  nodes, edges = dag["nodes"], dag["edges"]
  classified = [classify(n["name"]) for n in nodes]
  by_population: dict[str, list[dict]] = defaultdict(list)
  for node, (population, role, exact) in zip(nodes, classified):
    by_population[population].append({"id": node["id"], "name": node["name"], "population": population,
                                      "role": role, "exact": exact, "duration_us": float(node["duration_us"])})

  # Direct-child-of-anchor census: a support node with an incoming edge from a
  # quant or flash node.  The capture graph carries planner alias edges, so this
  # is a candidate census, not an exact dataflow claim.
  parents: dict[int, list[int]] = defaultdict(list)
  for edge in edges: parents[edge["to"]].append(edge["from"])
  node_pop = [pop for pop, _, _ in classified]
  candidate_ids: set[int] = set()
  for i, pop in enumerate(node_pop):
    if pop not in SUPPORT: continue
    if any(node_pop[p] in ANCHOR for p in parents[i]): candidate_ids.add(i)

  populations: dict[str, dict] = {}
  for population in sorted(by_population):
    rows = by_population[population]
    durations = [r["duration_us"] for r in rows]
    members = [r["id"] for r in rows]
    candidates = [r for r in rows if r["id"] in candidate_ids]
    candidate_epilogues = [r for r in candidates if clean(r["name"]).startswith("E_")]
    custom = sum(1 for r in rows if clean(r["name"]).startswith(("flash_", "q4k_", "q6k_", "quantize_q8")) or
                 any(marker in r["name"] for marker in ("lanemap_gemv", "inkernel", "decode_rmsnorm", "epi_resadd")))
    reductions = sum(1 for r in rows if clean(r["name"]).startswith("r_"))
    epilogues = sum(1 for r in rows if clean(r["name"]).startswith("E_"))
    populations[population] = {
      "node_count": len(rows),
      "total_us": round(sum(durations), 3),
      "mean_us": round(statistics.mean(durations), 3),
      "max_us": round(max(durations), 3),
      "min_us": round(min(durations), 3),
      "exact_count": sum(1 for r in rows if r["exact"]),
      "heuristic_count": sum(1 for r in rows if not r["exact"]),
      "roles": dict(Counter(r["role"] for r in rows)),
      "custom_kernel_count": custom,
      "reduction_count": reductions,
      "epilogue_count": epilogues,
      "boundary_free_eligible": len(rows) > 0 and epilogues == len(rows) and custom == 0,
      "fusion_candidate_count": len(candidates),
      "fusion_candidate_us": round(sum(r["duration_us"] for r in candidates), 3),
      "fusion_candidate_epilogue_count": len(candidate_epilogues),
      "fusion_candidate_epilogue_us": round(sum(r["duration_us"] for r in candidate_epilogues), 3),
      "member_ids": members,
    }

  per_node = [{"id": n["id"], "name": n["name"], "population": pop, "role": role,
               "exact": exact, "duration_us": float(n["duration_us"]),
               "fusion_candidate": n["id"] in candidate_ids}
              for n, (pop, role, exact) in zip(nodes, classified)]
  total = sum(n["duration_us"] for n in nodes)
  _require(round(total, 3) == round(sum(p["total_us"] for p in populations.values()), 3),
           "population durations do not sum to the capture total")
  return {
    "schema": SCHEMA,
    "status": "PASS" if populations.get(POP_OTHER, {}).get("node_count", 0) == 0 else "PARTIAL",
    "capture": {
      "node_count": len(nodes), "edge_count": len(edges),
      "group_count": len({n["group_id"] for n in nodes}),
      "total_duration_us": round(total, 3),
      "name_digest": hashlib.sha256("\n".join(n["name"] for n in nodes).encode()).hexdigest(),
    },
    "classifier": {
      "method": "identity rules from the exact 731-node native role partition, then prefix heuristics; unknown programs are flagged not silent",
      "exact_node_count": sum(1 for _, _, exact in classified if exact),
      "heuristic_node_count": sum(1 for _, _, exact in classified if not exact),
      "unclassified_node_count": populations.get(POP_OTHER, {}).get("node_count", 0),
    },
    "populations": populations,
    "fusion_candidates": sorted([{"node": r["id"], "name": r["name"], "population": r["population"]}
                                  for rows in by_population.values() for r in rows if r["id"] in candidate_ids],
                                key=lambda r: r["node"]),
    "per_node": per_node,
  }


def _fmt(row: tuple) -> str:
  name, count, total, mean, max_us, exact, heuristic, cand, cand_us, eligible = row
  return (f"{name:26s} {count:5d} {total:10.3f} {mean:8.3f} {max_us:8.3f} "
          f"{exact:5d} {heuristic:5d} {cand:5d} {cand_us:9.3f} {eligible!s:5s}")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--dag", required=True); ap.add_argument("--out", required=True)
  args = ap.parse_args()
  result = analyze(load(args.dag))
  path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  header = (f"{'population':26s} {'n':>5s} {'total_us':>10s} {'mean_us':>8s} {'max_us':>8s} "
            f"{'exact':>5s} {'heur':>5s} {'cand':>5s} {'cand_us':>9s} {'bfree':>5s}")
  print(header)
  print("-" * len(header))
  for population in sorted(result["populations"]):
    row = result["populations"][population]
    print(_fmt((population, row["node_count"], row["total_us"], row["mean_us"], row["max_us"],
                row["exact_count"], row["heuristic_count"], row["fusion_candidate_count"],
                row["fusion_candidate_us"], row["boundary_free_eligible"])))
  print(f"\ncapture: {result['capture']['node_count']} nodes, {result['capture']['edge_count']} edges, "
        f"{result['capture']['total_duration_us']:.3f} us, groups {result['capture']['group_count']}, "
        f"status {result['status']}")


if __name__ == "__main__": main()
