#!/usr/bin/env python3
"""Build the same-clock gate/up -> down edge ledger from an HCQ profile JSONL."""
from __future__ import annotations

import argparse, hashlib, json, pathlib, re, statistics, subprocess
from collections import Counter

HASH = re.compile(r"_[0-9a-f]{40,64}$")
GATE = "q4k_gate_up_four_warp_vec_fp16_12288_4096"
DOWN_TOKEN = "4096_12288_epi_ffnresadd"


def canon(name: str) -> str: return HASH.sub("", str(name)).strip()


def complete_replays(lines: list[dict]) -> tuple[tuple[int, ...], list[list[dict]]]:
  sizes = [len(x.get("entries", [])) for x in lines]
  tails = Counter(sizes[i+3] for i in range(len(sizes)-3) if tuple(sizes[i:i+3]) == (32, 64, 128))
  if not tails: raise RuntimeError("no 32/64/128 production replay prefix")
  group = (32, 64, 128, tails.most_common(1)[0][0])
  out, i = [], 0
  while i + len(group) <= len(lines):
    if tuple(sizes[i:i+len(group)]) == group:
      out.append(lines[i:i+len(group)]); i += len(group)
    else: i += 1
  return group, out


def percentile(values: list[float], q: float) -> float:
  return sorted(values)[min(len(values)-1, int(q * len(values)))]


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--profile-jsonl", type=pathlib.Path, required=True)
  ap.add_argument("--warmup", type=int, default=3)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  lines = [json.loads(x) for x in args.profile_jsonl.read_text().splitlines() if x.strip()]
  group, replays = complete_replays(lines)
  steady = replays[args.warmup:]
  rows, token_sums, violations = [], [], []
  for replay_index, replay in enumerate(steady):
    token_waits = []
    for graph_index, graph in enumerate(replay):
      entries, deps = graph.get("entries", []), graph.get("deps", [])
      for node_index, producer in enumerate(entries):
        if canon(producer.get("name", "")) != GATE: continue
        if node_index + 1 >= len(entries):
          violations.append({"kind": "producer_at_graph_tail", "replay": replay_index,
            "graph": graph_index, "producer": node_index}); continue
        consumer = entries[node_index+1]
        consumer_name = canon(consumer.get("name", ""))
        direct_dep = node_index in deps[node_index+1]
        if DOWN_TOKEN not in consumer_name or not direct_dep:
          violations.append({"kind": "non_direct_down_consumer", "replay": replay_index,
            "graph": graph_index, "producer": node_index, "consumer": consumer_name,
            "consumer_deps": deps[node_index+1]}); continue
        wait = float(consumer["start"]) - float(producer["end"])
        token_waits.append(wait)
        rows.append({"replay": replay_index, "graph": graph_index, "producer_index": node_index,
          "consumer_index": node_index+1, "producer": GATE, "consumer": consumer_name,
          "producer_start": float(producer["start"]), "producer_end": float(producer["end"]),
          "consumer_start": float(consumer["start"]), "consumer_end": float(consumer["end"]),
          "producer_body_us": float(producer["duration"]), "edge_wait_us": wait,
          "consumer_body_us": float(consumer["duration"]), "direct_dependency": direct_dep})
    token_sums.append(sum(token_waits))

  waits = [x["edge_wait_us"] for x in rows]
  if not waits: raise RuntimeError("no steady FFN edges found")
  result = {
    "schema": "tinygrad.nv_dense_ffn_edge_ledger.v1",
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "profile_jsonl": str(args.profile_jsonl),
    "profile_sha256": hashlib.sha256(args.profile_jsonl.read_bytes()).hexdigest(),
    "group_sizes": list(group), "complete_replays": len(replays), "warmup_replays": args.warmup,
    "steady_replays": len(steady), "edge_count": len(rows), "violations": violations,
    "invariants": {"edges_per_token": len(rows) // len(steady),
      "every_token_has_36_edges": len(rows) == 36 * len(steady),
      "all_consumers_are_direct_down": not violations,
      "all_waits_nonnegative": all(x >= 0 for x in waits)},
    "edge_wait_us": {"min": min(waits), "median": statistics.median(waits),
      "mean": statistics.mean(waits), "p95": percentile(waits, 0.95), "max": max(waits)},
    "per_token_edge_wait_sum_us": {"min": min(token_sums), "median": statistics.median(token_sums),
      "mean": statistics.mean(token_sums), "max": max(token_sums)},
    "hypothesis": "H1_CLOSED_ZERO_MEASURABLE_EDGE_WAIT" if max(waits) == 0 else "H1_REQUIRES_ATTRIBUTION",
    # Retain one complete 36-layer physical census in the committed artifact;
    # aggregate statistics above cover every steady replay and the raw profile
    # hash preserves provenance without duplicating ten thousand timestamp rows.
    "rows": [{**row, "layer": layer} for layer, row in enumerate(rows[:36])],
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k:v for k,v in result.items() if k != "rows"}, indent=2, sort_keys=True))
  return 0 if not violations and len(rows) == 36 * len(steady) else 1


if __name__ == "__main__": raise SystemExit(main())
