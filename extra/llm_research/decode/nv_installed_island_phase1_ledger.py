#!/usr/bin/env python3
"""Phase 1 profiler-tax ledger (measurement tooling only).

Consumes the HCQ graph-profile JSONL emitted under PROFILE=1 and computes, for
each complete decode-token replay, the node_sum / interval-union / overlap /
span identities in the profiled domain. A decode replay is the repeating
[32, 64, 128, 256, 116] node-group sequence (596 nodes), matching the locked
role census cardinality.

This is measurement tooling only; it changes no production file.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter


DECODE_GROUP_SIZES = (32, 64, 128, 256, 116)


def _union_us(entries: list[dict]) -> float:
  ivs = sorted((float(e["start"]), float(e["end"])) for e in entries)
  if not ivs:
    return 0.0
  total = 0.0
  cur_s, cur_e = ivs[0]
  for s, e in ivs[1:]:
    if s <= cur_e:
      cur_e = max(cur_e, e)
    else:
      total += cur_e - cur_s
      cur_s, cur_e = s, e
  total += cur_e - cur_s
  return total


def _replay_metrics(entries: list[dict]) -> dict:
  node_sum = sum(float(e["duration"]) for e in entries)
  union = _union_us(entries)
  span = max(float(e["end"]) for e in entries) - min(float(e["start"]) for e in entries)
  return {
    "node_count": len(entries),
    "node_sum_us": round(node_sum, 3),
    "union_us": round(union, 3),
    "overlap_us": round(node_sum - union, 3),
    "span_us": round(span, 3),
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--profile-jsonl", required=True)
  ap.add_argument("--warmup", type=int, default=3)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()

  lines = []
  with open(args.profile_jsonl, encoding="utf-8") as f:
    for line in f:
      lines.append(json.loads(line))

  sizes = [len(d.get("entries", [])) for d in lines]
  size_hist = dict(sorted(Counter(sizes).items()))

  # Group consecutive lines into complete decode replays by the repeated
  # canonical [32, 64, 128, 256, 116] group-size pattern.
  replays: list[dict] = []
  i = 0
  while i + len(DECODE_GROUP_SIZES) <= len(lines):
    window = sizes[i:i + len(DECODE_GROUP_SIZES)]
    if tuple(window) == DECODE_GROUP_SIZES:
      entries = []
      for j in range(len(DECODE_GROUP_SIZES)):
        entries.extend(lines[i + j].get("entries", []))
      replays.append(_replay_metrics(entries))
      i += len(DECODE_GROUP_SIZES)
    else:
      i += 1

  if not replays:
    raise SystemExit("no complete decode replays matched the canonical group-size pattern")

  steady = replays[args.warmup:]
  keys = ("node_count", "node_sum_us", "union_us", "overlap_us", "span_us")
  median = {k: round(statistics.median([r[k] for r in steady]), 3) for k in keys}
  result = {
    "schema": "tinygrad.nv_installed_island_phase1_ledger.v1",
    "profile_jsonl": args.profile_jsonl,
    "group_size_histogram": size_hist,
    "canonical_decode_group_sizes": list(DECODE_GROUP_SIZES),
    "canonical_decode_node_count": sum(DECODE_GROUP_SIZES),
    "replay_count": len(replays),
    "steady_replay_count": len(steady),
    "warmup_dropped": args.warmup,
    "steady_median": median,
    "all_replays": replays,
  }
  with open(args.out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
  print(json.dumps({"group_size_histogram": size_hist,
                    "replay_count": len(replays),
                    "steady_replay_count": len(steady),
                    "steady_median": median}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
