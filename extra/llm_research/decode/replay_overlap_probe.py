#!/usr/bin/env python3
"""Decode replay overlap probe: per-graph HCQ timestamps vs node-sums.

nsys --cuda-graph-trace=node is blocked on this host (captures carry no GPU
trace data), so this probe measures the scope's P0 question through the NV
runtime's own PROFILE timestamps: with PROFILE=1 and
HCQ_GRAPH_PROFILE_JSON set, each HCQGraph replay writes per-kernel start/end
timestamps (microseconds). For each graph group of one measured decode token,
the probe compares the group's span (max end - min start) with its node-sum
(sum of member durations). span < node-sum means the group overlaps internally;
totals vs wall give the replay-wide overlap that the P0 protocol would produce.

Usage:
  PYTHONPATH=/home/ubuntu/tinygrad-arkey python3 \
    extra/llm_research/decode/replay_overlap_probe.py --depth 512 [--out X.json]

Fused prefill attention is OFF (house convention; the NV fused prefill ABI is
deterministically broken at HEAD, b3-tuned-schedule-characterization-record).
"""
from __future__ import annotations

import argparse, json, os, pathlib, subprocess, sys, time

os.environ["PROFILE"] = "1"
os.environ["HCQ_GRAPH_PROFILE_JSON"] = "/tmp/replay_overlap_graph.jsonl"
sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

import tinygrad.llm.model as tgm
from tinygrad.device import Device
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer
from tinygrad.runtime.graph.hcq import HCQGraph

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--out", type=str, default=None)
  args = ap.parse_args()

  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()

  launches: list[dict] = []
  capture = {"on": True}
  _orig_call = HCQGraph.__call__

  def traced_call(self, input_uops, var_vals, wait=False):
    t0 = time.perf_counter()
    ret = _orig_call(self, input_uops, var_vals, wait=wait)
    names = tuple(c[1].arg.name for c in self.calls)
    if capture["on"]:
      launches.append({"names": names, "host_us": (time.perf_counter() - t0) * 1e6})
    print(f"CALL inst={names[0][:28]} k={self.kickoff_value} members={len(names)} "
          f"host={time.perf_counter()-t0:.4f}s", file=sys.stderr, flush=True)
    return ret

  HCQGraph.__call__ = traced_call

  model, kv = Transformer.from_gguf(MODEL, 4608)
  prompt = [1] * args.depth
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  pathlib.Path(os.environ["HCQ_GRAPH_PROFILE_JSON"]).unlink(missing_ok=True)
  print("--- warmup1 (capture)", file=sys.stderr, flush=True)
  with Context(DEBUG=0):
    next(gen)  # warmup: graph capture + compile (no replay, no jsonl line)
  print("--- warmup2 (T2)", file=sys.stderr, flush=True)
  with Context(DEBUG=0):
    next(gen)  # T2: first replay, kickoff 1 (no collect)
  print("--- measured (T3)", file=sys.stderr, flush=True)
  t0 = time.perf_counter()
  with Context(DEBUG=0):
    next(gen)  # T3: measured token (wall W, no sync)
  wall = time.perf_counter() - t0
  Device["NV"].synchronize()
  wall_sync = time.perf_counter() - t0
  capture["on"] = False
  print("--- flush (T4)", file=sys.stderr, flush=True)
  with Context(DEBUG=0):
    next(gen)  # T4: flush; collects T3's timestamps into the jsonl

  lines = [json.loads(l) for l in open(os.environ["HCQ_GRAPH_PROFILE_JSON"]) if l.strip()]
  k = len(launches)
  if len(lines) != k:
    raise RuntimeError(f"profile line accounting: {len(lines)} lines vs {k} measured launches "
                       f"(expected one jsonl line per measured replay, collected at the flush replay)")
  for i, la in enumerate(launches):
    if len(lines[i]["entries"]) != len(la["names"]):
      raise RuntimeError(f"instance {i}: {len(lines[i]['entries'])} entries vs {len(la['names'])} members")

  rows = []
  node_sum_us = 0
  span_sum_us = 0
  host_us = 0
  for i, la in enumerate(launches):
    line = lines[i]  # collected at the flush replay; timestamps are the measured token's
    ents = line["entries"]
    member_us = sum(float(e["duration"]) for e in ents)
    span_us = max(float(e["end"]) for e in ents) - min(float(e["start"]) for e in ents)
    node_sum_us += member_us
    span_sum_us += span_us
    host_us += la["host_us"]
    rows.append({"group": i, "kernels": len(ents), "node_sum_us": round(member_us, 1),
                 "span_us": round(span_us, 1), "overlap_us": round(max(0.0, member_us - span_us), 1),
                 "overlap_pct": round(100 * max(0.0, member_us - span_us) / member_us, 1),
                 "host_submit_us": round(la["host_us"], 1)})

  out = {
    "depth": args.depth,
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, cwd="/home/ubuntu/tinygrad-arkey").stdout.strip(),
    "model": MODEL,
    "groups": len(rows), "kernels": sum(r["kernels"] for r in rows),
    "node_sum_us": round(node_sum_us, 1), "span_sum_us": round(span_sum_us, 1),
    "host_submit_us": round(host_us, 1),
    "wall_s_ms": round(wall * 1000, 2), "wall_sync_ms": round(wall_sync * 1000, 2),
    "replay_overlap_pct": round(100 * max(0.0, node_sum_us - span_sum_us) / node_sum_us, 2),
    "rows": rows,
  }
  print(json.dumps(out, indent=1))
  if args.out:
    with open(args.out, "w") as f:
      json.dump(out, f, indent=1)
      f.write("\n")


if __name__ == "__main__":
  main()
