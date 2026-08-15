#!/usr/bin/env python3
"""Diagnostic: does the decode DAG actually distribute calls across streams?

Wraps plan_multi_stream and reports, per captured graph, the call-count per
stream. This distinguishes "multi-stream ran but the DAG is chain-like" from
"multi-stream silently did nothing", before trusting a flat wall result.
"""
from __future__ import annotations

import collections, json, os

from tinygrad import Device
import tinygrad.runtime.graph.cuda as gc

_orig = gc.plan_multi_stream
_reports = []

def _wrap(n_calls, preds, costs, n_streams):
  streams = _orig(n_calls, preds, costs, n_streams)
  _reports.append({
    "n_calls": n_calls,
    "n_streams": n_streams,
    "stream_counts": collections.Counter(streams),
    "cross_edges": len(gc.cross_stream_edges(preds, streams)),
  })
  return streams

gc.plan_multi_stream = _wrap

from tinygrad.llm.model import Transformer

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"

def main():
  model, _kv = Transformer.from_gguf(MODEL, 4608)
  gen = model.generate([1] * 512, chunk_size=32, temperature=0.0)
  next(gen)
  Device[Device.DEFAULT].synchronize()
  for _ in range(3):
    next(gen)
  Device[Device.DEFAULT].synchronize()
  gen.close()
  out = {
    "n_streams": int(os.environ.get("CUDA_GRAPH_STREAMS", "1")),
    "graphs": _reports,
  }
  print(json.dumps(out, indent=1))
  with open(f"/tmp/route_b_stream_dist_s{out['n_streams']}.json", "w") as f:
    json.dump(out, f, indent=1)

if __name__ == "__main__":
  main()
