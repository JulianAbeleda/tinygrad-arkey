#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import statistics
import time
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_graph_node_feasibility_result.json"


def main() -> int:
  if not os.environ.get("PROFILE"):
    print("ERROR: run with PROFILE=1 so HCQGraph emits ProfileGraphEvent")
    return 2

  from tinygrad import Tensor, Device, TinyJit, dtypes
  from tinygrad.device import Compiled
  from extra.gemm import rdna3_wmma_matmul as ref

  dev = Device["AMD"]
  m, n, k = 512, 12288, 4096
  waves_m, waves_n, wm, wn, bk, pad, dbuf, plra = 2, 2, 4, 4, 32, 16, 0, 1
  bm, bn, threads = waves_m * wm * 16, waves_n * wn * 16, waves_m * waves_n * 32
  lds_bytes = max((bk * 2 + pad) * (bm + bn) * (2 if dbuf else 1), 65536 // 8)
  insts = ref.build_gemm_lds2(m, n, k, waves_m, waves_n, wm, wn, bk, pad, dbuf, PLRA=plra)

  def route_fxn(a: Tensor, bt: Tensor, c: Tensor) -> Tensor:
    _, out = ref._run_insts_lds(insts, a, bt, c, m, n, k, "prefill_graph_route_gemm", lds_bytes, bm, bn, threads)
    return out.realize()

  route = TinyJit(route_fxn)
  Tensor.manual_seed(11)
  a = (Tensor.randn(m, k, dtype=dtypes.half) * 0.05).contiguous().realize()
  bt = (Tensor.randn(n, k, dtype=dtypes.half) * 0.05).contiguous().realize()
  c = Tensor.empty(m, n, dtype=dtypes.half).contiguous().realize()
  dev.synchronize()

  # call 0 eager, call 1 capture, call 2+ graph replay
  samples = []
  for i in range(5):
    t0 = time.perf_counter()
    out = route(a, bt, c)
    dev.synchronize()
    if i >= 2:
      samples.append(time.perf_counter() - t0)
  _ = out
  base = len(Compiled.profile_events)
  out = route(a, bt, c)
  dev.synchronize()
  dev._at_profile_finalize()
  evs = [e for e in Compiled.profile_events[base:] if type(e).__name__ == "ProfileGraphEvent"]
  graph_entries: list[dict[str, Any]] = []
  for e in evs:
    sigs = [float(s) for s in e.sigs]
    for ent in e.ents:
      graph_entries.append({
        "device": ent.device,
        "name": str(ent.name),
        "dur_us": sigs[ent.en_id] - sigs[ent.st_id],
      })
  matching = [x for x in graph_entries if "prefill_graph_route_gemm" in x["name"]]

  sample = out[0, 0:16].float().numpy()
  finite_sample = bool(np.isfinite(sample).all())
  nonzero_sample = bool(np.any(sample != 0))
  wall_ms = [x * 1000.0 for x in samples]
  result = {
    "date": "2026-06-20",
    "phase": "PREFILL_GRAPH_NODE_FEASIBILITY",
    "schema": "prefill_graph_node_feasibility_v1",
    "verdict": "PASS_PREFILL_GRAPH_NODE_FEASIBILITY" if matching and finite_sample and nonzero_sample else "BLOCKED_PREFILL_GRAPH_NODE_FEASIBILITY",
    "gate_pass": bool(matching and finite_sample and nonzero_sample),
    "default_behavior_changed": False,
    "performance_claim": False,
    "shape": {"M": m, "N": n, "K": k, "layout": "A[M,K], Bt[N,K], C[M,N]"},
    "kernel": {"waves_m": waves_m, "waves_n": waves_n, "wm": wm, "wn": wn, "bk": bk, "pad": pad, "dbuf": dbuf, "plra": plra, "lds_bytes": lds_bytes},
    "graph": {
      "profile_graph_events": len(evs),
      "entries": len(graph_entries),
      "matching_entries": matching,
    },
    "timing": {
      "wall_ms_samples_replay": [round(x, 3) for x in wall_ms],
      "wall_ms_median_replay": round(statistics.median(wall_ms), 3) if wall_ms else None,
    },
    "sample": {
      "first_16": [float(x) for x in sample],
      "finite": finite_sample,
      "nonzero": nonzero_sample,
    },
    "decision": {
      "if_pass": "custom dependency-free GEMM can be an HCQGraph-captured node at the real PREFILL_V2 gate/up shape; next gate is one-role correctness against PREFILL_V2 linear",
      "if_blocked": "graph-route transfer is blocked before correctness/timing; inspect JIT graph support or custom kernel layout",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "graph": result["graph"],
    "timing": result["timing"],
    "sample": {"finite": finite_sample, "nonzero": nonzero_sample},
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
