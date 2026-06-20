#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_graph_route_one_role_correctness_result.json"


def main() -> int:
  if not os.environ.get("PROFILE"):
    print("ERROR: run with PROFILE=1")
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
    _, out = ref._run_insts_lds(insts, a, bt, c, m, n, k, "prefill_graph_route_gemm_correctness", lds_bytes, bm, bn, threads)
    return out.realize()

  route = TinyJit(route_fxn)
  Tensor.manual_seed(13)
  a = (Tensor.randn(m, k, dtype=dtypes.half) * 0.05).contiguous().realize()
  bt = (Tensor.randn(n, k, dtype=dtypes.half) * 0.05).contiguous().realize()
  c = Tensor.empty(m, n, dtype=dtypes.half).contiguous().realize()
  dev.synchronize()

  # Capture and replay the custom route.
  for _ in range(3):
    out = route(a, bt, c)
    dev.synchronize()

  base = len(Compiled.profile_events)
  out = route(a, bt, c)
  dev.synchronize()
  dev._at_profile_finalize()
  evs = [e for e in Compiled.profile_events[base:] if type(e).__name__ == "ProfileGraphEvent"]
  matching: list[dict[str, Any]] = []
  for e in evs:
    sigs = [float(s) for s in e.sigs]
    for ent in e.ents:
      if "prefill_graph_route_gemm_correctness" in str(ent.name):
        matching.append({"name": str(ent.name), "dur_us": sigs[ent.en_id] - sigs[ent.st_id]})

  oracle = (a.float() @ bt.float().transpose()).realize()
  dev.synchronize()
  got_np = out.float().numpy()
  ref_np = oracle.numpy()
  diff = got_np - ref_np
  rmse = float(np.sqrt(np.mean(diff * diff)))
  refn = float(np.sqrt(np.mean(ref_np * ref_np)))
  rel_rmse = rmse / (refn + 1e-9)
  max_abs = float(np.max(np.abs(diff)))
  gates = {
    "graph_captured": bool(matching),
    "rel_rmse_lte_1e_2": rel_rmse <= 1.0e-2,
    "max_abs_lte_2e_1": max_abs <= 2.0e-1,
  }
  verdict = "PASS_PREFILL_GRAPH_ROUTE_ONE_ROLE_CORRECTNESS" if all(gates.values()) else "BLOCKED_PREFILL_GRAPH_ROUTE_ONE_ROLE_CORRECTNESS"
  result = {
    "date": "2026-06-20",
    "phase": "PREFILL_GRAPH_ROUTE_ONE_ROLE_CORRECTNESS",
    "schema": "prefill_graph_route_one_role_correctness_v1",
    "verdict": verdict,
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": False,
    "shape": {"M": m, "N": n, "K": k, "layout": "A[M,K] @ Bt[N,K].T -> C[M,N]"},
    "graph_matching_entries": matching,
    "correctness": {"rel_rmse": rel_rmse, "rmse": rmse, "ref_rms": refn, "max_abs": max_abs},
    "gates": gates,
    "decision": {
      "if_pass": "graph route is feasible and numerically correct for one real PREFILL_V2 gate/up-shape matmul; next gate is one-role in-graph timing and full-bucket projection",
      "if_blocked": "fix graph capture or layout/math before timing",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "correctness": result["correctness"],
    "graph_matching_entries": matching,
    "gates": gates,
    "out": str(OUT.relative_to(ROOT)),
  }, indent=2))
  return 0 if all(gates.values()) else 1


if __name__ == "__main__":
  raise SystemExit(main())
