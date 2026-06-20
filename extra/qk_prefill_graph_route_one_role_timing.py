#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import statistics
from typing import Callable


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-inmodel-integration-penalty/prefill_graph_route_one_role_timing_result.json"
AUDIT = ROOT / "bench/qk-inmodel-integration-penalty/inmodel_integration_penalty_audit_result.json"


def graph_durations_since(base: int, needle: str) -> list[float]:
  from tinygrad.device import Compiled
  out = []
  for e in [e for e in Compiled.profile_events[base:] if type(e).__name__ == "ProfileGraphEvent"]:
    sigs = [float(s) for s in e.sigs]
    for ent in e.ents:
      if needle in str(ent.name):
        out.append(sigs[ent.en_id] - sigs[ent.st_id])
  return out


def run_profile(fn: Callable, args: tuple, needle: str, reps: int = 8) -> list[float]:
  from tinygrad import Device
  from tinygrad.device import Compiled
  dev = Device["AMD"]
  for _ in range(3):
    fn(*args)
    dev.synchronize()
  ds = []
  for _ in range(reps):
    base = len(Compiled.profile_events)
    fn(*args)
    dev.synchronize()
    dev._at_profile_finalize()
    got = graph_durations_since(base, needle)
    if got: ds.append(got[-1])
  return ds


def full_speedup(share: float, component_speedup: float) -> float:
  return 1.0 / ((1.0 - share) + share / component_speedup)


def main() -> int:
  if not os.environ.get("PROFILE"):
    print("ERROR: run with PROFILE=1")
    return 2

  from tinygrad import Tensor, Device, TinyJit, dtypes
  from extra.gemm import rdna3_wmma_matmul as ref

  dev = Device["AMD"]
  m, n, k = 512, 12288, 4096
  waves_m, waves_n, wm, wn, bk, pad, dbuf, plra = 2, 2, 4, 4, 32, 16, 0, 1
  bm, bn, threads = waves_m * wm * 16, waves_n * wn * 16, waves_m * waves_n * 32
  lds_bytes = max((bk * 2 + pad) * (bm + bn) * (2 if dbuf else 1), 65536 // 8)
  insts = ref.build_gemm_lds2(m, n, k, waves_m, waves_n, wm, wn, bk, pad, dbuf, PLRA=plra)

  def route_fxn(a: Tensor, bt: Tensor, c: Tensor) -> Tensor:
    _, out = ref._run_insts_lds(insts, a, bt, c, m, n, k, "prefill_graph_route_gemm_timing", lds_bytes, bm, bn, threads)
    return out.realize()

  def baseline_fxn(a: Tensor, bt: Tensor) -> Tensor:
    return (a @ bt.transpose()).realize()

  route = TinyJit(route_fxn)
  baseline = TinyJit(baseline_fxn)
  Tensor.manual_seed(17)
  a = (Tensor.randn(m, k, dtype=dtypes.half) * 0.05).contiguous().realize()
  bt = (Tensor.randn(n, k, dtype=dtypes.half) * 0.05).contiguous().realize()
  c = Tensor.empty(m, n, dtype=dtypes.half).contiguous().realize()
  dev.synchronize()

  route_us = run_profile(route, (a, bt, c), "prefill_graph_route_gemm_timing")
  # Baseline graph node names are generated; use the only graph entry from GRAPH_ONE_KERNEL.
  baseline_us = []
  from tinygrad.device import Compiled
  for _ in range(3):
    baseline(a, bt)
    dev.synchronize()
  for _ in range(8):
    base = len(Compiled.profile_events)
    baseline(a, bt)
    dev.synchronize()
    dev._at_profile_finalize()
    evs = [e for e in Compiled.profile_events[base:] if type(e).__name__ == "ProfileGraphEvent"]
    for e in evs:
      sigs = [float(s) for s in e.sigs]
      if len(e.ents) == 1:
        ent = e.ents[0]
        baseline_us.append(sigs[ent.en_id] - sigs[ent.st_id])

  audit = json.loads(AUDIT.read_text())
  matmul_share = float(audit["amdahl"]["matmul_share_of_span"])
  route_med = statistics.median(route_us)
  base_med = statistics.median(baseline_us)
  component_speedup = base_med / route_med
  projected = full_speedup(matmul_share, component_speedup)
  gates = {
    "route_profiled": bool(route_us),
    "baseline_profiled": bool(baseline_us),
    "component_speedup_gte_1p2245": component_speedup >= 1.2245,
    "projected_full_prefill_gte_1p15": projected >= 1.15,
  }
  verdict = "PASS_PREFILL_GRAPH_ROUTE_ONE_ROLE_TIMING_MATERIAL" if all(gates.values()) else "BLOCKED_PREFILL_GRAPH_ROUTE_ONE_ROLE_TIMING_NOT_MATERIAL"
  result = {
    "date": "2026-06-20",
    "phase": "PREFILL_GRAPH_ROUTE_ONE_ROLE_TIMING",
    "schema": "prefill_graph_route_one_role_timing_v1",
    "verdict": verdict,
    "gate_pass": all(gates.values()),
    "default_behavior_changed": False,
    "performance_claim": True,
    "shape": {"M": m, "N": n, "K": k},
    "timing": {
      "route_us_samples": [round(x, 3) for x in route_us],
      "baseline_us_samples": [round(x, 3) for x in baseline_us],
      "route_us_median": round(route_med, 3),
      "baseline_us_median": round(base_med, 3),
      "component_speedup": round(component_speedup, 4),
      "projected_full_prefill_speedup": round(projected, 4),
    },
    "gates": gates,
    "decision": {
      "if_pass": "one-role graph timing is material; next run full PREFILL_V2 measurement after model route wiring",
      "if_blocked": "graph route captures and is correct, but timing does not justify model wiring",
    },
  }
  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": verdict, "timing": result["timing"], "gates": gates, "out": str(OUT.relative_to(ROOT))}, indent=2))
  return 0 if all(gates.values()) else 1


if __name__ == "__main__":
  raise SystemExit(main())
