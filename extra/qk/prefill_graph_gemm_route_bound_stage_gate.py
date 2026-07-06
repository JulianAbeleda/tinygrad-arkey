#!/usr/bin/env python3
"""Route-bound diagnostic for the strict-pure fp16 PREFILL_V2 matmul path.

This gate exercises the actual default prefill linear route, not a custom microkernel:

  route_prefill_linear(..., prefill_graph_gemm=False, prefill_chunked=False)

The expected current state is intentionally not a pass for performance recovery. The route is
strict-pure and emits fp16 WMMA, but it does not route-bind generated LOCAL operand staging yet.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "prefill-graph-gemm-route-bound-stage-gate.v1"

_PROBE = r'''
import json
import numpy as np
from types import SimpleNamespace
from tinygrad import Tensor
from tinygrad.llm.prefill_routes import route_prefill_linear

rng = np.random.default_rng(20260707)
x = Tensor(rng.normal(size=(1, 512, 512)).astype(np.float16))
w = Tensor(rng.normal(size=(512, 512)).astype(np.float16))
lin = SimpleNamespace(_pf16_w=w, bias=None, weight=w, name="route_bound_probe")
out = route_prefill_linear(lin, x, prefill_graph_gemm=False, prefill_chunked=False).realize()
arr = out.numpy()
print("PROBE_RESULT " + json.dumps({
  "shape": list(arr.shape),
  "finite": bool(np.isfinite(arr).all()),
  "head": arr.reshape(-1)[:4].astype(float).tolist(),
}))
'''


def _run_amd_probe() -> dict[str, Any]:
  env = {**os.environ, "DEV": "AMD", "DEBUG": "4", "PYTHONPATH": "."}
  proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=Path.cwd(), env=env, capture_output=True, text=True)
  result = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-12000:], "stderr_tail": proc.stderr[-4000:]}
  for line in proc.stdout.splitlines():
    if line.startswith("PROBE_RESULT "):
      result.update(json.loads(line[len("PROBE_RESULT "):]))
      break
  src = proc.stdout
  result.update({
    "has_fp16_wmma": "wmma_f16_16x16x16_f16" in src or "WMMA_16_16_16_half" in src,
    "has_shared_local": "__attribute__((shared" in src,
    "has_barrier": "s_barrier" in src,
    "has_raw_ops_ins_marker": "Ops.INS" in src or "extra/qk/prefill/wmma.py" in src,
    "kernel_name_seen": "r_" in src,
  })
  return result


def build_report(*, run_amd: bool = False) -> dict[str, Any]:
  probe = _run_amd_probe() if run_amd else {"skipped": "pass --run-amd to execute the route-bound AMD probe"}
  route_bound_ok = bool(run_amd and probe.get("returncode") == 0 and probe.get("finite") and probe.get("has_fp16_wmma"))
  raw_excluded = bool(run_amd and not probe.get("has_raw_ops_ins_marker"))
  local_stage_present = bool(run_amd and probe.get("has_shared_local") and probe.get("has_barrier"))

  verdict = "PREFILL_GRAPH_GEMM_ROUTE_BOUND_LOCAL_STAGE_PASS" if route_bound_ok and raw_excluded and local_stage_present \
    else "PREFILL_GRAPH_GEMM_ROUTE_BOUND_LOCAL_STAGE_MISSING"

  return {
    "schema": SCHEMA,
    "route_id": "prefill_v2_scheduler_matmul_default",
    "shape": {"m": 512, "n": 512, "k": 512},
    "verdict": verdict,
    "evidence": {
      "route_bound_executes": route_bound_ok,
      "route_bound_fp16_wmma": bool(probe.get("has_fp16_wmma")),
      "route_bound_no_raw_ops_ins_marker": raw_excluded,
      "route_bound_shared_local": bool(probe.get("has_shared_local")),
      "route_bound_barrier": bool(probe.get("has_barrier")),
      "route_bound_local_stage_present": local_stage_present,
    },
    "probe": probe,
    "remaining_blocker": None if verdict.endswith("_PASS") else "default fp16 prefill route emits WMMA but not generated LOCAL operand staging",
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  ap.add_argument("--run-amd", action="store_true", help="execute the route-bound AMD prefill probe")
  args = ap.parse_args(argv)
  report = build_report(run_amd=args.run_amd)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
