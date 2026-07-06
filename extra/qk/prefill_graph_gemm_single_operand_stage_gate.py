#!/usr/bin/env python3
"""Diagnostic gate for generated iu8 WMMA single-operand LOCAL staging.

This gate is intentionally conservative. It does not claim the lowering exists until a generated
path can show all three required properties:

1. scheduler/codegen-owned WMMA surface is available,
2. LOCAL STAGE/BufferizeOpts API is available,
3. emitted AMD source for the staged probe contains LDS traffic and no raw prefill `Ops.INS` path.

This is a shaped-WMMA substrate probe. It proves generated LOCAL staging can preserve one iu8
WMMA operand layout in a tiny kernel; it does not claim the 8B fp16 graph-GEMM route is recovered.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "prefill-graph-gemm-single-operand-stage-gate.v1"
HANDOFF = Path("docs/HANDOFF-routeB-lds-codegen-20260706.md")
SURFACE_ARTIFACT = Path("bench/q4k-wmma-scheduler-surface/latest.json")

_PROBE = r'''
import json
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.schedule.indexing import BufferizeOpts
from tinygrad.schedule.wmma import shaped_wmma
from tinygrad.uop.ops import UOp, KernelInfo


def _frags(buf, row):
  vals = [buf[(row * 16) + i] for i in range(16)]
  return vals[0].vectorize(*vals[1:])


def direct_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
  lane = UOp.special(32, "lidx0")
  row = lane & 15
  zero = UOp.const(dtypes.int32, 0)
  acc = zero.vectorize(*([zero] * 7))
  w = shaped_wmma(_frags(a, row), _frags(b, row), acc, dims=(16, 16, 16), device="AMD", threads=32)
  stores = [out[lane + e * 32].store(w.gep(e)) for e in range(8)]
  return UOp.group(*stores).sink(arg=KernelInfo(name="prefill_stage_probe_direct", opts_to_apply=()))


def staged_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
  lane = UOp.special(32, "lidx0")
  row = lane & 15
  bvec = _frags(b, row)
  bstaged = bvec.bufferize(lane, arg=BufferizeOpts(None, AddrSpace.LOCAL, removable=False)).index(lane)
  zero = UOp.const(dtypes.int32, 0)
  acc = zero.vectorize(*([zero] * 7))
  w = shaped_wmma(_frags(a, row), bstaged, acc, dims=(16, 16, 16), device="AMD", threads=32)
  stores = [out[lane + e * 32].store(w.gep(e)) for e in range(8)]
  return UOp.group(*stores).sink(arg=KernelInfo(name="prefill_stage_probe_staged", opts_to_apply=()))


rng = np.random.default_rng(20260706)
a = rng.integers(-8, 8, size=(256,), dtype=np.int8)
b = rng.integers(0, 8, size=(256,), dtype=np.int8)
ta, tb = Tensor(a), Tensor(b)
direct = Tensor.empty(256, dtype=dtypes.int32).custom_kernel(ta, tb, fxn=direct_kernel)[0].realize().numpy()
staged = Tensor.empty(256, dtype=dtypes.int32).custom_kernel(ta, tb, fxn=staged_kernel)[0].realize().numpy()
print("PROBE_RESULT " + json.dumps({
  "output_match": bool(np.array_equal(direct, staged)),
  "max_abs": int(np.max(np.abs(direct - staged))),
  "direct_head": direct[:8].astype(int).tolist(),
  "staged_head": staged[:8].astype(int).tolist(),
}))
'''


def _load_json(path: Path) -> dict[str, Any]:
  try:
    return json.loads(path.read_text())
  except FileNotFoundError:
    return {"missing": str(path)}


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
    "has_shared_local": "__attribute__((shared" in src,
    "has_barrier": "s_barrier" in src,
    "has_wmma": "wmma_" in src or "__WMMA_" in src,
    "has_raw_ops_ins_marker": "Ops.INS" in src or "extra/qk/prefill/wmma.py" in src,
  })
  return result


def build_report(*, run_amd: bool = False) -> dict[str, Any]:
  from tinygrad.dtype import AddrSpace, dtypes
  from tinygrad.schedule.indexing import BufferizeOpts
  from tinygrad.schedule import rangeify
  from tinygrad.schedule.wmma import shaped_wmma
  from tinygrad.uop import Ops
  from tinygrad.uop.ops import UOp

  surface = _load_json(SURFACE_ARTIFACT)
  surface_ready = surface.get("verdict") == "Q4K_WMMA_SCHEDULER_SURFACE_SHAPED_READY"
  opts = BufferizeOpts(None, AddrSpace.LOCAL, removable=False)
  stage = UOp.const(dtypes.float32, 1.0).bufferize(UOp.range(32, 0), arg=opts)
  stage_api_ok = stage.op is Ops.STAGE and stage.arg.addrspace is AddrSpace.LOCAL and stage.arg.removable is False
  has_local_lowerer = hasattr(rangeify, "pm_add_buffers_local")
  has_shaped_wmma = callable(shaped_wmma)
  handoff_exists = HANDOFF.exists()

  probe = _run_amd_probe() if run_amd else {"skipped": "pass --run-amd to execute the tiny generated staging probe"}
  emitted_local_evidence = bool(probe.get("has_shared_local") and probe.get("has_barrier") and probe.get("has_wmma"))
  custom_probe_raw_markers_excluded = bool(run_amd and not probe.get("has_raw_ops_ins_marker") and probe.get("output_match"))
  passed = surface_ready and stage_api_ok and has_local_lowerer and has_shaped_wmma and emitted_local_evidence and custom_probe_raw_markers_excluded

  return {
    "schema": SCHEMA,
    "route_id": "generated_shaped_wmma_local_stage_probe",
    "target": "generated_iu8_shaped_wmma_local_stage_substrate_probe",
    "verdict": "PREFILL_GRAPH_GEMM_SINGLE_OPERAND_STAGE_PROBE_PASS" if passed
      else "PREFILL_GRAPH_GEMM_SINGLE_OPERAND_STAGE_BLOCKED_IMPLEMENTATION_MISSING",
    "api": {
      "ops_stage_available": stage_api_ok,
      "bufferize_opts_local_removable_false": stage_api_ok,
      "pm_add_buffers_local_available": has_local_lowerer,
      "shaped_wmma_helper_available": has_shaped_wmma,
      "surface_artifact": str(SURFACE_ARTIFACT),
      "surface_ready": surface_ready,
      "handoff_exists": handoff_exists,
    },
    "required_evidence": {
      "emitted_amd_source_has_shared_local": emitted_local_evidence,
      "emitted_amd_source_has_s_barrier": emitted_local_evidence,
      "emitted_amd_source_has_wmma": emitted_local_evidence,
      "custom_probe_has_no_raw_ops_ins_marker": custom_probe_raw_markers_excluded,
    },
    "probe": probe,
    "remaining_blocker": None if passed else "generated iu8 single-operand WMMA LOCAL staging probe not implemented or not run",
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  ap.add_argument("--run-amd", action="store_true", help="execute the tiny AMD shaped-WMMA LOCAL-stage probe")
  args = ap.parse_args(argv)
  report = build_report(run_amd=args.run_amd)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
