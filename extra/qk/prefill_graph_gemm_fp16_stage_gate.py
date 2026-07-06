#!/usr/bin/env python3
"""Diagnostic gate for generated fp16 WMMA single-operand LOCAL staging.

This is the fp16 counterpart to the iu8 shaped-WMMA substrate probe. It proves that
`Ops.STAGE` with `BufferizeOpts(..., AddrSpace.LOCAL, removable=False)` can preserve a
single fp16 WMMA operand layout in a tiny generated kernel. It is still not a route-bound
8B prefill graph-GEMM performance gate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from extra.qk.timing_harness import add_clock_pin_arg, set_clock_pin_env

SCHEMA = "prefill-graph-gemm-fp16-single-operand-stage-gate.v1"
HANDOFF = Path("docs/HANDOFF-routeB-lds-codegen-20260706.md")

_PROBE = r'''
import json
import os
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.schedule.indexing import BufferizeOpts
from tinygrad.schedule.wmma import shaped_wmma
from tinygrad.uop.ops import UOp, KernelInfo
from extra.qk.timing_harness import env_wants_clock_pin, pinned_peak_from_env


def _frags(buf, row):
  vals = [buf[(row * 16) + i] for i in range(16)]
  return vals[0].vectorize(*vals[1:])


def direct_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
  lane = UOp.special(32, "lidx0")
  row = lane & 15
  zero = UOp.const(dtypes.half, 0.0)
  acc = zero.vectorize(*([zero] * 7))
  w = shaped_wmma(_frags(a, row), _frags(b, row), acc, dims=(16, 16, 16), device="AMD", threads=32, dtype_out=dtypes.half)
  stores = [out[lane + e * 32].store(w.gep(e)) for e in range(8)]
  return UOp.group(*stores).sink(arg=KernelInfo(name="prefill_fp16_stage_probe_direct", opts_to_apply=()))


def staged_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
  lane = UOp.special(32, "lidx0")
  row = lane & 15
  stage_both = os.environ.get("PREFILL_FP16_STAGE_BOTH") == "1"
  avec = _frags(a, row)
  bvec = _frags(b, row)
  astaged = avec.bufferize(lane, arg=BufferizeOpts(None, AddrSpace.LOCAL, removable=False)).index(lane) if stage_both else avec
  bstaged = bvec.bufferize(lane, arg=BufferizeOpts(None, AddrSpace.LOCAL, removable=False)).index(lane)
  zero = UOp.const(dtypes.half, 0.0)
  acc = zero.vectorize(*([zero] * 7))
  w = shaped_wmma(astaged, bstaged, acc, dims=(16, 16, 16), device="AMD", threads=32, dtype_out=dtypes.half)
  stores = [out[lane + e * 32].store(w.gep(e)) for e in range(8)]
  return UOp.group(*stores).sink(arg=KernelInfo(name="prefill_fp16_stage_probe_staged", opts_to_apply=()))


rng = np.random.default_rng(20260706)
a = rng.normal(size=(256,)).astype(np.float16)
b = rng.normal(size=(256,)).astype(np.float16)
ta, tb = Tensor(a), Tensor(b)
with pinned_peak_from_env() as pin_prov:
  direct = Tensor.empty(256, dtype=dtypes.half).custom_kernel(ta, tb, fxn=direct_kernel)[0].realize().numpy()
  staged = Tensor.empty(256, dtype=dtypes.half).custom_kernel(ta, tb, fxn=staged_kernel)[0].realize().numpy()
print("PROBE_RESULT " + json.dumps({
  "pin_clock": env_wants_clock_pin(),
  "clock_pin": pin_prov,
  "output_match": bool(np.array_equal(direct, staged)),
  "max_abs": float(np.max(np.abs(direct.astype(np.float32) - staged.astype(np.float32)))),
  "direct_head": direct[:8].astype(float).tolist(),
  "staged_head": staged[:8].astype(float).tolist(),
}))
'''


def _run_amd_probe(*, both_operands: bool = False, pin_clock: bool = False) -> dict[str, Any]:
  env = {**os.environ, "DEV": "AMD", "DEBUG": "4", "PYTHONPATH": "."}
  if both_operands: env["PREFILL_FP16_STAGE_BOTH"] = "1"
  set_clock_pin_env(env, pin_clock)
  proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=Path.cwd(), env=env, capture_output=True, text=True)
  result = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-12000:], "stderr_tail": proc.stderr[-4000:]}
  for line in proc.stdout.splitlines():
    if line.startswith("PROBE_RESULT "):
      result.update(json.loads(line[len("PROBE_RESULT "):]))
      break
  src = proc.stdout
  result.update({
    "has_shared_local": "__attribute__((shared" in src,
    "shared_local_count": src.count("__attribute__((shared"),
    "has_barrier": "s_barrier" in src,
    "barrier_count": src.count("s_barrier"),
    "has_fp16_wmma": "wmma_f16_16x16x16_f16" in src or "WMMA_16_16_16_half" in src,
    "has_raw_ops_ins_marker": "Ops.INS" in src or "extra/qk/prefill/wmma.py" in src,
  })
  return result


def build_report(*, run_amd: bool = False, both_operands: bool = False, pin_clock: bool = False) -> dict[str, Any]:
  from tinygrad.dtype import AddrSpace, dtypes
  from tinygrad.schedule.indexing import BufferizeOpts
  from tinygrad.schedule import rangeify
  from tinygrad.schedule.wmma import shaped_wmma
  from tinygrad.uop import Ops
  from tinygrad.uop.ops import UOp

  opts = BufferizeOpts(None, AddrSpace.LOCAL, removable=False)
  stage = UOp.const(dtypes.half, 1.0).bufferize(UOp.range(32, 0), arg=opts)
  stage_api_ok = stage.op is Ops.STAGE and stage.arg.addrspace is AddrSpace.LOCAL and stage.arg.removable is False
  has_local_lowerer = hasattr(rangeify, "pm_add_buffers_local")
  has_shaped_wmma = callable(shaped_wmma)
  handoff_exists = HANDOFF.exists()

  probe = _run_amd_probe(both_operands=both_operands, pin_clock=pin_clock) if run_amd else {"skipped": "pass --run-amd to execute the tiny generated fp16 staging probe"}
  emitted_local_evidence = bool(probe.get("has_shared_local") and probe.get("has_barrier") and probe.get("has_fp16_wmma"))
  staged_operand_count_ok = bool(probe.get("shared_local_count", 0) >= (2 if both_operands else 1))
  custom_probe_raw_markers_excluded = bool(run_amd and not probe.get("has_raw_ops_ins_marker") and probe.get("output_match"))
  passed = stage_api_ok and has_local_lowerer and has_shaped_wmma and emitted_local_evidence and staged_operand_count_ok and custom_probe_raw_markers_excluded

  pass_verdict = "PREFILL_GRAPH_GEMM_FP16_BOTH_OPERANDS_STAGE_PROBE_PASS" if both_operands \
    else "PREFILL_GRAPH_GEMM_FP16_SINGLE_OPERAND_STAGE_PROBE_PASS"
  blocked_verdict = "PREFILL_GRAPH_GEMM_FP16_BOTH_OPERANDS_STAGE_BLOCKED_IMPLEMENTATION_MISSING" if both_operands \
    else "PREFILL_GRAPH_GEMM_FP16_SINGLE_OPERAND_STAGE_BLOCKED_IMPLEMENTATION_MISSING"
  blocker = "generated fp16 both-operand WMMA LOCAL staging probe not implemented or not run" if both_operands \
    else "generated fp16 single-operand WMMA LOCAL staging probe not implemented or not run"

  return {
    "schema": SCHEMA,
    "route_id": "generated_fp16_shaped_wmma_local_stage_probe",
    "target": "target_1_8b_fp16_graph_gemm_recovery_substrate_probe",
    "operand_mode": "both_operands" if both_operands else "single_operand",
    "verdict": pass_verdict if passed else blocked_verdict,
    "api": {
      "ops_stage_available": stage_api_ok,
      "bufferize_opts_local_removable_false": stage_api_ok,
      "pm_add_buffers_local_available": has_local_lowerer,
      "shaped_wmma_helper_available": has_shaped_wmma,
      "handoff_exists": handoff_exists,
    },
    "required_evidence": {
      "emitted_amd_source_has_shared_local": emitted_local_evidence,
      "emitted_amd_source_has_expected_local_buffer_count": staged_operand_count_ok,
      "emitted_amd_source_has_s_barrier": emitted_local_evidence,
      "emitted_amd_source_has_fp16_wmma": emitted_local_evidence,
      "custom_probe_has_no_raw_ops_ins_marker": custom_probe_raw_markers_excluded,
      "pin_clock": pin_clock,
    },
    "probe": probe,
    "remaining_blocker": None if passed else blocker,
  }


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  ap.add_argument("--run-amd", action="store_true", help="execute the tiny AMD fp16 shaped-WMMA LOCAL-stage probe")
  ap.add_argument("--both-operands", action="store_true", help="stage both A and B operands in the tiny fp16 probe")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)
  report = build_report(run_amd=args.run_amd, both_operands=args.both_operands, pin_clock=args.pin_clock)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
