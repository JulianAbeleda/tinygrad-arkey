#!/usr/bin/env python3
"""Tile-loop gate for generated fp16 WMMA LOCAL staging.

The medium warmstart gate showed that wrapping final WMMA operands in STAGE
captures too much loop shape and produces wrong or unbuildable kernels. This
gate proves the narrower substrate the scheduler needs next: a shaped WMMA
operand can be staged in LOCAL inside a tile loop while the STAGE index remains
tile-shaped, not whole-GEMM-shaped.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from extra.qk.timing_harness import add_clock_pin_arg, set_clock_pin_env

SCHEMA = "prefill-graph-gemm-tile-loop-stage-gate.v1"
ARTIFACT_DIR = pathlib.Path("bench/prefill-graph-gemm-tile-loop-stage")

_PROBE = r'''
import json
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.schedule.indexing import BufferizeOpts
from tinygrad.schedule.wmma import shaped_wmma
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from extra.qk.timing_harness import env_wants_clock_pin, pinned_peak_from_env

KT = 2

def _frags(buf, row, kt):
  off = kt * 256 + row * 16
  vals = [buf[off + i] for i in range(16)]
  return vals[0].vectorize(*vals[1:])

def direct_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
  lane = UOp.special(32, "lidx0")
  row = lane & 15
  kt = UOp.range(KT, 0, axis_type=AxisType.LOOP)
  zero = UOp.const(dtypes.half, 0.0)
  acc = zero.vectorize(*([zero] * 7))
  w = shaped_wmma(_frags(a, row, kt), _frags(b, row, kt), acc, dims=(16, 16, 16),
                  device="AMD", threads=32, dtype_out=dtypes.half)
  stores = [out[kt * 256 + lane + e * 32].store(w.gep(e)) for e in range(8)]
  return UOp.group(*stores).end(kt).sink(arg=KernelInfo(name="prefill_tile_loop_stage_direct", opts_to_apply=()))

def staged_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
  lane = UOp.special(32, "lidx0")
  row = lane & 15
  kt = UOp.range(KT, 0, axis_type=AxisType.LOOP)
  zero = UOp.const(dtypes.half, 0.0)
  acc = zero.vectorize(*([zero] * 7))
  bvec = _frags(b, row, kt)
  # Keep kt as an enclosing tile loop, not a STAGE index. This is the key
  # distinction from the failed medium-shape final-operand wrapping.
  bstaged = bvec.bufferize(lane, arg=BufferizeOpts(None, AddrSpace.LOCAL, removable=False)).index(lane)
  w = shaped_wmma(_frags(a, row, kt), bstaged, acc, dims=(16, 16, 16),
                  device="AMD", threads=32, dtype_out=dtypes.half)
  stores = [out[kt * 256 + lane + e * 32].store(w.gep(e)) for e in range(8)]
  return UOp.group(*stores).end(kt).sink(arg=KernelInfo(name="prefill_tile_loop_stage_staged", opts_to_apply=()))

rng = np.random.default_rng(20260707)
a_np = rng.normal(size=(KT * 256,)).astype(np.float16)
b_np = rng.normal(size=(KT * 256,)).astype(np.float16)
ta, tb = Tensor(a_np), Tensor(b_np)
with pinned_peak_from_env() as pin_prov:
  direct = Tensor.empty(KT * 256, dtype=dtypes.half).custom_kernel(ta, tb, fxn=direct_kernel)[0].realize().numpy()
  staged = Tensor.empty(KT * 256, dtype=dtypes.half).custom_kernel(ta, tb, fxn=staged_kernel)[0].realize().numpy()
print("PROBE_RESULT " + json.dumps({
  "pin_clock": env_wants_clock_pin(),
  "clock_pin": pin_prov,
  "output_match": bool(np.array_equal(direct, staged)),
  "max_abs": float(np.max(np.abs(direct.astype(np.float32) - staged.astype(np.float32)))),
  "direct_head": direct[:8].astype(float).tolist(),
  "staged_head": staged[:8].astype(float).tolist(),
}))
'''


def _local_buffer_sizes(src: str) -> list[int]:
  return [int(x) for x in re.findall(r"__attribute__\(\(shared.*?\)\)\s*half\s+\w+\[(\d+)\]", src, re.S)]


def _run_amd_probe(*, pin_clock: bool = False) -> dict[str, Any]:
  env = {**os.environ, "DEV": "AMD", "DEBUG": "4", "PYTHONPATH": "."}
  set_clock_pin_env(env, pin_clock)
  proc = subprocess.run([sys.executable, "-c", _PROBE], cwd=Path.cwd(), env=env, capture_output=True, text=True)
  result = {"returncode": proc.returncode, "stdout_tail": proc.stdout[-12000:], "stderr_tail": proc.stderr[-4000:]}
  for line in proc.stdout.splitlines():
    if line.startswith("PROBE_RESULT "):
      result.update(json.loads(line[len("PROBE_RESULT "):]))
      break
  sizes = _local_buffer_sizes(proc.stdout)
  result.update({
    "has_shared_local": "__attribute__((shared" in proc.stdout,
    "shared_local_sizes": sizes,
    "max_shared_local_elems": max(sizes, default=0),
    "has_tile_sized_local": any(0 < s <= 1024 for s in sizes),
    "has_whole_gemm_sized_local": any(s >= 65536 for s in sizes),
    "has_barrier": "s_barrier" in proc.stdout,
    "has_fp16_wmma": "wmma_f16_16x16x16_f16" in proc.stdout or "WMMA_16_16_16_half" in proc.stdout,
    "has_tile_loop": "for (int gidx0 = 0; gidx0 < 2; gidx0++)" in proc.stdout,
    "has_raw_ops_ins_marker": "Ops.INS" in proc.stdout or "extra/qk/prefill/wmma.py" in proc.stdout,
  })
  return result


def build_report(*, run_amd: bool = False, pin_clock: bool = False, artifact: bool = True) -> dict[str, Any]:
  from tinygrad.dtype import AddrSpace, dtypes
  from tinygrad.schedule.indexing import BufferizeOpts
  from tinygrad.schedule import rangeify
  from tinygrad.schedule.wmma import shaped_wmma
  from tinygrad.uop import Ops
  from tinygrad.uop.ops import AxisType, UOp

  lane = UOp.range(32, 0)
  kt = UOp.range(2, 1, axis_type=AxisType.LOOP)
  stage = (UOp.const(dtypes.half, 1.0) + kt.cast(dtypes.half)).bufferize(
    lane, arg=BufferizeOpts(None, AddrSpace.LOCAL, removable=False))
  stage_api_ok = stage.op is Ops.STAGE and stage.arg.addrspace is AddrSpace.LOCAL and stage.arg.removable is False
  stage_shape_is_lane_only = len(stage.src) == 2 and stage.src[1] is lane
  has_local_lowerer = hasattr(rangeify, "pm_add_buffers_local")
  has_shaped_wmma = callable(shaped_wmma)
  probe = _run_amd_probe(pin_clock=pin_clock) if run_amd else {
    "skipped": "pass --run-amd to execute the tile-loop generated fp16 staging probe"}

  emitted_ok = bool(probe.get("has_shared_local") and probe.get("has_barrier") and probe.get("has_fp16_wmma") and
                    probe.get("has_tile_loop") and probe.get("has_tile_sized_local") and
                    not probe.get("has_whole_gemm_sized_local"))
  raw_excluded = bool(run_amd and not probe.get("has_raw_ops_ins_marker"))
  numeric_ok = bool(run_amd and probe.get("returncode") == 0 and probe.get("output_match") and probe.get("max_abs", 1.0) == 0.0)
  passed = stage_api_ok and stage_shape_is_lane_only and has_local_lowerer and has_shaped_wmma and emitted_ok and raw_excluded and numeric_ok

  report = {
    "schema": SCHEMA,
    "route_id": "generated_fp16_shaped_wmma_tile_loop_local_stage_probe",
    "target": "target_1_8b_fp16_graph_gemm_tile_loop_stage_substrate",
    "verdict": "PREFILL_GRAPH_GEMM_TILE_LOOP_LOCAL_STAGE_PASS" if passed
      else "PREFILL_GRAPH_GEMM_TILE_LOOP_LOCAL_STAGE_BLOCKED",
    "api": {
      "ops_stage_available": stage_api_ok,
      "bufferize_opts_local_removable_false": stage_api_ok,
      "stage_shape_is_lane_only": stage_shape_is_lane_only,
      "pm_add_buffers_local_available": has_local_lowerer,
      "shaped_wmma_helper_available": has_shaped_wmma,
    },
    "required_evidence": {
      "run_amd": run_amd,
      "pin_clock": pin_clock,
      "emitted_amd_source_has_shared_local": bool(probe.get("has_shared_local")),
      "emitted_amd_source_has_s_barrier": bool(probe.get("has_barrier")),
      "emitted_amd_source_has_fp16_wmma": bool(probe.get("has_fp16_wmma")),
      "emitted_amd_source_has_tile_loop": bool(probe.get("has_tile_loop")),
      "emitted_amd_source_has_tile_sized_local": bool(probe.get("has_tile_sized_local")),
      "emitted_amd_source_avoids_whole_gemm_sized_local": bool(run_amd and not probe.get("has_whole_gemm_sized_local")),
      "custom_probe_has_no_raw_ops_ins_marker": raw_excluded,
      "custom_probe_output_matches_direct": numeric_ok,
    },
    "probe": probe,
    "remaining_blocker": None if passed else
      "generated tile-loop WMMA LOCAL staging has not been proven on AMD with bounded tile-shaped LDS",
  }
  if artifact:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "latest.json").write_text(json.dumps(report, indent=2))
  return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  ap.add_argument("--run-amd", action="store_true", help="execute the AMD tile-loop fp16 shaped-WMMA LOCAL-stage probe")
  ap.add_argument("--no-artifact", action="store_true")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)
  report = build_report(run_amd=args.run_amd, pin_clock=args.pin_clock, artifact=not args.no_artifact)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
