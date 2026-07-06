#!/usr/bin/env python3
"""Cooperative-partition probe for generated fp16 WMMA B-tile staging.

The tile-loop stage gate proved a generated LOCAL stage can preserve a WMMA
operand inside a tile loop, but it still stored a 512-half B fragment because
wave32 lanes duplicate the 16 useful B rows. This probe proves the next mapping
in pure generated UOps: lanes 0..15 cooperatively stage the unique 16x16 B tile
into 256 halfs, lanes 16..31 read the same row via lane&15, and WMMA output
matches the direct kernel.
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

from extra.qk.cooperative_stage_lanemap import CooperativeStageLaneMap
from extra.qk.timing_harness import add_clock_pin_arg, set_clock_pin_env

SCHEMA = "prefill-graph-gemm-coop-partition-gate.v1"
ARTIFACT_DIR = pathlib.Path("bench/prefill-graph-gemm-coop-partition")

_PROBE = r'''
import json
import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.dtype import AddrSpace
from tinygrad.schedule.wmma import shaped_wmma
from tinygrad.uop.ops import UOp, KernelInfo, AxisType
from extra.qk.timing_harness import env_wants_clock_pin, pinned_peak_from_env

KT = 2

def _frags(buf, row, kt):
  off = kt * 256 + row * 16
  vals = [buf[off + i] for i in range(16)]
  return vals[0].vectorize(*vals[1:])

def _bfrag_local(buf, row):
  vals = [buf[row * 16 + i] for i in range(16)]
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
  return UOp.group(*stores).end(kt).sink(arg=KernelInfo(name="prefill_coop_partition_direct", opts_to_apply=()))

def coop_kernel(out:UOp, a:UOp, b:UOp) -> UOp:
  lane = UOp.special(32, "lidx0")
  row = lane & 15
  kt = UOp.range(KT, 0, axis_type=AxisType.LOOP)
  bsh = UOp.placeholder((256,), dtypes.half, 777, addrspace=AddrSpace.LOCAL)
  st = UOp.range(2, 4, axis_type=AxisType.REDUCE)
  wv = UOp.range(8, 5, axis_type=AxisType.LOOP)
  # Cooperative partition: lanes 0..15 own the unique B tile rows. The upper
  # half-wave reads the same row through row=lane&15, matching RDNA3 WMMA B use.
  elem = row * 16 + st * 8 + wv
  stage = bsh[elem].store(b[kt * 256 + elem], lane < 16).end(wv).end(st)
  bar = UOp.barrier(UOp.group(stage))
  zero = UOp.const(dtypes.half, 0.0)
  acc = zero.vectorize(*([zero] * 7))
  w = shaped_wmma(_frags(a, row, kt), _bfrag_local(bsh.after(bar), row), acc, dims=(16, 16, 16),
                  device="AMD", threads=32, dtype_out=dtypes.half)
  stores = [out[kt * 256 + lane + e * 32].store(w.gep(e)) for e in range(8)]
  return UOp.group(*stores).end(kt).sink(arg=KernelInfo(name="prefill_coop_partition_probe", opts_to_apply=()))

rng = np.random.default_rng(20260707)
a_np = rng.normal(size=(KT * 256,)).astype(np.float16)
b_np = rng.normal(size=(KT * 256,)).astype(np.float16)
ta, tb = Tensor(a_np), Tensor(b_np)
with pinned_peak_from_env() as pin_prov:
  direct = Tensor.empty(KT * 256, dtype=dtypes.half).custom_kernel(ta, tb, fxn=direct_kernel)[0].realize().numpy()
  coop = Tensor.empty(KT * 256, dtype=dtypes.half).custom_kernel(ta, tb, fxn=coop_kernel)[0].realize().numpy()
print("PROBE_RESULT " + json.dumps({
  "pin_clock": env_wants_clock_pin(),
  "clock_pin": pin_prov,
  "output_match": bool(np.array_equal(direct, coop)),
  "max_abs": float(np.max(np.abs(direct.astype(np.float32) - coop.astype(np.float32)))),
  "direct_head": direct[:8].astype(float).tolist(),
  "coop_head": coop[:8].astype(float).tolist(),
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
    "has_unique_b_tile_local": 256 in sizes,
    "has_barrier": "s_barrier" in proc.stdout,
    "has_fp16_wmma": "wmma_f16_16x16x16_f16" in proc.stdout or "WMMA_16_16_16_half" in proc.stdout,
    "has_lane_guard": "lidx0<16" in proc.stdout,
    "has_tile_loop": "for (int gidx0 = 0; gidx0 < 2; gidx0++)" in proc.stdout,
    "has_raw_ops_ins_marker": "Ops.INS" in proc.stdout or "extra/qk/prefill/wmma.py" in proc.stdout,
  })
  return result


def build_report(*, run_amd: bool = False, pin_clock: bool = False, artifact: bool = True) -> dict[str, Any]:
  lanemap = CooperativeStageLaneMap(total=256, threads=16, width=8)
  try:
    lanemap.validate()
    lanemap_ok = lanemap.stages == 2
  except ValueError:
    lanemap_ok = False

  probe = _run_amd_probe(pin_clock=pin_clock) if run_amd else {
    "skipped": "pass --run-amd to execute the cooperative B-tile partition probe"}
  emitted_ok = bool(probe.get("has_shared_local") and probe.get("has_unique_b_tile_local") and
                    probe.get("has_barrier") and probe.get("has_fp16_wmma") and probe.get("has_lane_guard") and
                    probe.get("has_tile_loop"))
  raw_excluded = bool(run_amd and not probe.get("has_raw_ops_ins_marker"))
  numeric_ok = bool(run_amd and probe.get("returncode") == 0 and probe.get("output_match") and probe.get("max_abs", 1.0) == 0.0)
  passed = lanemap_ok and emitted_ok and raw_excluded and numeric_ok
  report = {
    "schema": SCHEMA,
    "route_id": "generated_fp16_shaped_wmma_coop_b_tile_partition_probe",
    "target": "target_1_8b_fp16_graph_gemm_cooperative_partition_substrate",
    "verdict": "PREFILL_GRAPH_GEMM_COOP_PARTITION_PROBE_PASS" if passed
      else "PREFILL_GRAPH_GEMM_COOP_PARTITION_PROBE_BLOCKED",
    "api": {
      "cooperative_stage_lanemap_available": True,
      "b_tile_lanemap_valid": lanemap_ok,
      "b_tile_unique_elements": 256,
      "producer_lanes": 16,
      "consumer_lanes": 32,
    },
    "required_evidence": {
      "run_amd": run_amd,
      "pin_clock": pin_clock,
      "emitted_amd_source_has_unique_256_half_local": bool(probe.get("has_unique_b_tile_local")),
      "emitted_amd_source_has_lane_guard": bool(probe.get("has_lane_guard")),
      "emitted_amd_source_has_s_barrier": bool(probe.get("has_barrier")),
      "emitted_amd_source_has_fp16_wmma": bool(probe.get("has_fp16_wmma")),
      "custom_probe_has_no_raw_ops_ins_marker": raw_excluded,
      "custom_probe_output_matches_direct": numeric_ok,
    },
    "probe": probe,
    "remaining_blocker": None if passed else
      "generated cooperative B-tile partition has not been proven on AMD",
  }
  if artifact:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "latest.json").write_text(json.dumps(report, indent=2))
  return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
  ap = argparse.ArgumentParser()
  ap.add_argument("--compact", action="store_true")
  ap.add_argument("--run-amd", action="store_true", help="execute the AMD cooperative B-tile partition probe")
  ap.add_argument("--no-artifact", action="store_true")
  add_clock_pin_arg(ap)
  args = ap.parse_args(argv)
  report = build_report(run_amd=args.run_amd, pin_clock=args.pin_clock, artifact=not args.no_artifact)
  print(json.dumps(report, indent=None if args.compact else 2))
  return report


if __name__ == "__main__":
  main()
