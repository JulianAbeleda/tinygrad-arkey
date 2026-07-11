#!/usr/bin/env python3
"""Controlled AMD differential probes for active lanes and store transactions."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

from extra.qk.mmq_amd_pmc import _decode_event, _extract_json

SCHEMA = "tinygrad.mmq_differential_probe.v1"


def summarize_store_calibration(points: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
  rows = [sample for point in points for sample in point.get("samples", []) if sample.get("status") == "live"]
  exact = bool(rows) and all(sample.get("counters", {}).get("GL2C_MC_WRREQ") == sample.get("unique_64b_lines") for sample in rows)
  return {"status": "live" if exact else "zero_suspect", "truth_status": "derived",
          "rule": "GL2C_MC_WRREQ equals unique touched 64B output lines" if exact else None,
          "supporting_samples": len(rows), "all_samples_exact": exact}


def _store_child(active_lanes: int, stride: int, waves: int) -> dict[str, Any]:
  if not 1 <= active_lanes <= 32 or stride < 1 or waves < 1: raise ValueError("invalid store probe geometry")
  from tinygrad import Tensor, dtypes
  from tinygrad.device import Compiled, Device
  from tinygrad.uop.ops import KernelInfo, UOp
  size = waves * 32 * stride
  def kernel(out: UOp) -> UOp:
    wave, lane = UOp.special(waves, "gidx0"), UOp.special(32, "lidx0")
    index = (wave * 32 + lane) * stride
    return out[index].store(lane.cast(dtypes.float32), gate=lane < active_lanes).sink(
      arg=KernelInfo(name=f"store_probe_a{active_lanes}_s{stride}_w{waves}", opts_to_apply=()))
  Compiled.profile_events.clear()
  Tensor.empty(size, dtype=dtypes.float32, device="AMD").custom_kernel(fxn=kernel)[0].realize()
  Device["AMD"].synchronize()
  events = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"]
  addresses = [(wave * 32 + lane) * stride * 4 for wave in range(waves) for lane in range(active_lanes)]
  return {"status": "live" if events else "blocked", "event_count": len(events),
          "counters": _decode_event(events[-1]) if events else {}, "active_lanes_per_wave": active_lanes,
          "stride_elements": stride, "waves": waves, "logical_lane_stores": len(addresses),
          "unique_64b_lines": len({address // 64 for address in addresses})}


def _run_store_child(active_lanes: int, stride: int, waves: int, counters: tuple[str, ...], timeout: int) -> dict[str, Any]:
  root = str(Path(__file__).resolve().parents[2])
  env = dict(os.environ, PROFILE="1", PMC="1", PMC_COUNTERS=",".join(counters), VIZ="0")
  env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
  argv = [sys.executable, str(Path(__file__).resolve()), "--store-child", str(active_lanes), str(stride), str(waves)]
  try:
    proc = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    result = _extract_json(proc.stdout) if proc.returncode == 0 else {"status": "blocked", "returncode": proc.returncode}
    result["stderr"] = proc.stderr[-4000:]
    return result
  except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
    return {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}


def run_store_transaction_calibration(*, active_lanes: Iterable[int] = (1, 2, 4, 8, 16, 32),
                                      strides: Iterable[int] = (1, 16), waves: int = 256, repetitions: int = 3,
                                      system_snapshot_id: str, timeout: int = 60) -> dict[str, Any]:
  if repetitions < 2: raise ValueError("repetitions must be >= 2")
  counters = ("GL2C_MC_WRREQ", "GL2C_EA_WRREQ_64B")
  points = []
  for stride in strides:
    for active in active_lanes:
      samples = [_run_store_child(active, stride, waves, counters, timeout) for _ in range(repetitions)]
      points.append({"active_lanes_per_wave": active, "stride_elements": stride, "waves": waves,
                     "samples": samples, "status": "live" if all(s.get("status") == "live" for s in samples) else "blocked"})
  artifact = {"schema": SCHEMA, "probe": "store_active_lane_transaction_calibration",
          "system_snapshot_id": system_snapshot_id, "collector": "tinygrad_kfd_native_pmc",
          "counter_semantics": {"GL2C_MC_WRREQ": {"truth_status": "measured_proxy", "unit": "memory_controller_write_requests"},
                                "GL2C_EA_WRREQ_64B": {"truth_status": "measured_proxy", "unit": "GL2_64B_write_requests"}},
          "repetitions": repetitions, "points": points,
          "notes": ["logical_lane_stores and unique_64b_lines are derived from probe addresses",
                    "counter requests are not physical bytes until this mapping is validated"]}
  artifact["calibration_result"] = summarize_store_calibration(points)
  return artifact


def run_controlled_probe(pair: Mapping[str, Any]) -> dict[str, Any]:
  if pair.get("kind") != "store_transaction_calibration": raise ValueError("unsupported controlled probe kind")
  return run_store_transaction_calibration(active_lanes=pair.get("active_lanes", (1, 2, 4, 8, 16, 32)),
    strides=pair.get("strides", (1, 16)), waves=pair.get("waves", 256), repetitions=pair.get("repetitions", 3),
    system_snapshot_id=pair["system_snapshot_id"])


def validate_differential_probe(artifact: Mapping[str, Any]) -> None:
  if artifact.get("schema") != SCHEMA: raise ValueError(f"schema must be {SCHEMA}")
  if not artifact.get("system_snapshot_id"): raise ValueError("system_snapshot_id is required")
  for pidx, point in enumerate(artifact.get("points", [])):
    if point.get("status") not in ("live", "blocked", "zero_suspect", "unsupported"):
      raise ValueError(f"points[{pidx}].status is invalid")


if __name__ == "__main__":
  if len(sys.argv) == 5 and sys.argv[1] == "--store-child":
    print("MMQ_PMC_JSON=" + json.dumps(_store_child(*map(int, sys.argv[2:])), sort_keys=True))
  else: print(json.dumps(run_store_transaction_calibration(system_snapshot_id="manual"), indent=2, sort_keys=True))
