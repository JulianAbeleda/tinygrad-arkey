#!/usr/bin/env python3
"""Host-side calibration for tinygrad custom-kernel invocation construction costs."""
from __future__ import annotations

import argparse, hashlib, json, platform, random, statistics, subprocess, sys, time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import KernelInfo, UOp

SCHEMA = "tinygrad.mmq_host_invocation_calibration.v1"
FALSE_SITE_POINTS = (0, 64, 128, 256)
PHASES = ("uop_construction", "schedule_creation", "compile_cache_lookup", "output_readback")


def _kernel(false_sites:int) -> Callable[..., UOp]:
  """Structurally match the residual probe without launching its kernel."""
  def kernel(out:UOp, inp:UOp) -> UOp:
    gid, lane = UOp.special(257, "gidx0"), UOp.special(64, "lidx0")
    value = inp[lane]
    stores = [out[0].store(value)]
    for site in range(false_sites):
      stores.append(out[site + 1].store(value, gate=gid >= (256 + site)))
    return UOp.group(*stores).sink(arg=KernelInfo(name=f"mmq_host_false_{false_sites}", opts_to_apply=()))
  return kernel


def _sink(false_sites:int) -> UOp:
  return _kernel(false_sites)(UOp.placeholder((false_sites + 1,), dtypes.float32, 0),
                              UOp.placeholder((64,), dtypes.float32, 1))


def _uop_count(root:UOp) -> int:
  return len(root.toposort())


def _clock_ns(fxn:Callable[[], Any]) -> tuple[int, Any]:
  start = time.perf_counter_ns()
  result = fxn()
  return time.perf_counter_ns() - start, result


def _empty_clock_ns() -> int:
  start = time.perf_counter_ns()
  return time.perf_counter_ns() - start


def _summary(samples:list[int], overhead_ns:float) -> dict[str, Any]:
  median = float(statistics.median(samples))
  return {"samples_ns": samples, "median_ns": median, "min_ns": min(samples), "max_ns": max(samples),
          "corrected_median_ns": max(0.0, median - overhead_ns)}


def _fit(rows:list[dict[str, Any]], phase:str) -> dict[str, Any]:
  x = np.asarray([[1.0, float(row["false_sites"])] for row in rows])
  y = np.asarray([row["phases"][phase]["corrected_median_ns"] for row in rows])
  coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
  prediction = x @ coefficients
  denominator = float(np.sum((y - y.mean()) ** 2))
  r2 = 1.0 if denominator == 0.0 else 1.0 - float(np.sum((y - prediction) ** 2)) / denominator
  return {"relationship": "intercept_ns + per_false_site_ns * false_sites",
          "intercept_ns": float(coefficients[0]), "per_false_site_ns": float(coefficients[1]), "r2": r2}


def _git_revision() -> str | None:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
  except (OSError, subprocess.CalledProcessError): return None


def _system_identity(device:str, supplied_snapshot_id:str | None) -> dict[str, Any]:
  dev = Device[device]
  facts = {"hostname": platform.node(), "platform": platform.platform(), "machine": platform.machine(),
           "python": platform.python_version(), "device": device, "device_class": type(dev).__name__,
           "renderer_class": type(dev.renderer).__name__, "tinygrad_git_revision": _git_revision()}
  digest = hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest()
  return {"system_snapshot_id": supplied_snapshot_id or f"sha256:{digest}",
          "system_snapshot_source": "supplied" if supplied_snapshot_id else "host_facts", "host_facts": facts}


def _static_metrics(false_sites:int, device:str) -> dict[str, Any]:
  sink = _sink(false_sites)
  program = to_program(sink, Device[device].renderer)
  source = program.src[3].arg
  return {"sink_uops": _uop_count(sink), "source_bytes": len(source.encode()),
          "source_lines": len(source.splitlines()), "source_nonempty_lines": sum(bool(x.strip()) for x in source.splitlines()),
          "rendered_statement_count": source.count(";"),
          "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "program_key": program.key.hex()}


def run_host_invocation_calibration(*, device:str="AMD", warmups:int=5, rounds:int=30, seed:int=20260711,
                                    system_snapshot_id:str | None=None) -> dict[str, Any]:
  if warmups < 1: raise ValueError("warmups must be >= 1")
  if rounds < 30: raise ValueError("rounds must be >= 30")

  resident_input = Tensor.empty(64, dtype=dtypes.float32, device=device).realize()
  resident_outputs = {sites: Tensor.empty(sites + 1, dtype=dtypes.float32, device=device).realize()
                      for sites in FALSE_SITE_POINTS}

  # Prime renderer/program caches with the same structures used by measured cache-hit lookups.
  for sites in FALSE_SITE_POINTS:
    for _ in range(warmups):
      out = Tensor.empty(sites + 1, dtype=dtypes.float32, device=device).custom_kernel(resident_input, fxn=_kernel(sites))[0]
      linear, _ = out.linear_with_vars()
      compile_linear(linear)
      resident_outputs[sites].numpy()

  samples = {sites: {phase: [] for phase in PHASES} for sites in FALSE_SITE_POINTS}
  overhead_samples:list[int] = []
  randomized_order:list[list[int]] = []
  rng = random.Random(seed)
  for _ in range(rounds):
    order = list(FALSE_SITE_POINTS)
    rng.shuffle(order)
    randomized_order.append(order)
    for sites in order:
      overhead_samples.append(_empty_clock_ns())
      elapsed, out = _clock_ns(lambda: Tensor.empty(sites + 1, dtype=dtypes.float32, device=device).custom_kernel(
        resident_input, fxn=_kernel(sites))[0])
      samples[sites]["uop_construction"].append(elapsed)
      elapsed, scheduled = _clock_ns(lambda: out.linear_with_vars())
      samples[sites]["schedule_creation"].append(elapsed)
      linear, _ = scheduled
      elapsed, _ = _clock_ns(lambda: compile_linear(linear))
      samples[sites]["compile_cache_lookup"].append(elapsed)
      elapsed, _ = _clock_ns(lambda: resident_outputs[sites].numpy())
      samples[sites]["output_readback"].append(elapsed)

  overhead_ns = float(statistics.median(overhead_samples))
  rows = []
  for sites in FALSE_SITE_POINTS:
    metrics = _static_metrics(sites, device)
    rows.append({"false_sites": sites, "static": metrics,
                 "phases": {phase: _summary(samples[sites][phase], overhead_ns) for phase in PHASES}})
  identity = _system_identity(device, system_snapshot_id)
  return {"schema": SCHEMA, "provenance_class": "generated_host_microbenchmark", **identity,
          "protocol": {"device": device, "warmups": warmups, "rounds": rounds, "seed": seed,
                       "randomized_case_order": randomized_order, "clock": "time.perf_counter_ns",
                       "phase_scope": {"uop_construction": "Tensor.empty plus custom_kernel UOp graph only",
                                       "schedule_creation": "Tensor.linear_with_vars only",
                                       "compile_cache_lookup": "compile_linear after structural cache warmup; no execution",
                                       "output_readback": "numpy copy from separately resident realized output; no kernel"}},
          "instrumentation_overhead": {"method": "back-to-back perf_counter_ns per case trial",
                                       **_summary(overhead_samples, 0.0)},
          "rows": rows, "fits": {phase: _fit(rows, phase) for phase in PHASES},
          "production_dispatch_changed": False}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("output", type=Path)
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--warmups", type=int, default=5)
  parser.add_argument("--rounds", type=int, default=30)
  parser.add_argument("--seed", type=int, default=20260711)
  parser.add_argument("--system-snapshot-id")
  args = parser.parse_args()
  result = run_host_invocation_calibration(device=args.device, warmups=args.warmups, rounds=args.rounds,
                                           seed=args.seed, system_snapshot_id=args.system_snapshot_id)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
