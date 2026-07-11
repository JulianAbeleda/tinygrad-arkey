#!/usr/bin/env python3
"""Minimal four-cell high-base MMQ host topology interaction probe."""
from __future__ import annotations

import argparse, hashlib, json, random, statistics, time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.device import Device
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import KernelInfo, UOp

from extra.qk.mmq_bounded_harness import ACTIVATION_LAYOUT_MMQ_DS4, _finite_q4k_bytes, _q8_activation_inputs
from extra.qk.mmq_invocation_v1 import _identity
from extra.qk.mmq_q4k_q8_atom import _as_u32_words

SCHEMA = "tinygrad.mmq_invocation_v3.generated_host_topology.v1"
BASE_TARGETS = (1024, 1280)
FALSE_SITES = (0, 255)
PHASES = ("uop_construction", "schedule_creation", "warmed_compile_cache_lookup")


def generated_id(base_target:int, false_sites:int) -> str:
  if base_target not in BASE_TARGETS or false_sites not in FALSE_SITES: raise ValueError("unsupported invocation-v3 cell")
  return f"generated_noncandidate.mmq_invocation_v3.base{base_target}.false{false_sites}"


def _kernel(steps:int, base_target:int, false_sites:int) -> Callable[..., UOp]:
  name = generated_id(base_target, false_sites).replace(".", "_")
  def kernel(out:UOp, words:UOp, values:UOp, scales:UOp, sums:UOp) -> UOp:
    row, batch, lane = UOp.special(16, "gidx0"), UOp.special(16, "gidx1"), UOp.special(32, "lidx0")
    value = words[row].cast(dtypes.float32) + values[batch * 256 + lane].cast(dtypes.float32)
    value = value + scales[batch * 8].cast(dtypes.float32) + sums[batch * 8].cast(dtypes.float32)
    for step in range(steps):
      operand = words[(row + step + 1) % 288].cast(dtypes.float32)
      value = value * UOp.const(dtypes.float32, 1.000001 + (step % 17) * 1e-7) + operand
    stores = [out[batch, row].store(value, gate=lane.eq(0))]
    for site in range(false_sites):
      mi, ni = divmod(site, 16)
      # Mirrors candidate per-output (batch,row) ownership, with impossible lane=32.
      stores.append(out[mi, ni].store(value, gate=batch.eq(mi) & row.eq(ni) & lane.eq(32)))
    return UOp.group(*stores).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def _sink(steps:int, base_target:int, false_sites:int) -> UOp:
  return _kernel(steps, base_target, false_sites)(UOp.placeholder((16, 16), dtypes.float32, 0),
    UOp.placeholder((288,), dtypes.uint32, 1), UOp.placeholder((4096,), dtypes.int8, 2),
    UOp.placeholder((128,), dtypes.float32, 3), UOp.placeholder((128,), dtypes.float32, 4))


def _resolve_steps(target:int) -> tuple[int, int]:
  candidates = []
  for steps in range(400):
    count = len(_sink(steps, target, 0).toposort()); candidates.append((abs(count-target), steps, count))
    if count >= target: break
  _, steps, achieved = min(candidates); return steps, achieved


def _clock(fxn:Callable[[], Any]) -> tuple[Any, int]:
  start = time.perf_counter_ns(); value = fxn(); return value, time.perf_counter_ns() - start


def _summary(samples:list[int], overhead:int) -> dict[str, Any]:
  median = float(statistics.median(samples))
  return {"samples_ns": samples, "median_ns": median, "min_ns": min(samples), "max_ns": max(samples),
          "overhead_corrected_median_ns": max(0.0, median-overhead)}


def _interaction_fit(rows:list[dict[str, Any]], phase:str) -> dict[str, Any]:
  x = np.asarray([[1, row["base_achieved_uops"], row["false_sites"], row["base_achieved_uops"]*row["false_sites"]]
                  for row in rows], dtype=np.float64)
  y = np.asarray([row["phases"][phase]["overhead_corrected_median_ns"] for row in rows])
  coefficients = np.linalg.solve(x, y)
  return {"terms": ["intercept", "base_achieved_uops", "false_sites", "base_uops_x_false_sites"],
          "coefficients_ns": [float(v) for v in coefficients], "cells": 4, "degrees_of_freedom": 0,
          "saturated_design": True, "r2": 1.0,
          "interpretation_policy": "interaction contrast for these four cells only; no uncertainty or extrapolation claim"}


def run_invocation_v3(*, rounds:int=30, warmups:int=5, seed:int=20260711,
                      system_snapshot_id:str | None=None) -> dict[str, Any]:
  if rounds != 30: raise ValueError("invocation-v3 contract requires exactly 30 rounds")
  if warmups < 1: raise ValueError("invocation-v3 requires warmups >= 1")
  q4 = _finite_q4k_bytes(16, 256, seed); activation = _q8_activation_inputs(16, 256, seed+1, ACTIVATION_LAYOUT_MMQ_DS4)
  ds4 = activation.ds4_activation
  if ds4 is None: raise RuntimeError("DS4 construction failed")
  resident = {"words": Tensor(_as_u32_words(q4), dtype=dtypes.uint32, device="AMD").realize(),
              "values": Tensor(np.ascontiguousarray(ds4.values.reshape(-1)), dtype=dtypes.int8, device="AMD").realize(),
              "scales": Tensor(np.ascontiguousarray(ds4.scales.reshape(-1)), dtype=dtypes.float32, device="AMD").realize(),
              "sums": Tensor(np.ascontiguousarray(ds4.sums.reshape(-1)), dtype=dtypes.float32, device="AMD").realize()}
  resolved = {target: _resolve_steps(target) for target in BASE_TARGETS}; state = {}
  for target in BASE_TARGETS:
    steps, base_uops = resolved[target]
    for sites in FALSE_SITES:
      sink = _sink(steps, target, sites); program = to_program(sink, Device["AMD"].renderer); source = program.src[3].arg
      state[(target, sites)] = {"steps": steps, "base_uops": base_uops, "total_uops": len(sink.toposort()),
        "source_bytes": len(source.encode()), "source_lines": len(source.splitlines()), "rendered_statements": source.count(";"),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "program_key": program.key.hex()}
  if resolved[1024][1] >= resolved[1280][1]: raise RuntimeError("base axis collapsed")
  for sites in FALSE_SITES:
    if state[(1024, sites)]["source_bytes"] >= state[(1280, sites)]["source_bytes"]: raise RuntimeError("rendered base axis collapsed")

  overhead_samples = []
  for _ in range(2000): start = time.perf_counter_ns(); overhead_samples.append(time.perf_counter_ns()-start)
  overhead = int(statistics.median(overhead_samples)); samples = {cell: {phase: [] for phase in PHASES} for cell in state}
  for (target, sites), item in state.items():
    for _ in range(warmups):
      lazy = Tensor.empty(16, 16, dtype=dtypes.float32, device="AMD").custom_kernel(
        resident["words"], resident["values"], resident["scales"], resident["sums"],
        fxn=_kernel(item["steps"], target, sites))[0]
      linear, _ = lazy.linear_with_vars(); compile_linear(linear)
  order = [cell for cell in state for _ in range(rounds)]; random.Random(seed).shuffle(order)
  for target, sites in order:
    item = state[(target, sites)]
    lazy, elapsed = _clock(lambda: Tensor.empty(16, 16, dtype=dtypes.float32, device="AMD").custom_kernel(
      resident["words"], resident["values"], resident["scales"], resident["sums"],
      fxn=_kernel(item["steps"], target, sites))[0])
    samples[(target, sites)]["uop_construction"].append(elapsed)
    scheduled, elapsed = _clock(lambda: lazy.linear_with_vars()); samples[(target, sites)]["schedule_creation"].append(elapsed)
    linear, _ = scheduled
    _, elapsed = _clock(lambda: compile_linear(linear)); samples[(target, sites)]["warmed_compile_cache_lookup"].append(elapsed)
  rows = []
  for target in BASE_TARGETS:
    for sites in FALSE_SITES:
      item = state[(target, sites)]
      rows.append({"generated_id": generated_id(target, sites), "candidate_id": None, "base_target_uops": target,
        "base_steps": item["steps"], "base_achieved_uops": item["base_uops"], "false_sites": sites,
        "topology": {"real_owner_stores": 1, "false_owner_stores": sites,
                     "false_gate": "batch==mi && row==ni && lane==32", "dynamic_false": True},
        "total_sink_uops": item["total_uops"],
        "rendered_source": {key:item[key] for key in ("source_bytes", "source_lines", "rendered_statements", "source_sha256", "program_key")},
        "phases": {phase:_summary(samples[(target, sites)][phase], overhead) for phase in PHASES}})
  return {"schema": SCHEMA, "provenance_class": "generated_host_microbenchmark", **_identity(system_snapshot_id),
    "shape": {"M":16,"N":16,"K":256}, "candidate_ids": [], "candidate_binaries": [], "candidate_timings": [],
    "cells": 4, "base_targets_uops": list(BASE_TARGETS), "false_site_points": list(FALSE_SITES),
    "resident_buffer_policy": "all phases reuse pre-realized AMD inputs", "instrumentation_overhead": _summary(overhead_samples, 0),
    "protocol": {"rounds":rounds,"warmups":warmups,"seed":seed,"randomized_interleaved_order":[list(x) for x in order],
                 "device_time_policy":"host-only: no device launch in measured phases; device timings excluded and absent"},
    "rows":rows,"topology_interaction_fits":{phase:_interaction_fit(rows,phase) for phase in PHASES},
    "production_dispatch_changed":False}


def main() -> None:
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
  parser.add_argument("--rounds",type=int,default=30);parser.add_argument("--warmups",type=int,default=5)
  parser.add_argument("--seed",type=int,default=20260711);parser.add_argument("--system-snapshot-id")
  args=parser.parse_args();result=run_invocation_v3(rounds=args.rounds,warmups=args.warmups,seed=args.seed,system_snapshot_id=args.system_snapshot_id)
  args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


if __name__ == "__main__": main()
