#!/usr/bin/env python3
"""Compile/run probe for the smallest scheduler-owned dynamic tile.

Validation-only.  This file intentionally does not import or modify emitter
implementation code beyond the public Tensor/UOp owner seam.
"""
from __future__ import annotations

import argparse
import json
import platform
import traceback
from pathlib import Path

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt.kernel_pipeline import SchedulerOutputTileLoop
from tinygrad.device import Device
from tinygrad.engine.realize import compile_linear, run_linear
from tinygrad.uop.ops import Ops, UOp
from tinygrad.helpers import GlobalCounters

from extra.qk.dynamic_tile_owner import dynamic_store, own_dynamic_tiles


def build_probe():
  output = Tensor.zeros(16, dtype=dtypes.float32, device="AMD")
  weights, activation, scales = [Tensor.zeros(16, dtype=dtypes.float32, device="AMD") for _ in range(3)]
  def body(tile):
    return dynamic_store(tile.output, tile.output_indices, tile.activation + tile.weights + tile.scales)
  sink = own_dynamic_tiles(SchedulerOutputTileLoop(1, loop_id=9771), weights, activation, scales, output,
    weight_rows=1, activation_rows=1, scale_rows=1, output_rows=1, row_width=4, body=body)
  # A void SINK is not a realization target. Attach the owned effect to the
  # actual output so Tensor's normal callify/schedule/compile path must see it.
  return output, sink


def run() -> dict:
  result = {"schema": "amd_dynamic_tile_owner_validation.v1", "target": "gfx1100",
            "hardware": platform.platform(), "tile_count": 1, "compile": "not_attempted",
            "run": "not_attempted", "classification": "not_attempted"}
  try:
    output, sink = build_probe()
    result["uops"] = len(sink.toposort())
    effected = Tensor(output.uop.after(*sink.src))
    linear = effected.schedule_linear()
    result["scheduled_kernels"] = len(linear.src)
    if not linear.src or not any(x.op is Ops.CALL for x in linear.src):
      raise RuntimeError("dynamic owner produced no scheduled CALL; refusing no-op validation")
    stores = [u for call in linear.src for u in call.src[0].toposort() if u.op is Ops.STORE]
    if not any(u.src[0].op is Ops.INDEX and u.src[1].op is not Ops.CONST for u in stores):
      raise RuntimeError("scheduled dynamic owner has no computed indexed writeback; refusing launch-only validation")
    compiled = compile_linear(linear)
    programs = [u for u in compiled.toposort() if u.op is Ops.PROGRAM]
    result["compiled_programs"] = len(programs)
    if not programs:
      raise RuntimeError("dynamic owner compiled to no PROGRAM; refusing no-op validation")
    result["kernel_names"] = [p.arg.name for p in programs]
    before_kernels = GlobalCounters.kernel_count
    run_linear(compiled, jit=True, wait=True)
    Device["AMD"].synchronize()
    result["kernel_count_delta"] = GlobalCounters.kernel_count - before_kernels
    if result["kernel_count_delta"] < 1:
      raise RuntimeError("dynamic owner execution launched no kernel; refusing no-op validation")
    result["compile"], result["run"], result["classification"] = "passed", "passed", "passed"
  except Exception as exc:
    result["exception_type"] = type(exc).__name__
    result["exact_failure"] = str(exc)
    result["traceback"] = traceback.format_exc()
    result["classification"] = ("INDEX_or_dynamic_store_unsupported" if
      any(x in str(exc) for x in ("INDEX", "dynamic store", "STORE")) else "pre_compiler_graph_failure")
  return result


if __name__ == "__main__":
  ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path)
  args = ap.parse_args(); record = run(); print(json.dumps(record, indent=2))
  if args.out: args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(record, indent=2) + "\n")
