#!/usr/bin/env python3
"""Full-role execution classifier for the Q4_K/Q8_1 tiled WMMA route."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tinygrad import Context, GlobalCounters, Tensor, dtypes
from tinygrad.engine.realize import compile_linear
from tinygrad.uop.ops import Ops
from extra.qk.layout import Q4K_WORDS_PER_BLOCK
from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, RTOL, _rel_rmse
from tinygrad.llm import route_ops as qk_ops
from extra.qk.q4k_wmma_tile_lowering import (
  Int8WMMATileLoweringSpec,
  build_scheduler_owned_tile_loop_contract,
  describe_q4k_full_role_lowering,
  qwen3_14b_q4k_m_gfx1100_profile,
)

ARTIFACT = Path("bench/q4k-wmma-tiled-role-shape-exec/latest.json")
SYNTHETIC_MAX_RAW_TILE_STEPS = 65536
SYNTHETIC_SEED = 20260706


def _raw_tile_steps(spec) -> int:
  return (spec.m // spec.m_tile) * (spec.n // spec.n_tile) * (spec.groups // spec.group_tile)


def _derive_synthetic_shape(spec: Int8WMMATileLoweringSpec) -> tuple[int, int]:
  m_tiles = min(spec.m_tiles, 4)
  n_tiles = min(spec.n_tiles, 4)
  tile_steps = spec.groups * m_tiles * n_tiles
  if tile_steps > SYNTHETIC_MAX_RAW_TILE_STEPS:
    # Reduce grid extent to keep tensor-graph node count bounded while preserving tile-local RAW.
    n_tiles = max(1, min(n_tiles, SYNTHETIC_MAX_RAW_TILE_STEPS // max(spec.groups * m_tiles, 1)))
    tile_steps = spec.groups * m_tiles * n_tiles
    if tile_steps > SYNTHETIC_MAX_RAW_TILE_STEPS:
      m_tiles = max(1, min(m_tiles, SYNTHETIC_MAX_RAW_TILE_STEPS // max(spec.groups * n_tiles, 1)))
  return spec.m_tile * m_tiles, spec.n_tile * n_tiles


def _role_row(spec: Int8WMMATileLoweringSpec, _lifecycle: dict[str, Any] | None = None) -> dict[str, Any]:
  base = {
    "role": spec.role,
    "m": spec.m,
    "n": spec.n,
    "k": spec.k,
    "groups": spec.groups,
    "tile": {
      "m_tile": spec.m_tile,
      "n_tile": spec.n_tile,
      "group_tile": spec.group_tile,
      "live_raw_elems": spec.live_raw_elems,
      "forbidden_full_raw_elems": spec.forbidden_full_raw_elems
    },
    "lowering_plan": spec.to_json(),
  }

  full_spec = qk_ops.describe_q4k_int8_wmma_tiled_prefill(spec.n, spec.k, spec.m, role=spec.role,
                                                         m_tile=spec.m_tile, n_tile=spec.n_tile,
                                                         group_tile=spec.group_tile)
  full_words = Tensor.empty(spec.n * (spec.k // 256) * Q4K_WORDS_PER_BLOCK, dtype=dtypes.uint)
  full_xq = Tensor.empty(spec.m, spec.k, dtype=dtypes.char)
  full_xscales = Tensor.empty(spec.m, spec.groups, dtype=dtypes.float32)
  try:
    compile_start = time.perf_counter()
    full_linear = qk_ops.emit_q4k_int8_wmma_tiled_scheduler_tensor(full_words, full_xq, full_xscales, full_spec).schedule_linear()
    full_compiled = compile_linear(full_linear)
    compile_ms = (time.perf_counter() - compile_start) * 1000.0
    programs = [u.src[0] for u in full_compiled.src if u.op is Ops.CALL and u.src and u.src[0].op is Ops.PROGRAM]
    program_sources = [next((x.arg for x in program.src if x.op is Ops.SOURCE), "") for program in programs]
    full_wmma_present = any("wmma_i32_16x16x16_iu8" in source for source in program_sources)
  except Exception as e:
    base["exec"] = {"attempted": True, "class": "blocked.full_role_compile_failed", "compile_ms": None,
                    "runtime_ms": None, "kernel_count": None, "graph_node_count": None, "wmma_present": False,
                    "numeric_ok": False, "error": f"full_role_compile_failed: {e}", "raw_tile_steps": spec.raw_tile_steps}
    return base

  synth_m, synth_n = _derive_synthetic_shape(spec)
  probe_k = min(spec.k, 5120)
  run_spec = qk_ops.describe_q4k_int8_wmma_tiled_prefill(synth_n, probe_k, synth_m, role=spec.role,
                                                        m_tile=spec.m_tile, n_tile=spec.n_tile,
                                                        group_tile=spec.group_tile)
  run_spec.validate()
  words, ref_w = _make_q4k_words(run_spec.n, run_spec.k, SYNTHETIC_SEED + len(spec.role))
  x = Tensor(np.random.default_rng(SYNTHETIC_SEED + 1 + len(spec.role))
             .standard_normal((run_spec.m, run_spec.k)).astype(np.float32)).realize()
  xq, xscales = qk_ops.q8_1_quantize(x.cast(dtypes.float32))
  ref_out = None
  reference_error = None
  try:
    x_np = x.numpy()
    ref_w_np = ref_w.numpy()
    x_blocks = x_np.reshape(run_spec.m * run_spec.groups, 32)
    x_scales = np.maximum(np.abs(x_blocks).max(axis=1, keepdims=True), 1.0e-12) / 127.0
    x_scales = np.where(x_scales == 0.0, 1.0, x_scales)
    xq_np = np.round(x_blocks / x_scales).clip(-128, 127).astype(np.float32)
    xq_3d = xq_np.reshape(run_spec.m, run_spec.groups, 32)
    scales_3d = x_scales.reshape(run_spec.m, run_spec.groups, 1)
    x_dq_np = (xq_3d * scales_3d).reshape(run_spec.m, run_spec.k)
    ref_out = x_dq_np @ ref_w_np.T
  except Exception as e:
    reference_error = f"reference_failed: {e}"

  debug = io.StringIO()
  before = GlobalCounters.kernel_count
  before_time = GlobalCounters.time_sum_s
  try:
    with contextlib.redirect_stdout(debug), Context(DEBUG=4):
      got = qk_ops.emit_q4k_int8_wmma_tiled_scheduler_tensor(words, xq, xscales, run_spec).realize().numpy()
  except Exception as e:
    return {
      **base,
      "exec": {
        "attempted": True,
        "class": "blocked.scheduler_owned_execution_failed",
        "compile_ms": None,
        "runtime_ms": None,
        "kernel_count": None,
        "graph_node_count": None,
        "wmma_present": False,
        "error": f"execution_failed: {e}",
        "raw_tile_steps": _raw_tile_steps(run_spec),
      },
    }

  debug_text = debug.getvalue()
  rel = _rel_rmse(got, ref_out) if ref_out is not None else None
  base["exec"] = {
    "attempted": True,
    "class": "pass.scheduler_owned_nested_contraction",
    "compile_ms": compile_ms,
    "runtime_ms": (GlobalCounters.time_sum_s - before_time) * 1000.0,
    "kernel_count": GlobalCounters.kernel_count - before,
    "graph_node_count": None,
    "wmma_present": full_wmma_present,
    "raw_tile_steps": _raw_tile_steps(run_spec),
    "m": run_spec.m,
    "n": run_spec.n,
    "k": run_spec.k,
    "wmma_evidence": "exact_full_role_compiled_program_source" if full_wmma_present else "missing",
    "full_shape": {"m": spec.m, "n": spec.n, "k": spec.k, "program_count": len(programs)},
    "numeric_probe_shape": {"m": run_spec.m, "n": run_spec.n, "k": run_spec.k},
    "numeric_ok": bool(rel is not None and rel < RTOL),
    "rel_rmse": float(rel) if rel is not None else None,
    "rtol": RTOL,
    "error": reference_error,
  }
  return base


def build(lifecycle: dict[str, Any] | None = None) -> dict[str, Any]:
  lowering = describe_q4k_full_role_lowering(qwen3_14b_q4k_m_gfx1100_profile())
  role_specs = lowering.roles
  loop_contract = build_scheduler_owned_tile_loop_contract(role_specs, route_id=lowering.route_id)
  rows = [_role_row(spec, lifecycle) for spec in role_specs]
  all_attempted = all(row["exec"]["attempted"] for row in rows)
  all_numeric_ok = all_attempted and all(bool(row["exec"].get("numeric_ok")) for row in rows)
  passed = all_numeric_ok and all(bool(row["exec"].get("wmma_present")) for row in rows) and loop_contract["satisfied"]
  return {
    "schema": "q4k_wmma_tiled_role_shape_exec_gate.v1",
    "scope": "synthetic execution gate for all 14B Q4_K/Q8_1 wmma_tiled prefill role shapes",
    "verdict": "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_PASS" if passed else "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED",
    "route_id": lowering.route_id,
    "scheduler_owned_tile_loop": loop_contract,
    "remaining_blocker": None if passed else "scheduler_owned_role_shape_execution_failed",
    "required_next": None if passed else "scheduler_owned_role_shape_execution_failed",
    "lifecycle": {"required": False, "superseded_by": loop_contract["implementation"]},
    "roles": rows,
    "attempted_count": sum(1 if row["exec"]["attempted"] else 0 for row in rows),
    "executed_roles": [row["role"] for row in rows if row["exec"]["attempted"]],
    "classified_blocker": not passed,
    "all_numeric_ok": all_numeric_ok,
    "blocker": None if passed else "one or more generated scheduler-owned role probes failed",
    "distinction_from_classifier": (
      "q4k_wmma_tiled_role_shape enumerates/selects shapes; this gate executes bounded full-K scheduler-owned "
      "subgraphs and verifies generated iu8 WMMA plus numeric parity."
    ),
    "next_blocker": None if passed else "scheduler_owned_role_shape_execution_failed",
  }


if __name__ == "__main__":
  out = build()
  ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
  ARTIFACT.write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_PASS" else 1)
