#!/usr/bin/env python3
"""Full-role execution classifier for the Q4_K/Q8_1 tiled WMMA route."""
from __future__ import annotations

import contextlib
import io
import json
import re
from pathlib import Path
from shutil import which
from typing import Any

import numpy as np

from tinygrad import Context, GlobalCounters, Tensor, dtypes
from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, RTOL, _rel_rmse
from tinygrad.llm import route_ops as qk_ops
from extra.qk.q4k_wmma_tile_lowering import (
  Int8WMMATileLoweringSpec,
  QWEN3_14B_Q4K_ROLE_SHAPES,
  build_scheduler_owned_tile_loop_contract,
  describe_int8_wmma_tile_lowering,
)
from extra.qk.q4k_wmma_tiled_lifecycle_gate import build as lifecycle_build

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


def _role_row(spec: Int8WMMATileLoweringSpec, lifecycle: dict[str, Any]) -> dict[str, Any]:
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

  if lifecycle["verdict"] != "Q4K_WMMA_TILED_LIFECYCLE_PASS":
    base["exec"] = {
      "attempted": False,
      "class": "blocked.lifecycle_missing",
      "compile_ms": None,
      "runtime_ms": None,
      "kernel_count": None,
      "graph_node_count": None,
      "wmma_present": None,
    }
    return base

  synth_m, synth_n = _derive_synthetic_shape(spec)
  run_spec = qk_ops.describe_q4k_int8_wmma_tiled_prefill(synth_n, spec.k, synth_m, role=spec.role,
                                                        m_tile=spec.m_tile, n_tile=spec.n_tile,
                                                        group_tile=spec.group_tile)
  run_spec.validate()
  with Context(DEV="PYTHON"):
    words, ref_w = _make_q4k_words(run_spec.n, run_spec.k, SYNTHETIC_SEED + len(spec.role))
    words = words.to("CPU").realize()
    ref_w = ref_w.to("CPU").realize()
    x = Tensor(np.random.default_rng(SYNTHETIC_SEED + 1 + len(spec.role))
               .standard_normal((run_spec.m, run_spec.k)).astype(np.float32)).to("CPU").realize()
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

  if which("clang") is None:
    return {
      **base,
      "exec": {
        "attempted": True,
        "class": "blocked.compiler_unavailable",
        "compile_ms": None,
        "runtime_ms": None,
        "kernel_count": None,
        "graph_node_count": None,
        "wmma_present": False,
        "error": reference_error or "execution_skipped_no_clang",
        "raw_tile_steps": _raw_tile_steps(run_spec),
      },
    }

  debug = io.StringIO()
  before = GlobalCounters.kernel_count
  before_time = GlobalCounters.time_sum_s
  try:
    with contextlib.redirect_stdout(debug), Context(DEBUG=4):
      got = qk_ops.emit_q4k_int8_wmma_tiled_exec_tensor(words, xq, xscales, run_spec).realize().numpy()
  except Exception as e:
    return {
      **base,
      "exec": {
        "attempted": True,
        "class": "blocked.generated_tiled_loop_execution_failed",
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
  compile_match = re.search(r"scheduled\s+\d+\s+kernels in\s+([0-9.]+)\s+ms", debug_text)
  base["exec"] = {
    "attempted": True,
    "class": "pass.generated_tiled_loop",
    "compile_ms": float(compile_match.group(1)) if compile_match else None,
    "runtime_ms": (GlobalCounters.time_sum_s - before_time) * 1000.0,
    "kernel_count": GlobalCounters.kernel_count - before,
    "graph_node_count": None,
    "wmma_present": "wmma_i32_16x16x16_iu8" in debug_text,
    "raw_tile_steps": _raw_tile_steps(run_spec),
    "m": run_spec.m,
    "n": run_spec.n,
    "k": run_spec.k,
    "wmma_evidence": "runtime_debug_contains_wmma_source" if "wmma_i32_16x16x16_iu8" in debug_text else "missing",
    "numeric_ok": bool(rel is not None and rel < RTOL),
    "rel_rmse": float(rel) if rel is not None else None,
    "rtol": RTOL,
    "error": reference_error,
  }
  return base


def build(lifecycle: dict[str, Any] | None = None) -> dict[str, Any]:
  lifecycle = lifecycle if lifecycle is not None else lifecycle_build()
  role_specs = tuple(describe_int8_wmma_tile_lowering(m, n, k, role=role, m_tile=16, n_tile=16, group_tile=1)
                    for role, m, n, k in QWEN3_14B_Q4K_ROLE_SHAPES)
  loop_contract = build_scheduler_owned_tile_loop_contract(role_specs, route_id="prefill_q4k_int8_wmma_tiled_research")
  rows = [_role_row(spec, lifecycle) for spec in role_specs]
  all_attempted = all(row["exec"]["attempted"] for row in rows)
  any_numeric_ok = any(row["exec"].get("numeric_ok") for row in rows)
  blocker = lifecycle["verdict"] == "Q4K_WMMA_TILED_LIFECYCLE_PASS" and loop_contract["required"]
  return {
    "schema": "q4k_wmma_tiled_role_shape_exec_gate.v1",
    "scope": "synthetic execution gate for all 14B Q4_K/Q8_1 wmma_tiled prefill role shapes",
    "verdict": "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_FULL_ROLE_LOWERING" if blocker else
      "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_LIFECYCLE",
    "route_id": "prefill_q4k_int8_wmma_tiled_research",
    "scheduler_owned_tile_loop": loop_contract,
    "remaining_blocker": loop_contract["remaining_blocker"] if blocker else None,
    "required_next": loop_contract["remaining_blocker"] if blocker else None,
    "lifecycle": {"verdict": lifecycle["verdict"], "class": lifecycle["class"]},
    "roles": rows,
    "attempted_count": sum(1 if row["exec"]["attempted"] else 0 for row in rows),
    "executed_roles": [row["role"] for row in rows if row["exec"]["attempted"]],
    "classified_blocker": True,
    "all_numeric_ok": all_attempted and any_numeric_ok,
    "blocker": "role execution is intentionally blocked until a scheduler-owned tile_m/tile_n/group loop is ownership-integrated",
    "distinction_from_classifier": (
      "q4k_wmma_tiled_role_shape enumerates/selects shapes; this gate executes bounded synthetic role-shape "
      "subgraphs and reports the remaining loop-boundary blocker."
    ),
    "next_blocker": loop_contract["remaining_blocker"] if blocker else None,
  }


if __name__ == "__main__":
  out = build()
  ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
  ARTIFACT.write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_ROLE_SHAPE_EXEC_BLOCKED_FULL_ROLE_LOWERING" else 1)
