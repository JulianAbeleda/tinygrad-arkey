#!/usr/bin/env python3
"""Lifecycle gate for the Q4_K/Q8_1 tiled WMMA route."""
from __future__ import annotations

import contextlib, io, json, re
from pathlib import Path
from typing import Any

import numpy as np
from tinygrad import Context, GlobalCounters, Tensor, dtypes
from extra.qk.layout import q8_1_quantize
from extra.qk.prefill_int8_wmma_spec import describe_q4k_int8_wmma_tiled_prefill, emit_q4k_int8_wmma_tiled_lifecycle_tensor
from extra.qk.prefill_mmq_parity_gate import _make_q4k_words, _rel_rmse, RTOL
from extra.qk.q4k_wmma_tiled_microgate import build as microgate_build
from extra.qk.q4k_wmma_tiled_surface_gate import build as surface_build

ARTIFACT = Path("bench/q4k-wmma-tiled-lifecycle/latest.json")

def _run_lifecycle(surface:dict[str, Any]|None=None) -> dict[str, Any]:
  n, k, m = 32, 256, 32
  words, ref_w = _make_q4k_words(n, k, 20260707)
  x = Tensor(np.random.default_rng(20260708).standard_normal((m, k)).astype(np.float32)).realize()
  xq, xscales = q8_1_quantize(x.cast(dtypes.float32))
  x_dq = (xq.reshape(m, k // 32, 32).cast(dtypes.float32) *
          xscales.reshape(m, k // 32, 1).cast(dtypes.float32)).reshape(m, k)
  ref_out = (x_dq @ ref_w.T).numpy()
  spec = describe_q4k_int8_wmma_tiled_prefill(n, k, m, role="lifecycle", m_tile=16, n_tile=16, group_tile=1)
  before = GlobalCounters.kernel_count
  before_time = GlobalCounters.time_sum_s
  debug = io.StringIO()
  with contextlib.redirect_stdout(debug), Context(DEBUG=4):
    got = emit_q4k_int8_wmma_tiled_lifecycle_tensor(words, xq, xscales, spec).realize().numpy()
  debug_text = debug.getvalue()
  rel = _rel_rmse(got, ref_out)
  compile_match = re.search(r"scheduled\s+\d+\s+kernels in\s+([0-9.]+)\s+ms", debug_text)
  lifecycle_source_wmma = "wmma_i32_16x16x16_iu8" in debug_text
  surface_wmma = bool(surface and surface.get("has_iu8_wmma_isa_or_source"))
  return {"numeric_ok": bool(rel < RTOL), "rel_rmse": rel, "rtol": RTOL,
          "kernel_count": GlobalCounters.kernel_count - before,
          "runtime_ms": (GlobalCounters.time_sum_s - before_time) * 1000.0,
          "compile_ms": float(compile_match.group(1)) if compile_match else None,
          "graph_node_count": None,
          "wmma_present": lifecycle_source_wmma or surface_wmma,
          "wmma_evidence": "lifecycle_debug_source" if lifecycle_source_wmma else
                           "surface_probe_cached_lifecycle" if surface_wmma else "missing",
          "debug_tail": debug_text[-6000:],
          "live_raw_elems": spec.live_raw_elems,
          "forbidden_full_raw_elems": spec.forbidden_full_raw_elems,
          "output_shape": list(got.shape)}


def build(surface:dict[str, Any]|None=None, microgate:dict[str, Any]|None=None) -> dict[str, Any]:
  surface = surface if surface is not None else surface_build()
  microgate = microgate if microgate is not None else microgate_build()
  surface_ok = surface["verdict"] == "Q4K_WMMA_TILED_SURFACE_TC_MATCHER_SELECTED"
  microgate_ok = microgate["verdict"] == "Q4K_WMMA_TILED_MICROGATE_PASS"
  run = _run_lifecycle(surface) if surface_ok and microgate_ok else None
  lifecycle_ok = bool(run and run["numeric_ok"] and run["wmma_present"] and
                      run["live_raw_elems"] == 16 * 16 * 1 and
                      run["forbidden_full_raw_elems"] == 8 * 32 * 32 and
                      run["kernel_count"] <= 128)
  verdict = "Q4K_WMMA_TILED_LIFECYCLE_PASS" if lifecycle_ok else "Q4K_WMMA_TILED_LIFECYCLE_FAIL"
  return {"schema": "q4k_wmma_tiled_lifecycle_gate.v1",
          "scope": "M=32,N=32,K=256 four-output-tile bounded generated lifecycle for Q4_K/Q8_1 tiled WMMA",
          "verdict": verdict,
          "route_id": "prefill_q4k_int8_wmma_tiled_research",
          "target_shape": {"m": 32, "n": 32, "k": 256, "output_tiles": 4,
                           "m_tile": 16, "n_tile": 16, "group_tile": 1},
          "surface": {"ok": surface_ok, "verdict": surface["verdict"],
                      "selected_surface": surface.get("selected_surface")},
          "one_tile_numeric": {"ok": microgate_ok, "verdict": microgate["verdict"],
                               "has_iu8_wmma": microgate["probe"]["has_iu8_wmma"]},
          "implemented": lifecycle_ok,
          "class": "pass.bounded_multi_tile_lifecycle" if lifecycle_ok else "fail",
          "classified_blocker": False,
          "lifecycle": run,
          "required_next": ["scale the bounded lifecycle to synthetic 14B role-shape execution",
                            "reduce kernel_count once scheduler ownership exists for the full route"]}


if __name__ == "__main__":
  out = build()
  ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
  ARTIFACT.write_text(json.dumps(out, indent=2))
  print(json.dumps(out, indent=2))
  raise SystemExit(0 if out["verdict"] == "Q4K_WMMA_TILED_LIFECYCLE_PASS" else 1)
