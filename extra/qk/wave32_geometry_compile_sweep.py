#!/usr/bin/env python3
"""Compile-only sweep of the existing AMD wave32 WMMA candidate path.

This is deliberately a consumer-side validation tool: it creates a matmul,
attaches candidate geometry, and invokes the normal rewrite pipeline.  It does
not change runtime selection, lowering, or emitters.
"""
from __future__ import annotations

import hashlib, json, sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.kernel_pipeline import KernelStage1PipelinePlan
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import KernelCandidateContext, KernelLDSWindow, KernelTileGeometry, Ops

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "bench/wave32-geometry-compile-sweep/latest.json"
TILES = (16, 32, 64, 128)
WAVES = ((1, 1), (2, 2), (4, 2))
K = 256

def _one(tile: int, waves: tuple[int, int]) -> dict[str, Any]:
  threads = waves[0] * waves[1] * 32
  # Keep the same two-buffer stage contract used by the existing stage-1 gate.
  slot_bytes = tile * 64
  lds = 2 * slot_bytes
  geometry = KernelTileGeometry((tile, tile, 32), waves, threads, 32,
    (KernelLDSWindow("A", 0, slot_bytes, 64),
     KernelLDSWindow("B", slot_bytes, lds, 64)))
  context = KernelCandidateContext("boltbeam.full_kernel_candidate.v1",
    hashlib.sha256(f"wave32:{tile}:{waves}:{K}".encode()).hexdigest(), geometry,
    KernelStage1PipelinePlan(2, slot_bytes))
  result: dict[str, Any] = {"tile": tile, "waves": list(waves), "wave_size": 32,
    "k": K, "wmma_count": None, "lds_bytes": lds,
    "register_evidence": {"status": "unavailable", "source": "final allocator"}}
  try:
    a, b = Tensor.empty(tile, K, dtype=dtypes.half), Tensor.empty(K, tile, dtype=dtypes.half)
    sink = next(u for u in (a @ b).schedule_linear().toposort() if u.op is Ops.SINK)
    sink = sink.replace(arg=replace(sink.arg, opts_to_apply=(Opt(OptOps.TC, 0, (0, 0, 1)),
                                                              Opt(OptOps.UNROLL, 0, 0)), candidate_context=context))
    lowered = full_rewrite_to_sink(sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100")), optimize=True)
    nodes = lowered.toposort()
    wmmas = [u for u in nodes if u.op is Ops.WMMA]
    regs = [u for u in nodes if u.op is Ops.DEFINE_REG]
    result["wmma_count"] = len(wmmas)
    result["register_evidence"] = {"status": "structural", "define_reg_elements": [u.ptrdtype.size for u in regs],
      "source": "lowered UOp graph; not final VGPR allocation"}
    result["status"] = "PASS"
  except Exception as exc:
    result.update(status="FAIL", failure_stage="rewrite_or_lowering", failure_type=type(exc).__name__,
                  first_failure=str(exc))
  return result

def run() -> dict[str, Any]:
  rows = [_one(tile, waves) for waves in WAVES for tile in TILES]
  first_failure = next((r for r in rows if r["status"] == "FAIL"), None)
  return {"schema": "wave32_geometry_compile_sweep.v1", "scope": "compile-only; existing tensor/compiler path",
          "geometries": {"wave_size": 32, "waves": [list(x) for x in WAVES], "tiles": list(TILES), "k": K},
          "first_failure": {"tile": first_failure["tile"], "waves": first_failure["waves"]} if first_failure else None,
          "results": rows}

if __name__ == "__main__":
  out = run(); ARTIFACT.parent.mkdir(parents=True, exist_ok=True); ARTIFACT.write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
  # A compile failure is the measured result of this diagnostic sweep, not a
  # harness failure; callers inspect each row and first_failure.
  raise SystemExit(0)
