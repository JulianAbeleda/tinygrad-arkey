#!/usr/bin/env python3
"""Canonical gate for generated fused PV tile decode-attention work.

This gate deliberately starts as a blocker gate.  It prevents the project from
mistaking the already-refuted split x-lane PV route for the desired fused tile
route.  Once a real generated builder exists, this file becomes the standalone
numeric + structural gate before model routing and W==D.
"""
from __future__ import annotations

import inspect, json, pathlib, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-fused-pv-tile"
TARGET_BUILDER = "flash_fused_pv_tile_whole_cache_kernel"
TARGET_PROGRAM = "flash_fused_pv_tile_whole_cache_32_128"
REFUTED_BUILDER = "flash_xlane_pv_from_m_kernel"
REFUTED_PROGRAM = "flash_xlane_pv_from_m_32_128"


def _builder_source(name: str) -> str | None:
  import extra.qk_flash_decode as qfd
  fn = getattr(qfd, name, None)
  if fn is None: return None
  return inspect.getsource(fn)


def _marker_counts(src: str | None) -> dict[str, int]:
  if src is None: return {}
  markers = {
    "axis_global": "AxisType.GLOBAL",
    "axis_local": "AxisType.LOCAL",
    "axis_reduce": "AxisType.REDUCE",
    "special_lane": "UOp.special",
    "warp_reduce_sum": "_warp_reduce_sum_staged",
    "warp_reduce_max": "warp_reduce_max",
    "reg_placeholder": "AddrSpace.REG",
    "local_placeholder": "AddrSpace.LOCAL",
    "d_global_refuted_shape": "d = UOp.range(W, 2, AxisType.GLOBAL)",
    "d_local_required_shape": "d = UOp.range(W, 2, AxisType.LOCAL)",
    "sink": ".sink(",
  }
  return {k: src.count(v) for k, v in markers.items()}


def _selected_lines(src: str | None, limit: int = 80) -> list[str]:
  if src is None: return []
  needles = ("def ", "UOp.range", "UOp.special", "AddrSpace", "_warp_reduce", "warp_reduce", ".store", "sink(")
  rows = []
  for i, line in enumerate(src.splitlines(), 1):
    if any(n in line for n in needles): rows.append(f"{i}: {line.rstrip()}")
    if len(rows) >= limit: break
  return rows


def _owned_oracle_facts() -> dict[str, Any]:
  p = ROOT / "bench/qk-isa-primitive-audit/owned_decode_attention.json"
  if not p.exists(): return {"available": False, "path": str(p.relative_to(ROOT))}
  d = json.loads(p.read_text())
  return {
    "available": True,
    "path": str(p.relative_to(ROOT)),
    "verdict": d.get("verdict"),
    "instruction_flags": d.get("instruction_flags", {}),
    "instr_counts": d.get("instr_counts", {}),
    "resources": d.get("resources", {}),
  }


def _wall_audit_facts() -> dict[str, Any]:
  p = ROOT / "bench/qk-decode-attention-generated-pv-kernel-audit/latest.json"
  if not p.exists(): return {"available": False, "path": str(p.relative_to(ROOT))}
  d = json.loads(p.read_text())
  diag = d.get("diagnosis", {})
  return {
    "available": True,
    "path": str(p.relative_to(ROOT)),
    "verdict": d.get("verdict"),
    "generated_pv_shape_flags": diag.get("generated_pv_shape_flags", {}),
    "blockers": diag.get("blockers", []),
  }


def build() -> dict[str, Any]:
  target_src = _builder_source(TARGET_BUILDER)
  refuted_src = _builder_source(REFUTED_BUILDER)
  target_markers = _marker_counts(target_src)
  refuted_markers = _marker_counts(refuted_src)
  target_exists = target_src is not None
  target_has_local_d = target_markers.get("d_local_required_shape", 0) > 0 or target_markers.get("axis_local", 0) > 0
  target_avoids_refuted_global_d = target_markers.get("d_global_refuted_shape", 0) == 0

  if not target_exists:
    verdict = "FUSED_PV_TILE_BLOCKED__NO_GENERATED_TILE_BUILDER"
  elif not target_has_local_d or not target_avoids_refuted_global_d:
    verdict = "FUSED_PV_TILE_BLOCKED__REFUTED_GLOBAL_D_SHAPE"
  else:
    verdict = "FUSED_PV_TILE_STRUCTURAL_BUILDER_PRESENT__NUMERIC_GATE_REQUIRED"

  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "target": {
      "builder": TARGET_BUILDER,
      "program": TARGET_PROGRAM,
      "exists": target_exists,
      "marker_counts": target_markers,
      "selected_source_lines": _selected_lines(target_src),
    },
    "refuted_current_route": {
      "builder": REFUTED_BUILDER,
      "program": REFUTED_PROGRAM,
      "exists": refuted_src is not None,
      "marker_counts": refuted_markers,
      "selected_source_lines": _selected_lines(refuted_src),
    },
    "required_shape": {
      "d_axis": "local/cooperative ownership, not global output-column ownership",
      "tile_lifecycle": "score/state/PV inside one tile lifecycle before compact partial output",
      "must_not_be": REFUTED_PROGRAM,
      "must_include": ["tile-local K/V reuse", "register online state", "cross-lane score reduction", "vectorized loads or packed-dot lowering"],
    },
    "owned_oracle_facts": _owned_oracle_facts(),
    "wall_audit_facts": _wall_audit_facts(),
    "decision": (
      "Do not route or W==D yet. Build the generated fused PV tile builder first, then extend this gate with standalone numeric comparison."
      if not target_exists else
      "Builder exists structurally; next step is standalone numeric comparison before model routing."
    ),
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-fused-pv-tile-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
