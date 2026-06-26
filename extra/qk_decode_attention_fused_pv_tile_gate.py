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

def _standalone_numeric() -> dict[str, Any]:
  import numpy as np
  from tinygrad import Tensor, dtypes
  from extra.qk_flash_decode import flash_fused_pv_tile_whole_cache_kernel

  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  G, S, W = Hq // Hkv, (Tc + L - 1) // L, Hd + 1
  rng = np.random.default_rng(20260626)
  score = rng.normal(0.0, 0.5, size=(Hq, MAXC)).astype(np.float32)
  cache = np.zeros((2, Hkv, MAXC, Hd), dtype=np.float32)
  cache[1] = rng.normal(0.0, 0.25, size=(Hkv, MAXC, Hd)).astype(np.float32)
  pm = np.full((Hq, S), -np.inf, dtype=np.float32)
  for h in range(Hq):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      pm[h, s] = score[h, t0:t1].max() if t0 < t1 else -1e30

  got = Tensor.empty(Hq * S * W, dtype=dtypes.float32).custom_kernel(
    Tensor(pm.reshape(-1)), Tensor(score.reshape(-1)), Tensor(cache.reshape(-1)),
    fxn=flash_fused_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc))[0].realize().numpy().reshape(Hq, S, W)

  ref = np.zeros((Hq, S, W), dtype=np.float32)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      for g in range(G):
        h = kvh * G + g
        p = np.exp(score[h, t0:t1] - pm[h, s]).astype(np.float32)
        ref[h, s, :Hd] = p @ cache[1, kvh, t0:t1, :]
        ref[h, s, Hd] = p.sum()

  diff = got - ref
  max_abs = float(np.max(np.abs(diff)))
  rmse = float(np.sqrt(np.mean(diff * diff)))
  ref_scale = float(np.sqrt(np.mean(ref * ref)) + 1e-12)
  rel_rmse = float(rmse / ref_scale)
  return {
    "checked": True,
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "L": L, "Tc": Tc, "S": S, "W": W},
    "max_abs": max_abs,
    "rmse": rmse,
    "rel_rmse": rel_rmse,
    "pass": bool(max_abs <= 5e-4 and rel_rmse <= 5e-5),
    "thresholds": {"max_abs": 5e-4, "rel_rmse": 5e-5},
  }


def build() -> dict[str, Any]:
  target_src = _builder_source(TARGET_BUILDER)
  refuted_src = _builder_source(REFUTED_BUILDER)
  target_markers = _marker_counts(target_src)
  refuted_markers = _marker_counts(refuted_src)
  target_exists = target_src is not None
  target_has_local_d = target_markers.get("d_local_required_shape", 0) > 0 or target_markers.get("axis_local", 0) > 0
  target_avoids_refuted_global_d = target_markers.get("d_global_refuted_shape", 0) == 0

  numeric = {"checked": False, "reason": "target builder missing"}
  if not target_exists:
    verdict = "FUSED_PV_TILE_BLOCKED__NO_GENERATED_TILE_BUILDER"
  elif not target_has_local_d or not target_avoids_refuted_global_d:
    verdict = "FUSED_PV_TILE_BLOCKED__REFUTED_GLOBAL_D_SHAPE"
  else:
    numeric = _standalone_numeric()
    verdict = "FUSED_PV_TILE_STANDALONE_NUMERIC_PASS__ROUTE_GATE_REQUIRED" if numeric.get("pass") else "FUSED_PV_TILE_FAIL__STANDALONE_NUMERIC"

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
    "standalone_numeric": numeric,
    "owned_oracle_facts": _owned_oracle_facts(),
    "wall_audit_facts": _wall_audit_facts(),
    "decision": (
      "Do not route or W==D yet. Build the generated fused PV tile builder first, then extend this gate with standalone numeric comparison."
      if not target_exists else
      ("Standalone generated fused PV tile passed numeric gate. Next step is default-off model route wiring and route/materialization gate."
       if numeric.get("pass") else
       "Builder exists but standalone numeric failed. Fix kernel semantics before model routing.")
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
