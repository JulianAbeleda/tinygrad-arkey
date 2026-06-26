#!/usr/bin/env python3
"""Canonical gate for generated fused score+state+PV decode-attention work.

This starts as a scope/blocker gate.  It prevents confusing the already-routed
fused-PV-only candidate with the stronger target: one generated tile that fuses
score computation, online softmax state, and PV accumulation.
"""
from __future__ import annotations

import inspect, json, pathlib, time
import traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-fused-score-state-pv-tile"
TARGET_BUILDER = "flash_fused_score_state_pv_tile_whole_cache_kernel"
TARGET_PROGRAM = "flash_fused_score_state_pv_tile_whole_cache_32_128"
PREVIOUS_BUILDER = "flash_fused_pv_tile_whole_cache_kernel"
PREVIOUS_PROGRAM = "flash_fused_pv_tile_whole_cache_32_128"
OLD_SCORE_PROGRAM = "flash_score_whole_cache_32_128"
OLD_MAX_PROGRAM = "flash_max_32"


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
    "q_load_hint": "q[",
    "cache_k_load_hint": "0 * Hkv",
    "cache_v_load_hint": "1 * Hkv",
    "online_m_hint": "old_m",
    "online_l_hint": "old_l",
    "den_col_hint": "d.eq(Hd)",
    "max_col_hint": "d.eq(Hd + 1)",
    "sink": ".sink(",
  }
  return {k: src.count(v) for k, v in markers.items()}


def _selected_lines(src: str | None, limit: int = 100) -> list[str]:
  if src is None: return []
  needles = ("def ", "UOp.range", "UOp.special", "AddrSpace", "q[", "cache[", "old_m", "old_l", "corr", "p =", ".store", "sink(")
  rows = []
  for i, line in enumerate(src.splitlines(), 1):
    if any(n in line for n in needles): rows.append(f"{i}: {line.rstrip()}")
    if len(rows) >= limit: break
  return rows


def _latest_json(path: str) -> dict[str, Any]:
  p = ROOT / path
  if not p.exists(): return {"available": False, "path": path}
  d = json.loads(p.read_text())
  return {"available": True, "path": path, "verdict": d.get("verdict"), "diagnosis": d.get("diagnosis", {}), "wd": d.get("wd", {})}


def _fused_pv_wd_summary() -> dict[str, Any]:
  path = "bench/qk-decode-eval/runs/20260626T003837-decode_attention_fused_pv_tile.json"
  p = ROOT / path
  if not p.exists(): return {"available": False, "path": path}
  d = json.loads(p.read_text()).get("wd", {})
  return {
    "available": True,
    "path": path,
    "baseline_per_ctx": d.get("baseline_per_ctx", {}),
    "per_ctx": d.get("per_ctx", {}),
    "delta_pct": d.get("delta_pct", {}),
    "promotion_gate_passed": d.get("promotion_gate_passed"),
  }

def _standalone_numeric() -> dict[str, Any]:
  import numpy as np
  from tinygrad import Tensor, dtypes
  from extra.qk_flash_decode import flash_fused_score_state_pv_tile_whole_cache_kernel

  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  G, S, W = Hq // Hkv, (Tc + L - 1) // L, Hd + 2
  rng = np.random.default_rng(20260626)
  q = rng.normal(0.0, 0.25, size=(Hq, Hd)).astype(np.float32)
  cache = np.zeros((2, Hkv, MAXC, Hd), dtype=np.float32)
  cache[0] = rng.normal(0.0, 0.25, size=(Hkv, MAXC, Hd)).astype(np.float32)
  cache[1] = rng.normal(0.0, 0.25, size=(Hkv, MAXC, Hd)).astype(np.float32)

  got = Tensor.empty(Hq * S * W, dtype=dtypes.float32).custom_kernel(
    Tensor(q.reshape(-1)), Tensor(cache.reshape(-1)),
    fxn=flash_fused_score_state_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc))[0].realize().numpy().reshape(Hq, S, W)

  ref = np.zeros((Hq, S, W), dtype=np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      for g in range(G):
        h = kvh * G + g
        scores = (cache[0, kvh, t0:t1, :] @ q[h]) * scale
        m = np.max(scores).astype(np.float32)
        p = np.exp(scores - m).astype(np.float32)
        ref[h, s, :Hd] = p @ cache[1, kvh, t0:t1, :]
        ref[h, s, Hd] = p.sum()
        ref[h, s, Hd + 1] = m

  diff = got - ref
  finite = bool(np.isfinite(got).all())
  max_abs = float(np.max(np.abs(diff)))
  rmse = float(np.sqrt(np.mean(diff * diff)))
  ref_scale = float(np.sqrt(np.mean(ref * ref)) + 1e-12)
  rel_rmse = float(rmse / ref_scale)
  return {
    "checked": True,
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "L": L, "Tc": Tc, "S": S, "W": W},
    "finite": finite,
    "max_abs": max_abs,
    "rmse": rmse,
    "rel_rmse": rel_rmse,
    "pass": bool(finite and max_abs <= 1e-3 and rel_rmse <= 1e-5),
    "thresholds": {"max_abs": 1e-3, "rel_rmse": 1e-5},
  }

def _standalone_numeric_or_blocker() -> dict[str, Any]:
  try:
    return _standalone_numeric()
  except Exception as e:
    tb = traceback.format_exc()
    if "pop from empty list" in tb and "Estimates.from_uops" in tb:
      verdict = "FUSED_SCORE_STATE_PV_TILE_BLOCKED__MULTI_REDUCTION_STORE_SHAPE"
    else:
      verdict = "FUSED_SCORE_STATE_PV_TILE_FAIL__STANDALONE_EXCEPTION"
    return {
      "checked": True,
      "pass": False,
      "blocked": verdict.startswith("FUSED_SCORE_STATE_PV_TILE_BLOCKED"),
      "verdict": verdict,
      "exception_type": type(e).__name__,
      "exception": str(e),
      "traceback_tail": tb[-5000:],
    }


def build() -> dict[str, Any]:
  target_src = _builder_source(TARGET_BUILDER)
  previous_src = _builder_source(PREVIOUS_BUILDER)
  target_markers = _marker_counts(target_src)
  previous_markers = _marker_counts(previous_src)
  target_exists = target_src is not None
  previous_exists = previous_src is not None

  numeric = {"checked": False, "reason": "target builder missing or structurally blocked"}
  if not target_exists:
    verdict = "FUSED_SCORE_STATE_PV_TILE_BLOCKED__NO_GENERATED_TILE_BUILDER"
  elif target_markers.get("axis_reduce", 0) < 2:
    verdict = "FUSED_SCORE_STATE_PV_TILE_BLOCKED__MISSING_SCORE_AND_TOKEN_REDUCTIONS"
  elif target_markers.get("axis_local", 0) == 0:
    verdict = "FUSED_SCORE_STATE_PV_TILE_BLOCKED__NO_LOCAL_D_OWNERSHIP"
  elif target_markers.get("q_load_hint", 0) == 0 or target_markers.get("cache_k_load_hint", 0) == 0 or target_markers.get("cache_v_load_hint", 0) == 0:
    verdict = "FUSED_SCORE_STATE_PV_TILE_BLOCKED__INCOMPLETE_QKV_LIFECYCLE"
  else:
    numeric = _standalone_numeric_or_blocker()
    verdict = ("FUSED_SCORE_STATE_PV_TILE_STANDALONE_NUMERIC_PASS__ROUTE_GATE_REQUIRED" if numeric.get("pass") else
               numeric.get("verdict", "FUSED_SCORE_STATE_PV_TILE_FAIL__NUMERIC"))

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
    "previous_fused_pv_only": {
      "builder": PREVIOUS_BUILDER,
      "program": PREVIOUS_PROGRAM,
      "exists": previous_exists,
      "marker_counts": previous_markers,
      "selected_source_lines": _selected_lines(previous_src),
      "wd_summary": _fused_pv_wd_summary(),
    },
    "required_route_signature": {
      "must_include": [TARGET_PROGRAM, "flash_state_gmax_32_128", "flash_state_combine_32_128"],
      "must_exclude": [OLD_SCORE_PROGRAM, OLD_MAX_PROGRAM, PREVIOUS_PROGRAM],
      "default_off_flag": "DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE=1",
    },
    "required_output_layout": {
      "W": "Hd + 2",
      "d_lt_Hd": "unnormalized PV accumulator",
      "d_eq_Hd": "split denominator l",
      "d_eq_Hd_plus_1": "split max m",
    },
    "standalone_numeric": numeric,
    "kill_gate": "If UOp cannot express q.k score reduce + token online recurrence + local-d PV in one builder, classify as FUSED_SCORE_STATE_PV_TILE_BLOCKED__MULTI_REDUCTION_STORE_SHAPE.",
    "supporting_artifacts": {
      "fused_pv_tile_gate": _latest_json("bench/qk-decode-attention-fused-pv-tile/latest.json"),
      "generated_pv_wall_audit": _latest_json("bench/qk-decode-attention-generated-pv-kernel-audit/latest.json"),
    },
    "decision": (
      "Do not route or W==D yet. Build the target generated score+state+PV tile builder, then add standalone numeric comparison."
      if not target_exists else
      ("Standalone numeric passed. Next step is default-off model route wiring and route/materialization/lifecycle gate."
       if numeric.get("pass") else
       "Target builder exists but standalone numeric failed. Fix kernel semantics before route wiring.")
    ),
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-fused-score-state-pv-tile-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
