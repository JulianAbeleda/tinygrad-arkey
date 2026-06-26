#!/usr/bin/env python3
"""Canonical gate for generated fused PV tile decode-attention work.

This gate deliberately starts as a blocker gate.  It prevents the project from
mistaking the already-refuted split x-lane PV route for the desired fused tile
route.  Once a real generated builder exists, this file becomes the standalone
numeric + structural gate before model routing and W==D.
"""
from __future__ import annotations

import inspect, json, os, pathlib, subprocess, sys, time
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

def _route_env(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_FUSED_PV_TILE_CHILD": "1", "QK_FUSED_PV_TILE_ARM": arm}
  for k in ("DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
            "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_TILE_PLACEHOLDER", "DECODE_ATTN_TILE_SCORE_MAX",
            "DECODE_ATTN_TILE_PROB", "DECODE_ATTN_TILE_PARTIAL_PV", "DECODE_ATTN_TILE_PROB_PARTIAL_PV",
            "DECODE_ATTN_ONLINE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE",
            "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE", "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE",
            "DECODE_ATTN_FUSED_PV_TILE", "V_DOT2_LOWERING", "WARP_REDUCE_LOWERING"):
    env[k] = "0"
  if arm == "fused_pv_tile":
    env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
    env["DECODE_ATTN_FUSED_PV_TILE"] = "1"
  return env

def _programs(route: dict[str, Any]) -> list[str]:
  return list(route["route_fire"]["program_node_names"])

def _route_signature(route: dict[str, Any]) -> dict[str, Any]:
  names = _programs(route)
  generated = [n for n in names if n.startswith("flash_")]
  return {
    "generated_attention_programs": generated,
    "has_target_program": any(n.startswith(TARGET_PROGRAM) for n in generated),
    "has_refuted_program": any(n.startswith(REFUTED_PROGRAM) for n in generated),
    "has_score": any(n.startswith("flash_score_whole_cache") for n in generated),
    "has_max": any(n == "flash_max_32" for n in generated),
    "has_gmax": any(n == "flash_gmax_32" for n in generated),
    "has_den": any(n == "flash_den_32" for n in generated),
    "has_combine": any(n.startswith("flash_combine") for n in generated),
  }

def _child_route(arm: str) -> dict[str, Any]:
  from extra.qk_decode_attention_purity_capture import capture
  route = capture("a2" if arm == "fused_pv_tile" else "baseline")
  return {"arm": arm, "route": route, "signature": _route_signature(route)}

def _run_route_child(arm: str) -> dict[str, Any]:
  r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=_route_env(arm),
                     capture_output=True, text=True)
  if r.returncode != 0:
    return {"arm": arm, "failed": True, "returncode": r.returncode, "stdout_tail": r.stdout[-6000:], "stderr_tail": r.stderr[-6000:]}
  for line in reversed(r.stdout.strip().splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm": arm, "failed": True, "returncode": 0, "error": "no json", "stdout_tail": r.stdout[-6000:], "stderr_tail": r.stderr[-6000:]}

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

def _route_gate() -> dict[str, Any]:
  baseline = _run_route_child("baseline")
  fused = _run_route_child("fused_pv_tile")
  if baseline.get("failed") or fused.get("failed"):
    return {"checked": True, "pass": False, "verdict": "FUSED_PV_TILE_ROUTE_FAIL__CHILD", "baseline": baseline, "fused_pv_tile": fused}
  route = fused["route"]
  sig = fused["signature"]
  token_match = baseline["route"]["tokens_sample"] == route["tokens_sample"]
  materialization_clean = (not route["materialization"]["E_49152_present"]) and bool(route["materialization"]["selected_route_buffer_identity"])
  owned_absent = route["route_counts"]["owned_flash_tile_gqa_whole"] == 0 and route["route_counts"]["owned_flash_combine"] == 0
  generated_clean = route["verdict"] == "DECODE_ATTENTION_A2_GENERATED_WHOLECACHE_ROUTE_CLEAN"
  lifecycle_complete = sig["has_score"] and sig["has_max"] and sig["has_target_program"] and sig["has_gmax"] and sig["has_den"] and sig["has_combine"]
  passed = token_match and materialization_clean and owned_absent and generated_clean and lifecycle_complete and not sig["has_refuted_program"]
  if not token_match:
    verdict = "FUSED_PV_TILE_ROUTE_FAIL__TOKEN_MISMATCH"
  elif not materialization_clean:
    verdict = "FUSED_PV_TILE_ROUTE_FAIL__MATERIALIZATION"
  elif not owned_absent:
    verdict = "FUSED_PV_TILE_ROUTE_FAIL__OWNED_ROUTE_PRESENT"
  elif not sig["has_target_program"]:
    verdict = "FUSED_PV_TILE_ROUTE_FAIL__TARGET_PROGRAM_MISSING"
  elif sig["has_refuted_program"]:
    verdict = "FUSED_PV_TILE_ROUTE_FAIL__REFUTED_PROGRAM_PRESENT"
  elif not lifecycle_complete:
    verdict = "FUSED_PV_TILE_ROUTE_FAIL__INCOMPLETE_LIFECYCLE"
  elif not generated_clean:
    verdict = "FUSED_PV_TILE_ROUTE_FAIL__CAPTURE_NOT_CLEAN"
  else:
    verdict = "FUSED_PV_TILE_ROUTE_CLEAN__WD_REQUIRED"
  return {
    "checked": True,
    "pass": passed,
    "verdict": verdict,
    "token_match": token_match,
    "materialization_clean": materialization_clean,
    "owned_absent": owned_absent,
    "generated_clean": generated_clean,
    "lifecycle_complete": lifecycle_complete,
    "baseline": baseline,
    "fused_pv_tile": fused,
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
    route_gate = _route_gate() if numeric.get("pass") else {"checked": False, "reason": "standalone numeric failed"}
    verdict = route_gate["verdict"] if route_gate.get("checked") else "FUSED_PV_TILE_FAIL__STANDALONE_NUMERIC"

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
    "route_gate": route_gate if "route_gate" in locals() else {"checked": False, "reason": "target builder missing or structurally blocked"},
    "owned_oracle_facts": _owned_oracle_facts(),
    "wall_audit_facts": _wall_audit_facts(),
    "decision": (
      "Do not route or W==D yet. Build the generated fused PV tile builder first, then extend this gate with standalone numeric comparison."
      if not target_exists else
      ("Route/materialization gate passed. Next step is W==D candidate evaluation; do not promote without W==D."
       if (numeric.get("pass") and route_gate.get("pass")) else
       "Standalone numeric passed but route/materialization gate failed; fix route before W==D."
       if numeric.get("pass") else
       "Builder exists but standalone numeric failed. Fix kernel semantics before model routing.")
    ),
  }


def main() -> int:
  if os.environ.get("QK_FUSED_PV_TILE_CHILD") == "1":
    print(json.dumps(_child_route(os.environ.get("QK_FUSED_PV_TILE_ARM", "baseline"))))
    return 0
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
