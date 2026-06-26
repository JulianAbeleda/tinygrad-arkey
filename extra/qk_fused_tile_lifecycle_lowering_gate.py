#!/usr/bin/env python3
"""Gate for the lower-level fused tile lifecycle lowering blocker.

This gate is intentionally below decode attention. It records whether the repo
can lower a nested-reduce + recurrence-state + local-output/metadata-store UOp
shape, and links that to the fused score/state/PV attention blocker.
"""
from __future__ import annotations

import json, pathlib, time
import traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-fused-tile-lifecycle-lowering"
ATTN_BLOCKER = ROOT / "bench/qk-decode-attention-fused-score-state-pv-tile/latest.json"


def _read_json(path: pathlib.Path) -> dict[str, Any]:
  if not path.exists(): return {"available": False, "path": str(path.relative_to(ROOT))}
  d = json.loads(path.read_text())
  return {"available": True, "path": str(path.relative_to(ROOT)), "verdict": d.get("verdict"), "standalone_numeric": d.get("standalone_numeric", {})}

def _synthetic_lifecycle_kernel(D:int, J:int, E:int):
  from tinygrad import dtypes
  from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, UOp
  F32 = dtypes.float32
  W = D + 2
  def kernel(out:UOp, a:UOp, v:UOp) -> UOp:
    d = UOp.range(W, 0, AxisType.LOCAL)
    is_v = d < D
    is_l = d.eq(D)
    j = UOp.range(J, 1, axis_type=AxisType.REDUCE)
    e = UOp.range(E, 2, axis_type=AxisType.REDUCE)
    dot = UOp.placeholder((1,), F32, 250, addrspace=AddrSpace.REG)
    dot = dot.after(d, j)[0].set(0.0)
    dot_upd = dot[0].set(dot.after(e)[0] + a[j * E + e], end=e)
    acc = UOp.placeholder((1,), F32, 251, addrspace=AddrSpace.REG)
    den = UOp.placeholder((1,), F32, 252, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((1,), F32, 253, addrspace=AddrSpace.REG)
    init = acc.after(d)[0].set(0.0)
    init = den.after(init)[0].set(0.0)
    init = mx.after(init)[0].set(-float("inf"))
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    old_m = mx.after(j)[0]
    sc = dot.after(dot_upd)[0]
    new_m = old_m.maximum(sc)
    corr = (old_m - new_m).exp2()
    p = (sc - new_m).exp2()
    vd = is_v.where(v[j * D + is_v.where(d, d.const_like(0))], UOp.const(F32, 1.0))
    upd = acc[0].store(acc.after(j)[0] * corr + p * vd)
    upd = den.after(upd)[0].store(den.after(j)[0] * corr + p)
    upd = mx.after(upd)[0].store(new_m).end(j)
    af, lf, mf = acc.after(upd)[0], den.after(upd)[0], mx.after(upd)[0]
    val = is_v.where(af, is_l.where(lf, mf))
    return out[d].store(val).end(d).sink(arg=KernelInfo(name="synthetic_fused_tile_lifecycle", opts_to_apply=()))
  return kernel

def _synthetic_repro() -> dict[str, Any]:
  try:
    import numpy as np
    from tinygrad import Tensor, dtypes
    D, J, E = 4, 3, 5
    W = D + 2
    rng = np.random.default_rng(20260626)
    a = rng.normal(0, 0.25, size=(J, E)).astype(np.float32)
    v = rng.normal(0, 0.25, size=(J, D)).astype(np.float32)
    got = Tensor.empty(W, dtype=dtypes.float32).custom_kernel(
      Tensor(a.reshape(-1)), Tensor(v.reshape(-1)), fxn=_synthetic_lifecycle_kernel(D, J, E))[0].realize().numpy().reshape(W)
    ref = np.zeros(W, dtype=np.float32)
    m, l = -np.inf, np.float32(0.0)
    acc = np.zeros(D, dtype=np.float32)
    for j in range(J):
      sc = np.float32(a[j].sum())
      mn = max(m, sc)
      corr = np.exp2(np.float32(m - mn))
      p = np.exp2(np.float32(sc - mn))
      acc = acc * corr + p * v[j]
      l = l * corr + p
      m = mn
    ref[:D], ref[D], ref[D + 1] = acc, l, m
    diff = got - ref
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return {"checked": True, "compiled": True, "pass": bool(max_abs <= 1e-5), "max_abs": max_abs, "rmse": rmse,
            "shape": {"D": D, "J": J, "E": E, "W": W}}
  except Exception as e:
    tb = traceback.format_exc()
    if "pop from empty list" in tb and "Estimates.from_uops" in tb:
      verdict = "FUSED_TILE_LIFECYCLE_SYNTHETIC_BLOCKED__ESTIMATE_SCOPE_STACK"
    else:
      verdict = "FUSED_TILE_LIFECYCLE_SYNTHETIC_FAIL__EXCEPTION"
    return {"checked": True, "compiled": False, "pass": False, "verdict": verdict, "exception_type": type(e).__name__,
            "exception": str(e), "has_estimates_from_uops": "Estimates.from_uops" in tb,
            "has_pop_from_empty_list": "pop from empty list" in tb, "traceback_tail": tb[-5000:]}

def _synthetic_global_g_kernel(T:int, G:int, D:int, J:int, E:int):
  from tinygrad import dtypes
  from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, UOp
  F32 = dtypes.float32
  W = D + 2
  def kernel(out:UOp, a:UOp, v:UOp) -> UOp:
    tile = UOp.range(T, 0, AxisType.GLOBAL)
    d = UOp.range(W, 1, AxisType.LOCAL)
    is_v = d < D
    is_l = d.eq(D)
    j = UOp.range(J, 2, axis_type=AxisType.REDUCE)
    e = UOp.range(E, 3, axis_type=AxisType.REDUCE)
    g = UOp.range(G, 4)
    dot = UOp.placeholder((G,), F32, 260, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 5)
    dot_init = dot.after(tile, d, j)[zi].store(0.0).end(zi)
    dot = dot.after(dot_init)
    dot_upd = dot[g].store(dot.after(e)[g] + a[((tile * G + g) * J + j) * E + e]).end(g).end(e)
    dot_f = dot.after(dot_upd)
    acc = UOp.placeholder((G,), F32, 261, addrspace=AddrSpace.REG)
    den = UOp.placeholder((G,), F32, 262, addrspace=AddrSpace.REG)
    mx = UOp.placeholder((G,), F32, 263, addrspace=AddrSpace.REG)
    za = UOp.range(G, 6)
    init = acc.after(tile, d)[za].store(0.0).end(za)
    zl = UOp.range(G, 7)
    init = den.after(init)[zl].store(0.0).end(zl)
    zm = UOp.range(G, 8)
    init = mx.after(init)[zm].store(-float("inf")).end(zm)
    acc, den, mx = acc.after(init), den.after(init), mx.after(init)
    g2 = UOp.range(G, 9)
    old_m = mx.after(j)[g2]
    sc = dot_f[g2]
    new_m = old_m.maximum(sc)
    corr = (old_m - new_m).exp2()
    p = (sc - new_m).exp2()
    vd = is_v.where(v[((tile * G + g2) * J + j) * D + is_v.where(d, d.const_like(0))], UOp.const(F32, 1.0))
    upd = acc[g2].store(acc.after(j)[g2] * corr + p * vd)
    upd = den.after(upd)[g2].store(den.after(j)[g2] * corr + p)
    upd = mx.after(upd)[g2].store(new_m).end(g2).end(j)
    go = UOp.range(G, 10)
    af, lf, mf = acc.after(upd), den.after(upd), mx.after(upd)
    val = is_v.where(af[go], is_l.where(lf[go], mf[go]))
    return out[((tile * G + go) * W + d)].store(val).end(go).end(tile, d).sink(
      arg=KernelInfo(name="synthetic_global_g_fused_tile_lifecycle", opts_to_apply=()))
  return kernel

def _synthetic_global_g_repro() -> dict[str, Any]:
  try:
    import numpy as np
    from tinygrad import Tensor, dtypes
    T, G, D, J, E = 2, 3, 4, 3, 5
    W = D + 2
    rng = np.random.default_rng(20260627)
    a = rng.normal(0, 0.25, size=(T, G, J, E)).astype(np.float32)
    v = rng.normal(0, 0.25, size=(T, G, J, D)).astype(np.float32)
    got = Tensor.empty(T * G * W, dtype=dtypes.float32).custom_kernel(
      Tensor(a.reshape(-1)), Tensor(v.reshape(-1)), fxn=_synthetic_global_g_kernel(T, G, D, J, E))[0].realize().numpy().reshape(T, G, W)
    ref = np.zeros((T, G, W), dtype=np.float32)
    for t in range(T):
      for g in range(G):
        m, l = -np.inf, np.float32(0.0)
        acc = np.zeros(D, dtype=np.float32)
        for j in range(J):
          sc = np.float32(a[t, g, j].sum())
          mn = max(m, sc)
          corr = np.exp2(np.float32(m - mn))
          p = np.exp2(np.float32(sc - mn))
          acc = acc * corr + p * v[t, g, j]
          l = l * corr + p
          m = mn
        ref[t, g, :D], ref[t, g, D], ref[t, g, D + 1] = acc, l, m
    diff = got - ref
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return {"checked": True, "compiled": True, "pass": bool(max_abs <= 1e-5), "max_abs": max_abs, "rmse": rmse,
            "shape": {"T": T, "G": G, "D": D, "J": J, "E": E, "W": W}}
  except Exception as e:
    tb = traceback.format_exc()
    if "pop from empty list" in tb and "Estimates.from_uops" in tb:
      verdict = "FUSED_TILE_LIFECYCLE_GLOBAL_G_BLOCKED__ESTIMATE_SCOPE_STACK"
    elif "UOp verification failed" in tb:
      verdict = "FUSED_TILE_LIFECYCLE_GLOBAL_G_BLOCKED__UOP_VERIFY"
    else:
      verdict = "FUSED_TILE_LIFECYCLE_GLOBAL_G_FAIL__EXCEPTION"
    return {"checked": True, "compiled": False, "pass": False, "verdict": verdict, "exception_type": type(e).__name__,
            "exception": str(e), "has_estimates_from_uops": "Estimates.from_uops" in tb,
            "has_pop_from_empty_list": "pop from empty list" in tb,
            "has_uop_verify": "UOp verification failed" in tb, "traceback_tail": tb[-5000:]}


def _minimal_repro() -> dict[str, Any]:
  synthetic = _synthetic_repro()
  global_g = _synthetic_global_g_repro()
  attn = _read_json(ATTN_BLOCKER)
  numeric = attn.get("standalone_numeric", {}) if attn.get("available") else {}
  tb = numeric.get("traceback_tail", "")
  if global_g.get("verdict") == "FUSED_TILE_LIFECYCLE_GLOBAL_G_BLOCKED__ESTIMATE_SCOPE_STACK":
    verdict = "FUSED_TILE_LIFECYCLE_BLOCKED__GLOBAL_G_ESTIMATE_SCOPE_STACK"
    classified = True
  elif global_g.get("verdict") == "FUSED_TILE_LIFECYCLE_GLOBAL_G_BLOCKED__UOP_VERIFY":
    verdict = "FUSED_TILE_LIFECYCLE_BLOCKED__GLOBAL_G_UOP_VERIFY"
    classified = True
  elif global_g.get("compiled") and global_g.get("pass"):
    verdict = "FUSED_TILE_LIFECYCLE_GLOBAL_G_NUMERIC_PASS__ATTENTION_REPRO_STRONGER"
    classified = True
  elif synthetic.get("verdict") == "FUSED_TILE_LIFECYCLE_SYNTHETIC_BLOCKED__ESTIMATE_SCOPE_STACK":
    verdict = "FUSED_TILE_LIFECYCLE_BLOCKED__SYNTHETIC_ESTIMATE_SCOPE_STACK"
    classified = True
  elif synthetic.get("compiled") and synthetic.get("pass"):
    verdict = "FUSED_TILE_LIFECYCLE_SYNTHETIC_NUMERIC_PASS__ATTENTION_REPRO_STRONGER"
    classified = True
  elif numeric.get("verdict") == "FUSED_SCORE_STATE_PV_TILE_BLOCKED__MULTI_REDUCTION_STORE_SHAPE" and "Estimates.from_uops" in tb and "pop from empty list" in tb:
    verdict = "FUSED_TILE_LIFECYCLE_BLOCKED__ESTIMATE_SCOPE_STACK"
    classified = True
  elif attn.get("available"):
    verdict = "FUSED_TILE_LIFECYCLE_BLOCKED__UNCLASSIFIED_FROM_ATTENTION_ARTIFACT"
    classified = False
  else:
    verdict = "FUSED_TILE_LIFECYCLE_REPRO_MISSING"
    classified = False
  return {
    "checked": True,
    "verdict": verdict,
    "classified": classified,
    "source": "synthetic_repro" if verdict.startswith("FUSED_TILE_LIFECYCLE_BLOCKED__SYNTHETIC") else "attention_builder_blocker_artifact",
    "synthetic": synthetic,
    "synthetic_global_g": global_g,
    "known_failure_signature": {
      "exception": numeric.get("exception"),
      "exception_type": numeric.get("exception_type"),
      "has_estimates_from_uops": "Estimates.from_uops" in tb,
      "has_pop_from_empty_list": "pop from empty list" in tb,
    },
  }


def build() -> dict[str, Any]:
  repro = _minimal_repro()
  if repro["verdict"] == "FUSED_TILE_LIFECYCLE_GLOBAL_G_NUMERIC_PASS__ATTENTION_REPRO_STRONGER":
    next_step = "Global+G synthetic lifecycle passes; next isolator should add attention-like cache indexing and Hd=128/W=130 scale."
  elif repro["verdict"] == "FUSED_TILE_LIFECYCLE_SYNTHETIC_NUMERIC_PASS__ATTENTION_REPRO_STRONGER":
    next_step = "Synthetic single-tile lifecycle passes; build the next isolator with GLOBAL tile axes and G-vector recurrence state to find the attention-specific lowering delta."
  elif repro["verdict"] in ("FUSED_TILE_LIFECYCLE_BLOCKED__GLOBAL_G_ESTIMATE_SCOPE_STACK", "FUSED_TILE_LIFECYCLE_BLOCKED__GLOBAL_G_UOP_VERIFY"):
    next_step = "Global+G synthetic repro captured the lowering wall; fix or encapsulate this axis/state pattern before returning to attention."
  elif repro["verdict"] == "FUSED_TILE_LIFECYCLE_BLOCKED__SYNTHETIC_ESTIMATE_SCOPE_STACK":
    next_step = "Synthetic repro is sufficient; fix or encapsulate the generic nested lifecycle lowering pattern."
  else:
    next_step = "Attention blocker remains the authority; reduce it further until the synthetic repro captures the same failure."
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": repro["verdict"],
    "minimal_repro": repro,
    "attention_blocker_artifact": _read_json(ATTN_BLOCKER),
    "required_lowering_capability": {
      "generic_shape": "nested reduce + recurrence tuple + local output axis + compact metadata store",
      "attention_shape": "q.k score reduce inside token recurrence with local-d PV and l/m metadata columns",
      "first_fix_target": "scope-balanced lowering/estimation for nested END scopes",
      "not_yet": ["LDS tuning", "v_dot2 lowering", "W==D promotion"],
    },
    "next_step": next_step,
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"fused-tile-lifecycle-lowering-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
