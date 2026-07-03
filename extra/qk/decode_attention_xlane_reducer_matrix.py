#!/usr/bin/env python3
"""P13 reducer matrix for decode x-lane audit.

This isolates whether the online-state x-lane decode failure is caused by the
decode formula or by the generated warp-reducer shape.  Each arm keeps the same
global case axis + lidx0 lane binding used by the failing decode route, then
adds one stressor at a time.
"""
from __future__ import annotations

import json, pathlib, time
from collections.abc import Callable
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from extra.qk.amd_warp_reduce import warp_reduce_max
from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-xlane-reducer-matrix"
LANES, CASES, FEATS = 32, 5, 16


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * 1.4426950408889634).exp2()


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(13)
  m = (rng.standard_normal((CASES, LANES)) * 0.7).astype(np.float32)
  x = (rng.standard_normal((CASES, LANES)) * 0.25).astype(np.float32)
  xf = (rng.standard_normal((CASES, FEATS, LANES)) * 0.25).astype(np.float32)
  l = (rng.random((CASES, LANES)) * 2.0 + 0.05).astype(np.float32)
  acc = (rng.standard_normal((CASES, LANES)) * 0.35).astype(np.float32)
  active = np.array([32, 17, 9, 1, 29], dtype=np.int32)
  m[1, :] = -20.0; m[1, 7] = 1.5
  x[3, :] = 0.0; x[3, 0] = 2.0
  m[4, :] = np.linspace(-8, 8, LANES, dtype=np.float32)
  l[4, :] = 1.0
  acc[4, :] = np.linspace(-1, 1, LANES, dtype=np.float32)
  xf[4, :, :] = np.linspace(-1, 1, LANES, dtype=np.float32)
  return m, x, xf, l, acc, active


def _max_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    m = m_in[c * LANES + lane]
    gm = warp_reduce_max(m, lane, LANES)
    return out[c].store(gm, lane.eq(0)).end(c).sink(arg=_fki("p13_xlane_max_only"))
  return kernel


def _sum_kernel() -> Callable:
  def kernel(out: UOp, x_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    sx = _warp_reduce_sum_staged(x_in[c * LANES + lane], lane, LANES)
    return out[c].store(sx, lane.eq(0)).end(c).sink(arg=_fki("p13_xlane_sum_only"))
  return kernel


def _masked_sum_kernel() -> Callable:
  def kernel(out: UOp, x_in: UOp, active_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    keep = lane < active_in[c]
    x = keep.where(x_in[c * LANES + lane], 0.0)
    sx = _warp_reduce_sum_staged(x, lane, LANES)
    return out[c].store(sx, lane.eq(0)).end(c).sink(arg=_fki("p13_xlane_masked_sum"))
  return kernel


def _feature_sum_kernel() -> Callable:
  def kernel(out: UOp, xf_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    f = UOp.range(FEATS, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    x = xf_in[(c * FEATS + f) * LANES + lane]
    sx = _warp_reduce_sum_staged(x, lane, LANES, 90)
    return out[c * FEATS + f].store(sx, lane.eq(0)).end(c, f).sink(arg=_fki("p13_xlane_feature_sum"))
  return kernel


def _column_select_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp, x_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    col = UOp.range(2, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    gm = warp_reduce_max(m_in[c * LANES + lane], lane, LANES, 90)
    sx = _warp_reduce_sum_staged(x_in[c * LANES + lane], lane, LANES, 96)
    val = col.eq(0).where(gm, sx)
    return out[c * 2 + col].store(val, lane.eq(0)).end(c, col).sink(arg=_fki("p13_xlane_column_select"))
  return kernel


def _max_column_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    col = UOp.range(2, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    gm = warp_reduce_max(m_in[c * LANES + lane], lane, LANES, 90)
    return out[c * 2 + col].store(gm, lane.eq(0)).end(c, col).sink(arg=_fki("p13_xlane_max_column"))
  return kernel


def _weighted_ratio_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    m = m_in[c * LANES + lane]
    gm = warp_reduce_max(m, lane, LANES, 90)
    w = _fexp(m - gm)
    den = _warp_reduce_sum_staged(l_in[c * LANES + lane] * w, lane, LANES, 96)
    num = _warp_reduce_sum_staged(acc_in[c * LANES + lane] * w, lane, LANES, 102)
    return out[c].store(num / den, lane.eq(0)).end(c).sink(arg=_fki("p13_xlane_weighted_ratio"))
  return kernel


def _weighted_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    col = UOp.range(4, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    m = m_in[c * LANES + lane]
    gm = warp_reduce_max(m, lane, LANES, 90)
    w = _fexp(m - gm)
    den = _warp_reduce_sum_staged(l_in[c * LANES + lane] * w, lane, LANES, 96)
    num = _warp_reduce_sum_staged(acc_in[c * LANES + lane] * w, lane, LANES, 102)
    val = col.eq(0).where(gm, col.eq(1).where(den, col.eq(2).where(num, num / den)))
    return out[c * 4 + col].store(val, lane.eq(0)).end(c, col).sink(arg=_fki("p13_xlane_weighted"))
  return kernel


def _run_arm(name: str, got: Callable[[], np.ndarray], ref: np.ndarray) -> dict[str, Any]:
  try:
    arr = got().astype(np.float32)
    err = np.abs(arr - ref.astype(np.float32))
    return {
      "name": name,
      "verdict": "PASS" if bool(np.all(np.isfinite(arr))) and float(np.max(err)) <= 2e-4 else "FAIL",
      "max_error": float(np.max(err)),
      "has_nan": bool(np.isnan(arr).any()),
      "got": arr.tolist(),
      "ref": ref.astype(np.float32).tolist(),
    }
  except Exception as e:  # noqa: BLE001
    return {"name": name, "verdict": "CAPTURE_FAIL", "error": repr(e)}


def build() -> dict[str, Any]:
  m, x, xf, l, acc, active = _data()
  refs = {
    "max_only": np.max(m, axis=1),
    "sum_only": np.sum(x, axis=1, dtype=np.float32),
    "masked_sum": np.array([np.sum(x[i, :active[i]], dtype=np.float32) for i in range(CASES)], dtype=np.float32),
    "feature_sum": np.sum(xf, axis=2, dtype=np.float32),
    "column_select": np.stack([np.max(m, axis=1), np.sum(x, axis=1, dtype=np.float32)], axis=1),
    "max_column": np.stack([np.max(m, axis=1), np.max(m, axis=1)], axis=1),
  }
  gm = np.max(m, axis=1)
  w = np.exp(m - gm[:, None]).astype(np.float32)
  den = np.sum(l * w, axis=1, dtype=np.float32)
  num = np.sum(acc * w, axis=1, dtype=np.float32)
  refs["weighted_ratio"] = (num / den).astype(np.float32)
  refs["weighted"] = np.stack([gm, den, num, num / den], axis=1).astype(np.float32)

  mt = Tensor(m.reshape(-1))
  xt = Tensor(x.reshape(-1))
  xft = Tensor(xf.reshape(-1))
  lt = Tensor(l.reshape(-1))
  acct = Tensor(acc.reshape(-1))
  at = Tensor(active)
  arms = [
    _run_arm("max_only", lambda: Tensor.empty(CASES, dtype=dtypes.float32).custom_kernel(mt, fxn=_max_kernel())[0].realize().numpy(), refs["max_only"]),
    _run_arm("sum_only", lambda: Tensor.empty(CASES, dtype=dtypes.float32).custom_kernel(xt, fxn=_sum_kernel())[0].realize().numpy(), refs["sum_only"]),
    _run_arm("masked_sum", lambda: Tensor.empty(CASES, dtype=dtypes.float32).custom_kernel(xt, at, fxn=_masked_sum_kernel())[0].realize().numpy(), refs["masked_sum"]),
    _run_arm("feature_sum", lambda: Tensor.empty(CASES * FEATS, dtype=dtypes.float32).custom_kernel(xft, fxn=_feature_sum_kernel())[0].realize().numpy().reshape(CASES, FEATS), refs["feature_sum"]),
    _run_arm("column_select", lambda: Tensor.empty(CASES * 2, dtype=dtypes.float32).custom_kernel(mt, xt, fxn=_column_select_kernel())[0].realize().numpy().reshape(CASES, 2), refs["column_select"]),
    _run_arm("max_column", lambda: Tensor.empty(CASES * 2, dtype=dtypes.float32).custom_kernel(mt, fxn=_max_column_kernel())[0].realize().numpy().reshape(CASES, 2), refs["max_column"]),
    _run_arm("weighted_ratio", lambda: Tensor.empty(CASES, dtype=dtypes.float32).custom_kernel(mt, lt, acct, fxn=_weighted_ratio_kernel())[0].realize().numpy(), refs["weighted_ratio"]),
    _run_arm("weighted", lambda: Tensor.empty(CASES * 4, dtype=dtypes.float32).custom_kernel(mt, lt, acct, fxn=_weighted_kernel())[0].realize().numpy().reshape(CASES, 4), refs["weighted"]),
  ]
  hard_fails = [a for a in arms if a["verdict"] != "PASS"]
  if not hard_fails: verdict = "XLANE_REDUCER_MATRIX_PASS"
  elif any(a["name"] in ("max_only", "sum_only") and a["verdict"] != "PASS" for a in arms):
    verdict = "XLANE_REDUCER_MATRIX_FAIL__BASIC_REDUCER"
  elif any(a["name"] == "masked_sum" and a["verdict"] != "PASS" for a in arms):
    verdict = "XLANE_REDUCER_MATRIX_FAIL__MASK"
  elif any(a["name"] == "feature_sum" and a["verdict"] != "PASS" for a in arms):
    verdict = "XLANE_REDUCER_MATRIX_FAIL__FEATURE_AXIS"
  elif any(a["name"] == "column_select" and a["verdict"] != "PASS" for a in arms):
    verdict = "XLANE_REDUCER_MATRIX_FAIL__COLUMN_AXIS"
  elif any(a["name"] == "max_column" and a["verdict"] != "PASS" for a in arms):
    verdict = "XLANE_REDUCER_MATRIX_FAIL__COLUMN_AXIS"
  elif any(a["name"] == "weighted_ratio" and a["verdict"] != "PASS" for a in arms):
    verdict = "XLANE_REDUCER_MATRIX_FAIL__WEIGHTED_COMBINE"
  else:
    verdict = "XLANE_REDUCER_MATRIX_FAIL__WEIGHTED_COMBINE"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"cases": CASES, "lanes": LANES, "features": FEATS},
    "arms": arms,
    "diagnosis": {
      "basic_reducer_fail": "warp_reduce_max or _warp_reduce_sum_staged is wrong even without masks/columns/formula.",
      "mask_fail": "zeroing inactive lanes before the reducer is not preserved.",
      "feature_axis_fail": "a GLOBAL output-feature axis whose input depends on the feature changes reducer semantics.",
      "column_axis_fail": "adding a second GLOBAL output column changes reducer semantics or store placement.",
      "weighted_fail": "basic reducers pass, but softmax-style weight/denominator composition fails."
    }
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-xlane-reducer-matrix-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "XLANE_REDUCER_MATRIX_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
