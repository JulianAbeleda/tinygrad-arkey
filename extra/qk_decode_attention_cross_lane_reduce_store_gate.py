#!/usr/bin/env python3
"""Cross-lane reduction/store microgate for generated decode attention.

This is the reducer-first gate required before any new FusedScorePVLifecycle attention route.
It isolates the P11/P12 wall below the full attention tile:

  synthetic per-lane (m, l, acc) -> warp max/sum/LSE -> lane==0 gated store

The result classifies the fork:
- reducer math/lowering failure: fix cross-lane lowering, do not write attention routes.
- multi-store contract failure: fix generated store/AFTER contract, do not write attention routes.
- pass: fused tile work is expressible enough to continue.
"""
from __future__ import annotations

import json, pathlib, time
from typing import Any, Callable

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, UOp

from extra.amd_warp_reduce import warp_reduce_max
from extra.qk_warp_reduce_lowering import _warp_reduce_sum_staged

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-cross-lane-reduce-store"
LANES, CASES = 32, 4
TOL = 2e-4


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * 1.4426950408889634).exp2()


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(20260626)
  m = (rng.standard_normal((CASES, LANES)) * 0.5).astype(np.float32)
  x = (rng.standard_normal((CASES, LANES)) * 0.25).astype(np.float32)
  l = (rng.random((CASES, LANES)) * 2.0 + 0.1).astype(np.float32)
  acc = (rng.standard_normal((CASES, LANES)) * 0.3).astype(np.float32)
  m[1, :] = -20.0; m[1, 7] = 1.25; l[1, :] = 0.1; l[1, 7] = 1.7; acc[1, :] = 0.0; acc[1, 7] = 2.5
  m[2, :] = np.linspace(-8, 8, LANES, dtype=np.float32); l[2, :] = 1.0; acc[2, :] = np.linspace(-1, 1, LANES, dtype=np.float32)
  return m, x, l, acc


def _refs(m: np.ndarray, x: np.ndarray, l: np.ndarray, acc: np.ndarray) -> dict[str, np.ndarray]:
  gm = np.max(m, axis=1).astype(np.float32)
  sx = np.sum(x, axis=1, dtype=np.float32).astype(np.float32)
  w = np.exp(m - gm[:, None]).astype(np.float32)
  den = np.sum(l * w, axis=1, dtype=np.float32).astype(np.float32)
  lse = (np.sum(acc * w, axis=1, dtype=np.float32) / den).astype(np.float32)
  return {"max": gm, "sum": sx, "den": den, "lse": lse, "multi": np.stack([gm, sx, den, lse], axis=1).reshape(-1)}


def _component_kernel(kind: str):
  def kernel(out: UOp, m_in: UOp, x_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    m = m_in[c * LANES + lane]
    x = x_in[c * LANES + lane]
    l = l_in[c * LANES + lane]
    acc = acc_in[c * LANES + lane]
    gm = warp_reduce_max(m, lane, LANES, 90)
    sx = _warp_reduce_sum_staged(x, lane, LANES, 96)
    w = _fexp(m - gm)
    den = _warp_reduce_sum_staged(l * w, lane, LANES, 102)
    num = _warp_reduce_sum_staged(acc * w, lane, LANES, 108)
    if kind == "max": val = gm
    elif kind == "sum": val = sx
    elif kind == "den": val = den
    elif kind == "lse": val = num / den
    else: raise ValueError(kind)
    return out[c].store(val, lane.eq(0)).end(c).sink(arg=_fki(f"cross_lane_reduce_store_{kind}_32"))
  return kernel


def _multi_store_kernel():
  def kernel(out: UOp, m_in: UOp, x_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    m = m_in[c * LANES + lane]
    x = x_in[c * LANES + lane]
    l = l_in[c * LANES + lane]
    acc = acc_in[c * LANES + lane]
    gm = warp_reduce_max(m, lane, LANES, 120)
    sx = _warp_reduce_sum_staged(x, lane, LANES, 126)
    w = _fexp(m - gm)
    den = _warp_reduce_sum_staged(l * w, lane, LANES, 132)
    num = _warp_reduce_sum_staged(acc * w, lane, LANES, 138)
    gate = lane.eq(0)
    s0 = out[c * 4 + 0].store(gm, gate)
    s1 = out.after(s0)[c * 4 + 1].store(sx, gate)
    s2 = out.after(s1)[c * 4 + 2].store(den, gate)
    s3 = out.after(s2)[c * 4 + 3].store(num / den, gate)
    return s3.end(c).sink(arg=_fki("cross_lane_reduce_store_multi_32"))
  return kernel


def _run_vec(name: str, nout: int, fxn: Callable, ref: np.ndarray, inputs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, Any]:
  m, x, l, acc = inputs
  try:
    got = Tensor.empty(nout, dtype=dtypes.float32).custom_kernel(
      Tensor(m.reshape(-1)), Tensor(x.reshape(-1)), Tensor(l.reshape(-1)), Tensor(acc.reshape(-1)), fxn=fxn)[0].realize().numpy()
  except Exception as e:  # noqa: BLE001
    return {"name": name, "pass": False, "failure_class": "capture_or_verify", "error": repr(e)}
  err = float(np.max(np.abs(got - ref))) if not (np.isnan(got).any() or np.isnan(ref).any()) else float("nan")
  return {"name": name, "pass": bool(np.isfinite(err) and err <= TOL), "max_abs_error": err,
          "has_nan": bool(np.isnan(got).any()), "got": got.tolist(), "ref": ref.tolist()}


def build() -> dict[str, Any]:
  inputs = _inputs()
  refs = _refs(*inputs)
  rows = [
    _run_vec("max", CASES, _component_kernel("max"), refs["max"], inputs),
    _run_vec("sum", CASES, _component_kernel("sum"), refs["sum"], inputs),
    _run_vec("den", CASES, _component_kernel("den"), refs["den"], inputs),
    _run_vec("lse", CASES, _component_kernel("lse"), refs["lse"], inputs),
    _run_vec("multi_store", CASES * 4, _multi_store_kernel(), refs["multi"], inputs),
  ]
  by_name = {r["name"]: r for r in rows}
  if not by_name["max"].get("pass"): verdict = "CROSS_LANE_REDUCE_STORE_FAIL__MAX"
  elif not by_name["sum"].get("pass"): verdict = "CROSS_LANE_REDUCE_STORE_FAIL__SUM"
  elif not by_name["den"].get("pass"): verdict = "CROSS_LANE_REDUCE_STORE_FAIL__DEN"
  elif not by_name["lse"].get("pass"): verdict = "CROSS_LANE_REDUCE_STORE_FAIL__LSE"
  elif not by_name["multi_store"].get("pass"):
    verdict = "CROSS_LANE_REDUCE_STORE_FAIL__STORE_CONTRACT" if by_name["multi_store"].get("failure_class") else "CROSS_LANE_REDUCE_STORE_FAIL__MULTISTORE_NUMERIC"
  else: verdict = "CROSS_LANE_REDUCE_STORE_PASS"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict,
          "shape": {"cases": CASES, "lanes": LANES, "tolerance": TOL}, "rows": rows,
          "decision": "If PASS, FusedScorePVLifecycle may proceed. If FAIL, stop attention-route work and classify SEARCH_BLOCKED_BY_CODEGEN on CrossLaneReduceStore."}


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"cross-lane-reduce-store-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "CROSS_LANE_REDUCE_STORE_PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
