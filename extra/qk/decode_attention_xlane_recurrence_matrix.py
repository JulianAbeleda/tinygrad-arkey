#!/usr/bin/env python3
"""P14 recurrence matrix for decode x-lane online-state audit.

P11 proves the cross-lane merge once each lane already has (m, l, acc).
P13 proves isolated reducers and feature axes.  This file proves the missing
piece: each lane processes multiple tokens with the online softmax recurrence,
then the lanes merge.
"""
from __future__ import annotations

import json, math, pathlib, time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, UOp

from extra.qk.amd_warp_reduce import warp_reduce_max
from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-xlane-recurrence-matrix"
LANES, CASES, FEATS, R = 32, 4, 16, 3
TMAX = LANES * R
TOL = 2e-4


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * 1.4426950408889634).exp2()


def recurrence_kernel():
  def kernel(out: UOp, score_in: UOp, val_in: UOp, active_in: UOp) -> UOp:
    G = CASES * FEATS
    cidx = UOp.range(CASES, 0, AxisType.GLOBAL)
    fidx = UOp.range(FEATS, 1, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    ridx = UOp.range(R, 2, axis_type=AxisType.REDUCE)
    t = ridx * LANES + lane
    in_r = t < active_in[cidx]
    t_safe = in_r.where(t, t.const_like(0))
    sc_load = score_in[cidx * TMAX + t_safe]
    vd = val_in[(cidx * TMAX + t_safe) * FEATS + fidx]
    g = cidx * FEATS + fidx

    acc = UOp.placeholder((G,), dtypes.float32, 120, addrspace=AddrSpace.REG)
    lse = UOp.placeholder((G,), dtypes.float32, 121, addrspace=AddrSpace.REG)
    mval = UOp.placeholder((G,), dtypes.float32, 122, addrspace=AddrSpace.REG)
    zi = UOp.range(G, 3)
    init = acc[zi].store(0.0).end(zi)
    zi2 = UOp.range(G, 4)
    init = lse.after(init)[zi2].store(0.0).end(zi2)
    zi3 = UOp.range(G, 5)
    init = mval.after(init)[zi3].store(-float("inf")).end(zi3)
    acc, lse, mval = acc.after(init), lse.after(init), mval.after(init)

    old_m = mval.after(ridx)[g]
    sc = in_r.where(sc_load, old_m)
    mn = in_r.where(old_m.maximum(sc), old_m)
    corr = in_r.where(_fexp(old_m - mn), 1.0)
    p = in_r.where(_fexp(sc - mn), 0.0)
    upd = acc[g].store(acc.after(ridx)[g] * corr + p * vd)
    upd = lse.after(upd)[g].store(lse.after(ridx)[g] * corr + p)
    upd = mval.after(upd)[g].store(mn).end(ridx)
    acc, lse, mval = acc.after(upd), lse.after(upd), mval.after(upd)

    gm = warp_reduce_max(mval[g], lane, LANES, 90)
    w = _fexp(mval[g] - gm)
    num = _warp_reduce_sum_staged(acc[g] * w, lane, LANES, 96)
    den = _warp_reduce_sum_staged(lse[g] * w, lane, LANES, 102)
    return out[cidx * FEATS + fidx].store(num / den, lane.eq(0)).end(cidx, fidx).sink(arg=_fki("p14_xlane_recurrence"))
  return kernel


def _data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(14)
  score = (rng.standard_normal((CASES, TMAX)) * 0.6).astype(np.float32)
  val = (rng.standard_normal((CASES, TMAX, FEATS)) * 0.3).astype(np.float32)
  active = np.array([TMAX, 70, 33, 1], dtype=np.int32)
  score[1, :] = np.linspace(-4, 4, TMAX, dtype=np.float32)
  val[2, :, :] = np.linspace(-1, 1, TMAX, dtype=np.float32)[:, None]
  val[3, :, :] = 0.0
  val[3, 0, :] = np.linspace(-0.5, 0.5, FEATS, dtype=np.float32)
  return score, val, active


def _ref(score: np.ndarray, val: np.ndarray, active: np.ndarray) -> np.ndarray:
  out = np.empty((CASES, FEATS), dtype=np.float32)
  for c in range(CASES):
    sc = score[c, :active[c]].astype(np.float32)
    p = np.exp(sc - np.max(sc)).astype(np.float32)
    p /= np.sum(p, dtype=np.float32)
    out[c] = p @ val[c, :active[c]].astype(np.float32)
  return out


def build() -> dict[str, Any]:
  score, val, active = _data()
  ref = _ref(score, val, active)
  try:
    got = Tensor.empty(CASES * FEATS, dtype=dtypes.float32).custom_kernel(
      Tensor(score.reshape(-1)), Tensor(val.reshape(-1)), Tensor(active), fxn=recurrence_kernel()
    )[0].realize().numpy().reshape(CASES, FEATS)
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "XLANE_RECURRENCE_MATRIX_FAIL__CAPTURE", "error": repr(e)}
  err = float(np.max(np.abs(got - ref))) if not (np.isnan(got).any() or np.isnan(ref).any()) else float("nan")
  verdict = "XLANE_RECURRENCE_MATRIX_PASS" if np.isfinite(err) and err <= TOL else "XLANE_RECURRENCE_MATRIX_FAIL__OUTPUT"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"cases": CASES, "features": FEATS, "lanes": LANES, "tokens_per_lane": R, "tmax": TMAX},
    "active_tokens": active.tolist(),
    "tolerance": TOL,
    "max_abs_error": err,
    "got": got.tolist(),
    "ref": ref.tolist(),
    "has_nan": {"got": bool(np.isnan(got).any()), "ref": bool(np.isnan(ref).any())},
    "decision": "If pass, the x-lane recurrence primitive is sound; investigate full-route indexing/layout. If fail, fix recurrence update ordering."
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-xlane-recurrence-matrix-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "XLANE_RECURRENCE_MATRIX_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
