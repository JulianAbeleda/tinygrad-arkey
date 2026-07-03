#!/usr/bin/env python3
"""P12 component tests for x-lane merge primitives."""
from __future__ import annotations

import json, pathlib, time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from extra.qk.amd_warp_reduce import warp_reduce_max
from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-online-state-pv-p12-xlane-components"
LANES, CASES = 32, 4


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * 1.4426950408889634).exp2()

def component_kernel():
  def kernel(out: UOp, m_in: UOp, x_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    col = UOp.range(4, 1, AxisType.GLOBAL)
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
    val = col.eq(0).where(gm, col.eq(1).where(sx, col.eq(2).where(den, num / den)))
    return out[c * 4 + col].store(val, lane.eq(0)).end(c, col).sink(arg=_fki("p12_xlane_components_32"))
  return kernel


def _inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(12)
  m = (rng.standard_normal((CASES, LANES)) * 0.5).astype(np.float32)
  x = (rng.standard_normal((CASES, LANES)) * 0.25).astype(np.float32)
  l = (rng.random((CASES, LANES)) * 2.0 + 0.1).astype(np.float32)
  acc = (rng.standard_normal((CASES, LANES)) * 0.3).astype(np.float32)
  m[1, :] = -20.0; m[1, 7] = 1.25; l[1, :] = 0.1; l[1, 7] = 1.7; acc[1, :] = 0.0; acc[1, 7] = 2.5
  m[2, :] = np.linspace(-8, 8, LANES, dtype=np.float32); l[2, :] = 1.0; acc[2, :] = np.linspace(-1, 1, LANES, dtype=np.float32)
  return m, x, l, acc


def build() -> dict[str, Any]:
  m, x, l, acc = _inputs()
  gm = np.max(m, axis=1)
  sx = np.sum(x, axis=1, dtype=np.float32)
  w = np.exp(m - gm[:, None]).astype(np.float32)
  den = np.sum(l * w, axis=1, dtype=np.float32)
  out = np.sum(acc * w, axis=1, dtype=np.float32) / den
  ref = np.stack([gm, sx, den, out], axis=1).astype(np.float32)
  try:
    got = Tensor.empty(CASES * 4, dtype=dtypes.float32).custom_kernel(
      Tensor(m.reshape(-1)), Tensor(x.reshape(-1)), Tensor(l.reshape(-1)), Tensor(acc.reshape(-1)), fxn=component_kernel())[0].realize().numpy().reshape(CASES, 4)
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "ONLINE_STATE_PV_P12_FAIL__CAPTURE", "error": repr(e)}
  errs = {
    "max": float(np.max(np.abs(got[:, 0] - ref[:, 0]))),
    "sum": float(np.max(np.abs(got[:, 1] - ref[:, 1]))),
    "den": float(np.max(np.abs(got[:, 2] - ref[:, 2]))),
    "lse": float(np.max(np.abs(got[:, 3] - ref[:, 3]))),
  }
  if np.isnan(got).any(): verdict = "ONLINE_STATE_PV_P12_FAIL__NAN"
  elif errs["max"] > 2e-4: verdict = "ONLINE_STATE_PV_P12_FAIL__MAX"
  elif errs["sum"] > 2e-4: verdict = "ONLINE_STATE_PV_P12_FAIL__SUM"
  elif errs["den"] > 2e-4: verdict = "ONLINE_STATE_PV_P12_FAIL__DEN"
  elif errs["lse"] > 2e-4: verdict = "ONLINE_STATE_PV_P12_FAIL__LSE"
  else: verdict = "ONLINE_STATE_PV_P12_COMPONENTS_PASS"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"cases": CASES, "lanes": LANES, "columns": ["max", "sum", "den", "lse"]},
    "errors": errs,
    "got": got.tolist(),
    "ref": ref.tolist(),
    "has_nan": bool(np.isnan(got).any()),
    "decision": "If max/sum pass but den/lse fail, fix formula inputs; if max/sum fail, replace staged helper usage for this route."
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-online-state-pv-p12-xlane-components-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "ONLINE_STATE_PV_P12_COMPONENTS_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
