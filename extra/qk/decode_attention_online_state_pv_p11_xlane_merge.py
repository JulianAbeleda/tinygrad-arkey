#!/usr/bin/env python3
"""P11 synthetic x-lane online-softmax merge microproof."""
from __future__ import annotations

import json, pathlib, time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, UOp

from extra.qk.amd_warp_reduce import warp_reduce_max
from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-online-state-pv-p11-xlane-merge"
LANES, CASES = 32, 4


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * 1.4426950408889634).exp2()

def merge_kernel():
  def kernel(out: UOp, m_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(LANES, "lidx0")
    m = m_in[c * LANES + lane]
    l = l_in[c * LANES + lane]
    acc = acc_in[c * LANES + lane]
    gm = warp_reduce_max(m, lane, LANES, 90)
    w = _fexp(m - gm)
    den = _warp_reduce_sum_staged(l * w, lane, LANES, 96)
    num = _warp_reduce_sum_staged(acc * w, lane, LANES, 102)
    return out[c].store(num / den, lane.eq(0)).end(c).sink(arg=_fki("p11_xlane_merge_32"))
  return kernel


def _ref(m: np.ndarray, l: np.ndarray, acc: np.ndarray) -> np.ndarray:
  gm = np.max(m, axis=1)
  w = np.exp(m - gm[:, None]).astype(np.float32)
  return np.sum(acc * w, axis=1, dtype=np.float32) / np.sum(l * w, axis=1, dtype=np.float32)


def build() -> dict[str, Any]:
  rng = np.random.default_rng(11)
  m = (rng.standard_normal((CASES, LANES)) * 0.5).astype(np.float32)
  l = (rng.random((CASES, LANES)) * 2.0 + 0.1).astype(np.float32)
  acc = (rng.standard_normal((CASES, LANES)) * 0.3).astype(np.float32)
  # Include an all-but-one weak-lane case and a wide max-spread case.
  m[1, :] = -20.0; m[1, 7] = 1.25; l[1, :] = 0.1; l[1, 7] = 1.7; acc[1, :] = 0.0; acc[1, 7] = 2.5
  m[2, :] = np.linspace(-8, 8, LANES, dtype=np.float32); l[2, :] = 1.0; acc[2, :] = np.linspace(-1, 1, LANES, dtype=np.float32)
  ref = _ref(m, l, acc)
  try:
    got = Tensor.empty(CASES, dtype=dtypes.float32).custom_kernel(Tensor(m.reshape(-1)), Tensor(l.reshape(-1)), Tensor(acc.reshape(-1)),
      fxn=merge_kernel())[0].realize().numpy()
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "ONLINE_STATE_PV_P11_FAIL__CAPTURE", "error": repr(e)}
  err = float(np.max(np.abs(got - ref))) if not (np.isnan(got).any() or np.isnan(ref).any()) else float("nan")
  verdict = "ONLINE_STATE_PV_P11_MERGE_PASS" if np.isfinite(err) and err <= 2e-4 else "ONLINE_STATE_PV_P11_FAIL__MERGE"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"cases": CASES, "lanes": LANES},
    "max_abs_error": err,
    "got": got.tolist(),
    "ref": ref.tolist(),
    "has_nan": {"got": bool(np.isnan(got).any()), "ref": bool(np.isnan(ref).any())},
    "decision": "If pass, the staged x-lane merge primitive is correct; debug per-lane state generation in the full tile. If fail, fix merge primitive first."
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-online-state-pv-p11-xlane-merge-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "ONLINE_STATE_PV_P11_MERGE_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
