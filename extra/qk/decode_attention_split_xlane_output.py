#!/usr/bin/env python3
"""P15 isolated final-output numeric gate for split-state x-lane decode attention."""
from __future__ import annotations

import json, math, pathlib, time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes

from extra.qk.flash_decode import (
  flash_score_whole_cache_kernel,
  flash_online_state_pv_tile_whole_cache_kernel,
  flash_online_state_pv_tile_xlane_whole_cache_kernel,
  flash_state_gmax_kernel,
  flash_state_combine_kernel,
  flash_max_kernel,
  flash_xlane_split_m_kernel,
  flash_xlane_pv_from_m_kernel,
  flash_gmax_kernel,
  flash_den_kernel,
  flash_combine_kernel,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-split-xlane-output"
Hq, Hkv, Hd, MAXC, L = 32, 8, 128, 512, 64
CASES = (128, 130, 32, 256)
TOL = 5e-3


def _ref(q: np.ndarray, cache: np.ndarray, Tc: int) -> np.ndarray:
  G = Hq // Hkv
  qf = q.astype(np.float32)
  kf = cache[0].astype(np.float32)
  vf = cache[1].astype(np.float32)
  out = np.zeros((Hq, Hd), np.float32)
  scale = 1.0 / math.sqrt(Hd)
  for h in range(Hq):
    kv = h // G
    sc = (qf[h] @ kf[kv, :Tc].T) * scale
    pp = np.exp(sc - np.max(sc)).astype(np.float32)
    pp /= np.sum(pp, dtype=np.float32)
    out[h] = pp @ vf[kv, :Tc]
  return out


def _ref_split(q: np.ndarray, cache: np.ndarray, Tc: int) -> tuple[np.ndarray, np.ndarray]:
  G = Hq // Hkv
  Sval = math.ceil(Tc / L)
  qf = q.astype(np.float32)
  kf = cache[0].astype(np.float32)
  vf = cache[1].astype(np.float32)
  state = np.zeros((Hq, Sval, 2), np.float32)
  pv = np.zeros((Hq, Sval, Hd), np.float32)
  scale = 1.0 / math.sqrt(Hd)
  for h in range(Hq):
    kv = h // G
    for s in range(Sval):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      sc = (qf[h] @ kf[kv, t0:t1].T) * scale
      m = np.max(sc).astype(np.float32)
      p = np.exp(sc - m).astype(np.float32)
      state[h, s, 0] = np.sum(p, dtype=np.float32)
      state[h, s, 1] = m
      pv[h, s] = p @ vf[kv, t0:t1]
  return state, pv


def _run(q: np.ndarray, cache: np.ndarray, Tc: int, mode: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
  Sval = math.ceil(Tc / L)
  Smax = math.ceil(MAXC / L)
  W = Hd + 2
  q_t = Tensor(q).realize()
  cache_t = Tensor(cache).realize()
  q_f = q_t.reshape(Hq * Hd)
  cache_f = cache_t.reshape(2 * Hkv * MAXC * Hd)
  score = Tensor.empty(Hq * MAXC, dtype=dtypes.float32).custom_kernel(q_f, cache_f,
    fxn=flash_score_whole_cache_kernel(Hd, Hq, Hkv, MAXC, Tc))[0].realize()
  if mode == "split_xlane":
    W2 = Hd + 1
    pm = Tensor.empty(Hq * Smax, dtype=dtypes.float32).custom_kernel(score,
      fxn=flash_max_kernel(Hq, MAXC, L, Sval, Tc))[0].realize()
    po = Tensor.empty(Hq * Smax * W2, dtype=dtypes.float32).custom_kernel(pm, score, cache_f,
      fxn=flash_xlane_pv_from_m_kernel(Hd, Hq, Hkv, MAXC, L, Sval, Tc))[0].realize()
    gm = Tensor.empty(Hq, dtype=dtypes.float32).custom_kernel(pm, fxn=flash_gmax_kernel(Hq, Sval))[0].realize()
    dn = Tensor.empty(Hq, dtype=dtypes.float32).custom_kernel(po, pm, gm, fxn=flash_den_kernel(Hd, Hq, Sval))[0].realize()
    out = Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(po, pm, gm, dn,
      fxn=flash_combine_kernel(Hd, Hq, Sval))[0].realize()
    po_np = po.numpy()[:Hq * Sval * W2].reshape(Hq, Sval, W2)
    aux = {"pm": pm.numpy()[:Hq * Sval].reshape(Hq, Sval), "pl": po_np[:, :, Hd], "pv": po_np[:, :, :Hd]}
  else:
    kern = flash_online_state_pv_tile_xlane_whole_cache_kernel if mode == "xlane" else flash_online_state_pv_tile_whole_cache_kernel
    po = Tensor.empty(Hq * Smax * W, dtype=dtypes.float32).custom_kernel(score, cache_f,
      fxn=kern(Hd, Hq, Hkv, MAXC, L, Sval, Tc))[0].realize()
    gm = Tensor.empty(Hq, dtype=dtypes.float32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, Sval))[0].realize()
    out = Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(po, gm,
      fxn=flash_state_combine_kernel(Hd, Hq, Sval))[0].realize()
    aux = {"po": po.numpy().reshape(Hq, Smax, W)}
  return out.numpy().reshape(Hq, Hd), aux


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
  if np.isnan(a).any() or np.isnan(b).any(): return float("nan")
  return float(np.max(np.abs(a - b)))


def _case(Tc: int) -> dict[str, Any]:
  rng = np.random.default_rng(15000 + Tc)
  q = (rng.standard_normal((Hq, Hd)) * 0.2).astype(np.float16)
  cache = (rng.standard_normal((2, Hkv, MAXC, Hd)) * 0.2).astype(np.float16)
  ref = _ref(q, cache, Tc)
  ref_state, ref_pv = _ref_split(q, cache, Tc)
  scalar, scalar_aux = _run(q, cache, Tc, "scalar")
  xlane, _xlane_aux = _run(q, cache, Tc, "xlane")
  split, split_aux = _run(q, cache, Tc, "split_xlane")
  split_pm = split_aux["pm"]
  split_pl = split_aux["pl"]
  split_pv = split_aux["pv"]
  errs = {
    "scalar_vs_ref": _max_abs(scalar, ref),
    "xlane_vs_ref": _max_abs(xlane, ref),
    "split_vs_ref": _max_abs(split, ref),
    "split_vs_scalar": _max_abs(split, scalar),
    "split_vs_xlane": _max_abs(split, xlane),
    "split_m_vs_ref": _max_abs(split_pm, ref_state[:, :, 1]),
    "split_l_vs_ref": _max_abs(split_pl, ref_state[:, :, 0]),
    "split_pv_vs_ref": _max_abs(split_pv, ref_pv),
  }
  nans = {"scalar": bool(np.isnan(scalar).any()), "xlane": bool(np.isnan(xlane).any()),
          "split": bool(np.isnan(split).any()), "ref": bool(np.isnan(ref).any())}
  if any(nans.values()): verdict = "FAIL__NAN"
  elif errs["scalar_vs_ref"] > TOL: verdict = "FAIL__SCALAR_REF"
  elif errs["xlane_vs_ref"] > TOL: verdict = "FAIL__XLANE_REF"
  elif errs["split_vs_ref"] > TOL: verdict = "FAIL__SPLIT_REF"
  elif errs["split_vs_scalar"] > TOL: verdict = "FAIL__SPLIT_SCALAR"
  else: verdict = "PASS"
  return {"Tc": Tc, "L": L, "Sval": math.ceil(Tc / L), "verdict": verdict, "errors": errs, "nans": nans}


def build() -> dict[str, Any]:
  try:
    cases = [_case(tc) for tc in CASES]
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "SPLIT_XLANE_OUTPUT_FAIL__CAPTURE", "error": repr(e)}
  first = next((c for c in cases if c["verdict"] != "PASS"), None)
  verdict = "SPLIT_XLANE_OUTPUT_PASS" if first is None else "SPLIT_XLANE_OUTPUT_" + first["verdict"]
  return {
    "date": "2026-06-26",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "L": L},
    "tolerance": TOL,
    "cases": cases,
    "first_failure": first,
    "decision": "If pass, run route gate and W==D; if fail, debug split state/PV indexing."
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-split-xlane-output-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "SPLIT_XLANE_OUTPUT_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
