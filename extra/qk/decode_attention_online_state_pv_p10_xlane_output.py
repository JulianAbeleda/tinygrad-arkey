#!/usr/bin/env python3
"""P10 isolated x-lane final-output numeric gate."""
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
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-online-state-pv-p10-xlane-output"
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


def _run(q: np.ndarray, cache: np.ndarray, Tc: int, xlane: bool) -> np.ndarray:
  Sval = math.ceil(Tc / L)
  Smax = math.ceil(MAXC / L)
  W = Hd + 2
  q_t = Tensor(q).realize()
  cache_t = Tensor(cache).realize()
  q_f = q_t.reshape(Hq * Hd)
  cache_f = cache_t.reshape(2 * Hkv * MAXC * Hd)
  score = Tensor.empty(Hq * MAXC, dtype=dtypes.float32).custom_kernel(q_f, cache_f,
    fxn=flash_score_whole_cache_kernel(Hd, Hq, Hkv, MAXC, Tc))[0].realize()
  kern = flash_online_state_pv_tile_xlane_whole_cache_kernel if xlane else flash_online_state_pv_tile_whole_cache_kernel
  po = Tensor.empty(Hq * Smax * W, dtype=dtypes.float32).custom_kernel(score, cache_f,
    fxn=kern(Hd, Hq, Hkv, MAXC, L, Sval, Tc))[0].realize()
  gm = Tensor.empty(Hq, dtype=dtypes.float32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, Sval))[0].realize()
  out = Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(po, gm,
    fxn=flash_state_combine_kernel(Hd, Hq, Sval))[0].realize()
  return out.numpy().reshape(Hq, Hd)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
  if np.isnan(a).any() or np.isnan(b).any(): return float("nan")
  return float(np.max(np.abs(a - b)))


def _case(Tc: int) -> dict[str, Any]:
  rng = np.random.default_rng(10000 + Tc)
  q = (rng.standard_normal((Hq, Hd)) * 0.2).astype(np.float16)
  cache = (rng.standard_normal((2, Hkv, MAXC, Hd)) * 0.2).astype(np.float16)
  ref = _ref(q, cache, Tc)
  scalar = _run(q, cache, Tc, False)
  xlane = _run(q, cache, Tc, True)
  errs = {
    "scalar_vs_ref": _max_abs(scalar, ref),
    "xlane_vs_ref": _max_abs(xlane, ref),
    "xlane_vs_scalar": _max_abs(xlane, scalar),
  }
  nans = {"scalar": bool(np.isnan(scalar).any()), "xlane": bool(np.isnan(xlane).any()), "ref": bool(np.isnan(ref).any())}
  if any(nans.values()): verdict = "FAIL__NAN"
  elif errs["scalar_vs_ref"] > TOL: verdict = "FAIL__SCALAR_REF"
  elif errs["xlane_vs_ref"] > TOL: verdict = "FAIL__XLANE_REF"
  elif errs["xlane_vs_scalar"] > TOL: verdict = "FAIL__XLANE_SCALAR"
  else: verdict = "PASS"
  return {"Tc": Tc, "L": L, "Sval": math.ceil(Tc / L), "verdict": verdict, "errors": errs, "nans": nans}


def build() -> dict[str, Any]:
  try:
    cases = [_case(tc) for tc in CASES]
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "ONLINE_STATE_PV_P10_FAIL__CAPTURE", "error": repr(e)}
  first = next((c for c in cases if c["verdict"] != "PASS"), None)
  if first is None: verdict = "ONLINE_STATE_PV_P10_XLANE_OUTPUT_PASS"
  else: verdict = "ONLINE_STATE_PV_P10_" + first["verdict"]
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "L": L},
    "tolerance": TOL,
    "cases": cases,
    "first_failure": first,
    "decision": "If pass, rerun P7 in-model; if fail, debug x-lane final-output merge."
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-online-state-pv-p10-xlane-output-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "ONLINE_STATE_PV_P10_XLANE_OUTPUT_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
