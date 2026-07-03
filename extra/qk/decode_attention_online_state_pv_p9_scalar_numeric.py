#!/usr/bin/env python3
"""P9 scalar online-state+PV tile numeric proof against NumPy."""
from __future__ import annotations

import json, math, pathlib, time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes

from extra.qk.flash_decode import (
  flash_score_whole_cache_kernel,
  flash_online_state_pv_tile_whole_cache_kernel,
  flash_state_gmax_kernel,
  flash_state_combine_kernel,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-online-state-pv-p9-scalar-numeric"
Hq, Hkv, Hd, MAXC, L = 32, 8, 128, 512, 64
CASES = (128, 130, 32, 256)
TOL = {"score": 2e-3, "m": 2e-3, "l": 2e-3, "pv": 5e-3, "out": 5e-3}


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
  if np.isnan(a).any() or np.isnan(b).any(): return float("nan")
  return float(np.max(np.abs(a - b)))


def _ref(q: np.ndarray, cache: np.ndarray, Tc: int) -> dict[str, np.ndarray]:
  G = Hq // Hkv
  Sval = math.ceil(Tc / L)
  qf = q.astype(np.float32)
  kf = cache[0].astype(np.float32)
  vf = cache[1].astype(np.float32)
  score = np.zeros((Hq, MAXC), np.float32)
  m = np.full((Hq, Sval), -np.inf, np.float32)
  l = np.zeros((Hq, Sval), np.float32)
  pv = np.zeros((Hq, Sval, Hd), np.float32)
  scale = 1.0 / math.sqrt(Hd)
  for h in range(Hq):
    kv = h // G
    score[h, :Tc] = (qf[h] @ kf[kv, :Tc].T) * scale
    for s in range(Sval):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      sc = score[h, t0:t1]
      mm = float(np.max(sc))
      pp = np.exp(sc - mm).astype(np.float32)
      m[h, s] = mm
      l[h, s] = np.sum(pp, dtype=np.float32)
      pv[h, s] = pp @ vf[kv, t0:t1]
  gm = np.max(m, axis=1)
  out = np.zeros((Hq, Hd), np.float32)
  for h in range(Hq):
    w = np.exp(m[h] - gm[h]).astype(np.float32)
    den = np.sum(w * l[h], dtype=np.float32)
    out[h] = np.sum((w[:, None] * pv[h]), axis=0, dtype=np.float32) / den
  return {"score": score, "m": m, "l": l, "pv": pv, "out": out}


def _run_generated(q: np.ndarray, cache: np.ndarray, Tc: int) -> dict[str, np.ndarray]:
  Sval = math.ceil(Tc / L)
  Smax = math.ceil(MAXC / L)
  W = Hd + 2
  q_t = Tensor(q).realize()
  cache_t = Tensor(cache).realize()
  q_f = q_t.reshape(Hq * Hd)
  cache_f = cache_t.reshape(2 * Hkv * MAXC * Hd)
  score = Tensor.empty(Hq * MAXC, dtype=dtypes.float32).custom_kernel(q_f, cache_f,
    fxn=flash_score_whole_cache_kernel(Hd, Hq, Hkv, MAXC, Tc))[0].realize()
  po = Tensor.empty(Hq * Smax * W, dtype=dtypes.float32).custom_kernel(score, cache_f,
    fxn=flash_online_state_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, Sval, Tc))[0].realize()
  gm = Tensor.empty(Hq, dtype=dtypes.float32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, Sval))[0].realize()
  out = Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(po, gm,
    fxn=flash_state_combine_kernel(Hd, Hq, Sval))[0].realize()
  po_np = po.numpy().reshape(Hq, Smax, W)[:, :Sval, :]
  return {
    "score": score.numpy().reshape(Hq, MAXC),
    "m": po_np[:, :, Hd + 1],
    "l": po_np[:, :, Hd],
    "pv": po_np[:, :, :Hd],
    "out": out.numpy().reshape(Hq, Hd),
  }


def _case(Tc: int) -> dict[str, Any]:
  rng = np.random.default_rng(9000 + Tc)
  q = (rng.standard_normal((Hq, Hd)) * 0.2).astype(np.float16)
  cache = (rng.standard_normal((2, Hkv, MAXC, Hd)) * 0.2).astype(np.float16)
  ref = _ref(q, cache, Tc)
  got = _run_generated(q, cache, Tc)
  Sval = math.ceil(Tc / L)
  active_score = got["score"][:, :Tc]
  active_m, active_l, active_pv = got["m"], got["l"], got["pv"]
  score_err = _max_abs(ref["score"][:, :Tc], active_score)
  errs = {
    "score": score_err,
    "m": _max_abs(ref["m"], active_m),
    "l": _max_abs(ref["l"], active_l),
    "pv": _max_abs(ref["pv"], active_pv),
    "out": _max_abs(ref["out"], got["out"]),
  }
  nans = {
    "score": bool(np.isnan(active_score).any()),
    "m": bool(np.isnan(active_m).any()),
    "l": bool(np.isnan(active_l).any()),
    "pv": bool(np.isnan(active_pv).any()),
    "out": bool(np.isnan(got["out"]).any()),
  }
  verdict = "PASS"
  if any(nans.values()): verdict = "FAIL__NAN"
  elif errs["score"] > TOL["score"]: verdict = "FAIL__SCORE"
  elif errs["m"] > TOL["m"]: verdict = "FAIL__M"
  elif errs["l"] > TOL["l"]: verdict = "FAIL__L"
  elif errs["pv"] > TOL["pv"]: verdict = "FAIL__PV"
  elif errs["out"] > TOL["out"]: verdict = "FAIL__OUT"
  return {"Tc": Tc, "L": L, "Sval": Sval, "verdict": verdict, "errors": errs, "nans": nans}


def build() -> dict[str, Any]:
  try:
    cases = [_case(tc) for tc in CASES]
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "ONLINE_STATE_PV_P9_FAIL__CAPTURE", "error": repr(e)}
  first = next((c for c in cases if c["verdict"] != "PASS"), None)
  if first is None:
    verdict = "ONLINE_STATE_PV_P9_NUMERIC_PASS"
  else:
    suffix = first["verdict"].replace("FAIL__", "")
    verdict = "ONLINE_STATE_PV_P9_FAIL__" + suffix
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "L": L},
    "tolerances": TOL,
    "cases": cases,
    "first_failure": first,
    "decision": "If pass, return to x-lane numeric debugging; if fail, fix scalar online-state recurrence first."
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-online-state-pv-p9-scalar-numeric-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "ONLINE_STATE_PV_P9_NUMERIC_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
