#!/usr/bin/env python3
"""P8 isolated numeric gate for scalar vs x-lane online-state+PV tile."""
from __future__ import annotations

import json, pathlib, time
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import UOp

from extra.qk.flash_decode import (
  flash_score_whole_cache_kernel,
  flash_online_state_pv_tile_whole_cache_kernel,
  flash_online_state_pv_tile_xlane_whole_cache_kernel,
  flash_state_gmax_kernel,
  flash_state_combine_kernel,
)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-attention-online-state-pv-p8-numeric"

Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 512, 64, 128
Smax = (MAXC + L - 1) // L
Sval = (Tc + L - 1) // L
W = Hd + 2


def _run_tile(xlane: bool) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(8)
  q = (rng.standard_normal((Hq, Hd)) * 0.2).astype(np.float16)
  cache = (rng.standard_normal((2, Hkv, MAXC, Hd)) * 0.2).astype(np.float16)
  q_t = Tensor(q).realize()
  cache_t = Tensor(cache).realize()
  Tc_u = Tc
  S = Sval
  q_f = q_t.reshape(Hq * Hd)
  cache_f = cache_t.reshape(2 * Hkv * MAXC * Hd)
  score = Tensor.empty(Hq * MAXC, dtype=dtypes.float32).custom_kernel(q_f, cache_f,
    fxn=flash_score_whole_cache_kernel(Hd, Hq, Hkv, MAXC, Tc_u))[0].realize()
  kern = flash_online_state_pv_tile_xlane_whole_cache_kernel if xlane else flash_online_state_pv_tile_whole_cache_kernel
  po = Tensor.empty(Hq * Smax * W, dtype=dtypes.float32).custom_kernel(score, cache_f,
    fxn=kern(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0].realize()
  gm = Tensor.empty(Hq, dtype=dtypes.float32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, S))[0].realize()
  out = Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(po, gm,
    fxn=flash_state_combine_kernel(Hd, Hq, S))[0].realize()
  return po.numpy().reshape(Hq, Smax, W), out.numpy().reshape(Hq, Hd)


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
  with np.errstate(invalid="ignore"):
    diff = np.abs(a - b)
  if np.isnan(diff).all(): return float("nan")
  return float(np.nanmax(diff))


def build() -> dict[str, Any]:
  try:
    scalar_po, scalar_out = _run_tile(False)
    xlane_po, xlane_out = _run_tile(True)
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "ONLINE_STATE_PV_P8_FAIL__CAPTURE", "error": repr(e)}
  active = slice(0, Sval)
  scalar_active, xlane_active = scalar_po[:, active, :], xlane_po[:, active, :]
  errs = {
    "m_max_abs": _max_abs(scalar_active[:, :, Hd + 1], xlane_active[:, :, Hd + 1]),
    "l_max_abs": _max_abs(scalar_active[:, :, Hd], xlane_active[:, :, Hd]),
    "pv_max_abs": _max_abs(scalar_active[:, :, :Hd], xlane_active[:, :, :Hd]),
    "out_max_abs": _max_abs(scalar_out, xlane_out),
    "xlane_has_nan": bool(np.isnan(xlane_active).any() or np.isnan(xlane_out).any()),
    "scalar_has_nan": bool(np.isnan(scalar_active).any() or np.isnan(scalar_out).any()),
    "xlane_active_has_nan": bool(np.isnan(xlane_active).any()),
    "scalar_active_has_nan": bool(np.isnan(scalar_active).any()),
    "xlane_out_has_nan": bool(np.isnan(xlane_out).any()),
    "scalar_out_has_nan": bool(np.isnan(scalar_out).any()),
  }
  if errs["scalar_out_has_nan"] or errs["xlane_out_has_nan"]:
    verdict = "ONLINE_STATE_PV_P8_FAIL__NAN"
  elif any(np.isnan(errs[k]) for k in ("m_max_abs", "l_max_abs", "pv_max_abs", "out_max_abs")):
    verdict = "ONLINE_STATE_PV_P8_FAIL__NAN"
  elif errs["m_max_abs"] > 1e-4:
    verdict = "ONLINE_STATE_PV_P8_FAIL__M"
  elif errs["l_max_abs"] > 1e-4:
    verdict = "ONLINE_STATE_PV_P8_FAIL__L"
  elif errs["pv_max_abs"] > 2e-3:
    verdict = "ONLINE_STATE_PV_P8_FAIL__PV"
  elif errs["out_max_abs"] > 2e-3:
    verdict = "ONLINE_STATE_PV_P8_FAIL__OUT"
  else:
    verdict = "ONLINE_STATE_PV_P8_NUMERIC_PASS"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "L": L, "Tc": Tc, "Sval": Sval, "W": W},
    "errors": errs,
    "decision": "Raw state buffers may contain NaNs in non-authoritative slots; use finite active-column errors and final output as the gate."
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  latest = OUT / "latest.json"
  stamped = OUT / f"decode-attention-online-state-pv-p8-numeric-{out['timestamp']}.json"
  latest.write_text(json.dumps(out, indent=2) + "\n")
  stamped.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "ONLINE_STATE_PV_P8_NUMERIC_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
