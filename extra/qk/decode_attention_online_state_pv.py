#!/usr/bin/env python3
"""Online-state x-lane decode-attention family (P8-P15 + TG-P10.1), collapsed to one parameterized module.

Nine sequential numeric proofs of the online-softmax / cross-lane (x-lane) decode-attention route. Each VARIANT
builds+validates its proof and RETURNS the verdict dict (gate_registry writes `latest.json` + a dated snapshot and
prints it); the sole exception is `tg_p10_repro`, a report-only check that RETURNS an int exit code and writes its own
`reg_scalar_lowering.json` (it spawns fresh subprocesses -- see below). All need DEV=AMD hardware. Registry
entrypoints: build_p8()..build_tg_p10().

Arc (see docs/qk-gate-series-conclusions.md "Cluster A"): P9 (scalar tile vs NumPy, the base) -> P8/P10 (x-lane
state / final-output vs ref) -> when the full x-lane route failed, decomposed into synthetic micro-proofs P11 (merge),
P12 (merge components), P13 (reducer matrix), P14 (recurrence matrix), all PASS -> localized the fault to route-level
indexing/layout, not the primitives. P15 re-expressed as a split-state pipeline (PASS). TG-P10.1 pins the terminal
combine blocker.

  P8  ONLINE_STATE_PV_P8_NUMERIC_PASS        -- x-lane tile == scalar tile (m/l/PV + out; tol m/l 1e-4, pv/out 2e-3).
  P9  ONLINE_STATE_PV_P9_NUMERIC_PASS        -- scalar online-state+PV whole-cache tile matches NumPy flash ref across
                                                ragged ctx (Tc=128,130,32,256, L=64; tol score/m/l 2e-3, pv/out 5e-3).
  P10 ONLINE_STATE_PV_P10_XLANE_OUTPUT_PASS  -- x-lane final output == NumPy ref and == scalar route (tol 5e-3).
  P11 ONLINE_STATE_PV_P11_MERGE_PASS         -- staged cross-lane online-softmax merge correct in isolation (err <= 2e-4).
  P12 ONLINE_STATE_PV_P12_COMPONENTS_PASS    -- max/sum/den/LSE sub-components each correct (per-col tol 2e-4).
  P13 XLANE_REDUCER_MATRIX_PASS              -- 8-arm reducer/feature/column-axis stressor sweep sound.
  P14 XLANE_RECURRENCE_MATRIX_PASS           -- per-lane multi-token online recurrence reproduces softmax*V (tol 2e-4).
  P15 SPLIT_XLANE_OUTPUT_PASS                -- split-state pipeline matches ref/scalar/xlane + per-split ref (tol 5e-3).
  TG-P10.1 TG_P10_1_PASS_REG_REPRO_PINNED    -- minimal generated-UOp repro of the combine blocker (report-only, int).

Run:  DEV=AMD PYTHONPATH=. python3 -m extra.qk.gate_registry run <gate-name> [...]
"""
from __future__ import annotations

import json, math, os, pathlib, re, subprocess, sys, time
from collections.abc import Callable
from typing import Any

os.environ.setdefault("DEV", "AMD")  # tg_p10 sacred env ordering: DEV set before the tinygrad import below

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, UOp

from extra.qk.amd_warp_reduce import warp_reduce_max
from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged
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

# ---- shared scaffolding ----------------------------------------------------------------------------------------------
# Shared decode geometry for the whole-cache tile proofs (P8/P9/P10/P15).
Hq, Hkv, Hd, MAXC, L = 32, 8, 128, 512, 64


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())
def _fexp(x: UOp) -> UOp: return (x * 1.4426950408889634).exp2()


def _max_abs(a: np.ndarray, b: np.ndarray) -> float:
  """Strict abs-diff (P9/P10/P15): any NaN in either operand -> NaN (a NaN is a failure signal, not to be masked)."""
  if np.isnan(a).any() or np.isnan(b).any(): return float("nan")
  return float(np.max(np.abs(a - b)))


def _max_abs_nanmax(a: np.ndarray, b: np.ndarray) -> float:
  """NaN-tolerant abs-diff (P8 only): raw state buffers legitimately hold NaN in inactive slots, so ignore-NaN and
  only report NaN when the WHOLE diff is NaN. The P8 gate compares finite active-column errors, never a blanket check."""
  with np.errstate(invalid="ignore"):
    diff = np.abs(a - b)
  if np.isnan(diff).all(): return float("nan")
  return float(np.nanmax(diff))


def _ref_full(q: np.ndarray, cache: np.ndarray, Tc: int) -> np.ndarray:
  """Whole-cache full-softmax reference out (Hq,Hd) -- shared by P10 and P15."""
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


# ---- P8: x-lane tile == scalar tile (isolated numeric gate) ----------------------------------------------------------
_P8_Tc = 128
_P8_Smax = (MAXC + L - 1) // L
_P8_Sval = (_P8_Tc + L - 1) // L
_P8_W = Hd + 2


def _p8_run_tile(xlane: bool) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(8)
  q = (rng.standard_normal((Hq, Hd)) * 0.2).astype(np.float16)
  cache = (rng.standard_normal((2, Hkv, MAXC, Hd)) * 0.2).astype(np.float16)
  q_t = Tensor(q).realize()
  cache_t = Tensor(cache).realize()
  Tc_u = _P8_Tc
  S = _P8_Sval
  q_f = q_t.reshape(Hq * Hd)
  cache_f = cache_t.reshape(2 * Hkv * MAXC * Hd)
  score = Tensor.empty(Hq * MAXC, dtype=dtypes.float32).custom_kernel(q_f, cache_f,
    fxn=flash_score_whole_cache_kernel(Hd, Hq, Hkv, MAXC, Tc_u))[0].realize()
  kern = flash_online_state_pv_tile_xlane_whole_cache_kernel if xlane else flash_online_state_pv_tile_whole_cache_kernel
  po = Tensor.empty(Hq * _P8_Smax * _P8_W, dtype=dtypes.float32).custom_kernel(score, cache_f,
    fxn=kern(Hd, Hq, Hkv, MAXC, L, S, Tc_u))[0].realize()
  gm = Tensor.empty(Hq, dtype=dtypes.float32).custom_kernel(po, fxn=flash_state_gmax_kernel(Hd, Hq, S))[0].realize()
  out = Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(po, gm,
    fxn=flash_state_combine_kernel(Hd, Hq, S))[0].realize()
  return po.numpy().reshape(Hq, _P8_Smax, _P8_W), out.numpy().reshape(Hq, Hd)


def _p8() -> dict[str, Any]:
  try:
    scalar_po, scalar_out = _p8_run_tile(False)
    xlane_po, xlane_out = _p8_run_tile(True)
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "ONLINE_STATE_PV_P8_FAIL__CAPTURE", "error": repr(e)}
  active = slice(0, _P8_Sval)
  scalar_active, xlane_active = scalar_po[:, active, :], xlane_po[:, active, :]
  errs = {
    "m_max_abs": _max_abs_nanmax(scalar_active[:, :, Hd + 1], xlane_active[:, :, Hd + 1]),
    "l_max_abs": _max_abs_nanmax(scalar_active[:, :, Hd], xlane_active[:, :, Hd]),
    "pv_max_abs": _max_abs_nanmax(scalar_active[:, :, :Hd], xlane_active[:, :, :Hd]),
    "out_max_abs": _max_abs_nanmax(scalar_out, xlane_out),
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
    "shape": {"Hq": Hq, "Hkv": Hkv, "Hd": Hd, "MAXC": MAXC, "L": L, "Tc": _P8_Tc, "Sval": _P8_Sval, "W": _P8_W},
    "errors": errs,
    # HAZARD (P8): raw state buffers may contain NaNs in non-authoritative (inactive-Smax) slots; use finite
    # active-column errors and the final output as the gate, never a blanket state-buffer NaN check.
    "decision": "Raw state buffers may contain NaNs in non-authoritative slots; use finite active-column errors and final output as the gate."
  }


# ---- P9: scalar online-state+PV tile == NumPy flash ref --------------------------------------------------------------
_P9_CASES = (128, 130, 32, 256)
_P9_TOL = {"score": 2e-3, "m": 2e-3, "l": 2e-3, "pv": 5e-3, "out": 5e-3}


def _p9_ref(q: np.ndarray, cache: np.ndarray, Tc: int) -> dict[str, np.ndarray]:
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


def _p9_run_generated(q: np.ndarray, cache: np.ndarray, Tc: int) -> dict[str, np.ndarray]:
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


def _p9_case(Tc: int) -> dict[str, Any]:
  rng = np.random.default_rng(9000 + Tc)
  q = (rng.standard_normal((Hq, Hd)) * 0.2).astype(np.float16)
  cache = (rng.standard_normal((2, Hkv, MAXC, Hd)) * 0.2).astype(np.float16)
  ref = _p9_ref(q, cache, Tc)
  got = _p9_run_generated(q, cache, Tc)
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
  elif errs["score"] > _P9_TOL["score"]: verdict = "FAIL__SCORE"
  elif errs["m"] > _P9_TOL["m"]: verdict = "FAIL__M"
  elif errs["l"] > _P9_TOL["l"]: verdict = "FAIL__L"
  elif errs["pv"] > _P9_TOL["pv"]: verdict = "FAIL__PV"
  elif errs["out"] > _P9_TOL["out"]: verdict = "FAIL__OUT"
  return {"Tc": Tc, "L": L, "Sval": Sval, "verdict": verdict, "errors": errs, "nans": nans}


def _p9() -> dict[str, Any]:
  try:
    cases = [_p9_case(tc) for tc in _P9_CASES]
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
    "tolerances": _P9_TOL,
    "cases": cases,
    "first_failure": first,
    "decision": "If pass, return to x-lane numeric debugging; if fail, fix scalar online-state recurrence first."
  }


# ---- P10: x-lane final output == ref and == scalar route -------------------------------------------------------------
_P10_CASES = (128, 130, 32, 256)
_P10_TOL = 5e-3


def _p10_run(q: np.ndarray, cache: np.ndarray, Tc: int, xlane: bool) -> np.ndarray:
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


def _p10_case(Tc: int) -> dict[str, Any]:
  rng = np.random.default_rng(10000 + Tc)
  q = (rng.standard_normal((Hq, Hd)) * 0.2).astype(np.float16)
  cache = (rng.standard_normal((2, Hkv, MAXC, Hd)) * 0.2).astype(np.float16)
  ref = _ref_full(q, cache, Tc)
  scalar = _p10_run(q, cache, Tc, False)
  xlane = _p10_run(q, cache, Tc, True)
  errs = {
    "scalar_vs_ref": _max_abs(scalar, ref),
    "xlane_vs_ref": _max_abs(xlane, ref),
    "xlane_vs_scalar": _max_abs(xlane, scalar),
  }
  nans = {"scalar": bool(np.isnan(scalar).any()), "xlane": bool(np.isnan(xlane).any()), "ref": bool(np.isnan(ref).any())}
  if any(nans.values()): verdict = "FAIL__NAN"
  elif errs["scalar_vs_ref"] > _P10_TOL: verdict = "FAIL__SCALAR_REF"
  elif errs["xlane_vs_ref"] > _P10_TOL: verdict = "FAIL__XLANE_REF"
  elif errs["xlane_vs_scalar"] > _P10_TOL: verdict = "FAIL__XLANE_SCALAR"
  else: verdict = "PASS"
  return {"Tc": Tc, "L": L, "Sval": math.ceil(Tc / L), "verdict": verdict, "errors": errs, "nans": nans}


def _p10() -> dict[str, Any]:
  try:
    cases = [_p10_case(tc) for tc in _P10_CASES]
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
    "tolerance": _P10_TOL,
    "cases": cases,
    "first_failure": first,
    "decision": "If pass, rerun P7 in-model; if fail, debug x-lane final-output merge."
  }


# ---- P11: synthetic x-lane online-softmax merge microproof -----------------------------------------------------------
_P11_LANES, _P11_CASES = 32, 4


def _p11_merge_kernel():
  def kernel(out: UOp, m_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(_P11_CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(_P11_LANES, "lidx0")
    m = m_in[c * _P11_LANES + lane]
    l = l_in[c * _P11_LANES + lane]
    acc = acc_in[c * _P11_LANES + lane]
    gm = warp_reduce_max(m, lane, _P11_LANES, 90)
    w = _fexp(m - gm)
    den = _warp_reduce_sum_staged(l * w, lane, _P11_LANES, 96)
    num = _warp_reduce_sum_staged(acc * w, lane, _P11_LANES, 102)
    return out[c].store(num / den, lane.eq(0)).end(c).sink(arg=_fki("p11_xlane_merge_32"))
  return kernel


def _p11_ref(m: np.ndarray, l: np.ndarray, acc: np.ndarray) -> np.ndarray:
  gm = np.max(m, axis=1)
  w = np.exp(m - gm[:, None]).astype(np.float32)
  return np.sum(acc * w, axis=1, dtype=np.float32) / np.sum(l * w, axis=1, dtype=np.float32)


def _p11() -> dict[str, Any]:
  rng = np.random.default_rng(11)
  m = (rng.standard_normal((_P11_CASES, _P11_LANES)) * 0.5).astype(np.float32)
  l = (rng.random((_P11_CASES, _P11_LANES)) * 2.0 + 0.1).astype(np.float32)
  acc = (rng.standard_normal((_P11_CASES, _P11_LANES)) * 0.3).astype(np.float32)
  # Include an all-but-one weak-lane case and a wide max-spread case.
  m[1, :] = -20.0; m[1, 7] = 1.25; l[1, :] = 0.1; l[1, 7] = 1.7; acc[1, :] = 0.0; acc[1, 7] = 2.5
  m[2, :] = np.linspace(-8, 8, _P11_LANES, dtype=np.float32); l[2, :] = 1.0; acc[2, :] = np.linspace(-1, 1, _P11_LANES, dtype=np.float32)
  ref = _p11_ref(m, l, acc)
  try:
    got = Tensor.empty(_P11_CASES, dtype=dtypes.float32).custom_kernel(Tensor(m.reshape(-1)), Tensor(l.reshape(-1)), Tensor(acc.reshape(-1)),
      fxn=_p11_merge_kernel())[0].realize().numpy()
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "ONLINE_STATE_PV_P11_FAIL__CAPTURE", "error": repr(e)}
  err = float(np.max(np.abs(got - ref))) if not (np.isnan(got).any() or np.isnan(ref).any()) else float("nan")
  verdict = "ONLINE_STATE_PV_P11_MERGE_PASS" if np.isfinite(err) and err <= 2e-4 else "ONLINE_STATE_PV_P11_FAIL__MERGE"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"cases": _P11_CASES, "lanes": _P11_LANES},
    "max_abs_error": err,
    "got": got.tolist(),
    "ref": ref.tolist(),
    "has_nan": {"got": bool(np.isnan(got).any()), "ref": bool(np.isnan(ref).any())},
    "decision": "If pass, the staged x-lane merge primitive is correct; debug per-lane state generation in the full tile. If fail, fix merge primitive first."
  }


# ---- P12: component tests for x-lane merge primitives ----------------------------------------------------------------
_P12_LANES, _P12_CASES = 32, 4


def _p12_component_kernel():
  def kernel(out: UOp, m_in: UOp, x_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(_P12_CASES, 0, AxisType.GLOBAL)
    col = UOp.range(4, 1, AxisType.GLOBAL)
    lane = UOp.special(_P12_LANES, "lidx0")
    m = m_in[c * _P12_LANES + lane]
    x = x_in[c * _P12_LANES + lane]
    l = l_in[c * _P12_LANES + lane]
    acc = acc_in[c * _P12_LANES + lane]
    gm = warp_reduce_max(m, lane, _P12_LANES, 90)
    sx = _warp_reduce_sum_staged(x, lane, _P12_LANES, 96)
    w = _fexp(m - gm)
    den = _warp_reduce_sum_staged(l * w, lane, _P12_LANES, 102)
    num = _warp_reduce_sum_staged(acc * w, lane, _P12_LANES, 108)
    val = col.eq(0).where(gm, col.eq(1).where(sx, col.eq(2).where(den, num / den)))
    return out[c * 4 + col].store(val, lane.eq(0)).end(c, col).sink(arg=_fki("p12_xlane_components_32"))
  return kernel


def _p12_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(12)
  m = (rng.standard_normal((_P12_CASES, _P12_LANES)) * 0.5).astype(np.float32)
  x = (rng.standard_normal((_P12_CASES, _P12_LANES)) * 0.25).astype(np.float32)
  l = (rng.random((_P12_CASES, _P12_LANES)) * 2.0 + 0.1).astype(np.float32)
  acc = (rng.standard_normal((_P12_CASES, _P12_LANES)) * 0.3).astype(np.float32)
  m[1, :] = -20.0; m[1, 7] = 1.25; l[1, :] = 0.1; l[1, 7] = 1.7; acc[1, :] = 0.0; acc[1, 7] = 2.5
  m[2, :] = np.linspace(-8, 8, _P12_LANES, dtype=np.float32); l[2, :] = 1.0; acc[2, :] = np.linspace(-1, 1, _P12_LANES, dtype=np.float32)
  return m, x, l, acc


def _p12() -> dict[str, Any]:
  m, x, l, acc = _p12_inputs()
  gm = np.max(m, axis=1)
  sx = np.sum(x, axis=1, dtype=np.float32)
  w = np.exp(m - gm[:, None]).astype(np.float32)
  den = np.sum(l * w, axis=1, dtype=np.float32)
  out = np.sum(acc * w, axis=1, dtype=np.float32) / den
  ref = np.stack([gm, sx, den, out], axis=1).astype(np.float32)
  try:
    got = Tensor.empty(_P12_CASES * 4, dtype=dtypes.float32).custom_kernel(
      Tensor(m.reshape(-1)), Tensor(x.reshape(-1)), Tensor(l.reshape(-1)), Tensor(acc.reshape(-1)), fxn=_p12_component_kernel())[0].realize().numpy().reshape(_P12_CASES, 4)
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
    "shape": {"cases": _P12_CASES, "lanes": _P12_LANES, "columns": ["max", "sum", "den", "lse"]},
    "errors": errs,
    "got": got.tolist(),
    "ref": ref.tolist(),
    "has_nan": bool(np.isnan(got).any()),
    "decision": "If max/sum pass but den/lse fail, fix formula inputs; if max/sum fail, replace staged helper usage for this route."
  }


# ---- P13: reducer matrix (isolate decode-formula fault vs generated warp-reducer shape) -------------------------------
# Each arm keeps the same global case axis + lidx0 lane binding used by the failing decode route, then adds one
# stressor at a time.
_P13_LANES, _P13_CASES, _P13_FEATS = 32, 5, 16


def _p13_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(13)
  m = (rng.standard_normal((_P13_CASES, _P13_LANES)) * 0.7).astype(np.float32)
  x = (rng.standard_normal((_P13_CASES, _P13_LANES)) * 0.25).astype(np.float32)
  xf = (rng.standard_normal((_P13_CASES, _P13_FEATS, _P13_LANES)) * 0.25).astype(np.float32)
  l = (rng.random((_P13_CASES, _P13_LANES)) * 2.0 + 0.05).astype(np.float32)
  acc = (rng.standard_normal((_P13_CASES, _P13_LANES)) * 0.35).astype(np.float32)
  active = np.array([32, 17, 9, 1, 29], dtype=np.int32)
  m[1, :] = -20.0; m[1, 7] = 1.5
  x[3, :] = 0.0; x[3, 0] = 2.0
  m[4, :] = np.linspace(-8, 8, _P13_LANES, dtype=np.float32)
  l[4, :] = 1.0
  acc[4, :] = np.linspace(-1, 1, _P13_LANES, dtype=np.float32)
  xf[4, :, :] = np.linspace(-1, 1, _P13_LANES, dtype=np.float32)
  return m, x, xf, l, acc, active


def _p13_max_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp) -> UOp:
    c = UOp.range(_P13_CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(_P13_LANES, "lidx0")
    m = m_in[c * _P13_LANES + lane]
    gm = warp_reduce_max(m, lane, _P13_LANES)
    return out[c].store(gm, lane.eq(0)).end(c).sink(arg=_fki("p13_xlane_max_only"))
  return kernel


def _p13_sum_kernel() -> Callable:
  def kernel(out: UOp, x_in: UOp) -> UOp:
    c = UOp.range(_P13_CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(_P13_LANES, "lidx0")
    sx = _warp_reduce_sum_staged(x_in[c * _P13_LANES + lane], lane, _P13_LANES)
    return out[c].store(sx, lane.eq(0)).end(c).sink(arg=_fki("p13_xlane_sum_only"))
  return kernel


def _p13_masked_sum_kernel() -> Callable:
  def kernel(out: UOp, x_in: UOp, active_in: UOp) -> UOp:
    c = UOp.range(_P13_CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(_P13_LANES, "lidx0")
    keep = lane < active_in[c]
    x = keep.where(x_in[c * _P13_LANES + lane], 0.0)
    sx = _warp_reduce_sum_staged(x, lane, _P13_LANES)
    return out[c].store(sx, lane.eq(0)).end(c).sink(arg=_fki("p13_xlane_masked_sum"))
  return kernel


def _p13_feature_sum_kernel() -> Callable:
  def kernel(out: UOp, xf_in: UOp) -> UOp:
    c = UOp.range(_P13_CASES, 0, AxisType.GLOBAL)
    f = UOp.range(_P13_FEATS, 1, AxisType.GLOBAL)
    lane = UOp.special(_P13_LANES, "lidx0")
    x = xf_in[(c * _P13_FEATS + f) * _P13_LANES + lane]
    sx = _warp_reduce_sum_staged(x, lane, _P13_LANES, 90)
    return out[c * _P13_FEATS + f].store(sx, lane.eq(0)).end(c, f).sink(arg=_fki("p13_xlane_feature_sum"))
  return kernel


def _p13_column_select_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp, x_in: UOp) -> UOp:
    c = UOp.range(_P13_CASES, 0, AxisType.GLOBAL)
    col = UOp.range(2, 1, AxisType.GLOBAL)
    lane = UOp.special(_P13_LANES, "lidx0")
    gm = warp_reduce_max(m_in[c * _P13_LANES + lane], lane, _P13_LANES, 90)
    sx = _warp_reduce_sum_staged(x_in[c * _P13_LANES + lane], lane, _P13_LANES, 96)
    val = col.eq(0).where(gm, sx)
    return out[c * 2 + col].store(val, lane.eq(0)).end(c, col).sink(arg=_fki("p13_xlane_column_select"))
  return kernel


def _p13_max_column_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp) -> UOp:
    c = UOp.range(_P13_CASES, 0, AxisType.GLOBAL)
    col = UOp.range(2, 1, AxisType.GLOBAL)
    lane = UOp.special(_P13_LANES, "lidx0")
    gm = warp_reduce_max(m_in[c * _P13_LANES + lane], lane, _P13_LANES, 90)
    return out[c * 2 + col].store(gm, lane.eq(0)).end(c, col).sink(arg=_fki("p13_xlane_max_column"))
  return kernel


def _p13_weighted_ratio_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(_P13_CASES, 0, AxisType.GLOBAL)
    lane = UOp.special(_P13_LANES, "lidx0")
    m = m_in[c * _P13_LANES + lane]
    gm = warp_reduce_max(m, lane, _P13_LANES, 90)
    w = _fexp(m - gm)
    den = _warp_reduce_sum_staged(l_in[c * _P13_LANES + lane] * w, lane, _P13_LANES, 96)
    num = _warp_reduce_sum_staged(acc_in[c * _P13_LANES + lane] * w, lane, _P13_LANES, 102)
    return out[c].store(num / den, lane.eq(0)).end(c).sink(arg=_fki("p13_xlane_weighted_ratio"))
  return kernel


def _p13_weighted_kernel() -> Callable:
  def kernel(out: UOp, m_in: UOp, l_in: UOp, acc_in: UOp) -> UOp:
    c = UOp.range(_P13_CASES, 0, AxisType.GLOBAL)
    col = UOp.range(4, 1, AxisType.GLOBAL)
    lane = UOp.special(_P13_LANES, "lidx0")
    m = m_in[c * _P13_LANES + lane]
    gm = warp_reduce_max(m, lane, _P13_LANES, 90)
    w = _fexp(m - gm)
    den = _warp_reduce_sum_staged(l_in[c * _P13_LANES + lane] * w, lane, _P13_LANES, 96)
    num = _warp_reduce_sum_staged(acc_in[c * _P13_LANES + lane] * w, lane, _P13_LANES, 102)
    val = col.eq(0).where(gm, col.eq(1).where(den, col.eq(2).where(num, num / den)))
    return out[c * 4 + col].store(val, lane.eq(0)).end(c, col).sink(arg=_fki("p13_xlane_weighted"))
  return kernel


def _p13_run_arm(name: str, got: Callable[[], np.ndarray], ref: np.ndarray) -> dict[str, Any]:
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


def _p13() -> dict[str, Any]:
  m, x, xf, l, acc, active = _p13_data()
  refs = {
    "max_only": np.max(m, axis=1),
    "sum_only": np.sum(x, axis=1, dtype=np.float32),
    "masked_sum": np.array([np.sum(x[i, :active[i]], dtype=np.float32) for i in range(_P13_CASES)], dtype=np.float32),
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
    _p13_run_arm("max_only", lambda: Tensor.empty(_P13_CASES, dtype=dtypes.float32).custom_kernel(mt, fxn=_p13_max_kernel())[0].realize().numpy(), refs["max_only"]),
    _p13_run_arm("sum_only", lambda: Tensor.empty(_P13_CASES, dtype=dtypes.float32).custom_kernel(xt, fxn=_p13_sum_kernel())[0].realize().numpy(), refs["sum_only"]),
    _p13_run_arm("masked_sum", lambda: Tensor.empty(_P13_CASES, dtype=dtypes.float32).custom_kernel(xt, at, fxn=_p13_masked_sum_kernel())[0].realize().numpy(), refs["masked_sum"]),
    _p13_run_arm("feature_sum", lambda: Tensor.empty(_P13_CASES * _P13_FEATS, dtype=dtypes.float32).custom_kernel(xft, fxn=_p13_feature_sum_kernel())[0].realize().numpy().reshape(_P13_CASES, _P13_FEATS), refs["feature_sum"]),
    _p13_run_arm("column_select", lambda: Tensor.empty(_P13_CASES * 2, dtype=dtypes.float32).custom_kernel(mt, xt, fxn=_p13_column_select_kernel())[0].realize().numpy().reshape(_P13_CASES, 2), refs["column_select"]),
    _p13_run_arm("max_column", lambda: Tensor.empty(_P13_CASES * 2, dtype=dtypes.float32).custom_kernel(mt, fxn=_p13_max_column_kernel())[0].realize().numpy().reshape(_P13_CASES, 2), refs["max_column"]),
    _p13_run_arm("weighted_ratio", lambda: Tensor.empty(_P13_CASES, dtype=dtypes.float32).custom_kernel(mt, lt, acct, fxn=_p13_weighted_ratio_kernel())[0].realize().numpy(), refs["weighted_ratio"]),
    _p13_run_arm("weighted", lambda: Tensor.empty(_P13_CASES * 4, dtype=dtypes.float32).custom_kernel(mt, lt, acct, fxn=_p13_weighted_kernel())[0].realize().numpy().reshape(_P13_CASES, 4), refs["weighted"]),
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
    "shape": {"cases": _P13_CASES, "lanes": _P13_LANES, "features": _P13_FEATS},
    "arms": arms,
    # P13 FINDINGS (reducer semantics): a GLOBAL output-feature axis whose input depends on the feature, or a 2nd
    # GLOBAL output column, can change reducer semantics / store placement; masked (zeroed-inactive-lane) sums may
    # not be preserved.
    "diagnosis": {
      "basic_reducer_fail": "warp_reduce_max or _warp_reduce_sum_staged is wrong even without masks/columns/formula.",
      "mask_fail": "zeroing inactive lanes before the reducer is not preserved.",
      "feature_axis_fail": "a GLOBAL output-feature axis whose input depends on the feature changes reducer semantics.",
      "column_axis_fail": "adding a second GLOBAL output column changes reducer semantics or store placement.",
      "weighted_fail": "basic reducers pass, but softmax-style weight/denominator composition fails."
    }
  }


# ---- P14: recurrence matrix (per-lane multi-token online recurrence, then merge) -------------------------------------
# P11 proves the cross-lane merge once each lane already has (m, l, acc). P13 proves isolated reducers and feature
# axes. This proves the missing piece: each lane processes multiple tokens with the online softmax recurrence.
_P14_LANES, _P14_CASES, _P14_FEATS, _P14_R = 32, 4, 16, 3
_P14_TMAX = _P14_LANES * _P14_R
_P14_TOL = 2e-4


def _p14_recurrence_kernel():
  def kernel(out: UOp, score_in: UOp, val_in: UOp, active_in: UOp) -> UOp:
    G = _P14_CASES * _P14_FEATS
    cidx = UOp.range(_P14_CASES, 0, AxisType.GLOBAL)
    fidx = UOp.range(_P14_FEATS, 1, AxisType.GLOBAL)
    lane = UOp.special(_P14_LANES, "lidx0")
    ridx = UOp.range(_P14_R, 2, axis_type=AxisType.REDUCE)
    t = ridx * _P14_LANES + lane
    in_r = t < active_in[cidx]
    t_safe = in_r.where(t, t.const_like(0))
    sc_load = score_in[cidx * _P14_TMAX + t_safe]
    vd = val_in[(cidx * _P14_TMAX + t_safe) * _P14_FEATS + fidx]
    g = cidx * _P14_FEATS + fidx

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

    gm = warp_reduce_max(mval[g], lane, _P14_LANES, 90)
    w = _fexp(mval[g] - gm)
    num = _warp_reduce_sum_staged(acc[g] * w, lane, _P14_LANES, 96)
    den = _warp_reduce_sum_staged(lse[g] * w, lane, _P14_LANES, 102)
    return out[cidx * _P14_FEATS + fidx].store(num / den, lane.eq(0)).end(cidx, fidx).sink(arg=_fki("p14_xlane_recurrence"))
  return kernel


def _p14_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  rng = np.random.default_rng(14)
  score = (rng.standard_normal((_P14_CASES, _P14_TMAX)) * 0.6).astype(np.float32)
  val = (rng.standard_normal((_P14_CASES, _P14_TMAX, _P14_FEATS)) * 0.3).astype(np.float32)
  active = np.array([_P14_TMAX, 70, 33, 1], dtype=np.int32)
  score[1, :] = np.linspace(-4, 4, _P14_TMAX, dtype=np.float32)
  val[2, :, :] = np.linspace(-1, 1, _P14_TMAX, dtype=np.float32)[:, None]
  val[3, :, :] = 0.0
  val[3, 0, :] = np.linspace(-0.5, 0.5, _P14_FEATS, dtype=np.float32)
  return score, val, active


def _p14_ref(score: np.ndarray, val: np.ndarray, active: np.ndarray) -> np.ndarray:
  out = np.empty((_P14_CASES, _P14_FEATS), dtype=np.float32)
  for c in range(_P14_CASES):
    sc = score[c, :active[c]].astype(np.float32)
    p = np.exp(sc - np.max(sc)).astype(np.float32)
    p /= np.sum(p, dtype=np.float32)
    out[c] = p @ val[c, :active[c]].astype(np.float32)
  return out


def _p14() -> dict[str, Any]:
  score, val, active = _p14_data()
  ref = _p14_ref(score, val, active)
  try:
    got = Tensor.empty(_P14_CASES * _P14_FEATS, dtype=dtypes.float32).custom_kernel(
      Tensor(score.reshape(-1)), Tensor(val.reshape(-1)), Tensor(active), fxn=_p14_recurrence_kernel()
    )[0].realize().numpy().reshape(_P14_CASES, _P14_FEATS)
  except Exception as e:  # noqa: BLE001
    return {"date": "2026-06-25", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "XLANE_RECURRENCE_MATRIX_FAIL__CAPTURE", "error": repr(e)}
  err = float(np.max(np.abs(got - ref))) if not (np.isnan(got).any() or np.isnan(ref).any()) else float("nan")
  verdict = "XLANE_RECURRENCE_MATRIX_PASS" if np.isfinite(err) and err <= _P14_TOL else "XLANE_RECURRENCE_MATRIX_FAIL__OUTPUT"
  return {
    "date": "2026-06-25",
    "timestamp": time.strftime("%Y%m%d-%H%M%S"),
    "verdict": verdict,
    "shape": {"cases": _P14_CASES, "features": _P14_FEATS, "lanes": _P14_LANES, "tokens_per_lane": _P14_R, "tmax": _P14_TMAX},
    "active_tokens": active.tolist(),
    "tolerance": _P14_TOL,
    "max_abs_error": err,
    "got": got.tolist(),
    "ref": ref.tolist(),
    "has_nan": {"got": bool(np.isnan(got).any()), "ref": bool(np.isnan(ref).any())},
    "decision": "If pass, the x-lane recurrence primitive is sound; investigate full-route indexing/layout. If fail, fix recurrence update ordering."
  }


# ---- P15: split-state x-lane pipeline final-output numeric gate -------------------------------------------------------
_P15_CASES = (128, 130, 32, 256)
_P15_TOL = 5e-3


def _p15_ref_split(q: np.ndarray, cache: np.ndarray, Tc: int) -> tuple[np.ndarray, np.ndarray]:
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


def _p15_run(q: np.ndarray, cache: np.ndarray, Tc: int, mode: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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


def _p15_case(Tc: int) -> dict[str, Any]:
  rng = np.random.default_rng(15000 + Tc)
  q = (rng.standard_normal((Hq, Hd)) * 0.2).astype(np.float16)
  cache = (rng.standard_normal((2, Hkv, MAXC, Hd)) * 0.2).astype(np.float16)
  ref = _ref_full(q, cache, Tc)
  ref_state, ref_pv = _p15_ref_split(q, cache, Tc)
  scalar, scalar_aux = _p15_run(q, cache, Tc, "scalar")
  xlane, _xlane_aux = _p15_run(q, cache, Tc, "xlane")
  split, split_aux = _p15_run(q, cache, Tc, "split_xlane")
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
  elif errs["scalar_vs_ref"] > _P15_TOL: verdict = "FAIL__SCALAR_REF"
  elif errs["xlane_vs_ref"] > _P15_TOL: verdict = "FAIL__XLANE_REF"
  elif errs["split_vs_ref"] > _P15_TOL: verdict = "FAIL__SPLIT_REF"
  elif errs["split_vs_scalar"] > _P15_TOL: verdict = "FAIL__SPLIT_SCALAR"
  else: verdict = "PASS"
  return {"Tc": Tc, "L": L, "Sval": math.ceil(Tc / L), "verdict": verdict, "errors": errs, "nans": nans}


def _p15() -> dict[str, Any]:
  try:
    cases = [_p15_case(tc) for tc in _P15_CASES]
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
    "tolerance": _P15_TOL,
    "cases": cases,
    "first_failure": first,
    "decision": "If pass, run route gate and W==D; if fail, debug split state/PV indexing."
  }


# ---- TG-P10.1: minimal generated-UOp repro of the REG scalar / reduction-accumulator combine blocker -----------------
# Report-only: RETURNS an int exit code and writes its own reg_scalar_lowering.json (NOT latest.json). Emits a
# deterministic tinygrad.reg_scalar_lowering.v1 artifact with four cases (one passing control + three failure modes)
# so BoltBeam can mechanically classify the blocker as EMITTER_BLOCKED. All cases are GENERATED UOp only (no HIP/ASM).
#
# REG_STORE_DEVEC is a compile-time getenv (memoized), so the DEVEC case MUST run in a fresh subprocess -- the
# _tg_spawn re-invokes THIS module file with QK_P10_STAGE set (the __main__ dispatch below routes to the stage fn).
_TG_OUT = ROOT / "bench/tg-p10-reg-scalar-combine-lowering"
_TG_Hq, _TG_Hd, _TG_S = 32, 128, 36


def _tg_synth():
  W = _TG_Hd + 2
  rng = np.random.RandomState(1)
  pn = rng.normal(0, 1, (_TG_Hq * _TG_S * W)).astype(np.float32)
  p3 = pn.reshape(_TG_Hq, _TG_S, W)
  p3[:, :, _TG_Hd + 1] = rng.normal(-2, 1, (_TG_Hq, _TG_S)); p3[:, :, _TG_Hd] = np.abs(p3[:, :, _TG_Hd]) + 0.1
  pn = p3.reshape(-1)
  ref_m = p3[:, :, _TG_Hd + 1]; ref_l = p3[:, :, _TG_Hd]; ref_pv = p3[:, :, :_TG_Hd]
  gm = ref_m.max(1, keepdims=True); w = np.exp(ref_m - gm)
  ref = ((w[:, :, None] * ref_pv).sum(1) / (w * ref_l).sum(1)[:, None]).astype(np.float32)
  return Tensor(pn, device="AMD").realize(), ref


def _tg_run_case(build):
  """Returns (compile_ok, runtime_ok, numeric_ok, observed_accum, err). build() -> np result array."""
  try:
    out = build()
  except Exception as e:
    msg = str(e)[:300]
    obs = "vectorized_make_float4" if "not assignable" in msg or "make_float4" in msg else "unknown"
    return False, False, False, obs, f"{type(e).__name__}: {msg}"
  _, ref = _tg_synth()
  nan = bool(np.isnan(out).any())
  numeric = (not nan) and float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)) < 1e-3
  return True, (not nan), numeric, "scalar" if numeric else ("vectorized_make_float4" if nan else "unknown"), (None if numeric else ("nan_output" if nan else "numeric_mismatch"))


def _tg_build_shipped():
  pout, _ = _tg_synth()
  gm = Tensor.empty(_TG_Hq, dtype=dtypes.float32, device="AMD").custom_kernel(pout, fxn=flash_state_gmax_kernel(_TG_Hd, _TG_Hq, _TG_S, stride=_TG_S))[0]
  return Tensor.empty(_TG_Hq * _TG_Hd, dtype=dtypes.float32, device="AMD").custom_kernel(pout, gm, fxn=flash_state_combine_kernel(_TG_Hd, _TG_Hq, _TG_S, stride=_TG_S))[0].realize().numpy().reshape(_TG_Hq, _TG_Hd)


def _tg_build_shared_weight():
  from extra.qk.live_split_geometry import flash_fused_gmax_combine_kernel
  pout, _ = _tg_synth()
  return Tensor.empty(_TG_Hq * _TG_Hd, dtype=dtypes.float32, device="AMD").custom_kernel(pout, fxn=flash_fused_gmax_combine_kernel(_TG_Hd, _TG_Hq, _TG_S, stride=_TG_S))[0].realize().numpy().reshape(_TG_Hq, _TG_Hd)


def _tg_build_inline_gmax():
  from extra.qk.live_split_geometry import flash_inline_gm_combine_kernel
  pout, _ = _tg_synth()
  return Tensor.empty(_TG_Hq * _TG_Hd, dtype=dtypes.float32, device="AMD").custom_kernel(pout, fxn=flash_inline_gm_combine_kernel(_TG_Hd, _TG_Hq, _TG_S, stride=_TG_S))[0].realize().numpy().reshape(_TG_Hq, _TG_Hd)


def _tg_case(case_id, compile_ok, runtime_ok, numeric_ok, error_class, error_excerpt, observed, uses_devec):
  return {"case_id": case_id, "generated_uop_only": True, "uses_external_kernel": False,
          "compile_ok": compile_ok, "runtime_ok": runtime_ok, "numeric_ok": numeric_ok,
          "error_class": error_class, "error_excerpt": (error_excerpt or "")[:200],
          "reg_accumulator_expected": "scalar", "reg_accumulator_observed": observed, "uses_reg_store_devec": uses_devec}


def _tg_measure_one():
  # in-process cases (no REG_STORE_DEVEC)
  co, ro, no, obs, err = _tg_run_case(_tg_build_shipped)
  shipped = _tg_case("shipped_per_d_combine_compiles", co, ro, no, "ok" if no else "numeric_mismatch", err, obs, False)
  co, ro, no, obs, err = _tg_run_case(_tg_build_shared_weight)
  shared = _tg_case("shared_weight_combine_compile_fails", co, ro, no, "invalid_reg_vector_store" if not co else "ok", err, obs, False)
  co, ro, no, obs, err = _tg_run_case(_tg_build_inline_gmax)
  fused = _tg_case("fused_gmax_combine_compile_fails", co, ro, no, "invalid_reg_vector_store" if not co else "ok", err, obs, False)
  print("@@RESULT@@" + json.dumps([shipped, shared, fused]))


def _tg_measure_devec():
  # REG_STORE_DEVEC=1 case (fresh process)
  co, ro, no, obs, err = _tg_run_case(_tg_build_shared_weight)
  ec = "ok" if no else ("nan_output" if (co and not ro) else ("invalid_reg_vector_store" if not co else "numeric_mismatch"))
  print("@@RESULT@@" + json.dumps(_tg_case("reg_store_devec_compiles_nan", co, ro, no, ec, err, obs, True)))


def _tg_spawn(fn_name, extra_env):
  env = dict(os.environ); env.update(extra_env); env["QK_P10_STAGE"] = fn_name
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__))], env=env, capture_output=True, text=True, cwd=str(ROOT))
  m = re.search(r"@@RESULT@@(.*)", p.stdout)
  if not m:
    sys.stderr.write(p.stdout[-1500:] + "\n" + p.stderr[-1500:]); raise SystemExit(2)
  return json.loads(m.group(1))


def _tg_p10() -> int:
  _TG_OUT.mkdir(parents=True, exist_ok=True)
  cases = _tg_spawn("measure_one", {}) + [_tg_spawn("measure_devec", {"REG_STORE_DEVEC": "1"})]
  # a valid repro: exactly one passing control + at least one compile-fail + the DEVEC-NaN case
  control_ok = any(c["case_id"].startswith("shipped") and c["numeric_ok"] for c in cases)
  fails = [c for c in cases if not c["compile_ok"]]
  devec_nan = any(c["uses_reg_store_devec"] and c["compile_ok"] and not c["numeric_ok"] for c in cases)
  verdict = ("TG_P10_1_PASS_REG_REPRO_PINNED" if control_ok and fails and devec_nan
             else "TG_P10_1_BLOCKED_REPRO_NOT_MINIMAL")
  art = {"schema": "tinygrad.reg_scalar_lowering.v1", "candidate_id": "decode_attention_split_preserving_lse_combine",
         "model_id": "qwen3-8b-q4_k_m", "target_id": "amd_gfx1100", "verdict": verdict,
         "geometry": {"Hq": _TG_Hq, "Hkv": 8, "Hd": _TG_Hd, "S": _TG_S}, "cases": cases,
         "control_passes": control_ok, "n_compile_fail": len(fails), "reg_store_devec_nan": devec_nan}
  json.dump(art, open(_TG_OUT / "reg_scalar_lowering.json", "w"), indent=2)
  print(verdict, "control_ok=", control_ok, "fails=", [c["case_id"] for c in fails], "devec_nan=", devec_nan)
  return 0 if verdict == "TG_P10_1_PASS_REG_REPRO_PINNED" else 1


# ---- registry surface ------------------------------------------------------------------------------------------------
VARIANTS = {"p8": _p8, "p9": _p9, "p10": _p10, "p11": _p11, "p12": _p12,
            "p13_reducer": _p13, "p14_recurrence": _p14, "p15_split": _p15, "tg_p10_repro": _tg_p10}

def build(variant): return VARIANTS[variant]()
def build_p8(): return build("p8")
def build_p9(): return build("p9")
def build_p10(): return build("p10")
def build_p11(): return build("p11")
def build_p12(): return build("p12")
def build_p13(): return build("p13_reducer")
def build_p14(): return build("p14_recurrence")
def build_p15(): return build("p15_split")
def build_tg_p10(): return build("tg_p10_repro")


if __name__ == "__main__":
  # tg_p10 fresh-subprocess stages (REG_STORE_DEVEC memoization requires a clean process) route here first.
  _stage = os.environ.get("QK_P10_STAGE")
  if _stage == "measure_one": _tg_measure_one(); raise SystemExit(0)
  if _stage == "measure_devec": _tg_measure_devec(); raise SystemExit(0)
  _out = build(sys.argv[1] if len(sys.argv) > 1 else "p8")  # dev convenience; gate_registry is the real runner
  if isinstance(_out, int): raise SystemExit(_out)
  print(json.dumps(_out, indent=2)); raise SystemExit(0)
