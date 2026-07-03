#!/usr/bin/env python3
"""TG-P14.8: split-preserving combine reopen -- numeric correctness + directional timing + structural analysis.

Now that the AMD baseline reduce/upcast lowering lifts the emitter block (TG_P9_4_PASS_COMBINE_MICROGATE), re-run the generated-UOp
combine shapes for NUMERIC correctness vs a numpy reference (P10 only covered shared-weight + fused-gmax; this adds
the two-stage fexp-free weighted-sum), and report the STRUCTURAL win (fexp count, kernel count) plus a DIRECTIONAL
isolated timing. Isolated combine micro-timing is NOT the promotion authority (launch/clock-ramp confounds poison
short benches -- see the AMD decode measurement notes); the authoritative speed test is the full W==D at TG-P14.9.

Run: DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p14_combine_reopen.py
"""
from __future__ import annotations
import json, pathlib, time
import numpy as np

from extra.qk_tg_p10_reg_scalar_repro import _synth, Hq, Hd, S

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/tg-p14-amd-recovery-and-pure-attention-landing"


def _numeric(out: np.ndarray, ref: np.ndarray) -> dict:
  nan = bool(np.isnan(out).any()); inf = bool(np.isinf(out).any())
  rel = float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6))
  return {"nan": nan, "inf": inf, "rel_err": rel, "numeric_ok": (not nan and not inf and rel < 1e-3)}


def _sync():
  from tinygrad import Device
  Device["AMD"].synchronize()


def _time_call(build, iters=200, warmup=30) -> float:
  # directional only: many iters + warmup + explicit sync; still not the W==D authority.
  for _ in range(warmup): build()
  _sync()
  t0 = time.perf_counter()
  for _ in range(iters): build()
  _sync()
  return (time.perf_counter() - t0) / iters * 1e6  # us/call


def main() -> int:
  from tinygrad import Tensor, dtypes
  from extra.qk_flash_decode import flash_state_gmax_kernel, flash_state_combine_kernel
  from extra.qk_live_split_geometry import (flash_fused_gmax_combine_kernel, flash_gm_weights_kernel,
                                            flash_weighted_sum_kernel)
  pout, ref = _synth()

  def build_shipped():
    gm = Tensor.empty(Hq, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, fxn=flash_state_gmax_kernel(Hd, Hq, S, stride=S))[0]
    return Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, gm, fxn=flash_state_combine_kernel(Hd, Hq, S, stride=S))[0]

  def build_fused():
    return Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, fxn=flash_fused_gmax_combine_kernel(Hd, Hq, S, stride=S))[0]

  def build_two_stage():
    w = Tensor.empty(Hq * S, dtype=dtypes.float32, device="AMD").custom_kernel(
      pout, fxn=flash_gm_weights_kernel(Hd, Hq, S, stride=S))[0]
    return Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(
      w, pout, fxn=flash_weighted_sum_kernel(Hd, Hq, S, stride=S))[0]

  shapes = {}
  for name, build, nkern, fexp in (
      ("shipped_per_d",   build_shipped,   2, Hq * Hd * S),   # gmax + per-d combine (fexp recomputed per d)
      ("fused_lds_warp",  build_fused,     1, Hq * S),        # single fused kernel, weights shared in LDS
      ("two_stage_fexpfree", build_two_stage, 2, Hq * S)):    # weights (only fexp) + fexp-free weighted-sum
    try:
      out = build().realize().numpy().reshape(Hq, Hd)
      num = _numeric(out, ref)
      us = _time_call(lambda: build().realize())
    except Exception as e:
      shapes[name] = {"compile_ok": False, "err": f"{type(e).__name__}: {str(e)[:160]}"}
      continue
    shapes[name] = {"compile_ok": True, **num, "kernel_count": nkern, "fexp_count": fexp,
                    "uses_reg_store_devec": False, "directional_us_per_call": round(us, 2)}

  correct = all(s.get("numeric_ok") for s in shapes.values() if s.get("compile_ok"))
  all_compile = all(s.get("compile_ok") for s in shapes.values())
  base_fexp = shapes["shipped_per_d"]["fexp_count"]
  fexp_reduction = {k: round(base_fexp / v["fexp_count"], 1) for k, v in shapes.items() if v.get("compile_ok")}

  result = {
    "scope": "TG-P14.8 split-preserving combine reopen", "geometry": {"Hq": Hq, "Hd": Hd, "S": S},
    "all_compile": all_compile, "all_numeric_ok": correct,
    "shapes": shapes, "fexp_reduction_vs_shipped": fexp_reduction,
    "timing_authority_note": ("directional_us_per_call is an isolated micro-timing and is NOT the promotion "
                              "authority (launch/clock-ramp confounds). The authoritative speed test is the full "
                              "W==D at TG-P14.9 (generated attention integrated into model.generate)."),
    "verdict": ("TG_P14_8_PASS_COMBINE_REOPENED" if (all_compile and correct)
                else "TG_P14_8_BLOCKED_COMBINE_CORRECTNESS"),
  }
  OUT.mkdir(parents=True, exist_ok=True)
  json.dump(result, open(OUT / "combine_reopen.json", "w"), indent=2)
  print(result["verdict"], "all_compile=", all_compile, "all_numeric_ok=", correct)
  for k, v in shapes.items():
    print(f"  {k:20s} compile={v.get('compile_ok')} numeric_ok={v.get('numeric_ok')} "
          f"kern={v.get('kernel_count')} fexp={v.get('fexp_count')} us={v.get('directional_us_per_call')}")
  print("  fexp_reduction_vs_shipped:", fexp_reduction)
  return 0 if result["verdict"] == "TG_P14_8_PASS_COMBINE_REOPENED" else 1


if __name__ == "__main__":
  raise SystemExit(main())
