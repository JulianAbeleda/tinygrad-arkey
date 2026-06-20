#!/usr/bin/env python3
# AMD GEMM lean-addressing (Lever A) test: cut in-loop coop-load VALU by moving the K-advance from per-iter
# v_add (VALU) to scalar buffer-pointer s_add (SALU). The hard PMC audit named the residual = +23% VALU
# (8.66M ours vs 7.04M Tensile), dominated by ~1018 in-loop address-arith VALU/wave. This measures whether
# LEANADDR cuts SQ_INSTS_VALU toward Tensile's level (PMC) and whether it lifts TFLOPS (pinned, interleaved).
#
# Timing:  PYTHONPATH=. python3 extra/qk_amd_gemm_leanaddr_probe.py
# PMC:     DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_amd_gemm_leanaddr_probe.py
from __future__ import annotations

import importlib.util, json, os, pathlib, subprocess, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
RESULT = OUT / "amd_gemm_leanaddr_result.json"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"
M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
TENSILE_VALU = 7039488
CNT = int(os.environ.get("CNT", "150")); RAMP = int(os.environ.get("RAMP", "80"))
CONFIGS = [("lean0_plra1", 0), ("lean1_plra1", 1)]   # both PLRA1, BK32, PAD16, wg2


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)
def write_result(d): OUT.mkdir(parents=True, exist_ok=True); RESULT.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
def load_result(): return json.loads(RESULT.read_text()) if RESULT.exists() else {}
def stats(ets):
  s = sorted(ets); n = len(s); return {"best_tflops": FLOP / s[0] * 1e-12, "median_tflops": FLOP / s[n // 2] * 1e-12, "n": n}


def main() -> int:
  import numpy as np
  from tinygrad import Tensor, Device, Context
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import run_linear
  dev = Device["AMD"]; ref = load_mod(REF_SRC, "rdna3_ref")
  rng = np.random.default_rng(1)
  a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16), device="AMD")
  bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16), device="AMD")
  c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD"); Tensor.realize(a, bt, c)
  refmat = a.numpy().astype(np.float32) @ bt.numpy().astype(np.float32).T; refnorm = float(np.sqrt(np.mean(refmat ** 2)))
  def chk(out): d = out.float().numpy().astype(np.float32) - refmat; return float(np.sqrt(np.mean(d ** 2)) / (refnorm + 1e-9))

  pmc_mode = os.environ.get("PMC") == "1"
  result = load_result(); result.update({"date": "2026-06-20", "phase": "AMD_GEMM_LEANADDR", "schema": "amd_gemm_leanaddr_v1",
                                          "role": "ffn_gate/up", "default_behavior_changed": False, "tensile_valu_ref": TENSILE_VALU})
  built = {}
  for label, la in CONFIGS:
    insts = ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 16, 0, 1, 0, la)
    lin, out = ref._run_insts_lds(insts, a, bt, c, M, N, K, label, 32768, 128, 128, 128)
    run_linear(lin); dev.synchronize(); built[label] = {"lin": lin, "rel_rmse": chk(out), "correct": chk(out) < 0.02}

  if pmc_mode:
    from tinygrad.device import Compiled
    from extra.qk_pmc_capture import decode_pmc
    def cap(fn, warmup=4):
      for _ in range(warmup): fn()
      bse = len([e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"])
      fn(); dev.synchronize(); dev._at_profile_finalize()
      evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"][bse:]
      return decode_pmc(evs[0]) if evs else None
    pmc = {}
    for label, _ in CONFIGS:
      c0 = cap(lambda lin=built[label]["lin"]: run_linear(lin)); act = (c0 or {}).get("GRBM_GUI_ACTIVE", 0) or 1
      pmc[label] = None if not c0 else {"valu_total": c0.get("SQ_INSTS_VALU", 0), "salu_total": c0.get("SQ_INSTS_SALU", 0),
                                        "busy_per_active": round(c0.get("SQ_BUSY_CYCLES", 0) / act, 2), "L2_hit%": round(100 * c0.get("GL2C_HIT", 0) / (c0.get("GL2C_HIT", 0) + c0.get("GL2C_MISS", 0) + 1e-9), 1)}
    result["pmc"] = pmc
    v0, v1 = pmc["lean0_plra1"]["valu_total"], pmc["lean1_plra1"]["valu_total"]
    result["valu_analysis"] = {"lean0_valu": v0, "lean1_valu": v1, "valu_cut": v0 - v1,
                               "lean1_vs_tensile": round(v1 / TENSILE_VALU, 3), "tensile_valu": TENSILE_VALU}
  else:
    launches = [(l, lambda lin=built[l]["lin"]: run_linear(lin)) for l, _ in CONFIGS]
    try:
      ptm1 = load_mod(PTM1_SRC, "ptm1_bridge"); auth_lin, _m, _alive = ptm1.build_authority(); launches.append(("authority_llvm", lambda: run_linear(auth_lin)))
    except Exception: pass
    perflevel("high"); times = {l: [] for l, _ in launches}
    try:
      with Context(DEBUG=0):
        for _, fn in launches:
          for _ in range(RAMP): dev.synchronize(); fn()
          dev.synchronize()
        for _ in range(CNT):
          for l, fn in launches:
            dev.synchronize(); t0 = time.perf_counter(); fn(); dev.synchronize(); times[l].append(time.perf_counter() - t0)
    finally:
      perflevel("auto")
    result["timing"] = {l: {**stats(times[l]), **({"rel_rmse": built[l]["rel_rmse"]} if l in built else {"authority": True})} for l, _ in launches}
    t = result["timing"]; t0v = t["lean0_plra1"]["best_tflops"]; t1v = t["lean1_plra1"]["best_tflops"]
    result["timing_analysis"] = {"lean0_tflops": t0v, "lean1_tflops": t1v, "lean_gain_x": round(t1v / t0v, 3) if (t0v and t1v) else None,
                                 "authority_tflops": t.get("authority_llvm", {}).get("best_tflops")}

  # verdict when both present
  if "pmc" in result and "timing" in result:
    va = result["valu_analysis"]; ta = result["timing_analysis"]
    result["verdict"] = ("LEANADDR_CUTS_VALU_AND_HELPS" if ta["lean_gain_x"] and ta["lean_gain_x"] > 1.02 and va["valu_cut"] > 0
                         else "LEANADDR_CUTS_VALU_NEUTRAL_TFLOPS" if va["valu_cut"] > 0
                         else "LEANADDR_NO_VALU_CUT")
    result["why"] = f"VALU {va['lean0_valu']}->{va['lean1_valu']} (cut {va['valu_cut']}, ={va['lean1_vs_tensile']}x Tensile {TENSILE_VALU}); TFLOPS {ta['lean0_tflops']:.1f}->{ta['lean1_tflops']:.1f} ({ta['lean_gain_x']}x)."
  else:
    result.setdefault("verdict", "PARTIAL_RUN_BOTH_MODES")
  write_result(result)
  print(f"=== mode={'PMC' if pmc_mode else 'TIMING'} ===")
  print(json.dumps({"verdict": result.get("verdict"), "pmc": result.get("pmc"), "valu_analysis": result.get("valu_analysis"),
                    "timing": {k: round(v["best_tflops"], 1) for k, v in result.get("timing", {}).items()},
                    "timing_analysis": result.get("timing_analysis"), "why": result.get("why")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
