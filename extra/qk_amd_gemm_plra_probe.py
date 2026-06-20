#!/usr/bin/env python3
# AMD GEMM A-prefetch PLR test (REAL GPU, correctness-gated, interleaved one-clock + optional PMC, NO BEAM).
#
# Implements the bounded dependency-free PLR experiment the VGPR-scheduling audit pointed to: prefetch the next
# K-substep's A fragments into the DEAD coop-load temp registers (CTA/CTB, register-lifetime overlap like
# Tensile's pool) during the current substep's WMMAs, hiding the A ds_load latency. (Full A+B prefetch needs
# 64 VGPR > the ~32 dead; A-only fits exactly = partial PLR.) build_gemm_lds2(..., PLRA=1).
#
# Compares bk32_pad16_plra1 vs the bank-fix frontier (plra0, ~60.7) vs the LLVM authority, interleaved one
# clock. Does partial PLR lift toward Tensile (~66)? PMC mode reports lds-wait/busy to see latency hiding.
#
# Run timing:  PYTHONPATH=. python3 extra/qk_amd_gemm_plra_probe.py
# Run PMC:     DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_amd_gemm_plra_probe.py
from __future__ import annotations

import importlib.util, json, os, pathlib, re, subprocess, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
RESULT = OUT / "amd_gemm_plra_result.json"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"

M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
REL_RMSE_PASS = 0.02
TENSILE_PARITY = 64.0
LDS = 32768                        # wg2 occupancy
CNT = int(os.environ.get("CNT", "150")); RAMP = int(os.environ.get("RAMP", "80"))
# (label, PLRA)
CONFIGS = [("bk32_pad16_plra0", 0), ("bk32_pad16_plra1", 1)]


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod); return mod
def write_result(d): OUT.mkdir(parents=True, exist_ok=True); RESULT.write_text(json.dumps(d, indent=2, sort_keys=True) + "\n")
def load_result(): return json.loads(RESULT.read_text()) if RESULT.exists() else {}
def sample_power():
  try:
    out = subprocess.run(["rocm-smi", "--showpower"], capture_output=True, text=True, timeout=8).stdout
    m = re.search(r"Average Graphics Package Power \(W\):\s*([\d.]+)", out, re.I); return float(m.group(1)) if m else None
  except Exception: return None
def stats(ets):
  s = sorted(ets); n = len(s); return {"best_tflops": FLOP / s[0] * 1e-12, "median_tflops": FLOP / s[n // 2] * 1e-12, "n": n}


def main() -> int:
  import numpy as np
  from tinygrad import Tensor, Device, Context
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import run_linear
  ref = load_mod(REF_SRC, "rdna3_ref")
  rng = np.random.default_rng(1)
  a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16), device="AMD")
  bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16), device="AMD")
  c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD"); Tensor.realize(a, bt, c)
  refmat = a.numpy().astype(np.float32) @ bt.numpy().astype(np.float32).T; refnorm = float(np.sqrt(np.mean(refmat ** 2)))
  def chk(out): d = out.float().numpy().astype(np.float32) - refmat; return float(np.sqrt(np.mean(d ** 2)) / (refnorm + 1e-9))

  pmc_mode = os.environ.get("PMC") == "1"
  result = load_result(); result.update({"date": "2026-06-20", "phase": "AMD_GEMM_PLRA", "schema": "amd_gemm_plra_v1",
                                          "role": "ffn_gate/up", "default_behavior_changed": False, "is_diagnostic": True})
  built = {}
  for label, plra in CONFIGS:
    insts = ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 16, 0, plra)
    lin, out = ref._run_insts_lds(insts, a, bt, c, M, N, K, label, LDS, 128, 128, 128)
    run_linear(lin); Device["AMD"].synchronize(); rel = chk(out)
    built[label] = {"lin": lin, "rel_rmse": rel, "correct": rel < REL_RMSE_PASS, "insts": len(insts)}

  if pmc_mode:
    from tinygrad.device import Compiled
    from extra.qk_pmc_capture import decode_pmc
    def cap(fn, warmup=3):
      for _ in range(warmup): fn()
      bse = len([e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"])
      fn(); Device["AMD"].synchronize(); Device["AMD"]._at_profile_finalize()
      evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"][bse:]
      return decode_pmc(evs[0]) if evs else None
    pmc = {}
    for label, _ in CONFIGS:
      c0 = cap(lambda lin=built[label]["lin"]: run_linear(lin)); act = (c0 or {}).get("GRBM_GUI_ACTIVE", 0) or 1
      pmc[label] = None if not c0 else {"cycles": c0.get("GRBM_GUI_ACTIVE"),
                                        "busy_per_active": round(c0.get("SQ_BUSY_CYCLES", 0) / act, 2),
                                        "valu_per_active": round(c0.get("SQ_INSTS_VALU", 0) / act, 2),
                                        "lds_active_per_active": round(c0.get("SQC_LDS_IDX_ACTIVE", 0) / act, 2),
                                        "bankconf_per_active": round(c0.get("SQC_LDS_BANK_CONFLICT", 0) / act, 2)}
    result["pmc"] = pmc
  else:
    timed = [(label, built[label]["lin"]) for label, _ in CONFIGS if built[label]["correct"]]
    try:
      ptm1 = load_mod(PTM1_SRC, "ptm1_bridge"); auth_lin, _m, _alive = ptm1.build_authority(); timed.append(("authority_llvm", auth_lin))
    except Exception: pass
    power = []; times = {l: [] for l, _ in timed}
    with Context(DEBUG=0):
      for _, lin in timed:
        for _ in range(RAMP): Device["AMD"].synchronize(); run_linear(lin)
        Device["AMD"].synchronize()
      for rep in range(CNT):
        for l, lin in timed:
          Device["AMD"].synchronize(); t0 = time.perf_counter(); run_linear(lin); Device["AMD"].synchronize(); times[l].append(time.perf_counter() - t0)
        if rep % 12 == 0:
          p = sample_power()
          if p is not None: power.append(p)
    result["timing"] = {l: {**stats(times[l]), **({"rel_rmse": built[l]["rel_rmse"], "insts": built[l]["insts"]} if l in built else {"authority": True})} for l, _ in timed}
    result["timing_power_median_w"] = sorted(power)[len(power) // 2] if power else None

  if "timing" in result:
    t = result["timing"]
    p0 = t.get("bk32_pad16_plra0", {}).get("best_tflops"); p1 = t.get("bk32_pad16_plra1", {}).get("best_tflops")
    auth = t.get("authority_llvm", {}).get("best_tflops")
    result["analysis"] = {"plra0_tflops": p0, "plra1_tflops": p1, "authority_tflops": auth,
                          "plra_gain_x": round(p1 / p0, 3) if (p0 and p1) else None,
                          "reaches_parity_64": bool(p1 and p1 >= TENSILE_PARITY)}
    both_correct = all(built[l]["correct"] for l, _ in CONFIGS)
    if not both_correct: result["verdict"] = "FAIL_PLRA_CORRECTNESS"
    elif p1 and p1 >= TENSILE_PARITY: result["verdict"] = "PASS_PLRA_REACHES_TENSILE_PARITY"
    elif p0 and p1 and p1 > p0 * 1.02:
      result["verdict"] = "PLRA_HELPS_PARTIAL"; result["why"] = f"A-prefetch PLR lifts {p0:.1f}->{p1:.1f} (+{round((p1/p0-1)*100)}%), correct, but < 64 parity (B not prefetched). Toward Tensile."
    elif p0 and p1 and p1 < p0 * 0.98:
      result["verdict"] = "PLRA_REGRESSES"; result["why"] = f"A-prefetch PLR REGRESSES {p0:.1f}->{p1:.1f}; the reorder/extra-VGPR cost outweighs the partial latency hiding."
    else:
      result["verdict"] = "PLRA_NEUTRAL_RESTS_AT_TENSILE_CLASS"; result["why"] = f"A-prefetch PLR is ~neutral ({p0:.1f} vs {p1:.1f}); partial (A-only) latency hiding doesn't move it. Full A+B PLR needs >256 VGPR. Rest at Tensile-class ~{p0:.0f}."
  else:
    result.setdefault("verdict", "PARTIAL_RUN_BOTH_MODES")
  write_result(result)
  print(f"=== mode={'PMC' if pmc_mode else 'TIMING'} ===")
  print(json.dumps({"verdict": result.get("verdict"),
                    "timing": {k: round(v.get("best_tflops", 0), 1) for k, v in result.get("timing", {}).items()},
                    "pmc": result.get("pmc"), "analysis": result.get("analysis"), "why": result.get("why")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
