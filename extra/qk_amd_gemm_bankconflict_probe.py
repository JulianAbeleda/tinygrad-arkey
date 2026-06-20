#!/usr/bin/env python3
# AMD GEMM bank-conflict-free LDS layout test (REAL GPU, correctness-gated, NO BEAM, NO routing/default change).
#
# PMC named the BK32/LDS-family ceiling as LDS BANK CONFLICTS (31.5/cycle, 6.7x BK16, vs global-direct 0). This
# tests the named lever: pad the fragment-row stride to change the LDS bank mapping. In build_gemm_lds2 rows
# are stored with stride SA = BK*2 + PAD bytes; for BK32/PAD0, SA=64B=16 dwords=16 banks -> consecutive-row
# ds_load_b128 reads collide 16-way (RDNA3 has 32 banks). ds_load_b128 needs 16B-aligned addresses, so PAD must
# be a multiple of 16. Bank math lane->bank=(l*SA/4)%32: PAD0->16-way, PAD16(SA/4=20)->4-way, PAD32->8-way,
# PAD48->4-way. Predicted sweet spot: PAD16 (4x fewer conflicts).
#
# CLEAN ISOLATION: allocate the SAME total LDS (32768 = wg2 occupancy) for every PAD, so ONLY the internal bank
# mapping changes (occupancy held constant). Measure BOTH bankcf/cycle (PMC mode) and TFLOPS (timing mode).
#
# Run timing:  PYTHONPATH=. python3 extra/qk_amd_gemm_bankconflict_probe.py
# Run PMC:     DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_amd_gemm_bankconflict_probe.py
# (run both; results merge into one JSON, verdict computed when both present)
from __future__ import annotations

import importlib.util, json, os, pathlib, re, subprocess, time, traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
RESULT = OUT / "amd_gemm_bankconflict_result.json"
PMC_DOC = "bench/amd-broad-backend-roadmap/amd_gemm_pmc_result.json"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"

M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
REL_RMSE_PASS = 0.02
TENSILE_CLASS_TFLOPS = 60.0
FIXED_LDS = 32768                 # wg2 occupancy for every PAD (isolate bank-mapping from occupancy)
CNT = int(os.environ.get("CNT", "150")); RAMP = int(os.environ.get("RAMP", "80"))
# (PAD bytes, SA/4 dwords for BK32, predicted conflict-way)
PAD_LEVELS = [0, 16, 32, 48]
def sa_dwords(pad): return (32 * 2 + pad) // 4    # SA = BK*2+PAD = 64+PAD ; /4 dwords  (BK=32)
def predicted_conflict_way(pad):
  banks = set(((l * sa_dwords(pad)) % 32) for l in range(32))
  return 32 // len(banks)


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod); return mod


def write_result(data): OUT.mkdir(parents=True, exist_ok=True); RESULT.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
def load_result(): return json.loads(RESULT.read_text()) if RESULT.exists() else {}
def sample_power():
  try:
    out = subprocess.run(["rocm-smi", "--showpower"], capture_output=True, text=True, timeout=8).stdout
    m = re.search(r"Average Graphics Package Power \(W\):\s*([\d.]+)", out, re.I); return float(m.group(1)) if m else None
  except Exception: return None
def stats(ets):
  s = sorted(ets); n = len(s); return {"best_tflops": FLOP / s[0] * 1e-12, "median_tflops": FLOP / s[n // 2] * 1e-12, "n": n}


def build_lds2(ref, pad):
  insts = ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, pad, 0)
  bufsz = (32 * 2 + pad) * (128 + 128)   # SA*(BM+BN) for BK32 single-buffer
  assert bufsz <= FIXED_LDS, f"PAD{pad} BUFSZ {bufsz} > fixed {FIXED_LDS}"
  return insts


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
  result = load_result()
  result.setdefault("date", "2026-06-20"); result["phase"] = "AMD_GEMM_BANKCONFLICT"; result["schema"] = "amd_gemm_bankconflict_v1"
  result["role"] = "ffn_gate/up"; result["default_behavior_changed"] = False; result["is_diagnostic"] = True
  result["fixed_lds_bytes"] = FIXED_LDS
  result["predicted"] = {f"PAD{p}": {"sa_dwords": sa_dwords(p), "conflict_way": predicted_conflict_way(p)} for p in PAD_LEVELS}

  # build + correctness for each PAD (shared)
  built = {}
  for pad in PAD_LEVELS:
    try:
      insts = build_lds2(ref, pad)
      lin, out = ref._run_insts_lds(insts, a, bt, c, M, N, K, f"bk32_pad{pad}", FIXED_LDS, 128, 128, 128)
      run_linear(lin); Device["AMD"].synchronize(); rel = chk(out)
      built[pad] = {"lin": lin, "rel_rmse": rel, "correct": rel < REL_RMSE_PASS}
    except Exception as ex:
      built[pad] = {"error": repr(ex)[:120], "correct": False}

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
    for pad in PAD_LEVELS:
      b = built[pad]
      if not b.get("correct"): pmc[f"PAD{pad}"] = {"status": "INCORRECT_OR_FAIL"}; continue
      c0 = cap(lambda lin=b["lin"]: run_linear(lin))
      if not c0: pmc[f"PAD{pad}"] = {"status": "NO_PMC"}; continue
      act = c0.get("GRBM_GUI_ACTIVE", 0) or 1
      pmc[f"PAD{pad}"] = {"bankconf_per_active": round(c0.get("SQC_LDS_BANK_CONFLICT", 0) / act, 2),
                          "lds_active_per_active": round(c0.get("SQC_LDS_IDX_ACTIVE", 0) / act, 2),
                          "busy_per_active": round(c0.get("SQ_BUSY_CYCLES", 0) / act, 2),
                          "L2_hit%": round(100 * c0.get("GL2C_HIT", 0) / (c0.get("GL2C_HIT", 0) + c0.get("GL2C_MISS", 0) + 1e-9), 1)}
    result["pmc"] = pmc
    result["pmc_note"] = "bankcf/cycle per PAD at fixed wg2 LDS; lower = fewer conflicts (PMC timing perturbed)"
  else:
    timed = [(f"bk32_pad{pad}", built[pad]["lin"]) for pad in PAD_LEVELS if built[pad].get("correct")]
    try:
      ptm1 = load_mod(PTM1_SRC, "ptm1_bridge"); auth_lin, _m, _alive = ptm1.build_authority(); timed.append(("authority_llvm", auth_lin))
    except Exception: pass
    power = []; times = {lbl: [] for lbl, _ in timed}
    with Context(DEBUG=0):
      for _, lin in timed:
        for _ in range(RAMP): Device["AMD"].synchronize(); run_linear(lin)
        Device["AMD"].synchronize()
      for rep in range(CNT):
        for lbl, lin in timed:
          Device["AMD"].synchronize(); t0 = time.perf_counter(); run_linear(lin); Device["AMD"].synchronize()
          times[lbl].append(time.perf_counter() - t0)
        if rep % 12 == 0:
          p = sample_power()
          if p is not None: power.append(p)
    pv = sorted(power); pmed = pv[len(pv) // 2] if pv else None
    result["timing"] = {lbl: {**stats(times[lbl]),
                              **({"rel_rmse": built[int(lbl.split('pad')[1])]["rel_rmse"]} if "pad" in lbl else {"authority": True})}
                        for lbl, _ in timed}
    result["timing_clock"] = {"power_median_w": pmed, "samples": len(pv), "clock_valid": len(pv) >= 3 and (pmed or 0) >= 25}

  # verdict when both present
  if "pmc" in result and "timing" in result:
    pad_best = max((p for p in PAD_LEVELS if f"bk32_pad{p}" in result["timing"]),
                   key=lambda p: result["timing"][f"bk32_pad{p}"]["best_tflops"], default=None)
    pad0_bc = result["pmc"].get("PAD0", {}).get("bankconf_per_active")
    best_bc = result["pmc"].get(f"PAD{pad_best}", {}).get("bankconf_per_active") if pad_best is not None else None
    auth_t = result["timing"].get("authority_llvm", {}).get("best_tflops")
    best_t = result["timing"].get(f"bk32_pad{pad_best}", {}).get("best_tflops") if pad_best is not None else None
    pad0_t = result["timing"].get("bk32_pad0", {}).get("best_tflops")
    bc_cut = (pad0_bc / best_bc) if (pad0_bc and best_bc) else None
    tput_gain = (best_t / pad0_t) if (best_t and pad0_t) else None
    result["analysis"] = {"best_pad": pad_best, "bankconf_cut_x": round(bc_cut, 2) if bc_cut else None,
                          "throughput_gain_x": round(tput_gain, 2) if tput_gain else None,
                          "best_tflops": best_t, "pad0_tflops": pad0_t, "authority_tflops": auth_t,
                          "reaches_tensile_class": bool(best_t and best_t >= TENSILE_CLASS_TFLOPS)}
    if bc_cut and bc_cut >= 1.5 and tput_gain and tput_gain >= 1.05:
      if best_t and best_t >= TENSILE_CLASS_TFLOPS:
        result["verdict"] = "PASS_BANKCONFLICT_FIX_REACHES_TENSILE_CLASS"
      else:
        result["verdict"] = "PASS_BANKCONFLICT_FIX_LIFTS_THROUGHPUT"
      result["why"] = (f"PAD{pad_best} cuts bank conflicts {round(bc_cut,2)}x and lifts throughput {round(tput_gain,2)}x "
                       f"(to {best_t:.1f} TFLOPS) -> conflicts WERE a binding constraint; the named lever works.")
    elif bc_cut and bc_cut >= 1.5:
      result["verdict"] = "BANKCONFLICT_CUT_BUT_THROUGHPUT_FLAT"
      result["why"] = (f"PAD{pad_best} cut conflicts {round(bc_cut,2)}x but throughput is flat ({tput_gain}x) -> "
                       "bank conflicts were NOT the throughput binding constraint; re-examine (LDS bandwidth / read-prefetch).")
    else:
      result["verdict"] = "BANKCONFLICT_ROWPAD_INEFFECTIVE"
      result["why"] = ("row-padding did not cut conflicts -> the conflict is structural (needs a swizzle/XOR LDS "
                       "layout, not row-pad), or the alignment constraint blocks it. Bigger layout change required.")
  else:
    result.setdefault("verdict", "PARTIAL_RUN_BOTH_MODES_TO_COMPLETE")

  write_result(result)
  mode = "PMC" if pmc_mode else "TIMING"
  print(f"=== mode={mode} ===")
  print(json.dumps({"verdict": result.get("verdict"), "predicted": result["predicted"],
                    "pmc": result.get("pmc"), "timing": {k: round(v.get("best_tflops", 0), 1) for k, v in result.get("timing", {}).items()},
                    "analysis": result.get("analysis"), "why": result.get("why")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
