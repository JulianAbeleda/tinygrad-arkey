#!/usr/bin/env python3
# AMD GEMM OCCUPANCY-SENSITIVITY DIAGNOSTIC (REAL GPU, correctness-gated, interleaved one-clock, NO BEAM).
#
# Answers "why are we stuck at ~55 and not hitting Tensile ~66?" with a MEASUREMENT, not an assertion. The
# Tensile-source gap audit predicted our SIA0 kernel hides memory latency ONLY via occupancy (it has no
# software prefetch PLR/PGR). Decisive test: vary ONLY occupancy and watch throughput.
#
# Lever: LDS padding. _run_insts_lds allocates a DEFINE_LOCAL of `lds_bytes`; the BK32 kernel only USES 16384 B,
# so padding to 32768/49152/65536 leaves compute + correctness identical but cuts LDS-limited workgroups/CU
# (64KB/lds) from 4 -> 2 -> 1 -> forcing occupancy down. If throughput RISES with occupancy => latency-bound
# (confirms the audit: stuck because latency is hidden only by waves, and waves fight the VGPR/reuse budget).
# If FLAT => compute/issue-bound (would refute the audit). No new kernel family, no search.
from __future__ import annotations

import importlib.util, json, os, pathlib, re, subprocess, time, traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
AUDIT = "bench/amd-broad-backend-roadmap/amd_tensile_source_gap_audit_result.json"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"

M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
REL_RMSE_PASS = 0.02
ACTIVE_POWER_MIN_W = 25.0
CNT = int(os.environ.get("CNT", "150"))
RAMP = int(os.environ.get("RAMP", "80"))
LDS_PER_CU = 65536            # gfx1100 LDS bytes per WGP/CU
VGPR_PER_SIMD = 1536          # gfx1100 wave32 VGPR file per SIMD

# BK32 frontier kernel (W2x2 T4x4 BK32). Real LDS footprint = 16384 B. Pad to force lower occupancy.
BK32 = {"WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 32, "PAD": 0, "DBUF": 0, "BM": 128, "BN": 128, "THREADS": 128, "vgpr": 234, "real_lds": 16384}
PAD_LEVELS = [16384, 24576, 32768, 49152, 65536]   # -> LDS-limited WG/CU = 4,2(.6),2,1(.3),1


def read_json(rel: str) -> dict[str, Any]:
  p = ROOT / rel
  if not p.exists(): raise FileNotFoundError(rel)
  return json.loads(p.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod); return mod


def sample_power():
  try:
    out = subprocess.run(["rocm-smi", "--showpower"], capture_output=True, text=True, timeout=8).stdout
    m = re.search(r"Average Graphics Package Power \(W\):\s*([\d.]+)", out, re.I); return float(m.group(1)) if m else None
  except Exception:
    return None


def stats(ets):
  srt = sorted(ets); n = len(srt)
  return {"best_s": srt[0], "median_s": srt[n // 2], "best_tflops": FLOP / srt[0] * 1e-12, "median_tflops": FLOP / srt[n // 2] * 1e-12, "n": n}


def emit(result):
  write_json("amd_gemm_occupancy_diagnostic_result.json", result)
  print(json.dumps({k: result[k] for k in ("verdict", "gate_pass") if k in result}, indent=2))
  if "exact_blocker" in result: print("note:", result["exact_blocker"])
  for r in result.get("rows", []):
    s = r.get("stats")
    if s: print(f"  {r['label']:26} best={s['best_tflops']:6.2f} median={s['median_tflops']:6.2f} TFLOPS"
                + (f"  lds={r['lds_bytes']}B wg/cu={r['wg_per_cu']}" if "lds_bytes" in r else "  (authority)"))
    else: print(f"  {r['label']:26} {r.get('status')}")
  c = result.get("conclusion_metrics", {})
  if c: print(f"  => peak wg={c.get('peak_wg_per_cu')} = {c.get('peak_tflops')} TFLOPS "
              f"(peak/full={c.get('peak_over_full_x')}x, peak/authority={c.get('peak_over_authority_x')}x); {c.get('interpretation')}")
  return 0 if result.get("gate_pass") else 1


def base(verdict):
  return {"date": "2026-06-20", "phase": "AMD_GEMM_OCCUPANCY_DIAGNOSTIC", "schema": "amd_gemm_occupancy_diag_v1",
          "role": "ffn_gate/up", "verdict": verdict, "gate_pass": verdict.startswith("CONFIRMED"),
          "default_behavior_changed": False, "performance_claim": False, "is_diagnostic": True,
          "shape": {"M": M, "N": N, "K": K},
          "method": "isolate occupancy via LDS padding (compute+correctness identical); interleaved one-clock; "
                    "throughput-vs-occupancy slope distinguishes latency-bound from compute-bound",
          "input_artifacts": [AUDIT, "extra/gemm/rdna3_wmma_matmul.py"]}


def main() -> int:
  try:
    aud = read_json(AUDIT)
    if not aud.get("verdict", "").startswith("AUDIT_BANKED"):
      return emit({**base("BLOCKED_OCC_DIAG_PRECONDITION"), "exact_blocker": f"audit not banked: {aud.get('verdict')}"})
  except Exception as ex:
    return emit({**base("BLOCKED_OCC_DIAG_PRECONDITION"), "exact_blocker": f"missing audit artifact: {ex!r}"})

  try:
    import numpy as np
    from tinygrad import Tensor, Device, Context
    from tinygrad.dtype import dtypes
    from tinygrad.engine.realize import run_linear
    ref = load_mod(REF_SRC, "rdna3_ref")
    rng = np.random.default_rng(1)
    a_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float16); bt_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float16)
    a = Tensor(a_np, device="AMD"); bt = Tensor(bt_np, device="AMD"); c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD")
    Tensor.realize(a, bt, c)
    refmat = a_np.astype(np.float32) @ bt_np.astype(np.float32).T; refnorm = float(np.sqrt(np.mean(refmat ** 2)))
    insts = ref.build_gemm_lds2(M, N, K, BK32["WAVES_M"], BK32["WAVES_N"], BK32["WM"], BK32["WN"], BK32["BK"], BK32["PAD"], BK32["DBUF"])
  except Exception as ex:
    return emit({**base("BLOCKED_OCC_DIAG_LAUNCH"), "exact_blocker": f"setup/build failed: {ex!r}", "trace": traceback.format_exc().splitlines()[-4:]})

  def chk(out): cn = out.float().numpy().astype(np.float32); d = cn - refmat; return float(np.sqrt(np.mean(d**2))/(refnorm+1e-9))

  rows: list[dict[str, Any]] = []; timed: list[tuple[str, Any]] = []
  # occupancy ladder: same BK32 kernel, increasing LDS pad -> lower occupancy
  for lds in PAD_LEVELS:
    wg_cu = LDS_PER_CU // lds
    label = f"bk32_lds{lds}_wg{wg_cu}"
    try:
      lin, out = ref._run_insts_lds(insts, a, bt, c, M, N, K, label, lds, BK32["BM"], BK32["BN"], BK32["THREADS"])
      run_linear(lin); Device["AMD"].synchronize(); rel = chk(out)
    except Exception as ex:
      rows.append({"label": label, "status": "LAUNCH_FAIL", "error": repr(ex)}); continue
    rows.append({"label": label, "lds_bytes": lds, "wg_per_cu": wg_cu, "rel_rmse": rel,
                 "status": "CORRECT" if rel < REL_RMSE_PASS else "INCORRECT"})
    if rel < REL_RMSE_PASS: timed.append((label, lin))

  # authority reference row
  try:
    ptm1 = load_mod(PTM1_SRC, "ptm1_bridge"); auth_lin, auth_meta, _alive = ptm1.build_authority()
    rows.append({"label": "authority_llvm", "authority": True}); timed.append(("authority_llvm", auth_lin))
  except Exception as ex:
    rows.append({"label": "authority_llvm", "authority": True, "status": "UNAVAILABLE", "error": repr(ex)[:120]})

  if len(timed) < 2:
    return emit({**base("BLOCKED_OCC_DIAG_LAUNCH"), "exact_blocker": "insufficient correct rows to time", "rows": rows})

  power = []; times = {lbl: [] for lbl, _ in timed}
  try:
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
  except Exception as ex:
    return emit({**base("BLOCKED_OCC_DIAG_LAUNCH"), "exact_blocker": f"timing failed: {ex!r}", "rows": rows})

  for r in rows:
    if r["label"] in times: r["stats"] = stats(times[r["label"]])
  pv = sorted(power); power_median = pv[len(pv) // 2] if pv else None
  clock_valid = len(pv) >= 3 and (power_median or 0) >= ACTIVE_POWER_MIN_W

  occ = sorted([r for r in rows if r.get("status") == "CORRECT" and "stats" in r], key=lambda r: -r["wg_per_cu"])
  full = max((r for r in occ if r["wg_per_cu"] == max(r2["wg_per_cu"] for r2 in occ)), key=lambda r: r["stats"]["best_tflops"], default=None)
  occ1 = max((r for r in occ if r["wg_per_cu"] == 1), key=lambda r: r["stats"]["best_tflops"], default=None)
  peak = max(occ, key=lambda r: r["stats"]["best_tflops"], default=None)
  auth = next((r for r in rows if r.get("authority") and "stats" in r), None)
  sens = (peak["stats"]["best_tflops"] / full["stats"]["best_tflops"]) if (peak and full and full["stats"]["best_tflops"]) else None
  full_vs_occ1 = (full["stats"]["best_tflops"] / occ1["stats"]["best_tflops"]) if (full and occ1 and occ1["stats"]["best_tflops"]) else None
  peak_interior = bool(peak and full and occ1 and peak["wg_per_cu"] not in (full["wg_per_cu"], occ1["wg_per_cu"]))
  vgpr_waves = VGPR_PER_SIMD // BK32["vgpr"]

  # classify the occupancy curve SHAPE (peak position), not just the endpoints
  if not clock_valid:
    verdict, interp = "BLOCKED_OCC_DIAG_CLOCK_INVALID", None
    blocker = f"power could not witness active run (samples={len(pv)}, median={power_median})"
  elif sens is None or peak is None:
    verdict, interp = "BLOCKED_OCC_DIAG_LAUNCH", None; blocker = "could not compute occupancy curve"
  elif peak_interior and sens >= 1.08:
    verdict = "CONFIRMED_CONTENTION_LIMITED_OCC_OPTIMUM_INTERIOR"
    interp = (f"interior occupancy optimum at wg={peak['wg_per_cu']} ({peak['stats']['best_tflops']:.1f}); higher "
              f"occupancy (wg={full['wg_per_cu']}={full['stats']['best_tflops']:.1f}) and lower (wg=1="
              f"{occ1['stats']['best_tflops']:.1f}) both worse -> CONTENTION at high occupancy, not latency-hiding")
    blocker = None
  elif peak["wg_per_cu"] == max(r["wg_per_cu"] for r in occ) and sens >= 1.2:
    verdict, interp = "CONFIRMED_LATENCY_BOUND_OCCUPANCY_SENSITIVE", "monotonic: more occupancy = faster (latency-bound)"; blocker = None
  else:
    verdict = "OCC_DIAG_FLAT_OR_PURE_CONTENTION"
    interp = f"flat/monotonic; peak wg={peak['wg_per_cu']}, sens {round(sens,2)}x — not occupancy-latency-bound"
    blocker = "not the latency-via-occupancy mechanism; PMC needed to name the stall"

  result = {**base(verdict), "rows": rows,
            "conclusion_metrics": {"peak_tflops": peak["stats"]["best_tflops"] if peak else None,
                                   "peak_wg_per_cu": peak["wg_per_cu"] if peak else None,
                                   "peak_lds_bytes": peak["lds_bytes"] if peak else None,
                                   "full_occ_tflops": full["stats"]["best_tflops"] if full else None,
                                   "occ1_tflops": occ1["stats"]["best_tflops"] if occ1 else None,
                                   "peak_over_full_x": round(sens, 2) if sens else None,
                                   "full_over_occ1_x": round(full_vs_occ1, 2) if full_vs_occ1 else None,
                                   "peak_is_interior": peak_interior,
                                   "authority_tflops": auth["stats"]["best_tflops"] if auth else None,
                                   "peak_over_authority_x": round(peak["stats"]["best_tflops"] / auth["stats"]["best_tflops"], 2) if (peak and auth) else None,
                                   "vgpr_limited_waves_per_simd": vgpr_waves, "vgpr_per_wave": BK32["vgpr"],
                                   "interpretation": interp},
            "clock": {"power_median_w": power_median, "samples": len(pv), "clock_valid": clock_valid},
            "interleaving_order": [lbl for lbl, _ in timed], "cnt": CNT}
  if blocker: result["exact_blocker"] = blocker
  result["audit_mechanism_check"] = {
    "audit_predicted": "latency-bound, hidden only via occupancy (more occupancy => faster)",
    "measured": interp,
    "audit_latency_mechanism_supported": verdict == "CONFIRMED_LATENCY_BOUND_OCCUPANCY_SENSITIVE",
  }
  if verdict == "CONFIRMED_CONTENTION_LIMITED_OCC_OPTIMUM_INTERIOR":
    result["why_we_are_stuck"] = (
      f"MEASUREMENT CORRECTS THE AUDIT. BK32 is NOT latency-bound-via-occupancy: MORE occupancy (wg="
      f"{full['wg_per_cu']}) is SLOWER ({full['stats']['best_tflops']:.1f}) than the interior optimum wg="
      f"{peak['wg_per_cu']} ({peak['stats']['best_tflops']:.1f}); wg=1 is also slower ({occ1['stats']['best_tflops']:.1f}). "
      "The signature is CONTENTION at high occupancy (4 workgroups/CU thrash LDS bandwidth / barrier / L2), not a "
      "lack of waves to hide latency. ACTIONABLE: the BK32 default (wg4, 16384 B) leaves ~"
      f"{round((sens-1)*100)}% on the table; padding LDS to the wg={peak['wg_per_cu']} sweet spot reaches "
      f"~{peak['stats']['best_tflops']:.0f}, BEATING the authority (~{auth['stats']['best_tflops']:.0f}) by "
      f"~{round((peak['stats']['best_tflops']/auth['stats']['best_tflops']-1)*100)}%. The residual to Tensile (~66) "
      "is consistent with the contention story: Tensile's SIA1/SLW1/PLR1 scheduling reduces LDS/barrier pressure so "
      "it stays efficient WITHOUT dropping occupancy. Next: PMC (LDS-wait / barrier-stall) to name the contention, "
      "NOT more occupancy or tile sweeps.")
  return emit(result)


if __name__ == "__main__":
  raise SystemExit(main())
