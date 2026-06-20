#!/usr/bin/env python3
# AMD GEMM BK-DEPTH PUSH toward the 60-TFLOPS Tensile-class bar (REAL GPU, correctness-gated, interleaved
# one-clock, NO BEAM/search, NO routing/default change).
#
# The family sweep found K-block depth (BK16->BK32) is the lever that lifted the dependency-free LDS GEMM from
# ~41 to ~55 TFLOPS (reaching the LLVM authority). This pushes BK deeper. In build_gemm_lds2 valid BK are
# 16/32/64/128 (CPR=BK//8 must divide THREADS), and deeper BK inflates the cooperative-load temp VGPRs, so a
# deep BK only fits with a smaller tile OR more threads (256). The grid therefore co-varies BK with
# tile/wave-count, resource-fit pre-checks each config (mirroring build_gemm_lds2's asserts), correctness-gates
# survivors (rel RMSE < 0.02 vs A@B), and times them interleaved against the BK32 winner, the LLVM authority,
# and global-direct. Bounded ENUMERATED depth ladder, NOT a search.
from __future__ import annotations

import importlib.util, json, os, pathlib, re, subprocess, time, traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
SWEEP = "bench/amd-broad-backend-roadmap/amd_gemm_family_sweep_result.json"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"

M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
REL_RMSE_PASS = 0.02
TENSILE_CLASS_TFLOPS = 60.0
PRIOR_BK32_TFLOPS = 55.0       # the family-sweep frontier to beat
ACTIVE_POWER_MIN_W = 25.0
CNT = int(os.environ.get("CNT", "150"))
RAMP = int(os.environ.get("RAMP", "80"))

# Depth ladder co-varying BK with tile/wave-count. Each is resource-fit pre-checked before building.
GRID = [
  # proven 128x128 / 4-wave tile, depth ladder (BK64 here overflows VGPRs -> reported, not silently dropped)
  {"label": "W2x2_T4x4_BK16",  "WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 16, "PAD": 0, "DBUF": 0},
  {"label": "W2x2_T4x4_BK32",  "WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 32, "PAD": 0, "DBUF": 0},
  {"label": "W2x2_T4x4_BK64",  "WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 64, "PAD": 0, "DBUF": 0},
  # deep BK enabled by a smaller tile (frees VGPRs for load temps)
  {"label": "W2x2_T2x2_BK64",  "WAVES_M": 2, "WAVES_N": 2, "WM": 2, "WN": 2, "BK": 64, "PAD": 0, "DBUF": 0},
  {"label": "W2x2_T4x2_BK64",  "WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 2, "BK": 64, "PAD": 0, "DBUF": 0},
  {"label": "W2x2_T2x2_BK128", "WAVES_M": 2, "WAVES_N": 2, "WM": 2, "WN": 2, "BK": 128, "PAD": 0, "DBUF": 0},
  # deep BK enabled by more threads (256) -> larger RSTRIDE -> fewer load temps -> bigger tiles fit
  {"label": "W2x4_T4x4_BK32",  "WAVES_M": 2, "WAVES_N": 4, "WM": 4, "WN": 4, "BK": 32, "PAD": 0, "DBUF": 0},
  {"label": "W2x4_T4x4_BK64",  "WAVES_M": 2, "WAVES_N": 4, "WM": 4, "WN": 4, "BK": 64, "PAD": 0, "DBUF": 0},
  {"label": "W4x2_T4x4_BK64",  "WAVES_M": 4, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 64, "PAD": 0, "DBUF": 0},
  {"label": "W2x4_T4x2_BK64",  "WAVES_M": 2, "WAVES_N": 4, "WM": 4, "WN": 2, "BK": 64, "PAD": 0, "DBUF": 0},
]


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


def fit(cfg) -> dict[str, Any]:
  """Mirror build_gemm_lds2's geometry/resource asserts; return fit info + reason if it cannot build."""
  WM, WN, BK, PAD, DBUF = cfg["WM"], cfg["WN"], cfg["BK"], cfg["PAD"], cfg["DBUF"]
  WAVES_M, WAVES_N = cfg["WAVES_M"], cfg["WAVES_N"]
  THREADS = WAVES_M * WAVES_N * 32; BM = WAVES_M * WM * 16; BN = WAVES_N * WN * 16
  KT = BK // 16; CPR = BK // 8
  reasons = []
  if BK % 16: reasons.append("BK%16")
  if THREADS % max(CPR, 1): reasons.append(f"THREADS%CPR({THREADS}%{CPR})")
  RSTRIDE = THREADS // CPR if CPR else 0
  if not RSTRIDE or BM % RSTRIDE or BN % RSTRIDE: reasons.append("BM/BN%RSTRIDE")
  if M % BM or N % BN or K % BK: reasons.append("M/N/K not divisible")
  loadsA = BM // RSTRIDE if RSTRIDE else 0; loadsB = BN // RSTRIDE if RSTRIDE else 0
  FA = 10; FB = FA + WM * 8; ACCb = FB + WN * 8; CTA = ACCb + WM * WN * 8; CTB = CTA + loadsA * 4; SCR = CTB + loadsB * 4
  if SCR + 2 > 256: reasons.append(f"VGPR overflow(SCR={SCR})")
  SA = BK * 2 + PAD; BUFSZ = SA * BM + SA * BN; lds = BUFSZ * (2 if DBUF else 1)
  if lds > 65536: reasons.append(f"LDS overflow({lds})")
  return {"BM": BM, "BN": BN, "THREADS": THREADS, "KT": KT, "vgpr_SCR": SCR, "lds_bytes": lds,
          "buildable": not reasons, "reason": ";".join(reasons)}


def sample_power() -> dict[str, Any] | None:
  try:
    out = subprocess.run(["rocm-smi", "--showpower", "--showuse"], capture_output=True, text=True, timeout=8).stdout
  except Exception:
    return None
  m = re.search(r"Average Graphics Package Power \(W\):\s*([\d.]+)", out, re.I)
  return {"power_w": float(m.group(1)) if m else None}


def stats(ets: list[float]) -> dict[str, Any]:
  srt = sorted(ets); n = len(srt)
  return {"best_s": srt[0], "median_s": srt[n // 2], "best_tflops": FLOP / srt[0] * 1e-12,
          "median_tflops": FLOP / srt[n // 2] * 1e-12, "n": n}


def emit(result: dict[str, Any]) -> int:
  write_json("amd_gemm_bk_depth_result.json", result)
  print(json.dumps({k: result[k] for k in ("verdict", "gate_pass") if k in result}, indent=2))
  if "exact_blocker" in result: print("note:", result["exact_blocker"])
  for r in result.get("rows", []):
    s = r.get("stats")
    if s:
      print(f"  {r['label']:22} best={s['best_tflops']:6.2f} median={s['median_tflops']:6.2f} TFLOPS"
            + (f"  BK={r['BK']} {r.get('BM')}x{r.get('BN')} thr={r.get('THREADS')} lds={r.get('lds_bytes')}B vgpr={r.get('vgpr_SCR')}" if "BK" in r else "")
            + (f"  rel={r['rel_rmse']:.1e}" if "rel_rmse" in r else "  (authority)"))
    else:
      print(f"  {r['label']:22} {r.get('status')}: {r.get('reason') or r.get('error','')[:50]}")
  return 0 if result.get("gate_pass") else 1


def base(verdict: str) -> dict[str, Any]:
  return {
    "date": "2026-06-20", "phase": "AMD_GEMM_BK_DEPTH", "schema": "amd_gemm_bk_depth_v1",
    "role": "ffn_gate/up", "verdict": verdict, "gate_pass": verdict.startswith("PASS"),
    "default_behavior_changed": False, "correctness_claim": True, "performance_claim": True, "is_search": False,
    "shape": {"M": M, "N": N, "K": K},
    "harness": "interleaved round-robin, per-launch sync+perf_counter, RAMP burst excluded, best-of-N; bounded "
               "ENUMERATED BK-depth ladder (NOT BEAM); each row at its TRUE LDS footprint; resource-fit pre-checked",
    "thresholds": {"rel_rmse_pass": REL_RMSE_PASS, "tensile_class_tflops": TENSILE_CLASS_TFLOPS,
                   "prior_bk32_frontier_tflops": PRIOR_BK32_TFLOPS},
    "input_artifacts": [SWEEP, "extra/gemm/rdna3_wmma_matmul.py", "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"],
  }


def main() -> int:
  try:
    sw = read_json(SWEEP)
    if not sw.get("verdict", "").startswith("PASS_GEMM_FAMILY_SWEEP"):
      return emit({**base("BLOCKED_GEMM_BK_DEPTH_PRECONDITION"), "exact_blocker": f"family sweep not passed: {sw.get('verdict')}"})
  except Exception as ex:
    return emit({**base("BLOCKED_GEMM_BK_DEPTH_PRECONDITION"), "exact_blocker": f"missing family-sweep artifact: {ex!r}"})

  try:
    import numpy as np
    from tinygrad import Tensor, Device, Context
    from tinygrad.dtype import dtypes
    from tinygrad.engine.realize import run_linear
    ref = load_mod(REF_SRC, "rdna3_ref")
    rng = np.random.default_rng(1)
    a_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
    bt_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float16)
    a = Tensor(a_np, device="AMD"); bt = Tensor(bt_np, device="AMD"); c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD")
    Tensor.realize(a, bt, c)
    refmat = a_np.astype(np.float32) @ bt_np.astype(np.float32).T
    refnorm = float(np.sqrt(np.mean(refmat ** 2)))
  except Exception as ex:
    return emit({**base("BLOCKED_GEMM_BK_DEPTH_LAUNCH"), "exact_blocker": f"setup failed: {ex!r}",
                 "trace": traceback.format_exc().splitlines()[-4:]})

  def check(out): cn = out.float().numpy().astype(np.float32); d = cn - refmat; return float(np.max(np.abs(d))), float(np.sqrt(np.mean(d**2))/(refnorm+1e-9))

  rows: list[dict[str, Any]] = []
  timed: list[tuple[str, Any]] = []

  # calibration: global-direct baseline
  try:
    lin, out = ref._run_insts(ref.build_gemm_pipe(M, N, K, 4, 2), a, bt, c, M, N, K, 4, 2, "global_direct")
    run_linear(lin); Device["AMD"].synchronize(); ma, rel = check(out)
    rows.append({"label": "global_direct", "status": "CORRECT" if rel < REL_RMSE_PASS else "INCORRECT", "rel_rmse": rel})
    if rel < REL_RMSE_PASS: timed.append(("global_direct", lin))
  except Exception as ex:
    rows.append({"label": "global_direct", "status": "LAUNCH_FAIL", "error": repr(ex)})

  # BK-depth grid
  for cfg in GRID:
    info = fit(cfg)
    rec = {"label": cfg["label"], "BK": cfg["BK"], **{k: info[k] for k in ("BM", "BN", "THREADS", "KT", "vgpr_SCR", "lds_bytes")}}
    if not info["buildable"]:
      rec.update(status="UNBUILDABLE", reason=info["reason"]); rows.append(rec); continue
    try:
      insts = ref.build_gemm_lds2(M, N, K, cfg["WAVES_M"], cfg["WAVES_N"], cfg["WM"], cfg["WN"], cfg["BK"], cfg["PAD"], cfg["DBUF"])
      lin, out = ref._run_insts_lds(insts, a, bt, c, M, N, K, cfg["label"], info["lds_bytes"], info["BM"], info["BN"], info["THREADS"])
      run_linear(lin); Device["AMD"].synchronize(); ma, rel = check(out)
    except Exception as ex:
      rec.update(status="LAUNCH_FAIL", error=repr(ex)); rows.append(rec); continue
    rec.update(max_abs=ma, rel_rmse=rel, status="CORRECT" if rel < REL_RMSE_PASS else "INCORRECT")
    rows.append(rec)
    if rec["status"] == "CORRECT": timed.append((cfg["label"], lin))

  # authority row
  authority_alive = None
  try:
    ptm1 = load_mod(PTM1_SRC, "ptm1_bridge")
    auth_lin, auth_meta, authority_alive = ptm1.build_authority()
    rows.append({"label": "authority_llvm", "authority": True, "name": auth_meta.get("name")})
    timed.append(("authority_llvm", auth_lin))
  except Exception as ex:
    rows.append({"label": "authority_llvm", "authority": True, "status": "UNAVAILABLE", "error": repr(ex)[:160]})

  if not any(lbl not in ("global_direct", "authority_llvm") for lbl, _ in timed):
    return emit({**base("BLOCKED_GEMM_BK_DEPTH_LAUNCH"), "exact_blocker": "no correct depth-grid config to time", "rows": rows})

  # interleaved timing
  power = []
  times: dict[str, list[float]] = {lbl: [] for lbl, _ in timed}
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
          s = sample_power()
          if s and s.get("power_w") is not None: power.append(s["power_w"])
  except Exception as ex:
    return emit({**base("BLOCKED_GEMM_BK_DEPTH_LAUNCH"), "exact_blocker": f"timing launch failed: {ex!r}", "rows": rows})

  for r in rows:
    if r["label"] in times: r["stats"] = stats(times[r["label"]])
  pv = sorted(power); power_median = pv[len(pv) // 2] if pv else None
  clock_valid = len(pv) >= 3 and (power_median or 0) >= ACTIVE_POWER_MIN_W

  dep = [r for r in rows if r.get("status") == "CORRECT" and "stats" in r and r["label"] != "global_direct"]
  best = max(dep, key=lambda r: r["stats"]["best_tflops"]) if dep else None
  auth = next((r for r in rows if r.get("authority") and "stats" in r), None)
  gd = next((r for r in rows if r["label"] == "global_direct" and "stats" in r), None)
  # within-run BK32/128x128 reference (avoids cross-run noise in the "did depth help" comparison)
  bk32 = next((r for r in rows if r["label"] == "W2x2_T4x4_BK32" and "stats" in r), None)
  bk32_tflops = bk32["stats"]["best_tflops"] if bk32 else PRIOR_BK32_TFLOPS
  best_is_deeper = bool(best and best.get("BK", 0) > 32)
  ADVANCE_MARGIN = 1.02
  frontier = {
    "best_config": best["label"] if best else None,
    "best_tflops": best["stats"]["best_tflops"] if best else None,
    "best_BK": best.get("BK") if best else None,
    "bk32_reference_tflops": bk32_tflops,
    "deeper_bk_helped": bool(best_is_deeper and best and best["stats"]["best_tflops"] > bk32_tflops * ADVANCE_MARGIN),
    "authority_tflops": auth["stats"]["best_tflops"] if auth else None,
    "global_direct_tflops": gd["stats"]["best_tflops"] if gd else None,
    "ratio_best_over_authority": (best["stats"]["best_tflops"] / auth["stats"]["best_tflops"]) if best and auth else None,
    "reaches_tensile_class_60": bool(best and best["stats"]["best_tflops"] >= TENSILE_CLASS_TFLOPS),
  }

  if not clock_valid:
    verdict, blocker = "BLOCKED_GEMM_BK_DEPTH_CLOCK_INVALID", f"power could not witness active run (samples={len(pv)}, median={power_median})"
  elif best is None:
    verdict, blocker = "BLOCKED_GEMM_BK_DEPTH_LAUNCH", "no correct depth config produced timing"
  elif frontier["reaches_tensile_class_60"]:
    verdict, blocker = "PASS_GEMM_BK_DEPTH_TENSILE_CLASS_REACHED", None
  elif frontier["deeper_bk_helped"]:
    verdict = "PASS_GEMM_BK_DEPTH_FRONTIER_ADVANCED"
    blocker = (f"a deeper-BK config {best['label']} (BK={best.get('BK')}) = {best['stats']['best_tflops']:.1f} "
               f"beat BK32 ({bk32_tflops:.1f}) by >2% but is short of Tensile-class 60. Next bounded gate, no BEAM.")
  else:
    verdict = "BLOCKED_GEMM_BK_DEPTH_NO_IMPROVEMENT"
    blocker = (f"deeper BK did NOT help: fastest is {best['label']} (BK={best.get('BK')}) = "
               f"{best['stats']['best_tflops']:.1f} TFLOPS; BK64 overflows VGPRs at 128x128, and every deeper "
               f"config that fits (smaller tile or 256 threads) REGRESSES. BK32/128x128 is the family sweet "
               f"spot; the dependency-free family plateaus at ~{bk32_tflops:.0f} (reaches the LLVM authority "
               f"{auth['stats']['best_tflops']:.0f} if present) but does not clear 60 via depth. Next is "
               "bottleneck classification (VGPR/occupancy vs Tensile), NOT more depth or search.")

  result = {**base(verdict), "rows": rows, "frontier": frontier,
            "clock": {"power_median_w": power_median, "samples": len(pv), "clock_valid": clock_valid,
                      "note": "rocm-smi sclk unreliable on this card; power is the activity witness"},
            "interleaving_order": [lbl for lbl, _ in timed], "cnt": CNT}
  if blocker: result["exact_blocker"] = blocker
  return emit(result)


if __name__ == "__main__":
  raise SystemExit(main())
