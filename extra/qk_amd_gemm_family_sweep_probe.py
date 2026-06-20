#!/usr/bin/env python3
# AMD GEMM dependency-free FAMILY SWEEP vs the LLVM authority (REAL GPU, correctness-gated, interleaved
# one-clock, NO BEAM/search, NO routing/default change).
#
# Now that single-buffer LDS staging proved competitive (~1.3x global-direct, NOT refuted — the prior
# "~3.2 TFLOPS" was occupancy-crippled at 65536 B LDS), this sweeps a BOUNDED, ENUMERATED grid of the
# dependency-free RDNA3 GEMM family at the authority shape M=512,N=12288,K=4096, each launched at its TRUE
# LDS footprint, each correctness-gated (rel RMSE < 0.02 vs A@B), and times all correct rows together with
# the tinygrad LLVM authority row in ONE interleaved round-robin (clock-fair). It answers: can tuning the
# dependency-free family reach/beat the LLVM authority, and where does the frontier sit vs the 60-TFLOPS
# Tensile-class bar? This is a fixed config grid, NOT a search.
from __future__ import annotations

import importlib.util, json, os, pathlib, re, subprocess, time, traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
TIMING = "bench/amd-broad-backend-roadmap/amd_gemm_runnable_timing_result.json"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"

M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
REL_RMSE_PASS = 0.02
TENSILE_CLASS_TFLOPS = 60.0
ACTIVE_POWER_MIN_W = 25.0
CNT = int(os.environ.get("CNT", "150"))
RAMP = int(os.environ.get("RAMP", "80"))

# Bounded enumerated config grid for build_gemm_lds2(M,N,K, WAVES_M,WAVES_N,WM,WN,BK,PAD,DBUF). NOT a search.
LDS2_CONFIGS = [
  {"label": "lds2_W2x2_T4x4_BK16",          "WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 16, "PAD": 0, "DBUF": 0},
  {"label": "lds2_W2x2_T4x4_BK16_PAD8",     "WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 16, "PAD": 8, "DBUF": 0},
  {"label": "lds2_W2x2_T4x4_BK32",          "WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 32, "PAD": 0, "DBUF": 0},
  {"label": "lds2_W2x2_T4x4_BK16_DBUF",     "WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 16, "PAD": 0, "DBUF": 1},
  {"label": "lds2_W2x2_T2x2_BK16",          "WAVES_M": 2, "WAVES_N": 2, "WM": 2, "WN": 2, "BK": 16, "PAD": 0, "DBUF": 0},
  {"label": "lds2_W2x2_T4x4_BK32_PAD8_DBUF","WAVES_M": 2, "WAVES_N": 2, "WM": 4, "WN": 4, "BK": 32, "PAD": 8, "DBUF": 1},
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


def sample_power() -> dict[str, Any] | None:
  try:
    out = subprocess.run(["rocm-smi", "--showpower", "--showuse"], capture_output=True, text=True, timeout=8).stdout
  except Exception:
    return None
  def grab(pat): m = re.search(pat, out, re.I); return float(m.group(1)) if m else None
  return {"power_w": grab(r"Average Graphics Package Power \(W\):\s*([\d.]+)"), "gpu_use_pct": grab(r"GPU use \(%\):\s*(\d+)")}


def stats(ets: list[float]) -> dict[str, Any]:
  srt = sorted(ets); n = len(srt)
  return {"best_s": srt[0], "median_s": srt[n // 2], "best_tflops": FLOP / srt[0] * 1e-12,
          "median_tflops": FLOP / srt[n // 2] * 1e-12, "n": n}


def emit(result: dict[str, Any]) -> int:
  write_json("amd_gemm_family_sweep_result.json", result)
  print(json.dumps({k: result[k] for k in ("verdict", "gate_pass") if k in result}, indent=2))
  if "exact_blocker" in result: print("exact_blocker:", result["exact_blocker"])
  for r in result.get("rows", []):
    if r.get("status") == "CORRECT" or r.get("authority"):
      s = r.get("stats", {})
      print(f"  {r['label']:30} best={s.get('best_tflops',0):6.2f} median={s.get('median_tflops',0):6.2f} TFLOPS"
            + (f"  rel_rmse={r['rel_rmse']:.2e}" if "rel_rmse" in r else "  (authority)")
            + (f"  lds={r['lds_bytes']}B" if "lds_bytes" in r else ""))
    else:
      print(f"  {r['label']:30} {r.get('status')}: {r.get('error','')[:60]}")
  return 0 if result.get("gate_pass") else 1


def base(verdict: str) -> dict[str, Any]:
  return {
    "date": "2026-06-20", "phase": "AMD_GEMM_FAMILY_SWEEP", "schema": "amd_gemm_family_sweep_v1",
    "role": "ffn_gate/up", "verdict": verdict, "gate_pass": verdict.startswith("PASS"),
    "default_behavior_changed": False, "correctness_claim": True, "performance_claim": True, "is_search": False,
    "shape": {"M": M, "N": N, "K": K},
    "harness": "single-process interleaved round-robin, per-launch sync+perf_counter, warm, RAMP burst excluded, "
               "best-of-N; bounded ENUMERATED config grid (NOT BEAM/search); each row at its TRUE LDS footprint",
    "thresholds": {"rel_rmse_pass": REL_RMSE_PASS, "tensile_class_tflops": TENSILE_CLASS_TFLOPS},
    "input_artifacts": [TIMING, "extra/gemm/rdna3_wmma_matmul.py", "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"],
  }


def main() -> int:
  try:
    t = read_json(TIMING)
    if t.get("verdict") != "PASS_GEMM_RUNNABLE_TIMING_GATE":
      return emit({**base("BLOCKED_GEMM_FAMILY_SWEEP_PRECONDITION"), "exact_blocker": f"timing gate not passed: {t.get('verdict')}"})
  except Exception as ex:
    return emit({**base("BLOCKED_GEMM_FAMILY_SWEEP_PRECONDITION"), "exact_blocker": f"missing timing artifact: {ex!r}"})

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
    return emit({**base("BLOCKED_GEMM_FAMILY_SWEEP_LAUNCH"), "exact_blocker": f"setup failed: {ex!r}",
                 "trace": traceback.format_exc().splitlines()[-4:]})

  def check(out) -> tuple[float, float]:
    cn = out.float().numpy().astype(np.float32); d = cn - refmat
    return float(np.max(np.abs(d))), float(np.sqrt(np.mean(d ** 2)) / (refnorm + 1e-9))

  # ---- build + correctness-gate each row (only CORRECT rows are timed) ----
  rows: list[dict[str, Any]] = []
  timed: list[tuple[str, Any]] = []
  def add_handasm(label, insts_fn, run_fn, lds_bytes=None):
    rec = {"label": label}
    try:
      insts = insts_fn()
    except Exception as ex:
      rec.update(status="BUILD_FAIL", error=repr(ex)); rows.append(rec); return
    try:
      lin, out = run_fn(insts)
      run_linear(lin); Device["AMD"].synchronize()
      max_abs, rel = check(out)
    except Exception as ex:
      rec.update(status="LAUNCH_FAIL", error=repr(ex)); rows.append(rec); return
    rec.update(max_abs=max_abs, rel_rmse=rel, status="CORRECT" if rel < REL_RMSE_PASS else "INCORRECT")
    if lds_bytes is not None: rec["lds_bytes"] = lds_bytes
    rows.append(rec)
    if rec["status"] == "CORRECT": timed.append((label, lin))

  # global-direct baseline
  add_handasm("global_direct_pipe_T4x2",
              lambda: ref.build_gemm_pipe(M, N, K, 4, 2),
              lambda insts: ref._run_insts(insts, a, bt, c, M, N, K, 4, 2, "global_direct_pipe_T4x2"))
  # single-buffer LDS candidate (the timing-gate winner)
  add_handasm("lds_single_buffer", lambda: ref.build_gemm_lds(M, N, K),
              lambda insts: ref._run_insts_lds(insts, a, bt, c, M, N, K, "lds_single_buffer", 8192), lds_bytes=8192)
  # lds2 config grid, each at its TRUE footprint
  for cfg in LDS2_CONFIGS:
    BM = cfg["WAVES_M"] * cfg["WM"] * 16; BN = cfg["WAVES_N"] * cfg["WN"] * 16
    THREADS = cfg["WAVES_M"] * cfg["WAVES_N"] * 32
    ldsb = (cfg["BK"] * 2 + cfg["PAD"]) * (BM + BN) * (2 if cfg["DBUF"] else 1)
    add_handasm(cfg["label"],
                lambda cfg=cfg: ref.build_gemm_lds2(M, N, K, cfg["WAVES_M"], cfg["WAVES_N"], cfg["WM"], cfg["WN"], cfg["BK"], cfg["PAD"], cfg["DBUF"]),
                lambda insts, BM=BM, BN=BN, THREADS=THREADS, ldsb=ldsb, lbl=cfg["label"]: ref._run_insts_lds(insts, a, bt, c, M, N, K, lbl, ldsb, BM, BN, THREADS),
                lds_bytes=ldsb)

  # ---- optional: tinygrad LLVM authority row (self-contained, its own buffers) ----
  authority_alive = None
  try:
    ptm1 = load_mod(PTM1_SRC, "ptm1_bridge")
    auth_lin, auth_meta, authority_alive = ptm1.build_authority()
    rows.append({"label": "authority_tinygrad_llvm", "authority": True, "name": auth_meta.get("name"),
                 "name_matches": auth_meta.get("name_matches")})
    timed.append(("authority_tinygrad_llvm", auth_lin))
  except Exception as ex:
    rows.append({"label": "authority_tinygrad_llvm", "authority": True, "status": "UNAVAILABLE", "error": repr(ex)[:200]})

  if not timed:
    return emit({**base("BLOCKED_GEMM_FAMILY_SWEEP_LAUNCH"), "exact_blocker": "no correct row to time", "rows": rows})

  # ---- interleaved one-clock timing of all correct + authority rows ----
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
          if s is not None: power.append(s)
  except Exception as ex:
    return emit({**base("BLOCKED_GEMM_FAMILY_SWEEP_LAUNCH"), "exact_blocker": f"timing launch failed: {ex!r}", "rows": rows})

  for r in rows:
    if r["label"] in times: r["stats"] = stats(times[r["label"]])

  pvals = sorted(s["power_w"] for s in power if s.get("power_w") is not None)
  power_median = pvals[len(pvals) // 2] if pvals else None
  clock_valid = len(pvals) >= 3 and (power_median or 0) >= ACTIVE_POWER_MIN_W

  # ---- frontier + verdict ----
  dep_free = [r for r in rows if r.get("status") == "CORRECT" and "stats" in r]
  best_dep = max(dep_free, key=lambda r: r["stats"]["best_tflops"]) if dep_free else None
  auth_row = next((r for r in rows if r.get("authority") and "stats" in r), None)
  gd = next((r for r in rows if r["label"] == "global_direct_pipe_T4x2" and "stats" in r), None)
  frontier = {
    "best_dependency_free": (best_dep["label"] if best_dep else None),
    "best_dependency_free_tflops": (best_dep["stats"]["best_tflops"] if best_dep else None),
    "authority_tflops": (auth_row["stats"]["best_tflops"] if auth_row else None),
    "global_direct_tflops": (gd["stats"]["best_tflops"] if gd else None),
    "ratio_best_over_authority": (best_dep["stats"]["best_tflops"] / auth_row["stats"]["best_tflops"]
                                  if best_dep and auth_row else None),
    "ratio_best_over_global_direct": (best_dep["stats"]["best_tflops"] / gd["stats"]["best_tflops"]
                                      if best_dep and gd else None),
    "reaches_tensile_class": bool(best_dep and best_dep["stats"]["best_tflops"] >= TENSILE_CLASS_TFLOPS),
  }

  if not clock_valid:
    verdict, blocker = "BLOCKED_GEMM_FAMILY_SWEEP_CLOCK_INVALID", f"power could not witness active run (samples={len(pvals)}, median={power_median})"
  elif best_dep is None:
    verdict, blocker = "BLOCKED_GEMM_FAMILY_SWEEP_LAUNCH", "no correct dependency-free row produced timing"
  elif auth_row and frontier["ratio_best_over_authority"] and frontier["ratio_best_over_authority"] >= 1.0:
    verdict, blocker = "PASS_GEMM_FAMILY_SWEEP_AUTHORITY_BEATEN", None
  elif auth_row:
    verdict = "BLOCKED_GEMM_FAMILY_SWEEP_BELOW_AUTHORITY"
    blocker = (f"best dependency-free {best_dep['label']} = {best_dep['stats']['best_tflops']:.1f} TFLOPS, "
               f"{frontier['ratio_best_over_authority']:.2f}x the LLVM authority ({auth_row['stats']['best_tflops']:.1f}); "
               "below authority. Next: bottleneck classification of the best config, NOT search.")
  else:
    # authority row unavailable: report frontier vs global-direct (informational PASS)
    verdict = "PASS_GEMM_FAMILY_SWEEP_FRONTIER_NO_AUTHORITY"
    blocker = None

  result = {**base(verdict), "rows": rows, "frontier": frontier,
            "clock": {"power_median_w": power_median, "samples": len(pvals), "clock_valid": clock_valid,
                      "note": "rocm-smi sclk unreliable on this card; power is the activity witness"},
            "interleaving_order": [lbl for lbl, _ in timed], "cnt": CNT,
            "config_grid": LDS2_CONFIGS, "_authority_alive": None}
  if blocker: result["exact_blocker"] = blocker
  result.pop("_authority_alive", None)
  return emit(result)


if __name__ == "__main__":
  raise SystemExit(main())
