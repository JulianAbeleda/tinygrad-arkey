#!/usr/bin/env python3
# AMD GEMM SHAPE-SPECIFIC tile sweep (REAL GPU, pinned clock, correctness-gated, interleaved, NO BEAM).
#
# Hypothesis (from the Tensile tuning-table audit): our shape (M=512 small, N=12288 large, K=4096) is UNTUNED
# in Tensile -> the square 128x128 kernel is a nearest-neighbor fallback running ~22% off its potential. A
# NON-SQUARE tile matched to the skewed aspect (small M -> smaller BM for more M-parallelism; large N -> wider
# BN for reuse) may beat 128x128. Sweep tile shapes at our otherwise-best config (BK32, PAD16, PLRA where it
# fits), all at wg2 LDS=32768 (isolates tile shape from occupancy), pinned, interleaved vs current + authority.
#
# Run:  DEV=AMD PYTHONPATH=. python3 extra/qk_amd_gemm_shape_tile_sweep.py   (sets perflevel high, resets auto)
from __future__ import annotations

import importlib.util, json, os, pathlib, subprocess, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"
M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
LDS = 32768
CNT = int(os.environ.get("CNT", "150")); RAMP = int(os.environ.get("RAMP", "80"))
# (label, WAVES_M, WAVES_N, WM, WN) -> BM=WAVES_M*WM*16, BN=WAVES_N*WN*16. BK32, PAD16.
CONFIGS = [
  ("BM128xBN128_sq",  2, 2, 4, 4),   # current best (square) -- reference
  ("BM64xBN128",      2, 2, 2, 4),   # smaller M tile -> 8 M-blocks (more M-parallelism)
  ("BM64xBN256",      2, 2, 2, 8),   # small M, wide N
  ("BM64xBN192",      2, 2, 2, 6),   # small M, mid N (12288/192=64)
  ("BM128xBN256_2x4", 2, 4, 4, 4),   # wide N, 256 threads
  ("BM64xBN256_1x4",  1, 4, 4, 4),   # small M, wide N, 1x4 waves
  ("BM256xBN128_4x2", 4, 2, 4, 4),   # big M (only 2 M-blocks) -- expect worse, informative
]


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)
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

  result: dict[str, Any] = {"date": "2026-06-20", "phase": "AMD_GEMM_SHAPE_TILE_SWEEP", "schema": "amd_gemm_shape_tile_v1",
                            "role": "ffn_gate/up", "default_behavior_changed": False, "performance_claim": True, "is_search": False,
                            "hypothesis": "non-square tile beats square 128x128 for the M-small/N-large untuned shape"}
  built = {}
  for label, wm_, wn_, WM, WN in CONFIGS:
    BM = wm_ * WM * 16; BN = wn_ * WN * 16; THREADS = wm_ * wn_ * 32
    bufsz = (32 * 2 + 16) * (BM + BN)
    rec = {"BM": BM, "BN": BN, "THREADS": THREADS, "grid": [N // BN, M // BM, 1] if (N % BN == 0 and M % BM == 0) else None}
    try:
      assert M % BM == 0 and N % BN == 0, f"M/N not divisible by tile {BM}x{BN}"
      assert bufsz <= LDS, f"BUFSZ {bufsz} > {LDS}"
      plra = 1
      try: insts = ref.build_gemm_lds2(M, N, K, wm_, wn_, WM, WN, 32, 16, 0, plra)
      except AssertionError: plra = 0; insts = ref.build_gemm_lds2(M, N, K, wm_, wn_, WM, WN, 32, 16, 0, 0)
      lin, out = ref._run_insts_lds(insts, a, bt, c, M, N, K, label, LDS, BM, BN, THREADS)
      run_linear(lin); dev.synchronize(); rel = chk(out)
      rec.update(plra=plra, rel_rmse=rel, correct=rel < 0.02, lin=lin if rel < 0.02 else None)
    except Exception as ex:
      rec.update(correct=False, error=repr(ex)[:120])
    built[label] = rec

  launches = [(l, lambda lin=built[l]["lin"]: run_linear(lin)) for l, *_ in CONFIGS if built[l].get("correct")]
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

  result["rows"] = {l: {**({k: built[l][k] for k in ("BM", "BN", "THREADS", "grid", "plra", "rel_rmse")} if l in built else {"authority": True}), **stats(times[l])} for l, _ in launches}
  result["builds"] = {l: ("ok" if built[l].get("correct") else built[l].get("error", "incorrect")) for l, *_ in CONFIGS}
  t = result["rows"]
  sq = t.get("BM128xBN128_sq", {}).get("best_tflops"); auth = t.get("authority_llvm", {}).get("best_tflops")
  best = max(((l, t[l]["best_tflops"]) for l in t if l != "authority_llvm"), key=lambda x: x[1], default=(None, 0))
  result["analysis"] = {"square_128_tflops": sq, "best_tile": best[0], "best_tflops": best[1], "authority_tflops": auth,
                        "best_over_square_x": round(best[1] / sq, 3) if (sq and best[1]) else None}
  if best[0] and sq:
    if best[0] != "BM128xBN128_sq" and best[1] > sq * 1.02:
      result["verdict"] = "NONSQUARE_TILE_WINS"; result["why"] = f"{best[0]} ({best[1]:.1f}) beats square 128x128 ({sq:.1f}) by {round((best[1]/sq-1)*100)}% -> shape-specific tiling helps the skewed aspect."
    else:
      result["verdict"] = "SQUARE_128_STILL_BEST"; result["why"] = f"square 128x128 ({sq:.1f}) is still best (or within noise); non-square tiles don't beat it for this shape. best={best[0]} {best[1]:.1f}."
  else:
    result["verdict"] = "INCOMPLETE"
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "amd_gemm_shape_tile_sweep_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result.get("verdict"), "builds": result["builds"],
                    "tflops": {k: round(v["best_tflops"], 1) for k, v in sorted(t.items(), key=lambda x: -x[1]["best_tflops"])},
                    "analysis": result["analysis"], "why": result.get("why")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
