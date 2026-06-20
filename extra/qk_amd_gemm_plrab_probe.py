#!/usr/bin/env python3
# AMD GEMM full A+B PLR test (REAL GPU, pinned clock, interleaved, correctness-gated, NO BEAM).
#
# Full A+B prefetch (PLRAB) needs a 2nd A+B fragment buffer (WM*8+WN*8 VGPR) -> overflows the 128x128 (4x4)
# tile at 256 VGPR, but FITS at a slightly smaller tile (WM=4,WN=3 = 128x96). The experiment: does FULL
# prefetch at the smaller tile (less WMMA reuse, full latency hiding) beat A-ONLY prefetch at the full 4x4
# tile (more reuse, half hiding = the current ~57 clock-matched best)?
#
# Compares, pinned clock, interleaved: wn4 {plra0, plra1}, wn3 {plra0, plra1, plrab1}, authority.
# Run:  DEV=AMD PYTHONPATH=. python3 extra/qk_amd_gemm_plrab_probe.py   (sets perflevel high, resets auto)
from __future__ import annotations

import importlib.util, json, os, pathlib, subprocess, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"
M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
REL_RMSE_PASS = 0.02
LDS = 32768
CNT = int(os.environ.get("CNT", "150")); RAMP = int(os.environ.get("RAMP", "80"))
# (label, WN, PLRA, PLRAB)
CONFIGS = [
  ("wn4_plra0", 4, 0, 0), ("wn4_plra1", 4, 1, 0),
  ("wn3_plra0", 3, 0, 0), ("wn3_plra1", 3, 1, 0), ("wn3_plrab1", 3, 0, 1),
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

  result: dict[str, Any] = {"date": "2026-06-20", "phase": "AMD_GEMM_PLRAB", "schema": "amd_gemm_plrab_v1", "role": "ffn_gate/up",
                            "default_behavior_changed": False, "performance_claim": True}
  built = {}
  for label, wn, plra, plrab in CONFIGS:
    bn = 2 * wn * 16
    try:
      insts = ref.build_gemm_lds2(M, N, K, 2, 2, 4, wn, 32, 16, 0, plra, plrab)
      lin, out = ref._run_insts_lds(insts, a, bt, c, M, N, K, label, LDS, 128, bn, 128)
      run_linear(lin); dev.synchronize(); rel = chk(out)
      built[label] = {"lin": lin, "rel_rmse": rel, "correct": rel < REL_RMSE_PASS, "BN": bn, "tile": f"4x{wn}"}
    except Exception as ex:
      built[label] = {"error": repr(ex)[:120], "correct": False}
  launches = [(l, lambda lin=built[l]["lin"]: run_linear(lin)) for l, *_ in CONFIGS if built[l].get("correct")]
  try:
    ptm1 = load_mod(PTM1_SRC, "ptm1_bridge"); auth_lin, _m, _alive = ptm1.build_authority(); launches.append(("authority_llvm", lambda: run_linear(auth_lin)))
  except Exception: pass

  perflevel("high")
  times = {l: [] for l, _ in launches}
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

  result["timing"] = {l: {**stats(times[l]), **({"rel_rmse": built[l]["rel_rmse"], "tile": built[l]["tile"]} if l in built else {"authority": True})} for l, _ in launches}
  result["builds"] = {l: ("ok" if built[l].get("correct") else built[l].get("error", built[l].get("rel_rmse"))) for l, *_ in CONFIGS}
  t = result["timing"]
  a_only_44 = t.get("wn4_plra1", {}).get("best_tflops"); full_43 = t.get("wn3_plrab1", {}).get("best_tflops")
  auth = t.get("authority_llvm", {}).get("best_tflops")
  result["analysis"] = {"a_only_4x4_tflops": a_only_44, "full_AB_4x3_tflops": full_43, "authority_tflops": auth,
                        "full_over_a_only_x": round(full_43 / a_only_44, 3) if (a_only_44 and full_43) else None,
                        "best_config": max((l for l in t if l != "authority_llvm"), key=lambda l: t[l]["best_tflops"], default=None),
                        "best_tflops": max((t[l]["best_tflops"] for l in t if l != "authority_llvm"), default=None)}
  if a_only_44 and full_43:
    if full_43 > a_only_44 * 1.02:
      result["verdict"] = "FULL_AB_PLR_WINS"; result["why"] = f"full A+B PLR @4x3 ({full_43:.1f}) beats A-only @4x4 ({a_only_44:.1f}) by {round((full_43/a_only_44-1)*100)}% -- full latency hiding > the smaller-tile reuse loss."
    elif full_43 < a_only_44 * 0.98:
      result["verdict"] = "FULL_AB_PLR_LOSES_TILE_COST"; result["why"] = f"full A+B PLR @4x3 ({full_43:.1f}) < A-only @4x4 ({a_only_44:.1f}) -- the WN=3 reuse loss outweighs full latency hiding. A-only @4x4 stays best."
    else:
      result["verdict"] = "FULL_AB_PLR_TIE"; result["why"] = f"full A+B @4x3 ({full_43:.1f}) ~ A-only @4x4 ({a_only_44:.1f}); the tile-loss and the extra hiding cancel."
  else:
    result["verdict"] = "INCOMPLETE"
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "amd_gemm_plrab_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result.get("verdict"), "builds": result["builds"],
                    "timing": {k: round(v["best_tflops"], 1) for k, v in t.items()},
                    "analysis": result["analysis"], "why": result.get("why")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
