#!/usr/bin/env python3
# FACT-CHECK: is Tensile's M=384=79 TFLOPS a shape-intrinsic sweet-spot, a Tensile-specific win, or clock?
#
# Run OUR kernel (build_gemm_lds2 BK32+PAD16+PLRA1, square 128x128) on Tensile's TUNED shapes at PINNED clock
# and compare to the .dat's claimed speeds. If ours ALSO peaks at M=384 (>> M=256/768) -> the sweet-spot is
# SHAPE-INTRINSIC (real, alignment/tiling) and our M=512 genuinely can't hit it. If ours is FLAT across M ->
# the 79 is Tensile-specific or a boost-clock artifact in the offline .dat. Also tells us if 79 is even
# reachable at a stable pinned clock.
#
# Run:  DEV=AMD PYTHONPATH=. python3 extra/qk_amd_gemm_shape_factcheck.py   (sets perflevel high, resets auto)
from __future__ import annotations

import importlib.util, json, os, pathlib, subprocess, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
CNT = int(os.environ.get("CNT", "200")); RAMP = int(os.environ.get("RAMP", "100"))
# (M, N, K, tensile_dat_speed)  -- tuned shapes from the .dat (N=8192,K=4096 cluster) + our shape
SHAPES = [
  (256, 8192, 4096, 64.7),
  (384, 8192, 4096, 79.1),   # the claimed outlier sweet-spot
  (512, 8192, 4096, None),   # our M, tuned N -- interpolation point
  (768, 8192, 4096, 65.3),
  (1536, 8192, 4096, 68.2),
  (512, 12288, 4096, None),  # OUR shape (untuned)
]


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)


def main() -> int:
  import numpy as np
  from tinygrad import Tensor, Device, Context
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import run_linear
  dev = Device["AMD"]; ref = load_mod(REF_SRC, "rdna3_ref")
  perflevel("high")
  rows = []
  try:
    for M, N, K, dat in SHAPES:
      flop = 2 * M * N * K
      assert M % 128 == 0 and N % 128 == 0 and K % 32 == 0
      rng = np.random.default_rng(1)
      a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16), device="AMD")
      bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16), device="AMD")
      c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD"); Tensor.realize(a, bt, c)
      refmat = a.numpy().astype(np.float32) @ bt.numpy().astype(np.float32).T; refn = float(np.sqrt(np.mean(refmat ** 2)))
      insts = ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 16, 0, 1)  # BK32 PAD16 PLRA1 square-128
      lin, out = ref._run_insts_lds(insts, a, bt, c, M, N, K, f"s{M}x{N}", 32768, 128, 128, 128)
      run_linear(lin); dev.synchronize()
      rel = float(np.sqrt(np.mean((out.float().numpy().astype(np.float32) - refmat) ** 2)) / (refn + 1e-9))
      ets = []
      with Context(DEBUG=0):
        for _ in range(RAMP): dev.synchronize(); run_linear(lin)
        dev.synchronize()
        for _ in range(CNT):
          dev.synchronize(); t0 = time.perf_counter(); run_linear(lin); dev.synchronize(); ets.append(time.perf_counter() - t0)
      best = flop / min(ets) * 1e-12
      rows.append({"M": M, "N": N, "K": K, "ours_tflops": round(best, 1), "tensile_dat_speed": dat,
                   "ours_over_dat": round(best / dat, 3) if dat else None, "rel_rmse": round(rel, 6),
                   "grid": [N // 128, M // 128, 1], "wg": (N // 128) * (M // 128)})
  finally:
    perflevel("auto")

  # analyze: is M=384 a peak for OURS too? (vs M=256/768 at same N=8192)
  byM = {r["M"]: r["ours_tflops"] for r in rows if r["N"] == 8192}
  m384 = byM.get(384); m256 = byM.get(256); m768 = byM.get(768); m512 = byM.get(512)
  ours_384_peak = bool(m384 and m256 and m768 and m384 > m256 * 1.08 and m384 > m768 * 1.08)
  result = {"date": "2026-06-20", "phase": "AMD_GEMM_SHAPE_FACTCHECK", "schema": "amd_gemm_shape_factcheck_v1",
            "role": "ffn_gate/up", "default_behavior_changed": False, "performance_claim": True,
            "rows": rows, "ours_by_M_at_N8192": byM, "ours_384_is_peak": ours_384_peak,
            "finding": ("SHAPE_INTRINSIC: ours ALSO peaks hard at M=384 -> the 79 sweet-spot is real & shape-specific; our M=512 can't reach it"
                        if ours_384_peak else
                        "NOT_SHAPE_INTRINSIC: ours is FLAT across M (no 384 peak) -> the .dat 79 is Tensile-specific or a boost-clock artifact, not an intrinsic ceiling our shape misses")}
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "amd_gemm_shape_factcheck_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print("=== OUR kernel (pinned) vs Tensile .dat claimed speed, by shape ===")
  for r in rows:
    print(f"  {r['M']:5}x{r['N']:5}x{r['K']}: ours {r['ours_tflops']:5.1f} TFLOPS"
          + (f"  | .dat {r['tensile_dat_speed']:.1f}  (ours/{r['ours_over_dat']}x)" if r['tensile_dat_speed'] else "  | (untuned)")
          + f"  wg={r['wg']} rel={r['rel_rmse']:.1e}")
  print("ours by M @N8192:", byM)
  print("FINDING:", result["finding"])
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
