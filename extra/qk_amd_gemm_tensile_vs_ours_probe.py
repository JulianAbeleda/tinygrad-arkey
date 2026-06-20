#!/usr/bin/env python3
# Clock-matched Tensile-.co vs ours (REAL GPU, pinned clock, interleaved, NO BEAM). The DECISIVE comparison.
#
# The vendored Tensile .co is already traceable+launchable from tinygrad HCQ (extra/qk_tensile_hcq_launch.py,
# proven correct rel_err 3.5e-4 PASS). This TIMES the real Tensile kernel interleaved with our dependency-free
# best (build_gemm_lds2 BK32+PAD16+PLRA=1) and the LLVM authority, at pinned high clock -> settles whether
# Tensile is ~66 or ~61 at this clock, i.e. whether our ~61 is Tensile parity or ~92%.
#
# Run:  DEV=AMD PYTHONPATH=. python3 extra/qk_amd_gemm_tensile_vs_ours_probe.py   (sets perflevel high, resets auto)
from __future__ import annotations

import importlib.util, json, os, pathlib, struct, subprocess, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"
HCQ_SRC = ROOT / "extra/qk_tensile_hcq_launch.py"

M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
CNT = int(os.environ.get("CNT", "150")); RAMP = int(os.environ.get("RAMP", "80"))


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod); return mod
def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)
def stats(ets):
  s = sorted(ets); n = len(s); return {"best_tflops": FLOP / s[0] * 1e-12, "median_tflops": FLOP / s[n // 2] * 1e-12, "best_s": s[0], "n": n}


def main() -> int:
  import numpy as np
  from tinygrad import Tensor, Device, Context
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import run_linear
  dev = Device["AMD"]
  ref = load_mod(REF_SRC, "rdna3_ref")
  hcq = load_mod(HCQ_SRC, "hcq_launch")

  result: dict[str, Any] = {"date": "2026-06-20", "phase": "AMD_GEMM_TENSILE_VS_OURS", "schema": "amd_gemm_tensile_vs_ours_v1",
                            "role": "ffn_gate/up", "default_behavior_changed": False, "performance_claim": True, "shape": {"M": M, "N": N, "K": K}}
  launches: list[tuple[str, Any]] = []

  # --- our best: build_gemm_lds2 BK32 + PAD16 + A-prefetch PLR, wg2 ---
  rng = np.random.default_rng(1)
  a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16), device="AMD")
  bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16), device="AMD")
  c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD"); Tensor.realize(a, bt, c)
  refmat = a.numpy().astype(np.float32) @ bt.numpy().astype(np.float32).T; refnorm = float(np.sqrt(np.mean(refmat ** 2)))
  ours_lin, ours_out = ref._run_insts_lds(ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 16, 0, 1), a, bt, c, M, N, K, "ours_plra", 32768, 128, 128, 128)
  run_linear(ours_lin); dev.synchronize()
  ours_rel = float(np.sqrt(np.mean((ours_out.float().numpy().astype(np.float32) - refmat) ** 2)) / (refnorm + 1e-9))
  result["ours_rel_rmse"] = ours_rel
  launches.append(("ours_plra", lambda: run_linear(ours_lin)))

  # --- LLVM authority ---
  try:
    ptm1 = load_mod(PTM1_SRC, "ptm1_bridge"); auth_lin, _m, _alive = ptm1.build_authority()
    launches.append(("authority_llvm", lambda: run_linear(auth_lin)))
  except Exception as ex:
    result["authority_error"] = repr(ex)[:120]

  # --- vendored Tensile .co (traceable via NamedAMDProgram) ---
  tensile_ok = False
  try:
    cap = json.load(open("/tmp/kernarg.json")); raw = bytearray(cap["kernarg_bytes"]); assert len(raw) == 128
    sym = json.load(open(ROOT / "bench/qk-tensile-extraction/selection.json"))["selected"]["rocblas"]["kernel_symbol"]
    A_t = Tensor.randn(K, M, dtype=dtypes.half).contiguous().realize()        # col-major A
    B_t = Tensor.randn(N, K, dtype=dtypes.half).contiguous().realize()        # col-major B
    C_t = Tensor.zeros(N, M, dtype=dtypes.half).contiguous().realize()        # col-major C
    dev.synchronize()
    va = lambda t: t.uop.buffer._buf.va_addr
    struct.pack_into("<Q", raw, 16, va(C_t)); struct.pack_into("<Q", raw, 24, va(C_t))
    struct.pack_into("<Q", raw, 32, va(A_t)); struct.pack_into("<Q", raw, 40, va(B_t))
    elf = hcq.unbundle(); kd = hcq.kd_offset(elf, sym)
    tprg = hcq.NamedAMDProgram(dev, "tensile_ffn_gate_up", elf, kd, bytes(raw))
    tprg(global_size=(4, 96, 1), local_size=(128, 1, 1), wait=True, timeout=10000); dev.synchronize()
    tensile_ok = True
    launches.append(("tensile_co", lambda: tprg(global_size=(4, 96, 1), local_size=(128, 1, 1), wait=False)))
    result["tensile_loaded"] = True
  except Exception as ex:
    import traceback; result["tensile_error"] = repr(ex)[:200]; result["tensile_trace"] = traceback.format_exc().splitlines()[-4:]

  # --- interleaved timing at pinned high clock ---
  perflevel("high")
  times = {n_: [] for n_, _ in launches}
  try:
    with Context(DEBUG=0):
      for _, fn in launches:
        for _ in range(RAMP): dev.synchronize(); fn()
        dev.synchronize()
      for _ in range(CNT):
        for n_, fn in launches:
          dev.synchronize(); t0 = time.perf_counter(); fn(); dev.synchronize(); times[n_].append(time.perf_counter() - t0)
  finally:
    perflevel("auto")

  result["timing"] = {n_: stats(times[n_]) for n_, _ in launches}
  t = result["timing"]
  ours = t.get("ours_plra", {}).get("best_tflops"); tens = t.get("tensile_co", {}).get("best_tflops"); auth = t.get("authority_llvm", {}).get("best_tflops")
  result["comparison"] = {
    "ours_tflops": ours, "tensile_tflops": tens, "authority_tflops": auth,
    "ours_over_authority_x": round(ours / auth, 3) if (ours and auth) else None,
    "tensile_over_authority_x": round(tens / auth, 3) if (tens and auth) else None,
    "ours_over_tensile_x": round(ours / tens, 3) if (ours and tens) else None,
  }
  if ours and tens:
    r = ours / tens
    if r >= 0.97: result["verdict"] = "PARITY_CLOCK_MATCHED"; result["why"] = f"ours {ours:.1f} vs Tensile {tens:.1f} (clock-matched, pinned) = {round(r*100)}% -> effective parity."
    else: result["verdict"] = "BELOW_TENSILE_CLOCK_MATCHED"; result["why"] = f"ours {ours:.1f} vs Tensile {tens:.1f} (clock-matched) = {round(r*100)}% of Tensile; gap real even at matched clock."
  else:
    result["verdict"] = "TENSILE_UNAVAILABLE" if not tensile_ok else "INCOMPLETE"
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "amd_gemm_tensile_vs_ours_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result.get("verdict"), "ours_rel_rmse": ours_rel,
                    "timing": {k: round(v["best_tflops"], 1) for k, v in t.items()},
                    "comparison": result["comparison"], "why": result.get("why"),
                    "tensile_error": result.get("tensile_error")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
