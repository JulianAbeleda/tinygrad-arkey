#!/usr/bin/env python3
# PROMOTION BENCHMARK: ours vs Tensile .co vs LLVM authority on OUR shape (512x12288x4096), each measured
# ALONE (NOT interleaved -- interleaving a foreign-.co launch perturbs ours, per the fact-check), PINNED
# clock, multiple independent rounds. Confirms the parity finding is CONSISTENT before promoting+banking.
#
# Run:  DEV=AMD PYTHONPATH=. python3 extra/qk_amd_gemm_promotion_benchmark.py   (sets perflevel high, resets auto)
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
ROUNDS = int(os.environ.get("ROUNDS", "5")); CNT = int(os.environ.get("CNT", "120")); RAMP = int(os.environ.get("RAMP", "60"))


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)


def main() -> int:
  import numpy as np
  from tinygrad import Tensor, Device, Context
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import run_linear
  dev = Device["AMD"]; ref = load_mod(REF_SRC, "rdna3_ref"); hcq = load_mod(HCQ_SRC, "hcq")

  # ours
  rng = np.random.default_rng(1)
  a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16), device="AMD")
  bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16), device="AMD")
  c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD"); Tensor.realize(a, bt, c)
  refmat = a.numpy().astype(np.float32) @ bt.numpy().astype(np.float32).T; refn = float(np.sqrt(np.mean(refmat ** 2)))
  ours_lin, ours_out = ref._run_insts_lds(ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 16, 0, 1), a, bt, c, M, N, K, "ours", 32768, 128, 128, 128)
  run_linear(ours_lin); dev.synchronize()
  ours_rel = float(np.sqrt(np.mean((ours_out.float().numpy().astype(np.float32) - refmat) ** 2)) / (refn + 1e-9))

  launches = {"ours": lambda: run_linear(ours_lin)}
  # tensile .co
  try:
    capk = json.load(open("/tmp/kernarg.json")); raw = bytearray(capk["kernarg_bytes"])
    sym = json.load(open(ROOT / "bench/qk-tensile-extraction/selection.json"))["selected"]["rocblas"]["kernel_symbol"]
    A_t = Tensor.randn(K, M, dtype=dtypes.half).contiguous().realize(); B_t = Tensor.randn(N, K, dtype=dtypes.half).contiguous().realize(); C_t = Tensor.zeros(N, M, dtype=dtypes.half).contiguous().realize()
    dev.synchronize(); va = lambda t: t.uop.buffer._buf.va_addr
    for off, t in ((16, C_t), (24, C_t), (32, A_t), (40, B_t)): struct.pack_into("<Q", raw, off, va(t))
    elf = hcq.unbundle(); kd = hcq.kd_offset(elf, sym); tprg = hcq.NamedAMDProgram(dev, "tensile", elf, kd, bytes(raw))
    tprg(global_size=(4, 96, 1), local_size=(128, 1, 1), wait=True, timeout=10000); dev.synchronize()
    launches["tensile_co"] = lambda: tprg(global_size=(4, 96, 1), local_size=(128, 1, 1), wait=False)
  except Exception as ex:
    print("tensile .co unavailable:", repr(ex)[:120])
  # authority
  try:
    ptm1 = load_mod(PTM1_SRC, "ptm1"); auth_lin, _m, _alive = ptm1.build_authority(); launches["authority_llvm"] = lambda: run_linear(auth_lin)
  except Exception: pass

  def bench_alone(fn):  # measure ONE kernel in its own tight loop, best-of-CNT
    with Context(DEBUG=0):
      for _ in range(RAMP): dev.synchronize(); fn()
      dev.synchronize(); ets = []
      for _ in range(CNT):
        dev.synchronize(); t0 = time.perf_counter(); fn(); dev.synchronize(); ets.append(time.perf_counter() - t0)
    return FLOP / min(ets) * 1e-12

  perflevel("high")
  rounds = []
  try:
    for r in range(ROUNDS):
      row = {name: round(bench_alone(fn), 1) for name, fn in launches.items()}
      if "ours" in row and "tensile_co" in row: row["ours_over_tensile"] = round(row["ours"] / row["tensile_co"], 3)
      rounds.append(row); print(f"  round {r}: {row}")
  finally:
    perflevel("auto")

  # post-bench correctness re-check
  ours_rel_post = float(np.sqrt(np.mean((ours_out.float().numpy().astype(np.float32) - refmat) ** 2)) / (refn + 1e-9))
  def agg(key): vals = [r[key] for r in rounds if key in r]; return {"min": min(vals), "median": sorted(vals)[len(vals) // 2], "max": max(vals)} if vals else None
  ratios = [r["ours_over_tensile"] for r in rounds if "ours_over_tensile" in r]
  consistent = bool(ratios) and all(0.95 <= x <= 1.10 for x in ratios)
  result = {"date": "2026-06-20", "phase": "AMD_GEMM_PROMOTION_BENCHMARK", "schema": "amd_gemm_promotion_v1",
            "role": "ffn_gate/up", "shape": {"M": M, "N": N, "K": K}, "rounds": rounds,
            "method": "each kernel measured ALONE (not interleaved), pinned, best-of-%d, %d rounds" % (CNT, ROUNDS),
            "aggregate": {k: agg(k) for k in ("ours", "tensile_co", "authority_llvm", "ours_over_tensile")},
            "ours_rel_rmse": round(ours_rel, 6), "ours_rel_rmse_post": round(ours_rel_post, 6),
            "ours_over_tensile_ratios": ratios, "parity_consistent": consistent and ours_rel_post < 0.02,
            "config": "build_gemm_lds2(BK=32, PAD=16, PLRA=1) square-128 @ wg2 LDS=32768 (dependency-free)"}
  result["verdict"] = ("PASS_PARITY_CONSISTENT_PROMOTE" if result["parity_consistent"]
                       else "INCONSISTENT" if ratios else "TENSILE_UNAVAILABLE")
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "amd_gemm_promotion_benchmark_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result["verdict"], "aggregate": result["aggregate"], "ratios": ratios,
                    "ours_rel_rmse_post": result["ours_rel_rmse_post"], "parity_consistent": result["parity_consistent"]}, indent=2))
  return 0 if result["parity_consistent"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
