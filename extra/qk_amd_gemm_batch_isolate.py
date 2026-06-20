#!/usr/bin/env python3
# ISOLATE the residual: is the ~4% ours-vs-Tensile gap GPU execution time, or HOST launch overhead?
#
# The alone-pinned wall-clock per-launch includes host overhead, and ours (run_linear) vs Tensile (tprg, which
# REFILLS the 128-byte kernarg every call) have DIFFERENT host overhead. On a ~0.8ms kernel a ~30us host diff
# = ~4%. Test: BATCH K launches between syncs (host overhead pipelines/amortizes as K grows). If ours/Tensile
# -> 1.0 at high K, the ~4% was HOST overhead, not GPU. If it stays ~0.96, it's real GPU execution time.
#
# Run:  DEV=AMD PYTHONPATH=. python3 extra/qk_amd_gemm_batch_isolate.py   (pinned; resets auto)
from __future__ import annotations

import importlib.util, json, os, pathlib, struct, subprocess, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
HCQ_SRC = ROOT / "extra/qk_tensile_hcq_launch.py"
M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
BATCHES = [1, 8, 32]; CNT = int(os.environ.get("CNT", "60"))


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)


def main() -> int:
  import numpy as np
  from tinygrad import Tensor, Device, Context
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import run_linear
  dev = Device["AMD"]; ref = load_mod(REF_SRC, "rdna3_ref"); hcq = load_mod(HCQ_SRC, "hcq")
  rng = np.random.default_rng(1)
  a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16), device="AMD")
  bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16), device="AMD")
  c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD"); Tensor.realize(a, bt, c)
  ours_lin, _ = ref._run_insts_lds(ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 16, 0, 1), a, bt, c, M, N, K, "ours", 32768, 128, 128, 128)
  launches: dict[str, Any] = {"ours": lambda: run_linear(ours_lin)}
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
    print("tensile .co unavailable:", repr(ex)[:120]); return 1

  def bench(fn, batch):  # per-launch time = best over CNT of (total over `batch` back-to-back launches)/batch
    with Context(DEBUG=0):
      for _ in range(20): dev.synchronize(); fn()
      dev.synchronize(); best = 1e9
      for _ in range(CNT):
        dev.synchronize(); t0 = time.perf_counter()
        for _ in range(batch): fn()
        dev.synchronize(); best = min(best, (time.perf_counter() - t0) / batch)
    return best

  perflevel("high")
  result: dict[str, Any] = {"date": "2026-06-20", "phase": "AMD_GEMM_BATCH_ISOLATE", "schema": "amd_gemm_batch_isolate_v1",
                            "role": "ffn_gate/up", "shape": {"M": M, "N": N, "K": K}, "by_batch": {}}
  try:
    for B in BATCHES:
      row = {name: round(FLOP / bench(fn, B) * 1e-12, 1) for name, fn in launches.items()}
      row["ours_over_tensile"] = round(row["ours"] / row["tensile_co"], 3) if "tensile_co" in row else None
      result["by_batch"][str(B)] = row
      print(f"  batch={B:2}: {row}")
  finally:
    perflevel("auto")

  r1 = result["by_batch"][str(BATCHES[0])]["ours_over_tensile"]; rN = result["by_batch"][str(BATCHES[-1])]["ours_over_tensile"]
  result["ratio_batch1"] = r1; result["ratio_batchN"] = rN; result["ratio_moved"] = round(rN - r1, 3) if (r1 and rN) else None
  if rN and rN >= 0.985:
    result["verdict"] = "RESIDUAL_WAS_HOST_OVERHEAD"; result["why"] = f"ours/Tensile rose {r1}->{rN} as batch amortized host overhead -> the gap was per-launch HOST cost (tprg kernarg refill), GPU times ~equal. ACTUAL PARITY."
  elif rN and abs(rN - r1) < 0.02:
    result["verdict"] = "RESIDUAL_IS_REAL_GPU_TIME"; result["why"] = f"ours/Tensile stayed {r1}->{rN} across batch -> the ~{round((1-rN)*100)}% gap is REAL GPU execution time, not host overhead. Tensile genuinely does the work in fewer cycles."
  else:
    result["verdict"] = "PARTIAL_HOST_OVERHEAD"; result["why"] = f"ours/Tensile moved {r1}->{rN}: part host overhead, part GPU; residual GPU gap ~{round((1-rN)*100)}%."
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "amd_gemm_batch_isolate_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"verdict": result["verdict"], "by_batch": result["by_batch"], "ratio_1": r1, "ratio_N": rN, "why": result["why"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
