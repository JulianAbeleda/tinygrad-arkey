#!/usr/bin/env python3
# GOLD-STANDARD GPU-time test: measure PURE GPU execution time (HCQ wait=True signal timestamps) for ours and
# the Tensile .co, both as raw AMDProgram launched through the SAME mechanism. Removes ALL confounds of the
# batch test: host launch overhead (wait=True times only the GPU kernel via on-chip start/end signals), WAW
# serialization (one isolated dispatch), and launch-path differences (both raw AMDProgram). Settles definitively
# whether ours is GPU-faster than Tensile.
#
# Run:  DEV=AMD PYTHONPATH=. python3 extra/qk_amd_gemm_gputime_goldstandard.py   (pinned; resets auto)
from __future__ import annotations

import importlib.util, json, os, pathlib, struct, subprocess, statistics
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
HCQ_SRC = ROOT / "extra/qk_tensile_hcq_launch.py"
M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K
REPS = int(os.environ.get("REPS", "300"))


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod
def perflevel(x): subprocess.run(["rocm-smi", "--setperflevel", x], capture_output=True, text=True)


def main() -> int:
  import numpy as np
  from tinygrad import Tensor, Device
  from tinygrad.dtype import dtypes, AddrSpace
  from tinygrad.uop.ops import UOp, Ops
  from tinygrad.renderer.amd.elf import assemble_linear
  from tinygrad.runtime.ops_amd import AMDProgram
  dev = Device["AMD"]; ref = load_mod(REF_SRC, "rdna3_ref"); hcq = load_mod(HCQ_SRC, "hcq")
  rng = np.random.default_rng(1)
  a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16)).realize()
  bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16)).realize()
  c = Tensor.empty(M, N, dtype=dtypes.half).realize(); Tensor.realize(a, bt, c)
  refmat = a.numpy().astype(np.float32) @ bt.numpy().astype(np.float32).T; refn = float(np.sqrt(np.mean(refmat ** 2)))
  hb = lambda t: t.uop.buffer.ensure_allocated()._buf

  # ours -> raw AMDProgram (3 PARAM => kernarg 24B for A,Bt,C; DEFINE_LOCAL 32768 for the LDS)
  insts = ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 16, 0, 1)
  lin = UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))
  dl = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=32768, addrspace=AddrSpace.LOCAL), (), "lds")
  params = [UOp(Ops.PARAM, t.dtype.ptr(), (), i) for i, t in enumerate((a, bt, c))]
  g = [UOp.special(N // 128, "gidx0"), UOp.special(M // 128, "gidx1"), UOp.special(128, "lidx0")]
  prg_uop = UOp(Ops.PROGRAM, src=(UOp.sink(*params, dl, *g),))
  ours = AMDProgram(dev, "ours", assemble_linear(prg_uop, lin, "gfx1100"))
  def run_ours(wait): return ours(hb(a), hb(bt), hb(c), global_size=(N // 128, M // 128, 1), local_size=(128, 1, 1), wait=wait)
  run_ours(False); dev.synchronize()
  ours_rel = float(np.sqrt(np.mean((c.float().numpy().astype(np.float32) - refmat) ** 2)) / (refn + 1e-9))

  # tensile .co
  capk = json.load(open("/tmp/kernarg.json")); raw = bytearray(capk["kernarg_bytes"])
  sym = json.load(open(ROOT / "bench/qk-tensile-extraction/selection.json"))["selected"]["rocblas"]["kernel_symbol"]
  A_t = Tensor.randn(K, M, dtype=dtypes.half).contiguous().realize(); B_t = Tensor.randn(N, K, dtype=dtypes.half).contiguous().realize(); C_t = Tensor.zeros(N, M, dtype=dtypes.half).contiguous().realize(); dev.synchronize()
  va = lambda t: t.uop.buffer.ensure_allocated()._buf.va_addr
  for off, t in ((16, C_t), (24, C_t), (32, A_t), (40, B_t)): struct.pack_into("<Q", raw, off, va(t))
  elf = hcq.unbundle(); kd = hcq.kd_offset(elf, sym); tprg = hcq.NamedAMDProgram(dev, "tensile", elf, kd, bytes(raw))
  def run_tens(wait): return tprg(global_size=(4, 96, 1), local_size=(128, 1, 1), wait=wait, timeout=10000)
  run_tens(False); dev.synchronize()

  def gpu_times(run, reps):
    for _ in range(30): run(False)
    dev.synchronize()
    ts = []
    for _ in range(reps):
      t = run(True)  # wait=True returns GPU execution time (signal-timestamped), no host overhead
      if t is not None: ts.append(t)
    return ts

  perflevel("high")
  try:
    ot = gpu_times(run_ours, REPS); tt = gpu_times(run_tens, REPS)
  finally:
    perflevel("auto")

  def summ(ts): return {"min_us": round(min(ts) * 1e6, 1), "median_us": round(statistics.median(ts) * 1e6, 1),
                        "best_tflops": round(FLOP / min(ts) * 1e-12, 1), "median_tflops": round(FLOP / statistics.median(ts) * 1e-12, 1)}
  os_, ts_ = summ(ot), summ(tt)
  ratio_best = round(os_["best_tflops"] / ts_["best_tflops"], 3)
  ratio_med = round(os_["median_tflops"] / ts_["median_tflops"], 3)
  result = {"date": "2026-06-20", "phase": "AMD_GEMM_GPUTIME_GOLDSTANDARD", "schema": "amd_gemm_gputime_v1",
            "role": "ffn_gate/up", "shape": {"M": M, "N": N, "K": K},
            "method": "pure GPU execution time via HCQ wait=True signal timestamps; both raw AMDProgram; no host overhead, no batching, no WAW",
            "ours": os_, "tensile_co": ts_, "ours_rel_rmse": round(ours_rel, 6),
            "ratio_best_tflops": ratio_best, "ratio_median_tflops": ratio_med, "reps": REPS,
            "verdict": ("OURS_GPU_FASTER" if ratio_med >= 1.02 else "TENSILE_GPU_FASTER" if ratio_med <= 0.98 else "GPU_PARITY"),
            "why": f"GPU-time (wait=True): ours best {os_['best_tflops']} / median {os_['median_tflops']} TFLOPS vs Tensile best {ts_['best_tflops']} / median {ts_['median_tflops']}; ratio median {ratio_med}x."}
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "amd_gemm_gputime_goldstandard_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: result[k] for k in ("verdict", "ours", "tensile_co", "ratio_best_tflops", "ratio_median_tflops", "ours_rel_rmse", "why")}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
