#!/usr/bin/env python3
"""TPE-4 - isolated HCQ performance parity for the extracted rocBLAS Tensile ffn_gate/up kernel.

Measures the TPE-3 recovered kernel when launched from tinygrad HCQ on tinygrad-owned buffers:
  DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_tensile_hcq_perf.py

No HIP runtime is used in-process. The script reuses the TPE-3 named-descriptor loader and the committed
kernarg capture artifact, substitutes only the four buffer VAs, then records device-time samples from HCQ.
"""
from __future__ import annotations

import json, pathlib, statistics, struct
from tinygrad import Tensor, Device, dtypes

from extra.qk_tensile_hcq_launch import NamedAMDProgram, kd_offset, unbundle

SHAPE = {"M": 512, "N": 12288, "K": 4096}
FLOPS = 2 * SHAPE["M"] * SHAPE["N"] * SHAPE["K"]
CAPTURE = pathlib.Path("bench/qk-tensile-extraction/kernarg_capture.json")
SELECTION = pathlib.Path("bench/qk-tensile-extraction/selection.json")
OUT = pathlib.Path("bench/qk-tensile-extraction/hcq_perf.json")

WARMUPS = 8
RUNS = 40

def maps_contain_hip_runtime() -> bool:
  try:
    maps = pathlib.Path("/proc/self/maps").read_text(errors="ignore")
  except FileNotFoundError:
    return False
  return any(x in maps for x in ("libamdhip64", "librocblas", "libhipblaslt"))

def pctile(xs: list[float], p: float) -> float:
  assert xs
  ys = sorted(xs)
  i = min(len(ys)-1, max(0, round((len(ys)-1)*p)))
  return ys[i]

def stats_ms(samples: list[float]) -> dict:
  return {
    "samples_ms": [round(x, 6) for x in samples],
    "min_ms": round(min(samples), 6),
    "median_ms": round(statistics.median(samples), 6),
    "mean_ms": round(statistics.fmean(samples), 6),
    "p10_ms": round(pctile(samples, 0.10), 6),
    "p90_ms": round(pctile(samples, 0.90), 6),
    "max_ms": round(max(samples), 6),
    "stdev_ms": round(statistics.pstdev(samples), 6),
  }

def tflops(ms: float) -> float:
  return FLOPS / (ms * 1e9)

def load_raw_kernarg() -> bytearray:
  cap = json.loads(CAPTURE.read_text())
  raw = bytearray(cap["kernarg_bytes"])
  assert len(raw) == 128, len(raw)
  return raw

def substitute_ptrs(raw: bytearray, a: Tensor, b: Tensor, c: Tensor) -> None:
  va = lambda t: t.uop.buffer._buf.va_addr
  struct.pack_into("<Q", raw, 16, va(c))  # AddressD
  struct.pack_into("<Q", raw, 24, va(c))  # AddressC
  struct.pack_into("<Q", raw, 32, va(a))  # AddressA
  struct.pack_into("<Q", raw, 40, va(b))  # AddressB

def main() -> None:
  assert Device.DEFAULT == "AMD"
  dev = Device[Device.DEFAULT]
  hip_before = maps_contain_hip_runtime()

  selection = json.loads(SELECTION.read_text())
  sym = selection["selected"]["rocblas"]["kernel_symbol"]
  trace_ref = selection["selected"]["rocblas"]
  pxb_ref = selection["pxb1_reference"]

  Tensor.manual_seed(0)
  # Tensile contract is col-major C[512,12288] = A[512,4096] * B[4096,12288].
  # These row-major tinygrad tensors are laid out as the matching col-major buffers.
  a_t = Tensor.randn(4096, 512, dtype=dtypes.half).contiguous().realize()
  b_t = Tensor.randn(12288, 4096, dtype=dtypes.half).contiguous().realize()
  c_t = Tensor.zeros(12288, 512, dtype=dtypes.half).contiguous().realize()
  oracle = (b_t.float() @ a_t.float()).realize()
  dev.synchronize()

  raw = load_raw_kernarg()
  substitute_ptrs(raw, a_t, b_t, c_t)
  elf = unbundle()
  kd = kd_offset(elf, sym)
  prg = NamedAMDProgram(dev, "tensile_ffn_gate_up_perf", elf, kd, bytes(raw))

  warmup_ms = []
  for _ in range(WARMUPS):
    warmup_ms.append(float(prg(global_size=(4, 96, 1), local_size=(128, 1, 1), wait=True, timeout=10000)) * 1000.0)
  dev.synchronize()

  run_ms = []
  for _ in range(RUNS):
    run_ms.append(float(prg(global_size=(4, 96, 1), local_size=(128, 1, 1), wait=True, timeout=10000)) * 1000.0)
  dev.synchronize()

  diff = (c_t.float() - oracle).abs()
  max_abs = float(diff.max().item())
  rel_err = float((diff.max()/(oracle.abs().max()+1e-6)).item())
  median_ms = statistics.median(run_ms)
  median_tflops = tflops(median_ms)
  trace_tflops = float(trace_ref["tflops"])
  pxb_tflops = float(pxb_ref["rocblas_tflops"])
  tinygrad_tflops = float(pxb_ref.get("tinygrad_tflops", 42.0))
  min_gate_tflops = float(pxb_ref["gate_tflops"])
  parity90_tflops = 0.90 * trace_tflops
  hip_after = maps_contain_hip_runtime()

  res = {
    "schema": "qk_tensile_hcq_perf_v1",
    "phase": "TPE-4",
    "device": "RX 7900 XTX / gfx1100",
    "role": "ffn_gate/up",
    "shape": SHAPE,
    "flops": FLOPS,
    "kernel_symbol": sym,
    "kernel_symbol_short": sym[:80] + "...",
    "kd_offset": hex(kd),
    "launch": {"global": [4, 96, 1], "local": [128, 1, 1]},
    "warmup": stats_ms(warmup_ms),
    "timed": stats_ms(run_ms),
    "median_tflops": round(median_tflops, 3),
    "min_tflops": round(tflops(max(run_ms)), 3),
    "max_tflops": round(tflops(min(run_ms)), 3),
    "references": {
      "trace_rocblas_tflops": trace_tflops,
      "trace_rocblas_median_ms": round(float(trace_ref["median_us"])/1000.0, 6),
      "pxb1_rocblas_tflops": pxb_tflops,
      "pxb1_tinygrad_tflops": tinygrad_tflops,
      "minimum_gate_tflops": min_gate_tflops,
      "parity90_trace_tflops": round(parity90_tflops, 3),
    },
    "ratios": {
      "of_trace_rocblas_tflops": round(median_tflops / trace_tflops, 4),
      "of_pxb1_rocblas_tflops": round(median_tflops / pxb_tflops, 4),
      "of_pxb1_tinygrad_tflops": round(median_tflops / tinygrad_tflops, 4),
    },
    "correctness": {
      "max_abs": round(max_abs, 4),
      "rel_err": round(rel_err, 6),
      "pass": rel_err < 2e-2,
    },
    "process_libraries": {
      "hip_runtime_loaded_before": hip_before,
      "hip_runtime_loaded_after": hip_after,
    },
    "gates": {
      "correct": rel_err < 2e-2,
      "no_hip_runtime": not hip_after,
      "ge_90pct_trace": median_tflops >= parity90_tflops,
      "ge_62_tflops": median_tflops >= min_gate_tflops,
    },
  }
  res["verdict"] = "PASS" if all(res["gates"].values()) else "KILL"

  OUT.parent.mkdir(parents=True, exist_ok=True)
  OUT.write_text(json.dumps(res, indent=2) + "\n")
  print(json.dumps(res, indent=2))
  print("\nTPE-4 VERDICT:", res["verdict"])

if __name__ == "__main__":
  main()
