#!/usr/bin/env python3
"""EBT-1 — tinygrad-buffer pointer interop spike (Lane A) for external BLAS prefill.

Question: can rocBLAS (HIP runtime) run a GEMM DIRECTLY on tinygrad DEV=AMD (HCQ/KFD) VRAM pointers, no copies?
Allocates A/B/C as tinygrad AMD tensors, passes HCQBuffer.va_addr to a C shim that calls hipPointerGetAttributes +
rocBLAS GEMM on those pointers, then verifies the output against a tinygrad fp16 oracle and records timing vs the
standalone PXB-1 ceiling. Research-only; no model route, no defaults, decode untouched.

Conservative sync: tinygrad device.synchronize() before the BLAS call; the shim hipStreamSynchronize after.

  build: g++ -std=c++17 -D__HIP_PLATFORM_AMD__=1 -shared -fPIC -I/opt/rocm-7.2.4/include -L/opt/rocm-7.2.4/lib \
         -Wl,-rpath,/opt/rocm-7.2.4/lib extra/qk_prefill_bridge_shim.cpp -lamdhip64 -lrocblas -o /tmp/qk_bridge.so
  run:   DEV=AMD ROCBLAS_TENSILE_LIBPATH=/opt/rocm-7.2.4/lib/rocblas/library LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib \
         PYTHONPATH=. .venv/bin/python extra/qk_prefill_bridge_interop.py
"""
from __future__ import annotations
import ctypes, json, pathlib
from tinygrad import Tensor, Device, dtypes
from tinygrad.helpers import getenv

SHIM = getenv("SHIM", "/tmp/qk_bridge.so")
# PXB-1 standalone ffn_gate/up ceiling (best lib) for the within-10% timing gate
PXB1_FFN_MS = 0.738279  # hipBLASLt ffn_gate/up from bench/qk-prefill-external-blas/ceiling.json

def va(t:Tensor) -> int:
  return t.uop.buffer._buf.va_addr

def main():
  assert Device.DEFAULT == "AMD", f"need DEV=AMD, got {Device.DEFAULT}"
  M, N, K = 512, 12288, 4096                       # ffn_gate/up (the dominant prefill shape)
  lib = ctypes.CDLL(SHIM)
  lib.ebt1_pointer_attr.restype = ctypes.c_int
  lib.ebt1_pointer_attr.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
  lib.ebt1_gemm.restype = ctypes.c_int
  lib.ebt1_gemm.argtypes = [ctypes.c_uint64]*3 + [ctypes.c_int]*5 + [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int)]

  Tensor.manual_seed(0)
  A = Tensor.randn(M, K, dtype=dtypes.half).contiguous().realize()
  B = Tensor.randn(K, N, dtype=dtypes.half).contiguous().realize()
  C = Tensor.zeros(M, N, dtype=dtypes.half).contiguous().realize()
  oracle = (A.float() @ B.float()).realize()        # tinygrad fp16-input matmul oracle (fp32 accumulate)
  Device[Device.DEFAULT].synchronize()              # conservative: tinygrad done before BLAS touches the buffers

  a_va, b_va, c_va = va(A), va(B), va(C)
  res = {"shape": [M, N, K], "va": {"A": hex(a_va), "B": hex(b_va), "C": hex(c_va)}}

  # 1) does HIP accept the tinygrad VA pointers?
  attrs = {}
  for name, p in (("A", a_va), ("B", b_va), ("C", c_va)):
    mt, match = ctypes.c_int(0), ctypes.c_int(0)
    e = lib.ebt1_pointer_attr(ctypes.c_uint64(p), ctypes.byref(mt), ctypes.byref(match))
    attrs[name] = {"hip_err": e, "memtype": mt.value, "dev_ptr_matches": bool(match.value)}
  res["pointer_attr"] = attrs

  # 2) run the GEMM directly on tinygrad pointers
  gemm_ms, last_hip = ctypes.c_double(0), ctypes.c_int(-1)
  st = lib.ebt1_gemm(ctypes.c_uint64(a_va), ctypes.c_uint64(b_va), ctypes.c_uint64(c_va),
                     M, N, K, 10, 30, ctypes.byref(gemm_ms), ctypes.byref(last_hip))
  res["rocblas_status"] = st
  res["gemm_first_sync_hip_err"] = last_hip.value
  res["bridge_gemm_ms"] = gemm_ms.value
  res["pxb1_standalone_ms"] = PXB1_FFN_MS
  res["bridge_vs_standalone"] = (gemm_ms.value / PXB1_FFN_MS) if gemm_ms.value else None

  # 3) verify correctness vs the tinygrad oracle (read back the tinygrad-owned C buffer)
  if st == 0:
    Device[Device.DEFAULT].synchronize()
    diff = (C.float() - oracle).abs()
    rel = (diff.max() / (oracle.abs().max() + 1e-6)).item()
    res["correctness"] = {"max_abs": diff.max().item(), "rel_err": rel, "exact": rel < 2e-2}
  else:
    res["correctness"] = {"skipped": "gemm did not run"}

  # verdict
  hip_ok = all(a["hip_err"] == 0 for a in attrs.values())
  ran = (st == 0 and res["gemm_first_sync_hip_err"] == 0)
  correct = res.get("correctness", {}).get("exact", False)
  within10 = (res["bridge_vs_standalone"] is not None and res["bridge_vs_standalone"] <= 1.10)
  res["gates"] = {"hip_accepts_pointers": hip_ok, "gemm_ran": ran, "correct": correct, "timing_within_10pct": within10}
  res["verdict"] = "PASS" if (ran and correct and within10) else "KILL"
  res["verdict_note"] = ("HIP accepts tinygrad VA + correct GEMM + timing within 10%" if res["verdict"]=="PASS"
                         else "see gates; HIP may reject HCQ/KFD pointers, GEMM wrong, or timing off")

  out = pathlib.Path("bench/qk-prefill-external-bridge/interop.json"); out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(res, indent=2))
  print(json.dumps(res, indent=2))
  print(f"\nEBT-1 VERDICT: {res['verdict']}  | {res['verdict_note']}")

if __name__ == "__main__":
  main()
