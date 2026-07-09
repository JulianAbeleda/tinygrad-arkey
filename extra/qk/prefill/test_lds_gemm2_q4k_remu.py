"""GPU-FREE bit-exact correctness gate for the handwritten fused Q4_K->fp16-LDS->WMMA kernel
(build_gemm_lds2_q4k, extra/qk/prefill/wmma.py). Runs the kernel's raw RDNA3 instruction stream through
the remu functional emulator (github.com/Qazalin/remu) and compares to the in-tree ggml Q4_K dequant
reference (tinygrad/llm/gguf.py ggml_data_to_tensor, GGML_Q4_K) matmul. No GPU.

This reconstructs the deleted LDSGEMM2Q4K microbench so the SHIPPED prefill kernel (the 808 tok/s path)
has a runnable numeric oracle again -- and defines the reference a future *generated* codegen replacement
must match to let us delete the hand kernel.

Run (GPU-free):
  ALLOW_DEVICE_USAGE=1 PYTHONPATH=. .venv/bin/python extra/qk/prefill/test_lds_gemm2_q4k_remu.py
  # optional: pass MxNxK shapes, e.g. ... test_...py 128x128x256 512x128x512
"""
import os, sys, ctypes
os.environ["ALLOW_DEVICE_USAGE"] = "1"
import numpy as np
sys.path.insert(0, os.getcwd())
from tinygrad import Tensor
from tinygrad.llm.gguf import ggml_data_to_tensor
import extra.qk.prefill.wmma as wmma

GGML_Q4_K = 12
REF_DEV = "PYTHON"  # force the dequant reference onto the pure-python (CPU) backend -> never touches the GPU
LIBREMU = os.environ.get("LIBREMU", "/home/ubuntu/.claude/jobs/2f995982/tmp/libremu.so")
# config override via env: CFG="WAVES_M,WAVES_N,WM,WN". Default = the shipped route config (BM=BN=128, THREADS=128).
# CFG=1,1,2,2 -> single wave (THREADS=32): isolates decode/WMMA logic from remu's cross-wave barrier modeling.
WAVES_M, WAVES_N, WM, WN = (int(x) for x in os.environ.get("CFG", "2,2,4,4").split(","))


def make_packed_q4k(N, K, rng):
  """Random VALID packed Q4_K weight bytes [N, (K//256)*144]. d/dmin are forced to NORMAL fp16
  (the kernel's expand_f16 assumes exp!=0); scales+qs are fully random (any byte pattern is valid)."""
  NSB = K // 256
  BKPR = NSB * 144
  W = np.zeros((N, BKPR), dtype=np.uint8)
  for n in range(N):
    for sb in range(NSB):
      o = sb * 144
      d = np.float16(rng.uniform(0.01, 0.2) * rng.choice([-1.0, 1.0]))
      dmin = np.float16(rng.uniform(0.01, 0.2) * rng.choice([-1.0, 1.0]))
      W[n, o:o + 2] = np.frombuffer(d.tobytes(), dtype=np.uint8)
      W[n, o + 2:o + 4] = np.frombuffer(dmin.tobytes(), dtype=np.uint8)
      W[n, o + 4:o + 16] = rng.integers(0, 256, 12, dtype=np.uint8)
      W[n, o + 16:o + 144] = rng.integers(0, 256, 128, dtype=np.uint8)
  return W


def reference(A, W, N, K):
  """C_ref[m,n] = A[m,:] . dequant(W[n,:]).  W row = out-neuron (K in-features), so C = A @ dequant(W).T.
  Dequant runs on the PYTHON backend (CPU). The kernel stages the weight as fp16 in LDS, so round the
  reference weight to fp16 before the matmul to match the kernel's datapath."""
  t = Tensor(W.reshape(-1).copy(), device=REF_DEV)          # flat packed bytes (uint8)
  deq = ggml_data_to_tensor(t, N * K, GGML_Q4_K).reshape(N, K).numpy().astype(np.float32)
  deq_f16 = deq.astype(np.float16).astype(np.float32)        # match the kernel's fp16 LDS staging
  return A.astype(np.float32) @ deq_f16.T                    # [M, N]


def run_case(M, N, K):
  BM = WAVES_M * WM * 16; BN = WAVES_N * WN * 16; THREADS = WAVES_M * WAVES_N * 32
  assert M % BM == 0 and N % BN == 0 and K % 256 == 0, f"{M}x{N}x{K} not tileable (BM={BM} BN={BN})"
  rng = np.random.default_rng(0)
  A = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
  W = make_packed_q4k(N, K, rng)
  C = np.zeros((M, N), dtype=np.float16)

  insts = wmma.build_gemm_lds2_q4k(M, N, K, WAVES_M, WAVES_N, WM, WN)
  text = b"".join(i.to_bytes() for i in insts)
  assert len(text) % 4 == 0

  Ac = np.ascontiguousarray(A); Wc = np.ascontiguousarray(W); Cc = np.ascontiguousarray(C)
  args = (ctypes.c_uint64 * 3)(Ac.ctypes.data, Wc.ctypes.data, Cc.ctypes.data)  # kernarg order [A, W, C]
  lib = ctypes.CDLL(LIBREMU)
  lib.run_asm.restype = ctypes.c_int
  lib.run_asm.argtypes = [ctypes.c_char_p, ctypes.c_uint32] + [ctypes.c_uint32] * 6 + [ctypes.POINTER(ctypes.c_uint64)]
  gx, gy = N // BN, M // BM                                  # grid (N//BN, M//BM, 1); workgroup = THREADS
  rc = lib.run_asm(ctypes.c_char_p(text), len(text), gx, gy, 1, THREADS, 1, 1, args)

  ref = reference(A, W, N, K)
  got = Cc.astype(np.float32)
  nanfrac = float(np.isnan(got).mean())
  fin = np.isfinite(got)
  denom = np.sqrt((ref[fin] ** 2).mean()) if fin.any() else float("nan")
  rmse = float(np.sqrt(((got[fin] - ref[fin]) ** 2).mean())) if fin.any() else float("nan")
  rel = rmse / denom if denom else float("nan")
  maxabs = float(np.abs(got[fin] - ref[fin]).max()) if fin.any() else float("nan")
  PASS = (nanfrac == 0.0) and (rel < 5e-3)
  print(f"[{M}x{N}x{K}] rc={rc} nan_frac={nanfrac:.4f} rel_rmse={rel:.2e} max_abs={maxabs:.4f} "
        f"PASS={PASS} (grid={gx}x{gy}, threads={THREADS}, cfg={WAVES_M},{WAVES_N},{WM},{WN})")
  print(f"   got[0,:6]={got[0,:6]}")
  print(f"   ref[0,:6]={ref[0,:6]}")
  # per 32-col block max-abs-err on output row 0 -> shows if error localizes to foreign-wave columns (remu barrier blind spot)
  if N >= 32:
    blk = [float(np.abs(got[0, c:c+32] - ref[0, c:c+32]).max()) for c in range(0, N, 32)]
    print(f"   row0 max_abs per 32-col block: {[f'{b:.2f}' for b in blk]}")
  return PASS


if __name__ == "__main__":
  shapes = [tuple(int(x) for x in s.split("x")) for s in sys.argv[1:]] or [(128, 128, 256)]
  ok = all(run_case(*s) for s in shapes)
  sys.exit(0 if ok else 1)
