#!/usr/bin/env python3
"""Launch the exact compiler-owned pp512 Q4 gate body through CUDA for NCU.

The native NV backend bypasses the CUDA API and cannot be observed by CUPTI.
This research bridge loads the retained compiler cubin in a CUDA primary
context and binds the same real Q4 weight plus deterministic legal compact-Q8
record used by ``nv_compiler_q4k_production_gate.py``.  It never participates
in model routing.
"""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, re, statistics, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check

M, N, K = 512, 12288, 4096
GRID, BLOCK = (96, 4, 1), (32, 2, 4)


def _symbol(source: pathlib.Path) -> str:
  match = re.search(r'extern "C" __global__ void __launch_bounds__\(256\)\s+([^\s(]+)', source.read_text())
  if match is None: raise RuntimeError(f"no launch symbol in {source}")
  return match.group(1)


def _fixture(model: pathlib.Path, role: str) -> tuple[np.ndarray, np.ndarray]:
  from extra.llm_research.layout import GGML_Q4_K, packed_u32_slice, read_metadata
  metadata = read_metadata(model)
  info = next(x for x in metadata.infos if x.name == role)
  if info.typ != GGML_Q4_K or tuple(info.dims) != (K, N):
    raise RuntimeError(f"illegal gate fixture: {info}")
  words = packed_u32_slice(model, metadata, info, device="CPU").numpy().astype(np.uint32, copy=False).reshape(-1)
  q8 = (((np.arange(M*K, dtype=np.int64)*37+11)%255)-127).astype(np.int8).reshape(M, K)
  gids = np.arange(M*(K//32), dtype=np.int64).reshape(M, K//32)
  scales = (2.0**((gids%7)-5)).astype(np.float32)
  sums = q8.reshape(M, K//32, 32).astype(np.int32).sum(2).astype(np.float32)
  record = np.frombuffer(q8.reshape(-1).tobytes()+scales.reshape(-1).tobytes()+sums.reshape(-1).tobytes(), np.uint32).copy()
  if words.nbytes != 28_311_552 or record.nbytes != 2_621_440:
    raise RuntimeError(f"unexpected ABI sizes: words={words.nbytes}, record={record.nbytes}")
  return words, record


def _copy_htod(dst, src: np.ndarray) -> None:
  check(cuda.cuMemcpyHtoD_v2(dst, ctypes.c_void_p(src.ctypes.data), src.nbytes))


def _copy_dtoh(dst: np.ndarray, src) -> None:
  check(cuda.cuMemcpyDtoH_v2(ctypes.c_void_p(dst.ctypes.data), src, dst.nbytes))


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--role", default="blk.0.ffn_gate.weight")
  ap.add_argument("--cubin", type=pathlib.Path, default=pathlib.Path(
    "docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/production_gate_guarded_k64.cubin"))
  ap.add_argument("--source", type=pathlib.Path, default=pathlib.Path(
    "docs/task_workflow/evidence/nv-compiler-packed-fragment-20260828/production_gate_guarded_k64.cu"))
  ap.add_argument("--warmup", type=int, default=5)
  ap.add_argument("--reps", type=int, default=1)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()
  if args.warmup < 0 or args.reps < 1: raise SystemExit("invalid repetition count")

  words, record = _fixture(args.model, args.role)
  output = np.full(M*N, np.nan, np.float32)
  blob, symbol = args.cubin.read_bytes(), _symbol(args.source)

  check(cuda.cuInit(0))
  device, ctx = ctypes.c_int(0), cuda.CUcontext()
  check(cuda.cuDeviceGet(ctypes.byref(device), 0))
  check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device))
  check(cuda.cuCtxSetCurrent(ctx))
  module, function, stream = cuda.CUmodule(), cuda.CUfunction(), cuda.CUstream()
  bufs: list[cuda.CUdeviceptr] = []
  try:
    check(cuda.cuModuleLoadData(ctypes.byref(module), blob))
    check(cuda.cuModuleGetFunction(ctypes.byref(function), module, symbol.encode()))
    for host in (output, record, words):
      ptr = cuda.CUdeviceptr(); check(cuda.cuMemAlloc_v2(ctypes.byref(ptr), host.nbytes)); _copy_htod(ptr, host); bufs.append(ptr)
    params = (ctypes.c_void_p * 3)(*[ctypes.cast(ctypes.pointer(x), ctypes.c_void_p) for x in bufs])
    check(cuda.cuStreamCreate(ctypes.byref(stream), cuda.CU_STREAM_NON_BLOCKING))
    def launch() -> None:
      check(cuda.cuLaunchKernel(function, *GRID, *BLOCK, 0, stream, params, None))
    for _ in range(args.warmup): launch()
    check(cuda.cuStreamSynchronize(stream))
    samples = []
    for _ in range(args.reps):
      begin, end = cuda.CUevent(), cuda.CUevent()
      check(cuda.cuEventCreate(ctypes.byref(begin), 0)); check(cuda.cuEventCreate(ctypes.byref(end), 0))
      check(cuda.cuEventRecord(begin, stream)); launch(); check(cuda.cuEventRecord(end, stream)); check(cuda.cuEventSynchronize(end))
      elapsed = ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(elapsed), begin, end))
      samples.append(float(elapsed.value)*1000.0)
      check(cuda.cuEventDestroy_v2(begin)); check(cuda.cuEventDestroy_v2(end))
    _copy_dtoh(output, bufs[0])
    words_after, record_after = np.empty_like(words), np.empty_like(record)
    _copy_dtoh(words_after, bufs[2]); _copy_dtoh(record_after, bufs[1])
    check(cuda.cuStreamSynchronize(stream))
  finally:
    if stream: cuda.cuStreamDestroy_v2(stream)
    for ptr in bufs: cuda.cuMemFree_v2(ptr)
    if module: cuda.cuModuleUnload(module)
    cuda.cuDevicePrimaryCtxRelease(device)

  result = {
    "schema":"tinygrad.nv_prefill_gateup_ncu_bridge.v1", "model":str(args.model), "role":args.role,
    "cubin":str(args.cubin), "cubin_sha256":hashlib.sha256(blob).hexdigest(), "symbol":symbol,
    "grid":list(GRID), "block":list(BLOCK), "buffer_bytes":[output.nbytes, record.nbytes, words.nbytes],
    "warmup":args.warmup, "reps":args.reps, "samples_us":samples, "min_us":min(samples),
    "median_us":statistics.median(samples), "finite":bool(np.isfinite(output).all()),
    "unwritten":int(np.isnan(output).sum()), "nonzero":int(np.count_nonzero(output)),
    "output_sha256":hashlib.sha256(output.tobytes()).hexdigest(),
    "readonly":{"record":bool(np.array_equal(record, record_after)), "words":bool(np.array_equal(words, words_after))},
  }
  result["passed"] = bool(result["finite"] and result["unwritten"] == 0 and result["nonzero"] == M*N and all(result["readonly"].values()))
  args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
