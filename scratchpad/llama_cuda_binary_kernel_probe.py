#!/usr/bin/env python3
"""Feasibility probe for using a compiled llama.cpp CUDA kernel from tinygrad.

This is deliberately diagnostic-only.  It loads the instantiated
launch_mul_mat_vec_f_cuda<float, float, 1, false> wrapper from llama.cpp's
libggml-cuda, passes tinygrad-owned CUDA buffers to it, checks the numerical
result, and captures the same launch into a CUDA graph for replay.

The probe does not rebuild or copy the llama kernel and does not add it to a
tinygrad route.  --inspect-only performs the CPU-only artifact/ABI checks.
The default mode requires a live NVIDIA GPU and must be run under the repo's
GPU flock policy.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check


DEFAULT_LLAMA_ROOT = Path("/home/ubuntu/env/llama.cpp")
MMVF_SYMBOL = (
  "_Z25launch_mul_mat_vec_f_cudaIffLi1ELb0EEvPKT_PKfPKi"
  "31ggml_cuda_mm_fusion_args_devicePfllllllllllllllllllP11CUstream_st"
)


class FusionArgs(ctypes.Structure):
  # Mirrors ggml_cuda_mm_fusion_args_device in ggml-cuda/common.cuh.
  _fields_ = [
    ("x_bias", ctypes.c_void_p),
    ("gate", ctypes.c_void_p),
    ("gate_bias", ctypes.c_void_p),
    ("glu_op", ctypes.c_int),
  ]


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def resolve_artifacts(root: Path) -> tuple[Path, Path]:
  library = Path(os.environ.get("LLAMA_CUDA_LIB", root / "build-cuda/bin/libggml-cuda.so.0.14.0"))
  source = root / "ggml/src/ggml-cuda/mmvf.cu"
  if not library.is_file(): raise FileNotFoundError(f"llama CUDA library not found: {library}")
  if not source.is_file(): raise FileNotFoundError(f"llama MMVF source not found: {source}")
  return library.resolve(), source.resolve()


def inspect(library: Path, source: Path) -> tuple[ctypes.CDLL, dict[str, object]]:
  lib = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
  try:
    launcher = getattr(lib, MMVF_SYMBOL)
    symbol_present = launcher is not None
  except AttributeError:
    symbol_present = False

  cuobjdump = Path("/usr/local/cuda-13.2/bin/cuobjdump")
  cubins: list[str] = []
  if cuobjdump.is_file():
    result = subprocess.run([str(cuobjdump), "--list-elf", str(library)], check=True, text=True, capture_output=True)
    cubins = [line.strip() for line in result.stdout.splitlines() if line.startswith("ELF file")]

  report: dict[str, object] = {
    "schema": "tinygrad.llama_cuda_binary_kernel_probe.v1",
    "mode": "inspect",
    "library": str(library),
    "library_sha256": sha256(library),
    "source": str(source),
    "source_sha256": sha256(source),
    "symbol": MMVF_SYMBOL,
    "dynamic_symbol_present": symbol_present,
    "fusion_args_size": ctypes.sizeof(FusionArgs),
    "embedded_sm120a_cubins": len([line for line in cubins if "sm_120a.cubin" in line]),
    "binary_reuse_candidate": symbol_present and ctypes.sizeof(FusionArgs) == 32,
  }
  if not report["binary_reuse_candidate"]:
    raise RuntimeError(f"llama binary ABI preflight failed: {json.dumps(report, sort_keys=True)}")
  return lib, report


def device_pointer(buffer: Buffer) -> ctypes.c_void_p:
  opaque = buffer.ensure_allocated()._buf
  value = opaque.value if hasattr(opaque, "value") else int(opaque)
  return ctypes.c_void_p(value)


def configure_launcher(lib: ctypes.CDLL):
  launcher = getattr(lib, MMVF_SYMBOL)
  launcher.restype = None
  launcher.argtypes = (
    [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, FusionArgs, ctypes.c_void_p]
    + [ctypes.c_int64] * 18
    + [ctypes.c_void_p]
  )
  return launcher


def launch_mmvf(launcher, x: Buffer, y: Buffer, dst: Buffer, nrows: int, ncols: int, stream: cuda.CUstream) -> None:
  # Shapes are the no-IDs, one-column case:
  # src0=[ncols,nrows,1,1], src1=[ncols,1,1,1], dst=[nrows,1,1,1].
  launcher(
    device_pointer(x), device_pointer(y), None, FusionArgs(), device_pointer(dst),
    ncols, nrows,
    ncols, ncols, nrows,
    1, 1, 1,
    nrows * ncols, ncols, nrows, 1,
    1, nrows * ncols, ncols, nrows,
    1, 0, ctypes.cast(stream, ctypes.c_void_p),
  )


def copy_f32(buffer: Buffer) -> np.ndarray:
  raw = bytearray(buffer.nbytes)
  buffer.copyout(memoryview(raw))
  return np.frombuffer(raw, dtype=np.float32).copy()


def live_probe(lib: ctypes.CDLL, report: dict[str, object], nrows: int, ncols: int, replays: int) -> dict[str, object]:
  if ncols % 2: raise ValueError("llama MMVF requires an even ncols")
  launcher = configure_launcher(lib)

  # Creating Device[CUDA] establishes the context that owns these buffers.
  dev = Device["CUDA"]
  rng = np.random.default_rng(20260804)
  x_host = rng.normal(0.0, 0.25, size=(nrows, ncols)).astype(np.float32)
  y_host = rng.normal(0.0, 0.25, size=(ncols,)).astype(np.float32)
  reference = x_host @ y_host

  x = Buffer("CUDA", x_host.size, dtypes.float32, initial_value=bytearray(x_host.tobytes()))
  y = Buffer("CUDA", y_host.size, dtypes.float32, initial_value=bytearray(y_host.tobytes()))
  dst = Buffer("CUDA", nrows, dtypes.float32, preallocate=True)
  dev.synchronize()

  stream = cuda.CUstream()
  graph = cuda.CUgraph()
  instance = cuda.CUgraphExec()
  try:
    check(cuda.cuStreamCreate(ctypes.byref(stream), cuda.CU_STREAM_NON_BLOCKING))

    # Warm launch proves direct cross-runtime use and initializes llama's device
    # metadata before capture (host-side initialization is not capture work).
    launch_mmvf(launcher, x, y, dst, nrows, ncols, stream)
    check(cuda.cuStreamSynchronize(stream))
    direct = copy_f32(dst)
    direct_err = float(np.max(np.abs(direct - reference)))

    # Capture the exact same compiled llama launch in a CUDA graph owned and
    # controlled through tinygrad's CUDA driver bindings.
    check(cuda.cuStreamBeginCapture_v2(stream, cuda.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
    launch_mmvf(launcher, x, y, dst, nrows, ncols, stream)
    check(cuda.cuStreamEndCapture(stream, ctypes.byref(graph)))
    node_count = ctypes.c_size_t()
    check(cuda.cuGraphGetNodes(graph, None, ctypes.byref(node_count)))
    check(cuda.cuGraphInstantiate_v2(ctypes.byref(instance), graph, None, None, 0))
    for _ in range(replays): check(cuda.cuGraphLaunch(instance, stream))
    check(cuda.cuStreamSynchronize(stream))
    replay = copy_f32(dst)
    replay_err = float(np.max(np.abs(replay - reference)))

    tolerance = 2e-3
    report.update({
      "mode": "live",
      "gpu": getattr(dev, "device", "CUDA"),
      "shape": {"nrows": nrows, "ncols": ncols, "ncols_dst": 1},
      "direct_max_abs_err": direct_err,
      "graph_max_abs_err": replay_err,
      "graph_nodes": int(node_count.value),
      "graph_replays": replays,
      "direct_launch_ok": direct_err <= tolerance,
      "graph_capture_replay_ok": replay_err <= tolerance and node_count.value == 1,
      "compiled_llama_kernel_reusable": direct_err <= tolerance,
      "compiled_llama_kernel_graph_compatible": replay_err <= tolerance and node_count.value == 1,
    })
    return report
  finally:
    if instance: cuda.cuGraphExecDestroy(instance)
    if graph: cuda.cuGraphDestroy(graph)
    if stream: cuda.cuStreamDestroy_v2(stream)


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--llama-root", type=Path, default=DEFAULT_LLAMA_ROOT)
  parser.add_argument("--inspect-only", action="store_true")
  parser.add_argument("--nrows", type=int, default=64)
  parser.add_argument("--ncols", type=int, default=512)
  parser.add_argument("--replays", type=int, default=8)
  args = parser.parse_args()

  library, source = resolve_artifacts(args.llama_root)
  lib, report = inspect(library, source)
  if not args.inspect_only: report = live_probe(lib, report, args.nrows, args.ncols, args.replays)
  print(json.dumps(report, sort_keys=True))
  required = report["binary_reuse_candidate"] if args.inspect_only else (
    report["compiled_llama_kernel_reusable"] and report["compiled_llama_kernel_graph_compatible"]
  )
  return 0 if required else 1


if __name__ == "__main__":
  sys.exit(main())
