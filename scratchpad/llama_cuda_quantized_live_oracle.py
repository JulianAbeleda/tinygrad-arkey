#!/usr/bin/env python3
"""Launch an exact extracted llama Q6_K MMQ cubin on tinygrad-owned buffers.

Diagnostic only.  Inputs are packed with the pinned llama CPU reference
quantizers, then decoded independently in Python for the numerical oracle.
The exact cubin entry is launched directly through the CUDA Driver API and
captured as one node in a tinygrad-controlled CUDA graph.
"""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, statistics, sys
import numpy as np
import os, resource

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tinygrad.device import Buffer, Device
from tinygrad.dtype import dtypes
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check

QK_K, QK8_1 = 256, 32
Q4_BLOCK_BYTES, Q6_BLOCK_BYTES, Q8_BLOCK_BYTES = 144, 210, 36
ENTRY_Q4 = "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"
ENTRY_Q6 = "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj"
DEFAULT_CUBIN = pathlib.Path(__file__).with_name("llama_cuda_quantized_oracle_dump") / "libggml-cuda.so.0.14.36.sm_120a.cubin"
DEFAULT_Q8_CUBIN = pathlib.Path("/tmp/llama-oracle-cubins/libggml-cuda.so.0.14.44.sm_120a.cubin")
DEFAULT_BASE = pathlib.Path("/home/ubuntu/env/llama.cpp/build-cuda/bin/libggml-base.so.0.14.0")
ENTRY_Q8 = "_Z13quantize_q8_1PKfPvlllllj5uint3"


class UInt3(ctypes.Structure):
  _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32), ("z", ctypes.c_uint32)]


class FusionArgs(ctypes.Structure):
  _fields_ = [("x_bias", ctypes.c_void_p), ("gate", ctypes.c_void_p),
              ("gate_bias", ctypes.c_void_p), ("glu_op", ctypes.c_int32)]


def sha256(path: pathlib.Path) -> str:
  h = hashlib.sha256()
  with path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
  return h.hexdigest()


def fastdiv_values(divisor: int) -> UInt3:
  if divisor <= 0 or divisor > 0xFFFFFFFF: raise ValueError("invalid fastdiv divisor")
  level = 0
  while level < 32 and (1 << level) < divisor: level += 1
  multiplier = (((1 << 32) * ((1 << level)-divisor)) // divisor + 1) & 0xFFFFFFFF
  return UInt3(multiplier, level, divisor)


def _cpu_quantizers(base_path: pathlib.Path):
  lib = ctypes.CDLL(str(base_path), mode=ctypes.RTLD_LOCAL)
  q4, q6, q8 = lib.quantize_row_q4_K_ref, lib.quantize_row_q6_K_ref, lib.quantize_row_q8_1_ref
  for fn in (q4, q6, q8):
    fn.restype = None
    fn.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_int64]
  return q4, q6, q8


def pack_q6(values: np.ndarray, quantizer) -> bytes:
  values = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
  if values.size % QK_K: raise ValueError("Q6 values must be QK_K aligned")
  out = (ctypes.c_uint8 * (values.size // QK_K * Q6_BLOCK_BYTES))()
  quantizer(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out, values.size)
  return bytes(out)


def pack_q4(values: np.ndarray, quantizer) -> bytes:
  values = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
  if values.size % QK_K: raise ValueError("Q4 values must be QK_K aligned")
  out = (ctypes.c_uint8 * (values.size // QK_K * Q4_BLOCK_BYTES))()
  quantizer(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out, values.size)
  return bytes(out)


def pack_q8(values: np.ndarray, quantizer) -> bytes:
  values = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
  if values.size % QK8_1: raise ValueError("Q8 values must be QK8_1 aligned")
  out = (ctypes.c_uint8 * (values.size // QK8_1 * Q8_BLOCK_BYTES))()
  quantizer(values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), out, values.size)
  return bytes(out)


def decode_q8(payload: bytes) -> np.ndarray:
  if len(payload) % Q8_BLOCK_BYTES: raise ValueError("invalid Q8 payload")
  out = np.empty(len(payload) // Q8_BLOCK_BYTES * QK8_1, dtype=np.float32)
  for ib in range(len(payload) // Q8_BLOCK_BYTES):
    block = payload[ib*Q8_BLOCK_BYTES:(ib+1)*Q8_BLOCK_BYTES]
    d = float(np.frombuffer(block, dtype=np.float16, count=1, offset=0)[0])
    out[ib*QK8_1:(ib+1)*QK8_1] = np.frombuffer(block, dtype=np.int8, count=QK8_1, offset=4).astype(np.float32) * d
  return out


def decode_q6(payload: bytes) -> np.ndarray:
  if len(payload) % Q6_BLOCK_BYTES: raise ValueError("invalid Q6 payload")
  out = np.empty(len(payload) // Q6_BLOCK_BYTES * QK_K, dtype=np.float32)
  for ib in range(len(payload) // Q6_BLOCK_BYTES):
    block = payload[ib*Q6_BLOCK_BYTES:(ib+1)*Q6_BLOCK_BYTES]
    ql = np.frombuffer(block, dtype=np.uint8, count=128, offset=0)
    qh = np.frombuffer(block, dtype=np.uint8, count=64, offset=128)
    scales = np.frombuffer(block, dtype=np.int8, count=16, offset=192).astype(np.int32)
    d = float(np.frombuffer(block, dtype=np.float16, count=1, offset=208)[0])
    decoded = np.empty(QK_K, dtype=np.float32)
    ql_off = qh_off = scale_off = 0
    for half in (0, 128):
      for lane in range(32):
        scale = lane // 16
        h = int(qh[qh_off+lane])
        a, b = int(ql[ql_off+lane]), int(ql[ql_off+32+lane])
        qs = ((a & 15) | (((h >> 0) & 3) << 4),
              (b & 15) | (((h >> 2) & 3) << 4),
              (a >> 4) | (((h >> 4) & 3) << 4),
              (b >> 4) | (((h >> 6) & 3) << 4))
        for offset, scale_delta, q in zip((0, 32, 64, 96), (0, 2, 4, 6), qs):
          decoded[half+lane+offset] = d * scales[scale_off+scale+scale_delta] * (q-32)
      ql_off += 64; qh_off += 32; scale_off += 8
    out[ib*QK_K:(ib+1)*QK_K] = decoded
  return out


def decode_q4(payload: bytes) -> np.ndarray:
  """Independent block_q4_K decoder (ggml's six-bit scale/min packing)."""
  if len(payload) % Q4_BLOCK_BYTES: raise ValueError("invalid Q4 payload")
  out = np.empty(len(payload) // Q4_BLOCK_BYTES * QK_K, dtype=np.float32)
  for ib in range(len(payload) // Q4_BLOCK_BYTES):
    block = payload[ib*Q4_BLOCK_BYTES:(ib+1)*Q4_BLOCK_BYTES]
    d, dmin = (float(x) for x in np.frombuffer(block, dtype=np.float16, count=2))
    scales = np.frombuffer(block, dtype=np.uint8, count=12, offset=4)
    qs = np.frombuffer(block, dtype=np.uint8, count=128, offset=16)
    def get_scale(j: int):
      if j < 4: return int(scales[j] & 63), int(scales[j+4] & 63)
      return int((scales[j+4] & 15) | ((scales[j-4] >> 6) << 4)), int((scales[j+4] >> 4) | ((scales[j] >> 6) << 4))
    decoded = np.empty(QK_K, dtype=np.float32)
    qoff = outoff = 0
    for j in range(0, 8, 2):
      sc, mn = get_scale(j)
      for l in range(32):
        decoded[outoff+l] = d*sc*(int(qs[qoff+l]) & 15) - dmin*mn
      sc, mn = get_scale(j+1)
      for l in range(32):
        decoded[outoff+32+l] = d*sc*(int(qs[qoff+l]) >> 4) - dmin*mn
      qoff += 32; outoff += 64
    out[ib*QK_K:(ib+1)*QK_K] = decoded
  return out


def device_pointer(buffer: Buffer, offset_bytes: int = 0) -> ctypes.c_void_p:
  opaque = buffer.ensure_allocated()._buf
  return ctypes.c_void_p((opaque.value if hasattr(opaque, "value") else int(opaque)) + offset_bytes)


def _kernel_params(weight: Buffer, activation: Buffer, output: Buffer, nrows: int, k: int, guard: int):
  zero = UInt3(0, 0, 0)
  one = fastdiv_values(1)
  row_blocks, q8_blocks = k // QK_K, k // QK8_1
  args = [device_pointer(weight), device_pointer(activation), ctypes.c_void_p(), FusionArgs(),
          device_pointer(output, guard*4), ctypes.c_uint32(k), zero, ctypes.c_uint32(row_blocks),
          ctypes.c_uint32(q8_blocks), ctypes.c_uint32(nrows), one,
          ctypes.c_uint32(nrows*row_blocks), ctypes.c_uint32(q8_blocks), ctypes.c_uint32(nrows), one,
          ctypes.c_uint32(nrows*row_blocks), ctypes.c_uint32(q8_blocks), ctypes.c_uint32(nrows), ctypes.c_uint32(0)]
  params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in args])
  return args, params


def _copy_f32(buffer: Buffer) -> np.ndarray:
  raw = bytearray(buffer.nbytes); buffer.copyout(memoryview(raw))
  return np.frombuffer(raw, dtype=np.float32).copy()


def live(cubin: pathlib.Path, base: pathlib.Path, nrows: int, k: int, replays: int, timing_iters: int, timing_reps: int,
         gpu_q8_cubin: pathlib.Path | None = None, quant: str = "Q6_K", weight_fixture: pathlib.Path | None = None,
         activation_fixture: pathlib.Path | None = None, reference_fixture: pathlib.Path | None = None) -> dict:
  if k % 512 or nrows <= 0: raise ValueError("K must be 512-aligned and nrows positive")
  if ctypes.sizeof(FusionArgs) != 32 or ctypes.sizeof(UInt3) != 12: raise RuntimeError("ABI struct size mismatch")
  def mark(s):
    print(f"[vocab-replay] {s} rss_mb={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f}", flush=True)
  mark(f"start rows={nrows}")
  q4, q6, q8 = _cpu_quantizers(base); mark("quantizers loaded")
  if quant not in ("Q4_K", "Q6_K"): raise ValueError(f"unsupported quant {quant}")
  pack_weight, decode_weight, weight_bytes, entry = ((pack_q4, decode_q4, Q4_BLOCK_BYTES, ENTRY_Q4) if quant == "Q4_K" else
                                                       (pack_q6, decode_q6, Q6_BLOCK_BYTES, ENTRY_Q6))
  if (weight_fixture is None) != (activation_fixture is None): raise ValueError("weight and activation fixtures must be paired")
  if weight_fixture is not None:
    # tinygrad's CUDA copyin requires a writable buffer for its zero-copy address
    # lookup; mmap the immutable fixture copy-on-write so host bytes stay unchanged.
    weights_quant = np.memmap(weight_fixture, mode="c", dtype=np.uint8)
    activation_f32 = np.memmap(activation_fixture, mode="c", dtype=np.float32)
    expected_weight_bytes = nrows * (k // QK_K) * weight_bytes
    if len(weights_quant) != expected_weight_bytes or activation_f32.size != k: raise ValueError("fixture shape mismatch")
    activation_q8 = pack_q8(activation_f32, q8); mark("fixtures mapped and q8 packed")
  else:
    rng = np.random.default_rng(20260804)
    weights_f32 = rng.normal(0, 0.2, size=(nrows, k)).astype(np.float32)
    activation_f32 = rng.normal(0, 0.2, size=k).astype(np.float32)
    weights_quant, activation_q8 = pack_weight(weights_f32, q4 if quant == "Q4_K" else q6), pack_q8(activation_f32, q8)
  if reference_fixture is not None:
    reference = np.fromfile(reference_fixture, dtype=np.float32)
    if reference.size != nrows: raise ValueError("reference fixture shape mismatch")
    mark("reference mapped")
  else:
    reference = decode_weight(weights_quant).reshape(nrows, k) @ decode_q8(activation_q8)

  dev = Device["CUDA"]; mark("cuda device initialized")
  weight = Buffer("CUDA", len(weights_quant), dtypes.uint8, initial_value=memoryview(weights_quant))
  activation_f32_buf = Buffer("CUDA", activation_f32.size, dtypes.float32, initial_value=bytearray(activation_f32.tobytes()))
  activation = Buffer("CUDA", len(activation_q8), dtypes.uint8,
                      preallocate=True if gpu_q8_cubin is not None else False,
                      initial_value=None if gpu_q8_cubin is not None else bytearray(activation_q8))
  guard, sentinel = 32, np.float32(12345.25)
  output_init = np.full(nrows+2*guard, sentinel, dtype=np.float32)
  output = Buffer("CUDA", output_init.size, dtypes.float32, initial_value=bytearray(output_init.tobytes()))
  dev.synchronize(); mark("buffers allocated and uploaded")

  module, function, q8_module, q8_function, stream = cuda.CUmodule(), cuda.CUfunction(), cuda.CUmodule(), cuda.CUfunction(), cuda.CUstream()
  graph, instance = cuda.CUgraph(), cuda.CUgraphExec()
  start, end = cuda.CUevent(), cuda.CUevent()
  try:
    check(cuda.cuModuleLoad(ctypes.byref(module), str(cubin).encode()))
    check(cuda.cuModuleGetFunction(ctypes.byref(function), module, entry.encode()))
    if gpu_q8_cubin is not None:
      check(cuda.cuModuleLoad(ctypes.byref(q8_module), str(gpu_q8_cubin).encode()))
      check(cuda.cuModuleGetFunction(ctypes.byref(q8_function), q8_module, ENTRY_Q8.encode()))
    check(cuda.cuStreamCreate(ctypes.byref(stream), cuda.CU_STREAM_NON_BLOCKING))
    args, params = _kernel_params(weight, activation, output, nrows, k, guard)

    def launch_kernel():
      check(cuda.cuLaunchKernel(function, nrows, 1, 1, 32, 4, 1, 0, stream, params, None))

    q8_args = [device_pointer(activation_f32_buf), device_pointer(activation), ctypes.c_int64(k), ctypes.c_int64(k),
               ctypes.c_int64(k), ctypes.c_int64(k), ctypes.c_int64(k), ctypes.c_uint32(1), fastdiv_values(1)]
    q8_params = (ctypes.c_void_p * len(q8_args))(*[ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p) for arg in q8_args])

    def launch_sequence():
      if gpu_q8_cubin is not None:
        check(cuda.cuLaunchKernel(q8_function, (k+255)//256, 1, 1, 256, 1, 1, 0, stream, q8_params, None))
      launch_kernel()

    mark("launching producer and consumer")
    launch_sequence(); check(cuda.cuStreamSynchronize(stream)); mark("producer and consumer complete")
    direct = _copy_f32(output)
    direct_result = direct[guard:guard+nrows]
    max_abs = float(np.max(np.abs(direct_result-reference)))
    max_rel = float(np.max(np.abs(direct_result-reference) / np.maximum(np.abs(reference), 1e-3)))
    guards_ok = bool(np.all(direct[:guard] == sentinel) and np.all(direct[guard+nrows:] == sentinel))
    gpu_q8 = bytearray(activation.nbytes); activation.copyout(memoryview(gpu_q8))
    q8_byte_mismatches = sum(a != b for a, b in zip(gpu_q8, activation_q8))
    q8_blocks_gpu = np.frombuffer(gpu_q8, dtype=np.uint8).reshape(-1, Q8_BLOCK_BYTES)
    q8_blocks_cpu = np.frombuffer(activation_q8, dtype=np.uint8).reshape(-1, Q8_BLOCK_BYTES)
    q8_d_mismatch_blocks = int(np.any(q8_blocks_gpu[:, 0:2] != q8_blocks_cpu[:, 0:2], axis=1).sum())
    q8_s_mismatch_blocks = int(np.any(q8_blocks_gpu[:, 2:4] != q8_blocks_cpu[:, 2:4], axis=1).sum())
    q8_qs_mismatch_bytes = int((q8_blocks_gpu[:, 4:] != q8_blocks_cpu[:, 4:]).sum())
    q8_dequant_max_abs = float(np.max(np.abs(decode_q8(gpu_q8)-decode_q8(activation_q8))))
    q8_mismatch_fields = {"d": 0, "s": 0, "qs": 0}
    for off, (actual, expected) in enumerate(zip(gpu_q8, activation_q8)):
      if actual != expected: q8_mismatch_fields["d" if off % Q8_BLOCK_BYTES < 2 else "s" if off % Q8_BLOCK_BYTES < 4 else "qs"] += 1

    check(cuda.cuStreamBeginCapture_v2(stream, cuda.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
    launch_sequence()
    check(cuda.cuStreamEndCapture(stream, ctypes.byref(graph))); mark("graph captured")
    node_count = ctypes.c_size_t(); check(cuda.cuGraphGetNodes(graph, None, ctypes.byref(node_count)))
    edge_count = ctypes.c_size_t(); check(cuda.cuGraphGetEdges(graph, None, None, ctypes.byref(edge_count)))
    graph_nodes = (cuda.CUgraphNode * node_count.value)(); check(cuda.cuGraphGetNodes(graph, graph_nodes, ctypes.byref(node_count)))
    edge_from, edge_to = (cuda.CUgraphNode * edge_count.value)(), (cuda.CUgraphNode * edge_count.value)()
    if edge_count.value: check(cuda.cuGraphGetEdges(graph, edge_from, edge_to, ctypes.byref(edge_count)))
    graph_ids = {ctypes.cast(node, ctypes.c_void_p).value: i for i, node in enumerate(graph_nodes)}
    graph_node_functions = []
    for node in graph_nodes:
      params_node = cuda.CUDA_KERNEL_NODE_PARAMS_v1(); check(cuda.cuGraphKernelNodeGetParams(node, ctypes.byref(params_node)))
      fn = ctypes.cast(params_node.func, ctypes.c_void_p).value
      graph_node_functions.append("q8" if fn == ctypes.cast(q8_function, ctypes.c_void_p).value else "q6" if fn == ctypes.cast(function, ctypes.c_void_p).value else "unknown")
    graph_edges = [[graph_ids[ctypes.cast(edge_from[i], ctypes.c_void_p).value], graph_ids[ctypes.cast(edge_to[i], ctypes.c_void_p).value]]
                   for i in range(edge_count.value)]
    check(cuda.cuGraphInstantiate_v2(ctypes.byref(instance), graph, None, None, 0))
    for _ in range(replays): check(cuda.cuGraphLaunch(instance, stream))
    check(cuda.cuStreamSynchronize(stream))
    replay_result = _copy_f32(output)[guard:guard+nrows]
    replay_err = float(np.max(np.abs(replay_result-reference)))

    check(cuda.cuEventCreate(ctypes.byref(start), 0)); check(cuda.cuEventCreate(ctypes.byref(end), 0))
    timing_us = []
    for _ in range(timing_reps):
      check(cuda.cuEventRecord(start, stream))
      for _ in range(timing_iters): check(cuda.cuGraphLaunch(instance, stream))
      check(cuda.cuEventRecord(end, stream)); check(cuda.cuEventSynchronize(end))
      elapsed_ms = ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), start, end))
      timing_us.append(float(elapsed_ms.value)*1000/timing_iters)

    single_replay_us = []
    for _ in range(max(25, timing_reps)):
      check(cuda.cuEventRecord(start, stream)); check(cuda.cuGraphLaunch(instance, stream)); check(cuda.cuEventRecord(end, stream))
      check(cuda.cuEventSynchronize(end))
      elapsed_ms = ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), start, end))
      single_replay_us.append(float(elapsed_ms.value)*1000)

    q8_timing_us = []
    if gpu_q8_cubin is not None:
      for _ in range(timing_reps):
        check(cuda.cuEventRecord(start, stream))
        for _ in range(timing_iters):
          check(cuda.cuLaunchKernel(q8_function, (k+255)//256, 1, 1, 256, 1, 1, 0, stream, q8_params, None))
        check(cuda.cuEventRecord(end, stream)); check(cuda.cuEventSynchronize(end))
        elapsed_ms = ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), start, end))
        q8_timing_us.append(float(elapsed_ms.value)*1000/timing_iters)

    atol = max(2e-2, float(np.max(np.abs(reference))) * 2e-4)
    evidence = (f"DIAGNOSTIC_EXACT_Q8_PLUS_{quant}_FULL_PRIMITIVE" if gpu_q8_cubin is not None
                else f"DIAGNOSTIC_EXACT_{quant}_CUBIN_PREPACKED_Q8")
    return {"schema": "tinygrad.llama_cuda_quantized_live_oracle.v1", "evidence": evidence,
            "cubin": {"path": str(cubin), "sha256": sha256(cubin), "entry": entry},
            "q8_cubin": None if gpu_q8_cubin is None else {"path": str(gpu_q8_cubin), "sha256": sha256(gpu_q8_cubin), "entry": ENTRY_Q8},
            "cpu_library": {"path": str(base), "sha256": sha256(base), "use": "packing only; Python independently decodes reference"},
            "shape": {"quant": quant, "nrows": nrows, "k": k, "grid": [nrows,1,1], "block": [32,4,1], "weight_block_bytes": weight_bytes},
            "abi": {"argument_count": len(args), "fusion_args_bytes": ctypes.sizeof(FusionArgs), "uint3_bytes": ctypes.sizeof(UInt3)},
            "correctness": {"max_abs_err": max_abs, "max_rel_err": max_rel, "replay_max_abs_err": replay_err,
                            "q8_byte_mismatches_vs_cpu_reference": q8_byte_mismatches,
                            "q8_d_mismatch_blocks": q8_d_mismatch_blocks,
                            "q8_s_mismatch_blocks": q8_s_mismatch_blocks,
                            "q8_qs_mismatch_bytes": q8_qs_mismatch_bytes,
                            "q8_dequant_max_abs_diff": q8_dequant_max_abs,
                            "q8_byte_mismatch_fields": q8_mismatch_fields,
                            "atol": atol, "guards_ok": guards_ok, "pass": guards_ok and max_abs <= atol and replay_err <= atol},
            "graph": {"nodes": int(node_count.value), "edges": int(edge_count.value), "node_functions": graph_node_functions,
                      "dependency_edges": graph_edges, "replays": replays,
                      "capture_pass": node_count.value == (2 if gpu_q8_cubin is not None else 1) and
                                      edge_count.value == (1 if gpu_q8_cubin is not None else 0)},
            "timing": {"iterations_per_rep": timing_iters, "reps": timing_reps,
                       "event_us_per_replay_reps": timing_us,
                       "median_event_us_per_replay": float(statistics.median(timing_us)),
                       "single_replay_event_us_reps": single_replay_us,
                       "median_single_replay_event_us": float(statistics.median(single_replay_us)),
                       "q8_event_us_per_launch_reps": q8_timing_us,
                       "median_q8_event_us_per_launch": None if not q8_timing_us else float(statistics.median(q8_timing_us))},
            "non_claims": (["isolated full-primitive bound, not token wall"] if gpu_q8_cubin is not None else
                           ["prepacked q8 excludes activation production", "isolated exact-kernel bound, not token wall"])}
  finally:
    if end: cuda.cuEventDestroy_v2(end)
    if start: cuda.cuEventDestroy_v2(start)
    if instance: cuda.cuGraphExecDestroy(instance)
    if graph: cuda.cuGraphDestroy(graph)
    if stream: cuda.cuStreamDestroy_v2(stream)
    if q8_module: cuda.cuModuleUnload(q8_module)
    if module: cuda.cuModuleUnload(module)


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--cubin", type=pathlib.Path, default=DEFAULT_CUBIN)
  p.add_argument("--base-library", type=pathlib.Path, default=DEFAULT_BASE)
  p.add_argument("--nrows", type=int, default=1024)
  p.add_argument("--k", type=int, default=4096)
  p.add_argument("--replays", type=int, default=8)
  p.add_argument("--timing-iters", type=int, default=1000)
  p.add_argument("--timing-reps", type=int, default=5)
  p.add_argument("--gpu-q8-cubin", type=pathlib.Path)
  p.add_argument("--quant", choices=("Q4_K", "Q6_K"), default="Q6_K")
  p.add_argument("--weight-fixture", type=pathlib.Path)
  p.add_argument("--activation-fixture", type=pathlib.Path)
  p.add_argument("--reference-fixture", type=pathlib.Path)
  p.add_argument("--inspect-only", action="store_true")
  args = p.parse_args()
  if not args.cubin.is_file() or not args.base_library.is_file(): raise FileNotFoundError("oracle artifacts missing")
  if args.inspect_only:
    report = {"schema": "tinygrad.llama_cuda_quantized_live_oracle.v1", "mode": "inspect",
              "cubin": {"path": str(args.cubin), "sha256": sha256(args.cubin)},
              "cpu_library": {"path": str(args.base_library), "sha256": sha256(args.base_library)},
              "abi": {"entry": ENTRY_Q6, "argument_count": 19, "fusion_args_bytes": ctypes.sizeof(FusionArgs), "uint3_bytes": ctypes.sizeof(UInt3)}}
  else: report = live(args.cubin, args.base_library, args.nrows, args.k, args.replays, args.timing_iters, args.timing_reps,
                      args.gpu_q8_cubin, args.quant, args.weight_fixture, args.activation_fixture, args.reference_fixture)
  print(json.dumps(report, sort_keys=True))
  return 0 if args.inspect_only or (report["correctness"]["pass"] and report["graph"]["capture_pass"]) else 1


if __name__ == "__main__": sys.exit(main())
