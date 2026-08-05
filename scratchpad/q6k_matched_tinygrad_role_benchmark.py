#!/usr/bin/env python3
"""Diagnostic matched Q6_K K/V role benchmark.

This keeps the current tinygrad ``q6k_gen_partial_1024_4096_4`` primitive and
its required external reduction together in one CUDA graph.  Its weights are
the exact Q6_K byte payload produced by the pinned llama CPU packer used by
``llama_cuda_quantized_live_oracle.py``.  The activation starts from the same
exact Q8_1 payload, is independently decoded, then rounded to fp16 because
the current tinygrad role ABI takes fp16 (whereas llama's MMQ consumes Q8_1).

This is therefore an apples-to-apples *packed-weight / quantized-activation*
diagnostic, but not a claim that tinygrad executes llama's Q8 producer.  The
report carries both the tinygrad-fp16 and exact-Q8 reference errors.
"""
from __future__ import annotations

import argparse, ctypes, hashlib, json, pathlib, statistics, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scratchpad.llama_cuda_quantized_live_oracle import (DEFAULT_BASE, decode_q6, decode_q8, pack_q6, pack_q8, sha256, _cpu_quantizers)

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.engine.realize import get_graph_runtime
from tinygrad.llm.decode_kernels import emit_q6k_gemv_kernel, q6k_spec_for_role
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
from tinygrad.runtime.autogen import cuda
from tinygrad.runtime.ops_cuda import check
from tinygrad.uop.ops import Ops


def _graph_runner(jit: TinyJit):
  assert jit.captured is not None
  graph_calls = [call for call in jit.captured.linear.src
                 if call.src[0].op is Ops.CUSTOM_FUNCTION and call.src[0].arg == "graph"]
  if len(graph_calls) != 1: raise RuntimeError(f"expected one graph call, got {len(graph_calls)}")
  return get_graph_runtime(graph_calls[0].src[0])


def live(rows: int, k: int, replays: int, timing_iters: int, timing_reps: int, base: pathlib.Path) -> dict:
  if (rows, k) != (1024, 4096): raise ValueError("this benchmark is pinned to the observed K/V role 1024x4096")
  q6, q8 = _cpu_quantizers(base)
  rng = np.random.default_rng(20260804)
  weights_f32 = rng.normal(0, .2, size=(rows, k)).astype(np.float32)
  activation_f32 = rng.normal(0, .2, size=k).astype(np.float32)
  q6_payload, q8_payload = pack_q6(weights_f32, q6), pack_q8(activation_f32, q8)
  decoded_w, decoded_q8 = decode_q6(q6_payload).reshape(rows, k), decode_q8(q8_payload)
  x_fp16 = decoded_q8.astype(np.float16)
  ref_tiny_abi = decoded_w @ x_fp16.astype(np.float32)
  ref_exact_q8 = decoded_w @ decoded_q8

  spec = q6k_spec_for_role(rows, k, role="attn_kv", parts=4, use_coop=False, reduction="external_sum")
  weights = Tensor(np.frombuffer(q6_payload, dtype=np.uint16).copy(), dtype=dtypes.uint16, device="CUDA").contiguous().realize()
  x = Tensor(x_fp16.copy(), dtype=dtypes.float16, device="CUDA").contiguous().realize()

  @TinyJit
  def run(weight: Tensor, activation: Tensor):
    partials = execute_research_program(Tensor.empty((rows, spec.partial_axis_extent), dtype=dtypes.float32, device="CUDA"), weight, activation,
      program=KernelProgram("diagnostic.q6k_matched_role", spec.kernel_name, KernelProgramProvenance.RESEARCH_ONLY,
                            emit_q6k_gemv_kernel(spec)))
    # This sum is part of the measured role: callers consume a vector, not the
    # four independent K-slice partials emitted by the installed primitive.
    return partials.sum(axis=1).contiguous()

  run(weights, x).realize()       # eager compile/execute
  got_t = run(weights, x).realize() # capture/build graph
  got = got_t.numpy().astype(np.float32)
  runner = _graph_runner(run)
  Device["CUDA"].synchronize()

  # The graph has fixed buffers/parameters after the JIT capture.  Direct
  # launches let CUDA events measure the same complete graph path without
  # CPU submission or synchronize time.
  start, end = cuda.CUevent(), cuda.CUevent()
  try:
    check(cuda.cuEventCreate(ctypes.byref(start), 0)); check(cuda.cuEventCreate(ctypes.byref(end), 0))
    for _ in range(replays):
      check(cuda.cuGraphLaunch(runner.instance, None))
    Device["CUDA"].synchronize()
    replay = got_t.numpy().astype(np.float32)
    reps = []
    for _ in range(timing_reps):
      check(cuda.cuEventRecord(start, None))
      for _ in range(timing_iters): check(cuda.cuGraphLaunch(runner.instance, None))
      check(cuda.cuEventRecord(end, None)); check(cuda.cuEventSynchronize(end))
      elapsed_ms = ctypes.c_float(); check(cuda.cuEventElapsedTime(ctypes.byref(elapsed_ms), start, end))
      reps.append(float(elapsed_ms.value) * 1000.0 / timing_iters)
  finally:
    if end: cuda.cuEventDestroy_v2(end)
    if start: cuda.cuEventDestroy_v2(start)

  err = np.abs(got-ref_tiny_abi)
  exact_err = np.abs(got-ref_exact_q8)
  max_abs, max_rel = float(err.max()), float((err/np.maximum(np.abs(ref_tiny_abi), 1e-3)).max())
  atol = max(2e-2, float(np.max(np.abs(ref_tiny_abi))) * 2e-4)
  return {"schema": "tinygrad.q6k_matched_tinygrad_role_benchmark.v1", "evidence": "DIAGNOSTIC_MATCHED_PACKED_Q6K_ROLE",
          "role": {"semantic_role": "attn_kv", "shape": [rows, k], "tinygrad_kernel": spec.kernel_name,
                   "partial_shape": [rows, spec.partial_axis_extent], "required_tail": "Tensor.sum(axis=1).contiguous()"},
          "operand_contract": {"q6_payload_sha256": hashlib.sha256(q6_payload).hexdigest(), "q6_payload_bytes": len(q6_payload),
             "q8_payload_sha256": hashlib.sha256(q8_payload).hexdigest(), "q8_payload_bytes": len(q8_payload),
             "activation_bridge": "same Q8_1 payload independently decoded then rounded to fp16 for tinygrad ABI",
             "max_abs_fp16_bridge_vs_q8": float(np.max(np.abs(x_fp16.astype(np.float32)-decoded_q8)))},
          "cpu_library": {"path": str(base), "sha256": sha256(base), "use": "packing only; Python independently decodes references"},
          "graph": {"graph_calls": 1, "cuda_nodes": len(runner.nodes), "replays": replays},
          "correctness": {"tinygrad_abi_max_abs_err": max_abs, "tinygrad_abi_max_rel_err": max_rel,
             "exact_q8_reference_max_abs_delta": float(exact_err.max()), "replay_max_abs_err": float(np.max(np.abs(replay-ref_tiny_abi))),
             "atol": atol, "pass": bool(max_abs <= atol and np.max(np.abs(replay-ref_tiny_abi)) <= atol)},
          "timing": {"iterations_per_rep": timing_iters, "reps": timing_reps, "event_us_per_replay_reps": reps,
                     "median_event_us_per_replay": float(statistics.median(reps))},
          "non_claims": ["not token wall time", "not an exact llama Q8 producer replacement", "research-only route invocation"]}


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--base-library", type=pathlib.Path, default=DEFAULT_BASE)
  ap.add_argument("--replays", type=int, default=8); ap.add_argument("--timing-iters", type=int, default=1000)
  ap.add_argument("--timing-reps", type=int, default=5); ap.add_argument("--out", type=pathlib.Path)
  a = ap.parse_args()
  out = live(1024, 4096, a.replays, a.timing_iters, a.timing_reps, a.base_library)
  text = json.dumps(out, sort_keys=True)
  if a.out: a.out.write_text(text+"\n")
  print(text)
  return 0 if out["correctness"]["pass"] else 1


if __name__ == "__main__": raise SystemExit(main())
