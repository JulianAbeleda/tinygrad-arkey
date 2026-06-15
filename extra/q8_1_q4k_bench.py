#!/usr/bin/env python3
from __future__ import annotations

import argparse, pathlib, time
from math import prod

from tinygrad import Tensor, TinyJit, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.gguf import ggml_data_to_tensor

from extra.q4_k_gemv_primitive import (
  parse_opt, q8_1_bias_pack_u32_kernel, q4k_q8_1_gemv_partial_kernel, q4k_q8_1_intdot_partial_kernel,
  q4k_q8_1_vdot_parallel_partial_kernel, q4k_q8_1_vdot_partial_kernel, q4k_q8_1_vdot_builtin_partial_kernel,
  q4k_unpack_kernel,
)
from extra.qk_layout import (
  GGML_Q4_K, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, q4_k_reference, q8_1_dequantize, q8_1_quantize, read_metadata, tensor_shape,
)

def bench(label:str, iters:int, q4_bytes:int, fn) -> None:
  fn().realize()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(iters): fn().realize()
  wall_dt = (time.perf_counter() - st) / iters
  dev_dt = GlobalCounters.time_sum_s / iters
  dev_s = f"{dev_dt*1000:.3f} ms ({q4_bytes/dev_dt/1e9:.2f} Q4-GB/s)" if dev_dt > 0 else "n/a"
  print(f"{label}: wall={wall_dt*1000:.3f} ms ({q4_bytes/wall_dt/1e9:.2f} Q4-GB/s), "
        f"device={dev_s}, kernels={GlobalCounters.kernel_count/iters:.1f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Q4_K x q8_1 activation GEMV candidate benchmark")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", default="blk.0.ffn_gate.weight")
  parser.add_argument("--device", default=None)
  parser.add_argument("--rows", type=int)
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--parts", type=int, default=1)
  parser.add_argument("--opt", action="append", default=None)
  parser.add_argument("--kernel", choices=("float", "intdot", "vdot", "vdot_parallel", "vdot_builtin"), default="float")
  parser.add_argument("--unpack-check-rows", type=int, default=2)
  parser.add_argument("--seed", type=int, default=1337)
  parser.add_argument("--tol", type=float, default=1e-2)
  args = parser.parse_args()

  meta = read_metadata(args.gguf)
  matches = [x for x in meta.infos if x.name == args.tensor]
  if not matches: raise ValueError(f"tensor {args.tensor!r} not found")
  info = matches[0]
  if info.typ != GGML_Q4_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q4_K")
  n, shape = prod(info.dims), tensor_shape(info)
  if len(shape) != 2: raise ValueError(f"{info.name} is not a matrix: shape={shape}")
  rows, k = min(args.rows or shape[0], shape[0]), shape[1]
  if k % Q4_K_BLOCK_ELEMS != 0: raise ValueError(f"K={k} is not Q4_K block aligned")
  byte_start = meta.data_start + info.off
  if byte_start % 4 != 0: raise ValueError(f"Q4_K tensor byte offset is not uint32 aligned: {byte_start}")
  row_bytes = k // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
  q4_bytes = rows * row_bytes
  parts = min(args.parts, k // Q4_K_BLOCK_ELEMS)
  if parts < 1: raise ValueError("--parts must be >= 1")
  opt_specs = args.opt if args.opt is not None else ([] if args.kernel in ("vdot", "vdot_builtin") else ["LOCAL:0:64"])
  opts = tuple(parse_opt(x) for x in opt_specs)
  print(f"tensor={info.name} full_shape={shape} primitive_shape=({rows},{k}) q4_bytes={q4_bytes} "
        f"mode=q8_1_{args.kernel}_partial parts={parts} opts={[str(x) for x in opts]} device={args.device or 'default'}")

  raw = Tensor(args.gguf)
  raw_words = Tensor(args.gguf, dtype=dtypes.uint32)
  words = raw_words[byte_start//4:byte_start//4+q4_bytes//4].to(args.device).contiguous().realize()
  Tensor.manual_seed(args.seed)
  x = Tensor.randn(k, dtype=dtypes.float16, device=args.device).realize()
  partials = Tensor.empty(rows, parts, dtype=dtypes.float32, device=args.device)

  raw_u8 = raw[byte_start:byte_start+q4_bytes].to(args.device).contiguous().realize()
  decoded = ggml_data_to_tensor(raw_u8, rows*k, info.typ).reshape(rows, k).cast(dtypes.float16).realize()
  q_ref, scale_ref = q8_1_quantize(x.cast(dtypes.float32))
  x_deq = q8_1_dequantize(q_ref, scale_ref).reshape(k).realize()
  activation_max_abs = (x.cast(dtypes.float32) - x_deq).abs().max().item()
  print(f"q8_1_pack_correctness: q_blocks={scale_ref.shape[0]} activation_max_abs={activation_max_abs:.6g}")
  ref = (decoded.cast(dtypes.float32) * x_deq.reshape(1, k).cast(dtypes.float32)).sum(axis=1).realize()

  unpack_rows = min(args.unpack_check_rows, rows)
  if unpack_rows > 0:
    unpack_words = raw_words[byte_start//4:byte_start//4+(unpack_rows*row_bytes)//4].to(args.device).contiguous().realize()
    unpack_out = Tensor.empty(unpack_rows, k, dtype=dtypes.float32, device=args.device)
    unpack_got = unpack_out.custom_kernel(unpack_words, fxn=q4k_unpack_kernel(unpack_rows, k))[0].realize()
    unpack_ref = q4_k_reference(raw[byte_start:byte_start+unpack_rows*row_bytes].to(args.device), unpack_rows*k).reshape(unpack_rows, k).realize()
    unpack_max_abs = (unpack_got - unpack_ref).abs().max().item()
    print(f"unpack_correctness: rows={unpack_rows} max_abs={unpack_max_abs:.6g}")
    if unpack_max_abs != 0:
      raise AssertionError("Q4_K unpack primitive correctness failed")

  @TinyJit
  def candidate():
    q, scales = q8_1_quantize(x.cast(dtypes.float32))
    if args.kernel in ("vdot", "vdot_parallel", "vdot_builtin"):
      if opts and args.kernel in ("vdot", "vdot_builtin"):
        raise ValueError("q8_1 vdot/vdot_builtin candidate is fixed; --opt is not supported")
      q_bias_words = Tensor.empty(k//4, dtype=dtypes.uint32, device=args.device).custom_kernel(q, fxn=q8_1_bias_pack_u32_kernel(k))[0]
      kernel = {"vdot": q4k_q8_1_vdot_partial_kernel, "vdot_builtin": q4k_q8_1_vdot_builtin_partial_kernel,
                "vdot_parallel": q4k_q8_1_vdot_parallel_partial_kernel}[args.kernel]
      partial = partials.custom_kernel(words, q_bias_words, scales,
                                       fxn=kernel(rows, k, parts, "none", () if args.kernel != "vdot_parallel" else opts))[0]
    else:
      kernel = q4k_q8_1_intdot_partial_kernel if args.kernel == "intdot" else q4k_q8_1_gemv_partial_kernel
      partial = partials.custom_kernel(words, q, scales, fxn=kernel(rows, k, parts, "none", opts))[0]
    return partial.sum(axis=1)

  got = candidate().realize()
  max_abs = (got - ref).abs().max().item()
  print(f"correctness: max_abs={max_abs:.6g}")
  if max_abs > args.tol:
    print("got", got.numpy())
    print("ref", ref.numpy())
    raise AssertionError("Q4_K x q8_1 GEMV primitive correctness failed")
  bench({"float": "q4k_q8_1_gemv_partial", "intdot": "q4k_q8_1_intdot_partial", "vdot": "q4k_q8_1_vdot_partial",
         "vdot_parallel": "q4k_q8_1_vdot_parallel_partial", "vdot_builtin": "q4k_q8_1_vdot_builtin_partial"}[args.kernel],
        args.iters, q4_bytes, candidate)
