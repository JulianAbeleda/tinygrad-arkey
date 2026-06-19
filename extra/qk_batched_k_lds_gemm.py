#!/usr/bin/env python3
"""Arc A Phase 1 — Q4_K batched-K weight-reuse GEMM probe (register-blocked K reuse).

Question (the first decisive experiment): can one kernel compute T=K+1 outputs while loading/dequanting each Q4_K
weight block ONCE, reusing it across the T activation columns? The current q4k_gemm_kernel makes the column axis
`bb` a plain range + UPCAST:1:K opt, but measures ~linear in T (no reuse). This probe tests forcing `bb` to an
AxisType.UNROLL axis (and a Python-unrolled variant) so the bb-independent dequantted weight UOp is shared across
the T columns at the graph level.

fp-exact (no q8, no lossiness): this is a scheduling/reuse change only. Oracle = the current q4k_gemm_kernel and a
plain fp matmul of the dequantized weight. Reports correctness (max_abs vs ref), device time (DEBUG=2 time_sum_s),
and the per-kernel weight-load/dequant count from the emitted source.

  DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_batched_k_lds_gemm.py --tensor blk.0.ffn_gate.weight --T 5
"""
from __future__ import annotations
import argparse, io, contextlib, pathlib, statistics, sys
from math import prod
from tinygrad import Tensor, dtypes, Context, Device
from tinygrad.helpers import GlobalCounters, cdiv
from tinygrad.uop.ops import AxisType, UOp
from extra.qk_layout import (Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q4_K_BLOCK_BYTES, ggml_data_to_tensor,
                             pick_tensor, read_metadata, tensor_shape, q4_k_weight_bytes)
from extra.q4_k_gemv_primitive import q4k_gemm_kernel, _q4k_weight, _kernel_info, parse_opt

def q4k_gemm_reuse_kernel(rows:int, k:int, T:int, parts:int, opts:tuple):
  """T columns Python-UNROLLED with the dequantted weight computed ONCE per (blk,pos) and reused across all T
  columns (T separate accumulators, one shared weight UOp). The reuse is explicit at the graph level (not left to
  the UPCAST opt + CSE). fp-exact."""
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)
  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    part = UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 3, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    ws = [_q4k_weight(words, base, grp, pos) for grp in range(8)]   # dequant ONCE per (blk,pos), shared over T
    stores = []
    for t in range(T):
      contrib = UOp.const(dtypes.float32, 0.0)
      for grp in range(8):
        contrib = contrib + ws[grp] * x[t * k + blk * Q4_K_BLOCK_ELEMS + grp * 32 + pos].cast(dtypes.float32)
      contrib = in_range.where(contrib, UOp.const(dtypes.float32, 0.0))
      acc = partials[row, t, part].set(0.0)
      acc = partials[row, t, part].set(acc.after(blk_part, pos)[row, t, part] + contrib, end=pos)
      stores.append(acc)
    return UOp.group(*stores).end(row, part, blk_part).sink(arg=_kernel_info(f"q4k_gemm_reuse_{rows}_{k}_{T}_{parts}", "none", opts))
  return kernel

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("gguf", nargs="?", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--tensor", default="blk.0.ffn_gate.weight")
  ap.add_argument("--T", type=int, default=5)
  ap.add_argument("--device", default="AMD")
  ap.add_argument("--iters", type=int, default=60)
  args = ap.parse_args()
  T = args.T
  args.gguf = pathlib.Path(args.gguf)
  meta = read_metadata(args.gguf)
  info = pick_tensor(meta.infos, args.tensor)
  n, shape, q4_bytes = prod(info.dims), tensor_shape(info), q4_k_weight_bytes(info)
  rows, k = shape
  byte_start = meta.data_start + info.off
  raw_words = Tensor(args.gguf, dtype=dtypes.uint32)
  words = raw_words[byte_start//4:byte_start//4+q4_bytes//4].to(args.device).contiguous().realize()
  raw = Tensor(args.gguf)
  decoded = ggml_data_to_tensor(raw[byte_start:byte_start+q4_bytes].to(args.device).contiguous().realize(), n, info.typ
                                ).reshape(*shape).cast(dtypes.float16).contiguous().realize()
  Tensor.manual_seed(7)
  LOCAL = (parse_opt("LOCAL:0:64"),)                              # the model's ffn_gate opt (self.opts)
  xT = Tensor.randn((T, k), device=args.device, dtype=dtypes.float16).contiguous().realize()
  x1 = xT[:1].contiguous().realize()
  refT = xT.matmul(decoded.transpose()).realize()                 # [T, rows]
  ref1 = x1.matmul(decoded.transpose()).realize()                 # [1, rows]

  def run_gemm(fxn, xflat, t):                                    # current q4k_gemm_kernel path: partials[rows,t,1]
    partials = Tensor.empty(rows, t, 1, dtype=dtypes.float32, device=args.device)
    return partials.custom_kernel(words, xflat, fxn=fxn)[0].sum(axis=2).transpose(0, 1)
  def time_ms(thunk, iters):
    ts = []
    for _ in range(iters):
      GlobalCounters.reset()
      with contextlib.redirect_stdout(io.StringIO()), Context(DEBUG=2): thunk().realize()
      ts.append(GlobalCounters.time_sum_s)
    return statistics.median(ts) * 1e3

  # T==1 anchor (same kernel family, LOCAL:0:64) -> the "one pass" denominator for this role
  base1 = q4k_gemm_kernel(rows, k, 1, 1, "none", LOCAL)
  one_ms = time_ms(lambda: run_gemm(base1, x1.reshape(k), 1), args.iters)
  print(f"role {args.tensor} rows={rows} k={k} T={T} | LOCAL:0:64 | one T==1 pass = {one_ms:.3f}ms")

  variants = {
    "baseline_gemm(upcast)": lambda: (q4k_gemm_kernel(rows, k, T, 1, "none", LOCAL + (parse_opt(f"UPCAST:1:{min(T,16)}"),)),
                                      run_gemm, xT.reshape(T*k).contiguous().realize()),
    "reuse_unroll(shared_w)": lambda: (q4k_gemm_reuse_kernel(rows, k, T, 1, LOCAL),
                                      run_gemm, xT.reshape(T*k).contiguous().realize()),
  }
  for name, mk in variants.items():
    try:
      fxn, runner, xflat = mk()
      got = runner(fxn, xflat, T).realize()
      max_abs = (got - refT).abs().max().item()
      arg_ok = bool((got.argmax(-1) == refT.argmax(-1)).all().item())
      dev_ms = time_ms(lambda: runner(fxn, xflat, T), args.iters)
      print(f"  {name:24}: dev {dev_ms:7.3f}ms = {dev_ms/one_ms:.2f}x one pass | max_abs {max_abs:.4g} | argmax_exact {arg_ok}")
    except Exception as e:
      print(f"  {name:24}: FAILED {type(e).__name__}: {str(e)[:200]}")

if __name__ == "__main__":
  main()
