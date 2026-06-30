#!/usr/bin/env python3
from __future__ import annotations

import argparse, pathlib, time
from math import prod

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import GlobalCounters, cdiv
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from extra.qk_layout import (
  GGML_Q6_K, Q6K_HALFWORDS_PER_BLOCK, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_ELEMS, q6_k_reference, read_metadata, tensor_shape,
)

def parse_opt(spec:str) -> Opt:
  parts = spec.split(":")
  if len(parts) == 1:
    return Opt(OptOps[parts[0].upper()])
  if len(parts) != 3:
    raise ValueError(f"opt must be OP or OP:AXIS:ARG, got {spec!r}")
  op, axis, arg = parts
  return Opt(OptOps[op.upper()], int(axis), int(arg))

def _f16_half(half:UOp) -> UOp:
  return half.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)

def _q6k_byte(halfs:UOp, base:UOp, byte_idx:UOp|int) -> UOp:
  idx = UOp.const(dtypes.int32, byte_idx) if isinstance(byte_idx, int) else byte_idx
  return halfs[base + idx//2].rshift((idx%2)*8).bitwise_and(0xff)

def _i8(byte:UOp) -> UOp:
  return byte.cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.float32)

def _q6k_weight(halfs:UOp, base:UOp, grp:int, pos:UOp) -> UOp:
  half = grp // 8
  pgrp = grp % 8
  ql_byte_idx = half*64 + (pgrp%4)*16 + pos
  ql_shift = 4 if pgrp >= 4 else 0
  qh_byte_idx = 128 + half*32 + (pgrp%2)*16 + pos
  qh_shift = (pgrp//2) * 2
  ql = _q6k_byte(halfs, base, ql_byte_idx).rshift(ql_shift).bitwise_and(0xf)
  qh = _q6k_byte(halfs, base, qh_byte_idx).rshift(qh_shift).bitwise_and(0x3).lshift(4)
  q = ql.bitwise_or(qh).cast(dtypes.float32) - UOp.const(dtypes.float32, 32.0)
  scale = _i8(_q6k_byte(halfs, base, 192 + grp))
  d = _f16_half(halfs[base + 104])
  return d * q * scale

def _q6k_block_dot(halfs:UOp, x:UOp, base:UOp, x_block:UOp, pos:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(16):
    contrib = contrib + _q6k_weight(halfs, base, grp, pos) * x[x_block*Q6_K_BLOCK_ELEMS + grp*16 + pos].cast(dtypes.float32)
  return contrib

def _q6k_block_dot_gemm(halfs:UOp, x:UOp, base:UOp, x_block:UOp, pos:UOp, bb:UOp, k:int) -> UOp:
  # GEMM body: x is flattened [B*k]; each dequantized weight is reused across the B columns.
  # If bb is UPCAST'd, tinygrad unrolls it and CSEs the weight, so the Q6_K dequant runs once per weight.
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(16):
    w = _q6k_weight(halfs, base, grp, pos)
    contrib = contrib + w * x[bb*k + x_block*Q6_K_BLOCK_ELEMS + grp*16 + pos].cast(dtypes.float32)
  return contrib

def q6k_gemm_kernel(rows:int, k:int, b:int, parts:int, opts:tuple[Opt, ...]):
  k_blocks = k // Q6_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, halfs:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(b, 1)
    part = UOp.range(parts, 2)
    blk_part = UOp.range(blocks_per_part, 3, axis_type=AxisType.REDUCE)
    pos = UOp.range(16, 4, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = in_range.where(_q6k_block_dot_gemm(halfs, x, base, blk, pos, bb, k), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, bb, part].set(0.0)
    acc = partials[row, bb, part].set(acc.after(blk_part, pos)[row, bb, part] + contrib, end=pos)
    return acc.end(row, bb, part, blk_part).sink(arg=_kernel_info(f"q6k_gemm_{rows}_{k}_{b}_{parts}", opts))

  return kernel

def _kernel_info(name:str, opts:tuple[Opt, ...]) -> KernelInfo:
  return KernelInfo(name=name, opts_to_apply=opts)

def q6k_gemv_warp_kernel(rows:int, k:int):
  # FFN-down Q6_K WORK-DECOMPOSITION variant (lossless FP, same lever as q4k_gemv_warp): 32 threads/row = one
  # gfx1100 wave; lane = block_group(0..1)*16 + pos(0..15). pos = within-block byte index -> 16 adjacent lanes read
  # adjacent ql/qh bytes (coalesced, the coop insight); block_group splits the k_blocks into 2 K-parallel halves.
  # Each lane FP-accumulates its blocks (REG), then IN-KERNEL warp_reduce_sum (ds_bpermute) -> out[row] (one store,
  # no stage-2 partials buffer). Decode/math identical to the default (exact up to fp reassoc). k_blocks % 2 == 0.
  from tinygrad.dtype import AddrSpace
  from extra.amd_warp_reduce import warp_reduce_sum
  k_blocks = k // Q6_K_BLOCK_ELEMS
  if k_blocks % 2 != 0: raise ValueError(f"k_blocks={k_blocks} must be divisible by 2 for the 2-block_group Q6_K warp")
  bpb = k_blocks // 2

  def kernel(out:UOp, halfs:UOp, x:UOp) -> UOp:
    row = UOp.special(rows, "gidx0")
    lane = UOp.special(32, "lidx0")
    bg = lane // 16                                # block_group 0..1 (K-parallel)
    pos = lane % 16                                # within-block byte index 0..15 (coalesced)
    lblk = UOp.range(bpb, 0, axis_type=AxisType.REDUCE)
    blk = bg * bpb + lblk
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = _q6k_block_dot(halfs, x, base, blk, pos)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = warp_reduce_sum(acc[0], lane, 32)
    return out[row].store(total).sink(arg=_kernel_info(f"q6k_gemv_warp_{rows}_{k}", ()))

  return kernel

def q6k_halfwarp_partition_kernel(rows:int, k:int):
  # Q6K-2 HALF-WARP 2-ROW PARTITION (distinct from q6k_gemv_warp_kernel, which packs 2 K-groups of ONE row into a full
  # 32-lane warp). Here TWO INDEPENDENT rows share one 32-lane wave as two 16-lane partitions:
  #   half = lane // 16   (0 -> row A, 1 -> row B);   pos = lane % 16  (within-block byte index 0..15, coalesced)
  # Each lane FP-accumulates ITS row over ALL k_blocks (pos is the only within-row lane axis, no K-split), then a
  # HALF-WARP reduce -- warp_reduce_sum(acc, lane, width=16) with the FULL 32-lane `lane`: the xor ladder {8,4,2,1}
  # never crosses the 16-boundary (lane^8 keeps 0..15 in 0..15 and 16..31 in 16..31), so each half independently sums
  # its 16 pos lanes. Lane stores out[row] (rows A/B independent). NO partials buffer, NO external r_* reduce.
  # Decode/dequant byte-identical to the default (_q6k_weight verbatim; exact up to fp reassoc). rows must be even.
  from tinygrad.dtype import AddrSpace
  from extra.amd_warp_reduce import warp_reduce_sum
  if rows % 2 != 0: raise ValueError(f"rows={rows} must be even (2 rows per half-warp)")
  k_blocks = k // Q6_K_BLOCK_ELEMS

  def kernel(out:UOp, halfs:UOp, x:UOp) -> UOp:
    row_pair = UOp.special(rows // 2, "gidx0")    # one warp per row-pair
    lane = UOp.special(32, "lidx0")
    half = lane // 16                              # 0 -> row A, 1 -> row B
    pos = lane % 16                                # within-block pos 0..15 (coalesced)
    row = row_pair * 2 + half
    lblk = UOp.range(k_blocks, 0, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + lblk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = _q6k_block_dot(halfs, x, base, lblk, pos)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = warp_reduce_sum(acc[0], lane, 16)      # HALF-warp: each 16-lane half holds its own row's sum
    return out[row].store(total).sink(arg=_kernel_info(f"q6k_halfwarp_partition_{rows}_{k}", ()))

  return kernel

def q6k_gemv_partial_kernel(rows:int, k:int, parts:int, opts:tuple[Opt, ...]):
  k_blocks = k // Q6_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, halfs:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    part = UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    pos = UOp.range(16, 3, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = in_range.where(_q6k_block_dot(halfs, x, base, blk, pos), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part, pos)[row, part] + contrib, end=pos)
    return acc.end(row, part, blk_part).sink(arg=_kernel_info(f"q6k_gemv_partial_{rows}_{k}_{parts}", opts))

  return kernel


def q6k_coop_partial_kernel(rows:int, k:int, row_tile:int=4):
  # Cooperative-K Q6_K GEMV: the within-block position `pos` (0..15) becomes a LOCAL lane axis. In _q6k_weight,
  # ql/qh byte indices are `...+pos`, so adjacent lanes read ADJACENT bytes -> coalesced packed-weight loads (the
  # current q6k_gemv_partial maps one row per thread -> adjacent lanes read whole rows apart -> ~10% HBM peak).
  # Each lane writes its OWN partial partials[row, pos] (no cross-lane reduce in-kernel, like gqa_coop_vec); the
  # final reduction over the 16 pos-lanes is stage-2 `.sum(axis=1)`. row_tile rows share a workgroup so the
  # wavefront is row_tile*16 lanes wide (occupancy). Output: partials[rows, 16].
  k_blocks = k // Q6_K_BLOCK_ELEMS

  def kernel(partials:UOp, halfs:UOp, x:UOp) -> UOp:
    row_o = UOp.range(cdiv(rows, row_tile), 0)
    row_i = UOp.range(row_tile, 1, axis_type=AxisType.LOCAL)
    pos = UOp.range(16, 2, axis_type=AxisType.LOCAL)
    blk = UOp.range(k_blocks, 3, axis_type=AxisType.REDUCE)
    row = row_o * row_tile + row_i
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = _q6k_block_dot(halfs, x, base, blk, pos)

    acc = partials[row, pos].set(0.0)
    acc = partials[row, pos].set(acc.after(blk)[row, pos] + contrib, end=blk)
    return acc.end(row_o, row_i, pos).sink(arg=_kernel_info(f"q6k_coop_partial_{rows}_{k}", ()))

  return kernel

def q6k_unpack_kernel(rows:int, k:int):
  k_blocks = k // Q6_K_BLOCK_ELEMS

  def kernel(out:UOp, halfs:UOp) -> UOp:
    row = UOp.range(rows, 0)
    blk = UOp.range(k_blocks, 1)
    pos = UOp.range(16, 2)
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    stores = []
    for grp in range(16):
      stores.append(out[row, blk*Q6_K_BLOCK_ELEMS + grp*16 + pos].store(_q6k_weight(halfs, base, grp, pos)))
    return UOp.group(*stores).end(row, blk, pos).sink(arg=_kernel_info(f"q6k_unpack_{rows}_{k}", ()))

  return kernel

def bench(label:str, iters:int, quant_bytes:int, fn) -> None:
  fn().realize()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(iters): fn().realize()
  wall_dt = (time.perf_counter() - st) / iters
  dev_dt = GlobalCounters.time_sum_s / iters
  dev_s = f"{dev_dt*1000:.3f} ms ({quant_bytes/dev_dt/1e9:.2f} quant-GB/s)" if dev_dt > 0 else "n/a"
  print(f"{label}: wall={wall_dt*1000:.3f} ms ({quant_bytes/wall_dt/1e9:.2f} quant-GB/s), "
        f"device={dev_s}, kernels={GlobalCounters.kernel_count/iters:.1f}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Correctness-first custom Q6_K GEMV primitive probe")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", default="blk.0.ffn_down.weight")
  parser.add_argument("--device", default=None)
  parser.add_argument("--rows", type=int)
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--parts", type=int, default=1)
  parser.add_argument("--opt", action="append", default=None)
  parser.add_argument("--unpack-check-rows", type=int, default=2)
  parser.add_argument("--seed", type=int, default=1337)
  args = parser.parse_args()

  meta = read_metadata(args.gguf)
  matches = [x for x in meta.infos if x.name == args.tensor]
  if not matches: raise ValueError(f"tensor {args.tensor!r} not found")
  info = matches[0]
  if info.typ != GGML_Q6_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q6_K")
  n, shape = prod(info.dims), tensor_shape(info)
  if len(shape) != 2: raise ValueError(f"{info.name} is not a matrix: shape={shape}")
  rows, k = min(args.rows or shape[0], shape[0]), shape[1]
  if k % Q6_K_BLOCK_ELEMS != 0: raise ValueError(f"K={k} is not Q6_K block aligned")
  byte_start = meta.data_start + info.off
  if byte_start % 2 != 0: raise ValueError(f"Q6_K tensor byte offset is not uint16 aligned: {byte_start}")
  row_bytes = k // Q6_K_BLOCK_ELEMS * Q6_K_BLOCK_BYTES
  quant_bytes = rows * row_bytes
  if args.parts < 1: raise ValueError("--parts must be >= 1")
  parts = min(args.parts, k // Q6_K_BLOCK_ELEMS)
  opt_specs = args.opt if args.opt is not None else ["LOCAL:0:64"]
  opts = tuple(parse_opt(x) for x in opt_specs)
  print(f"tensor={info.name} full_shape={shape} primitive_shape=({rows},{k}) quant_bytes={quant_bytes} "
        f"mode=partial parts={parts} opts={[str(x) for x in opts]} device={args.device or 'default'}")

  raw = Tensor(args.gguf)
  raw_halfs = Tensor(args.gguf, dtype=dtypes.uint16)
  halfs = raw_halfs[byte_start//2:byte_start//2+quant_bytes//2].to(args.device).contiguous().realize()
  Tensor.manual_seed(args.seed)
  x = Tensor.randn(k, dtype=dtypes.float16, device=args.device).realize()
  partials = Tensor.empty(rows, parts, dtype=dtypes.float32, device=args.device)

  raw_u8 = raw[byte_start:byte_start+quant_bytes].to(args.device).contiguous().realize()
  decoded = ggml_data_to_tensor(raw_u8, rows*k, info.typ).reshape(rows, k).cast(dtypes.float16).realize()
  ref = (decoded.cast(dtypes.float32) * x.reshape(1, k).cast(dtypes.float32)).sum(axis=1).realize()

  def fused_graph():
    return (ggml_data_to_tensor(raw_u8, rows*k, info.typ).reshape(rows, k).cast(dtypes.float16).cast(dtypes.float32) *
            x.reshape(1, k).cast(dtypes.float32)).sum(axis=1)

  unpack_rows = min(args.unpack_check_rows, rows)
  if unpack_rows > 0:
    unpack_halfs = raw_halfs[byte_start//2:byte_start//2+(unpack_rows*row_bytes)//2].to(args.device).contiguous().realize()
    unpack_out = Tensor.empty(unpack_rows, k, dtype=dtypes.float32, device=args.device)
    unpack_got = unpack_out.custom_kernel(unpack_halfs, fxn=q6k_unpack_kernel(unpack_rows, k))[0].realize()
    unpack_ref = q6_k_reference(raw[byte_start:byte_start+unpack_rows*row_bytes].to(args.device), unpack_rows*k).reshape(unpack_rows, k).realize()
    unpack_max_abs = (unpack_got - unpack_ref).abs().max().item()
    print(f"unpack_correctness: rows={unpack_rows} max_abs={unpack_max_abs:.6g}")
    if unpack_max_abs != 0:
      raise AssertionError("Q6_K unpack primitive correctness failed")

  def primitive():
    partial = partials.custom_kernel(halfs, x, fxn=q6k_gemv_partial_kernel(rows, k, parts, opts))[0]
    return partial.sum(axis=1)

  got = primitive().realize()
  max_abs = (got - ref).abs().max().item()
  print(f"correctness: max_abs={max_abs:.6g}")
  if max_abs > 1e-2:
    print("got", got.numpy())
    print("ref", ref.numpy())
    raise AssertionError("Q6_K GEMV primitive correctness failed")
  bench("q6k_fused_graph", args.iters, quant_bytes, fused_graph)
  bench("q6k_gemv_primitive_partial", args.iters, quant_bytes, primitive)
