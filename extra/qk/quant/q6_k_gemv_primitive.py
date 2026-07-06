#!/usr/bin/env python3
from __future__ import annotations

from tinygrad import dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import cdiv
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from extra.qk.layout import Q6K_HALFWORDS_PER_BLOCK, Q6_K_BLOCK_ELEMS

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

def _q6k_block_dot_packed_load_gemm(halfs:UOp, x:UOp, base:UOp, x_block:UOp, lane2:UOp, bb:UOp, k:int) -> UOp:
  # GEMM body with adjacent Q6 payload bytes consumed from one uint16 halfword load. The standard body reduces over
  # pos=0..15 and extracts one byte at a time; here lane2=0..7 handles pos=(2*lane2, 2*lane2+1).
  contrib = UOp.const(dtypes.float32, 0.0)
  d = _f16_half(halfs[base + 104])
  for grp in range(16):
    half = grp // 8
    pgrp = grp % 8
    ql_byte_idx = half*64 + (pgrp%4)*16 + lane2*2
    ql_shift = 4 if pgrp >= 4 else 0
    qh_byte_idx = 128 + half*32 + (pgrp%2)*16 + lane2*2
    qh_shift = (pgrp//2) * 2
    ql_word = halfs[base + ql_byte_idx//2]
    qh_word = halfs[base + qh_byte_idx//2]
    scale = _i8(_q6k_byte(halfs, base, 192 + grp))
    for p in range(2):
      pos = lane2*2 + p
      ql = ql_word.rshift(p*8 + ql_shift).bitwise_and(0xf)
      qh = qh_word.rshift(p*8 + qh_shift).bitwise_and(0x3).lshift(4)
      q = ql.bitwise_or(qh).cast(dtypes.float32) - UOp.const(dtypes.float32, 32.0)
      w = d * q * scale
      contrib = contrib + w * x[bb*k + x_block*Q6_K_BLOCK_ELEMS + grp*16 + pos].cast(dtypes.float32)
  return contrib

def q6k_gemm_kernel(rows:int, k:int, b:int, parts:int, opts:tuple[Opt, ...], name:str="q6k_gemm"):
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
    return acc.end(row, bb, part, blk_part).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}_{parts}", opts))

  return kernel

def q6k_gemm_packed_load_kernel(rows:int, k:int, b:int, parts:int, opts:tuple[Opt, ...], name:str="q6k_gemm_packed_load"):
  k_blocks = k // Q6_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, halfs:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(b, 1)
    part = UOp.range(parts, 2)
    blk_part = UOp.range(blocks_per_part, 3, axis_type=AxisType.REDUCE)
    lane2 = UOp.range(8, 4, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = in_range.where(_q6k_block_dot_packed_load_gemm(halfs, x, base, blk, lane2, bb, k), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, bb, part].set(0.0)
    acc = partials[row, bb, part].set(acc.after(blk_part, lane2)[row, bb, part] + contrib, end=lane2)
    return acc.end(row, bb, part, blk_part).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}_{parts}", opts))

  return kernel

def q6k_gemm_packed_load_direct_out_kernel(rows:int, k:int, b:int, opts:tuple[Opt, ...],
                                           name:str="q6k_gemm_packed_load_direct_out"):
  k_blocks = k // Q6_K_BLOCK_ELEMS

  def kernel(out:UOp, halfs:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(b, 1)
    blk = UOp.range(k_blocks, 2, axis_type=AxisType.REDUCE)
    lane2 = UOp.range(8, 3, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = _q6k_block_dot_packed_load_gemm(halfs, x, base, blk, lane2, bb, k)

    acc = out[bb, row].set(0.0)
    acc = out[bb, row].set(acc.after(blk, lane2)[bb, row] + contrib, end=lane2)
    return acc.end(row, bb, blk).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}_1", opts))

  return kernel

def _kernel_info(name:str, opts:tuple[Opt, ...]) -> KernelInfo:
  return KernelInfo(name=name, opts_to_apply=opts)
