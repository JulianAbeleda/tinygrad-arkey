#!/usr/bin/env python3
from tinygrad import dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import cdiv
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS

def _f16_word(word:UOp, high:bool) -> UOp:
  bits = (word.rshift(16) if high else word).bitwise_and(0xffff)
  return bits.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)

def _q4k_group_params(words:UOp, base:UOp, grp:int) -> tuple[UOp, UOp, UOp, UOp]:
  scale_base = base + 1

  w0 = words[base]
  d, dmin = _f16_word(w0, False), _f16_word(w0, True)

  def scale_byte(idx:int) -> UOp:
    return words[scale_base + idx//4].rshift((idx%4)*8).bitwise_and(0xff)

  if grp < 4:
    sc = scale_byte(grp).bitwise_and(63)
    mn = scale_byte(4+grp).bitwise_and(63)
  else:
    high = scale_byte(8+grp-4)
    sc = high.bitwise_and(0xf).bitwise_or(scale_byte(grp-4).rshift(6).lshift(4))
    mn = high.rshift(4).bitwise_or(scale_byte(4+grp-4).rshift(6).lshift(4))
  return d, dmin, sc, mn

def _q4k_quant(words:UOp, base:UOp, grp:int, pos:UOp) -> UOp:
  qword = words[base + 4 + (grp//2)*8 + pos//4]
  return qword.rshift((pos%4)*8 + (grp%2)*4).bitwise_and(0xf)

def _q4k_group_qpack_lane4(words:UOp, base:UOp, grp:int, lane4:UOp) -> UOp:
  qword = words[base + 4 + (grp//2)*8 + lane4]
  return qword.rshift((grp%2)*4).bitwise_and(0x0F0F0F0F)

def _q4k_weight(words:UOp, base:UOp, grp:int, pos:UOp) -> UOp:
  d, dmin, sc, mn = _q4k_group_params(words, base, grp)
  q = _q4k_quant(words, base, grp, pos)
  return d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)

def w_f16(words:UOp, n, k:int, k_blocks:int) -> UOp:
  # 14B decode adapter (scope 14B task T1): scalar Q4_K weight decode -> fp16 for a single (n, k) element.
  # `n` may be a Python int or a UOp range (it only feeds the words base offset); `k` must be a Python int
  # because grp selects a Python branch inside _q4k_group_params. Reuses the verbatim Q4_K unpack primitives.
  blk = k // Q4_K_BLOCK_ELEMS
  grp = (k % Q4_K_BLOCK_ELEMS) // 32
  pos = k % 32
  base = (n * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
  return _q4k_weight(words, base, grp, pos).cast(dtypes.float16)

def _q4k_group_dot_packed_load(words:UOp, x:UOp, base:UOp, x_block:UOp, grp:int, lane4:UOp) -> UOp:
  d, dmin, sc, mn = _q4k_group_params(words, base, grp)
  qpack = _q4k_group_qpack_lane4(words, base, grp, lane4)
  contrib = UOp.const(dtypes.float32, 0.0)
  for nib in range(4):
    pos = lane4 * 4 + nib
    q = qpack.rshift(nib*8).bitwise_and(0xf)
    weight = d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)
    contrib = contrib + weight * x[x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos].cast(dtypes.float32)
  return contrib

def _q4k_block_dot_packed_load(words:UOp, x:UOp, base:UOp, x_block:UOp, lane4:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8):
    contrib = contrib + _q4k_group_dot_packed_load(words, x, base, x_block, grp, lane4)
  return contrib

def parse_opt(spec:str) -> Opt:
  parts = spec.split(":")
  if len(parts) == 1:
    return Opt(OptOps[parts[0].upper()])
  if len(parts) != 3:
    raise ValueError(f"opt must be OP or OP:AXIS:ARG, got {spec!r}")
  op, axis, arg = parts
  return Opt(OptOps[op.upper()], int(axis), int(arg))

def _kernel_info(name:str, schedule:str, opts:tuple[Opt, ...]) -> KernelInfo:
  if opts: return KernelInfo(name=name, opts_to_apply=opts)
  if schedule == "auto": return KernelInfo(name=name)
  return KernelInfo(name=name, opts_to_apply=())

def _q4k_block_dot_gemm(words:UOp, x:UOp, base:UOp, x_block:UOp, pos:UOp, bb:UOp, k:int) -> UOp:
  # GEMM body on the raw-words layout (same storage as the decode GEMV primitive); weight reused across bb.
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8):
    w = _q4k_weight(words, base, grp, pos)
    contrib = contrib + w * x[bb*k + x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos].cast(dtypes.float32)
  return contrib

def q4k_gemm_kernel(rows:int, k:int, b:int, parts:int, schedule:str, opts:tuple[Opt, ...], name:str="q4k_gemm"):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(b, 1)
    part = UOp.range(parts, 2)
    blk_part = UOp.range(blocks_per_part, 3, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 4, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot_gemm(words, x, base, blk, pos, bb, k), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, bb, part].set(0.0)
    acc = partials[row, bb, part].set(acc.after(blk_part, pos)[row, bb, part] + contrib, end=pos)
    return acc.end(row, bb, part, blk_part).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}_{parts}", schedule, opts))

  return kernel

def _q4k_block_dot_packed_load_gemm(words:UOp, x:UOp, base:UOp, x_block:UOp, lane4:UOp, bb:UOp, k:int) -> UOp:
  # GEMM body: x is flattened [B*K]; each dequantized weight is reused across the B columns.
  # If bb is UPCAST'd, tinygrad unrolls it and CSEs the weight, so the dequant runs once per weight.
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8):
    d, dmin, sc, mn = _q4k_group_params(words, base, grp)
    qword = words[base + 4 + (grp//2)*8 + lane4]
    for nib in range(4):
      pos = lane4 * 4 + nib
      q = qword.rshift(nib*8 + (grp%2)*4).bitwise_and(0xf).cast(dtypes.float32)
      weight = d * sc.cast(dtypes.float32) * q - dmin * mn.cast(dtypes.float32)
      contrib = contrib + weight * x[bb*k + x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos].cast(dtypes.float32)
  return contrib

def q4k_gemm_packed_load_kernel(rows:int, k:int, b:int, parts:int, schedule:str, opts:tuple[Opt, ...], name:str="q4k_gemm_packed_load"):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(b, 1)
    part = UOp.range(parts, 2)
    blk_part = UOp.range(blocks_per_part, 3, axis_type=AxisType.REDUCE)
    lane4 = UOp.range(8, 4, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot_packed_load_gemm(words, x, base, blk, lane4, bb, k), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, bb, part].set(0.0)
    acc = partials[row, bb, part].set(acc.after(blk_part, lane4)[row, bb, part] + contrib, end=lane4)
    return acc.end(row, bb, part, blk_part).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}_{parts}", schedule, opts))

  return kernel

def q4k_gemm_packed_load_direct_out_kernel(rows:int, k:int, b:int, schedule:str, opts:tuple[Opt, ...],
                                           name:str="q4k_gemm_packed_load_direct_out"):
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(b, 1)
    blk = UOp.range(k_blocks, 2, axis_type=AxisType.REDUCE)
    lane4 = UOp.range(8, 3, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load_gemm(words, x, base, blk, lane4, bb, k)

    acc = out[bb, row].set(0.0)
    acc = out[bb, row].set(acc.after(blk, lane4)[bb, row] + contrib, end=lane4)
    return acc.end(row, bb, blk).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}_1", schedule, opts))

  return kernel

def q4k_gemm_packed_load_reduce_out_kernel(rows:int, k:int, b:int, schedule:str, opts:tuple[Opt, ...],
                                           name:str="q4k_gemm_packed_load_reduce_out"):
  # Experimental direct-output variant using a real Ops.REDUCE instead of a manual store recurrence. GROUP/GROUPTOP
  # lowering only combines Ops.REDUCE correctly; the manual direct-out accumulator is fast with GROUP but wrong because
  # the GROUP_REDUCE local axis is masked at the global store instead of summed.
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(b, 1)
    blk = UOp.range(k_blocks, 2, axis_type=AxisType.REDUCE)
    lane4 = UOp.range(8, 3, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load_gemm(words, x, base, blk, lane4, bb, k)
    return out[bb, row].store(contrib.reduce(blk, lane4, arg=Ops.ADD)).end(row, bb).sink(
      arg=_kernel_info(f"{name}_{rows}_{k}_{b}_1", schedule, opts))

  return kernel
