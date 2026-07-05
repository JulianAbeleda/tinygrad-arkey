#!/usr/bin/env python3
import argparse, pathlib, time
from math import prod

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import GlobalCounters, cdiv
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp

from extra.qk.quant.q4_k_safety import assert_q4k_risky_search_allowed
from extra.qk.layout import (
  GGML_Q4_K, Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS, pick_tensor, q4_k_reference,
  read_metadata, tensor_shape,
)

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

def _q4k_weight(words:UOp, base:UOp, grp:int, pos:UOp) -> UOp:
  d, dmin, sc, mn = _q4k_group_params(words, base, grp)
  q = _q4k_quant(words, base, grp, pos)
  return d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)

def _q4k_block_dot(words:UOp, x:UOp, base:UOp, x_block:UOp, pos:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8):
    contrib = contrib + _q4k_weight(words, base, grp, pos) * x[x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos].cast(dtypes.float32)
  return contrib

def _q4k_group_dot_packed_load(words:UOp, x:UOp, base:UOp, x_block:UOp, grp:int, lane4:UOp) -> UOp:
  d, dmin, sc, mn = _q4k_group_params(words, base, grp)
  qword = words[base + 4 + (grp//2)*8 + lane4]
  contrib = UOp.const(dtypes.float32, 0.0)
  for nib in range(4):
    pos = lane4 * 4 + nib
    q = qword.rshift(nib*8 + (grp%2)*4).bitwise_and(0xf)
    weight = d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)
    contrib = contrib + weight * x[x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos].cast(dtypes.float32)
  return contrib

def _q4k_block_dot_packed_load(words:UOp, x:UOp, base:UOp, x_block:UOp, lane4:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8):
    contrib = contrib + _q4k_group_dot_packed_load(words, x, base, x_block, grp, lane4)
  return contrib

def _q4k_group_dot_vector_load(words:UOp, x:UOp, base:UOp, x_block:UOp, grp:int, lane_vec:UOp) -> UOp:
  d, dmin, sc, mn = _q4k_group_params(words, base, grp)
  qwords = words.index(base + 4 + (grp//2)*8 + lane_vec*4, ptr=True).load(dtype=dtypes.uint32.vec(4))
  dsc = d * sc.cast(dtypes.float32)
  dmn = dmin * mn.cast(dtypes.float32)
  srcs:list[UOp] = [dsc, qwords, dmn]
  lines = ["({{ float _total = 0.0f;"]
  for lane in range(4):
    lines.append(f"unsigned int _qw{lane} = (unsigned int)(({{1}})[{lane}]);")
    for nib in range(4):
      pos = (lane_vec*4 + lane)*4 + nib
      srcs.append(x[x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos].cast(dtypes.float32))
      lines.append(
        f"_total += (((float){{0}}) * (float)((_qw{lane} >> {nib*8 + (grp%2)*4}u) & 15u) - "
        f"((float){{2}})) * ((float){{{len(srcs)-1}}});"
      )
  lines.append("_total; }})")
  return UOp(Ops.CUSTOMI, dtypes.float32, tuple(srcs), arg=" ".join(lines))

def _q4k_block_dot_vector_load(words:UOp, x:UOp, base:UOp, x_block:UOp, lane_vec:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8):
    contrib = contrib + _q4k_group_dot_vector_load(words, x, base, x_block, grp, lane_vec)
  return contrib

def _q4k_block_dot_q8_1(words:UOp, xq:UOp, xscales:UOp, base:UOp, x_block:UOp, pos:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8):
    x_idx = x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos
    x = xq[x_idx].cast(dtypes.float32) * xscales[x_idx//Q8_1_BLOCK_ELEMS].cast(dtypes.float32)
    contrib = contrib + _q4k_weight(words, base, grp, pos) * x
  return contrib

def _q4k_block_dot_q8_1_gemm(words:UOp, xq:UOp, xscales:UOp, base:UOp, x_block:UOp, pos:UOp, bb:UOp, k:int) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  x_base = bb * k
  for grp in range(8):
    x_idx = x_base + x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos
    x = xq[x_idx].cast(dtypes.float32) * xscales[x_idx//Q8_1_BLOCK_ELEMS].cast(dtypes.float32)
    contrib = contrib + _q4k_weight(words, base, grp, pos) * x
  return contrib

def _q4k_group_dot_q8_1_intdot(words:UOp, xq:UOp, xscales:UOp, base:UOp, x_block:UOp, grp:int, afters:tuple[UOp, ...]) -> UOp:
  pos = UOp.range(32, 20 + grp, axis_type=AxisType.REDUCE)
  x_idx = x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos
  q4 = _q4k_quant(words, base, grp, pos).cast(dtypes.int32)
  q8 = xq[x_idx].cast(dtypes.int32)

  dot = UOp.placeholder((1,), dtypes.int32, 120 + grp, addrspace=AddrSpace.REG)
  dot = dot.after(*afters)[0].set(0)
  dot = dot[0].set(dot.after(pos)[0] + q4 * q8, end=pos)

  qsum = UOp.placeholder((1,), dtypes.int32, 140 + grp, addrspace=AddrSpace.REG)
  qsum = qsum.after(*afters)[0].set(0)
  qsum = qsum[0].set(qsum.after(pos)[0] + q8, end=pos)

  d, dmin, sc, mn = _q4k_group_params(words, base, grp)
  xscale = xscales[x_block*8 + grp].cast(dtypes.float32)
  return xscale * (d * sc.cast(dtypes.float32) * dot[0].cast(dtypes.float32) -
                   dmin * mn.cast(dtypes.float32) * qsum[0].cast(dtypes.float32))

def _q4k_block_dot_q8_1_intdot(words:UOp, xq:UOp, xscales:UOp, base:UOp, x_block:UOp, afters:tuple[UOp, ...]) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8):
    contrib = contrib + _q4k_group_dot_q8_1_intdot(words, xq, xscales, base, x_block, grp, afters)
  return contrib

def _vdot4_q4_q8_accum(acc:UOp, q8_bias:UOp, q4:UOp, d:UOp, dmin:UOp, sc:UOp, mn:UOp, xscale:UOp) -> UOp:
  return UOp(Ops.CUSTOMI, dtypes.float32, (acc, q8_bias, q4, d, dmin, sc, mn, xscale),
             arg='({{ float _acc = {0}; unsigned int _q8 = (unsigned int)({1}); unsigned int _q4 = (unsigned int)({2}); '
                 'unsigned int _dot = 0u; '
                 'asm volatile("v_dot4_u32_u8 %0, %1, %2, %0" : "+v"(_dot) : "v"(_q8), "v"(_q4)); '
                 'int _q4sum = (int)(_q4 & 255u) + (int)((_q4 >> 8) & 255u) + '
                 '(int)((_q4 >> 16) & 255u) + (int)((_q4 >> 24) & 255u); '
                 'int _q8sum = (int)(_q8 & 255u) + (int)((_q8 >> 8) & 255u) + '
                 '(int)((_q8 >> 16) & 255u) + (int)((_q8 >> 24) & 255u); '
                 'int _dot_signed = (int)_dot - _q4sum * 128; int _q8_signed_sum = _q8sum - 512; '
                 '_acc + ((float)({7})) * (((float)({3})) * ((float)({5})) * ((float)_dot_signed) - '
                 '((float)({4})) * ((float)({6})) * ((float)_q8_signed_sum)); }})')

def _q4k_group_dot_q8_1_vdot_parallel(words:UOp, xq_bias_words:UOp, xscales:UOp, base:UOp, x_block:UOp, grp:int,
                                      afters:tuple[UOp, ...]) -> UOp:
  lane4 = UOp.range(8, 160 + grp, axis_type=AxisType.REDUCE)
  qword = words[base + 4 + (grp//2)*8 + lane4]
  q4 = qword.rshift(4 if grp % 2 else 0).bitwise_and(0x0f0f0f0f)
  q8_bias = xq_bias_words[x_block*64 + grp*8 + lane4]

  d, dmin, sc, mn = _q4k_group_params(words, base, grp)
  xscale = xscales[x_block*8 + grp].cast(dtypes.float32)
  acc = UOp.placeholder((1,), dtypes.float32, 180 + grp, addrspace=AddrSpace.REG)
  acc = acc.after(*afters)[0].set(0.0)
  acc = acc[0].set(_vdot4_q4_q8_accum(acc.after(lane4)[0], q8_bias, q4, d, dmin, sc, mn, xscale), end=lane4)
  return acc[0]

def _q4k_block_dot_q8_1_vdot_parallel(words:UOp, xq_bias_words:UOp, xscales:UOp, base:UOp, x_block:UOp,
                                      afters:tuple[UOp, ...]) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8):
    contrib = contrib + _q4k_group_dot_q8_1_vdot_parallel(words, xq_bias_words, xscales, base, x_block, grp, afters)
  return contrib

def _u8_sum_expr(word:str) -> str:
  return " + ".join(f"((int)(({word} >> {shift}) & 255u))" for shift in (0, 8, 16, 24))

def _q4k_scale_byte_expr(base:str, idx:int) -> str:
  return f"(({base}[base+{1 + idx//4}] >> {8*(idx%4)}) & 255u)"

def _q4k_scale_min_expr(base:str, grp:int) -> tuple[str, str]:
  if grp < 4:
    return f"({_q4k_scale_byte_expr(base, grp)} & 63u)", f"({_q4k_scale_byte_expr(base, 4+grp)} & 63u)"
  high = _q4k_scale_byte_expr(base, 8+grp-4)
  return (f"(({high} & 15u) | (({_q4k_scale_byte_expr(base, grp-4)} >> 6u) << 4u))",
          f"(({high} >> 4u) | (({_q4k_scale_byte_expr(base, 4+grp-4)} >> 6u) << 4u))")

def _q4k_q8_1_vdot_source(k_blocks:int, parts:int, builtin:bool=False) -> str:
  if parts != 1: raise ValueError("q4k_q8_1_vdot_partial_kernel currently supports parts=1 only")
  p = [f"__P{i}__" for i in range(4)]
  lines = [
    "{",
    "  float total = 0.0f;",
    f"  for (int blk = 0; blk < {k_blocks}; blk++) {{",
    f"    int base = blk * {Q4K_WORDS_PER_BLOCK};",
    f"    unsigned int fp = {p[1]}[base];",
    "    float d = (float)(__builtin_bit_cast(_Float16, (unsigned short)(fp & 65535u)));",
    "    float dmin = (float)(__builtin_bit_cast(_Float16, (unsigned short)((fp >> 16u) & 65535u)));",
  ]
  for grp in range(8):
    scale, mn = _q4k_scale_min_expr(p[1], grp)
    shift = 4 if grp % 2 else 0
    qword_base = 4 + (grp//2)*8
    lines += [
      f"    {{ // Q4_K group {grp}",
      f"      unsigned int sc = {scale};",
      f"      unsigned int mn = {mn};",
      "      unsigned int dot = 0u;",
      "      int q4sum = 0;",
      "      int q8sum = 0;",
    ]
    for lane4 in range(8):
      q4_name, q8_name = f"q4_{grp}_{lane4}", f"q8_{grp}_{lane4}"
      lines += [
        f"      unsigned int {q4_name} = (({p[1]}[base+{qword_base+lane4}] >> {shift}u) & 0x0f0f0f0fu);",
        f"      unsigned int {q8_name} = {p[2]}[blk*64+{grp*8+lane4}];",
        (f"      dot = _dp4a({q8_name}, {q4_name}, dot);" if builtin else
         f"      asm volatile(\"v_dot4_u32_u8 %0, %1, %2, %0\" : \"+v\"(dot) : \"v\"({q8_name}), \"v\"({q4_name}));"),
        f"      q4sum += {_u8_sum_expr(q4_name)};",
        f"      q8sum += {_u8_sum_expr(q8_name)};",
      ]
    lines += [
      "      int dot_signed = ((int)dot) - q4sum * 128;",
      "      int q8_signed_sum = q8sum - 4096;",
      f"      float xscale = (float){p[3]}[blk*8+{grp}];",
      "      total += xscale * (d * (float)sc * (float)dot_signed - dmin * (float)mn * (float)q8_signed_sum);",
      "    }",
    ]
  lines += [
    "  }",
    f"  {p[0]}[0] = total;",
    "}",
  ]
  src = "\n".join(lines).replace("{", "{{").replace("}", "}}")
  for i, token in enumerate(p): src = src.replace(token, f"{{{i}}}")
  return src

def _q4k_tile_custom_partial_source(k_blocks:int, parts:int) -> str:
  blocks_per_part = cdiv(k_blocks, parts)
  p = [f"__P{i}__" for i in range(4)]
  part_line = "  int part = 0;" if parts == 1 else f"  int part = (int){p[3]};"
  lines = [
    "{",
    "  typedef unsigned int tg_uint4 __attribute__((ext_vector_type(4)));",
    "  float total = 0.0f;",
    part_line,
    f"  int start = part * {blocks_per_part};",
    f"  int stop = start + {blocks_per_part};",
    f"  if (stop > {k_blocks}) stop = {k_blocks};",
    "  for (int blk = start; blk < stop; blk++) {",
    f"    int base = blk * {Q4K_WORDS_PER_BLOCK};",
    f"    unsigned int fp = {p[1]}[base];",
    "    float d = (float)(__builtin_bit_cast(_Float16, (unsigned short)(fp & 65535u)));",
    "    float dmin = (float)(__builtin_bit_cast(_Float16, (unsigned short)((fp >> 16u) & 65535u)));",
  ]
  for pair in range(4):
    even, odd = pair * 2, pair * 2 + 1
    even_scale, even_min = _q4k_scale_min_expr(p[1], even)
    odd_scale, odd_min = _q4k_scale_min_expr(p[1], odd)
    lines += [
      f"    {{ // Q4_K group pair {even}/{odd}",
      f"      unsigned int sc_even = {even_scale};",
      f"      unsigned int mn_even = {even_min};",
      f"      unsigned int sc_odd = {odd_scale};",
      f"      unsigned int mn_odd = {odd_min};",
      "      for (int lane_vec = 0; lane_vec < 2; lane_vec++) {",
      f"        tg_uint4 qv = *((tg_uint4*)({p[1]} + base + {4 + pair*8} + lane_vec * 4));",
      "        for (int lane = 0; lane < 4; lane++) {",
      "          unsigned int w = qv[lane];",
      "          int pos_base = lane_vec * 16 + lane * 4;",
      "          for (int nib = 0; nib < 4; nib++) {",
      "            unsigned int byte = (w >> (8u * (unsigned int)nib)) & 255u;",
      "            unsigned int q_even = byte & 15u;",
      "            unsigned int q_odd = byte >> 4u;",
      f"            float x_even = (float){p[2]}[blk*{Q4_K_BLOCK_ELEMS}+{even*32}+pos_base+nib];",
      f"            float x_odd = (float){p[2]}[blk*{Q4_K_BLOCK_ELEMS}+{odd*32}+pos_base+nib];",
      "            total += (d * (float)sc_even * (float)q_even - dmin * (float)mn_even) * x_even;",
      "            total += (d * (float)sc_odd * (float)q_odd - dmin * (float)mn_odd) * x_odd;",
      "          }",
      "        }",
      "      }",
      "    }",
    ]
  lines += [
    "  }",
    f"  {p[0]}[0] = total;",
    "}",
  ]
  src = "\n".join(lines).replace("{", "{{").replace("}", "}}")
  for i, token in enumerate(p): src = src.replace(token, f"{{{i}}}")
  return src

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

def q4k_gemv_kernel(rows:int, k:int, schedule:str, opts:tuple[Opt, ...]):
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    blk = UOp.range(k_blocks, 1, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 2, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK

    acc = out[row].set(0.0)
    acc = out[row].set(acc.after(blk, pos)[row] + _q4k_block_dot(words, x, base, blk, pos), end=pos)
    return acc.end(row, blk).sink(arg=_kernel_info(f"q4k_gemv_ref_{rows}_{k}", schedule, opts))

  return kernel

def q4k_gemv_silu_gate_kernel(rows:int, k:int, schedule:str, opts:tuple[Opt, ...]):
  # FFN activation producer fusion (B1, decode scope): the 'up' GEMV writes silu(gate[row]) * (Sum w.x) at its
  # final store, eliminating the standalone silu(gate)*up elementwise launch (E_49152, ~1.24ms/token). Uses a REG
  # accumulator over a single flattened k reduce + an activated store (the flash_combine pattern). parts=1 only.
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(out:UOp, words:UOp, x:UOp, gate:UOp) -> UOp:
    row = UOp.range(rows, 0)
    kp = UOp.range(k_blocks * 32, 1, axis_type=AxisType.REDUCE)
    blk = kp // 32; pos = kp % 32
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot(words, x, base, blk, pos)
    acc = UOp.placeholder((1,), dtypes.float32, 130, addrspace=AddrSpace.REG)
    acc = acc.after(row)[0].set(0.0)
    acc = acc[0].set(acc.after(kp)[0] + contrib, end=kp)
    g = gate[row].cast(dtypes.float32)
    sig = UOp.const(dtypes.float32, 1.0) / (UOp.const(dtypes.float32, 1.0) + (g * -1.4426950408889634).exp2())  # silu = g*sigmoid(g)
    return out[row].store(g * sig * acc[0]).end(row).sink(arg=_kernel_info(f"q4k_gemv_silu_gate_{rows}_{k}", schedule, opts))

  return kernel

def q4k_gemv_silu_gate_v2_kernel(rows:int, k:int, schedule:str, opts:tuple[Opt, ...]):
  # B1 v2: keep the FAST nested blk+pos buffer-accumulator codegen of q4k_gemv_partial (scratch buffer), then the
  # final store applies silu(gate)*. out and scratch are separate buffers; the activated store ends blk.
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(out:UOp, scratch:UOp, words:UOp, x:UOp, gate:UOp) -> UOp:
    row = UOp.range(rows, 0)
    blk = UOp.range(k_blocks, 1, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 2, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    acc = scratch[row].set(0.0)
    acc = scratch[row].set(acc.after(blk, pos)[row] + _q4k_block_dot(words, x, base, blk, pos), end=pos)
    g = gate[row].cast(dtypes.float32)
    sig = UOp.const(dtypes.float32, 1.0) / (UOp.const(dtypes.float32, 1.0) + (g * -1.4426950408889634).exp2())
    return out[row].store(g * sig * acc.after(blk)[row]).end(row, blk).sink(arg=_kernel_info(f"q4k_gemv_silu_gate_v2_{rows}_{k}", schedule, opts))

  return kernel

def q4k_gemv_partial_kernel(rows:int, k:int, parts:int, schedule:str, opts:tuple[Opt, ...]):
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
    contrib = in_range.where(_q4k_block_dot(words, x, base, blk, pos), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part, pos)[row, part] + contrib, end=pos)
    return acc.end(row, part, blk_part).sink(arg=_kernel_info(f"q4k_gemv_partial_{rows}_{k}_{parts}", schedule, opts))

  return kernel

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

def q4k_gemv_packed_load_partial_kernel(rows:int, k:int, parts:int, schedule:str, opts:tuple[Opt, ...]):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    part = UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    lane4 = UOp.range(8, 3, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot_packed_load(words, x, base, blk, lane4), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part, lane4)[row, part] + contrib, end=lane4)
    return acc.end(row, part, blk_part).sink(arg=_kernel_info(f"q4k_gemv_packed_load_partial_{rows}_{k}_{parts}", schedule, opts))

  return kernel

def _sdot4_op(a:UOp, b:UOp, acc:UOp) -> UOp:
  # first-class-via-renderer-helper signed dot4: emits a `_sdot4(a,b,c)` call; the HIP renderer owns the helper
  # (cstyle.py, non-volatile inline-asm v_dot4_i32_i8 -- the only native dot4 on RDNA3) so the compiler can
  # schedule/reorder the dot4 calls (vs an opaque user `asm volatile`). Visible-enough op, not user asm.
  return UOp(Ops.CUSTOMI, dtypes.int32, (acc, a, b), arg='_sdot4({1}, {2}, {0})')

def q8_signed_pack_u32_kernel(k:int):
  # pack 4 consecutive SIGNED int8 q8 activations into one int32 (raw bits, NO +128 bias) for signed v_dot4_i32_i8.
  def kernel(out:UOp, q:UOp) -> UOp:
    idx = UOp.range(k // 4, 0); base = idx * 4
    word = UOp.const(dtypes.uint32, 0)
    for lane in range(4):
      word = word.bitwise_or(q[base + lane].cast(dtypes.int32).bitwise_and(255).cast(dtypes.uint32).lshift(8 * lane))
    return out[idx].store(word).end(idx).sink(arg=_kernel_info(f"q8_signed_pack_{k}", "none", ()))
  return kernel

def q4k_coop_sdot4_partial_kernel(rows:int, k:int, row_tile:int=8):
  # Deep-linearizer arc microkernel: llama-structure Q4_K MMVQ via the first-class _sdot4 op. Packed extract
  # (qword>>sh)&0x0F0F0F0F + signed _sdot4 dot + _sdot4(0x01010101,q8) qsum + per-group scale; block d/dmin once.
  # Inputs: words, q8packed (signed int32, 4 raw int8/word), xscales. Output partials[rows, 8].
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(partials:UOp, words:UOp, q8packed:UOp, xscales:UOp) -> UOp:
    row_o = UOp.range(cdiv(rows, row_tile), 0)
    row_i = UOp.range(row_tile, 1, axis_type=AxisType.LOCAL)
    lane4 = UOp.range(8, 2, axis_type=AxisType.LOCAL)
    blk = UOp.range(k_blocks, 3, axis_type=AxisType.REDUCE)
    row = row_o * row_tile + row_i
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    d_blk = _f16_word(words[base], False); dmin_blk = _f16_word(words[base], True)
    psd = UOp.const(dtypes.float32, 0.0); psm = UOp.const(dtypes.float32, 0.0)
    ones = UOp.const(dtypes.int32, 0x01010101); z = UOp.const(dtypes.int32, 0)
    for grp in range(8):
      _, _, sc, mn = _q4k_group_params(words, base, grp)
      qword = words[base + 4 + (grp // 2) * 8 + lane4]
      q4 = qword.rshift((grp % 2) * 4).bitwise_and(0x0F0F0F0F).cast(dtypes.int32)
      q8 = q8packed[blk * 64 + grp * 8 + lane4].cast(dtypes.int32)
      d8 = xscales[blk * 8 + grp].cast(dtypes.float32)
      # _sdot4 is v_dot4_i32_iu8: a=SIGNED (q8), b=UNSIGNED (q4 nibbles 0..15 / ones). dot=Σq8*q4, qsum=Σq8.
      psd = psd + d8 * _sdot4_op(q8, q4, z).cast(dtypes.float32) * sc.cast(dtypes.float32)
      psm = psm + d8 * _sdot4_op(q8, ones, z).cast(dtypes.float32) * mn.cast(dtypes.float32)
    contrib = d_blk * psd - dmin_blk * psm

    acc = partials[row, lane4].set(0.0)
    acc = partials[row, lane4].set(acc.after(blk)[row, lane4] + contrib, end=blk)
    return acc.end(row_o, row_i, lane4).sink(arg=_kernel_info(f"q4k_coop_sdot4_partial_{rows}_{k}", "", ()))

  return kernel

def q4k_gemv_warp_kernel(rows:int, k:int, lanes:int=32):
  # FFN-GEMV WORK-DECOMPOSITION variant (lossless FP, no q8/int-dot). llama's MMVQ shape: many threads/row +
  # K-block-parallel + IN-KERNEL warp-shuffle reduce + one output write (vs the default 1-thread/row serial
  # uncoalesced ~51% peak, and the coop's 8-lanes + stage-2 .sum). Here: `lanes` threads/row = ONE wave (32 on
  # gfx1100); lane = block_group*8 + lane4. lane4 (0..7) = within-block word index -> 8 adjacent lanes read 8
  # adjacent packed words (coalesced); block_group (0..3) splits the k_blocks into 4 K-parallel chunks across the
  # wave. Each lane FP-accumulates its blocks (REG), then warp_reduce_sum (ds_bpermute) -> out[row] (single store,
  # no stage-2 partials buffer). Decode/math identical to the default (exact up to fp reassoc). k_blocks % 4 == 0.
  from extra.qk.amd_warp_reduce import warp_reduce_sum
  if lanes != 32: raise ValueError("q4k_gemv_warp currently supports lanes=32 (one gfx1100 wave) only")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  if k_blocks % 4 != 0: raise ValueError(f"k_blocks={k_blocks} must be divisible by 4 for the 4-block_group warp split")
  bpb = k_blocks // 4

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.special(rows, "gidx0")              # one workgroup per row
    lane = UOp.special(32, "lidx0")               # 32 threads = one gfx1100 wave (warp_reduce needs a real lidx)
    bg = lane // 8                                 # block_group 0..3 (K-parallel across the wave)
    lane4 = lane % 8                               # within-block word index 0..7 (coalesced packed-word loads)
    lblk = UOp.range(bpb, 0, axis_type=AxisType.REDUCE)
    blk = bg * bpb + lblk
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load(words, x, base, blk, lane4)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = warp_reduce_sum(acc[0], lane, 32)      # every lane holds the row sum
    return out[row].store(total).sink(arg=KernelInfo(name=f"q4k_gemv_warp_{rows}_{k}", opts_to_apply=()))

  return kernel

def q4k_coop_partial_kernel(rows:int, k:int, row_tile:int=8):
  # Cooperative-K Q4_K GEMV (MMVQ_COOP, sibling of q6k_coop_partial_kernel). The Q4_K quant word index is
  # `4 + (grp//2)*8 + pos//4`, so the within-block word index `lane4` (= pos//4, 0..7) becomes a LOCAL lane
  # axis -> adjacent lanes read ADJACENT packed words -> coalesced (the default q4k_gemv_partial maps one row
  # per thread -> uncoalesced, ~40% peak). Each lane reads one qword and does its 4 nibbles across the 8 groups
  # (_q4k_block_dot_packed_load) and writes its OWN partial partials[row, lane4]; stage-2 `.sum(axis=1)` reduces
  # the 8 lanes (no in-kernel cross-lane reduce, like the Q6_K coop kernel). row_tile rows share a workgroup
  # (lanes = row_tile*8) for occupancy. Output: partials[rows, 8].
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row_o = UOp.range(cdiv(rows, row_tile), 0)
    row_i = UOp.range(row_tile, 1, axis_type=AxisType.LOCAL)
    lane4 = UOp.range(8, 2, axis_type=AxisType.LOCAL)
    blk = UOp.range(k_blocks, 3, axis_type=AxisType.REDUCE)
    row = row_o * row_tile + row_i
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load(words, x, base, blk, lane4)

    acc = partials[row, lane4].set(0.0)
    acc = partials[row, lane4].set(acc.after(blk)[row, lane4] + contrib, end=blk)
    return acc.end(row_o, row_i, lane4).sink(arg=_kernel_info(f"q4k_coop_partial_{rows}_{k}", "", ()))

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

def q4k_gemv_vector_load_partial_kernel(rows:int, k:int, parts:int, schedule:str, opts:tuple[Opt, ...]):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    part = UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    lane_vec = UOp.range(2, 3, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot_vector_load(words, x, base, blk, lane_vec), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part, lane_vec)[row, part] + contrib, end=lane_vec)
    return acc.end(row, part, blk_part).sink(arg=_kernel_info(f"q4k_gemv_vector_load_partial_{rows}_{k}_{parts}", schedule, opts))

  return kernel

def q4k_gemv_grouped_partial_kernel(rows:int, k:int, parts:int, row_group:int, schedule:str, opts:tuple[Opt, ...]):
  if row_group < 1: raise ValueError("row_group must be >= 1")
  if rows % row_group != 0:
    raise ValueError(f"row_group={row_group} must divide rows={rows} for grouped Q4_K GEMV")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)
  row_groups = rows // row_group

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    group = UOp.range(row_groups, 0)
    lane = UOp.range(row_group, 1)
    part = UOp.range(parts, 2)
    blk_part = UOp.range(blocks_per_part, 3, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 4, axis_type=AxisType.REDUCE)
    row = group * row_group + lane
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot(words, x, base, blk, pos), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part, pos)[row, part] + contrib, end=pos)
    return acc.end(group, lane, part, blk_part).sink(
      arg=_kernel_info(f"q4k_gemv_grouped_partial_{rows}_{k}_{parts}_rg{row_group}", schedule, opts))

  return kernel

def q4k_q8_1_gemv_partial_kernel(rows:int, k:int, parts:int, schedule:str, opts:tuple[Opt, ...]):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, xq:UOp, xscales:UOp) -> UOp:
    row = UOp.range(rows, 0)
    part = UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 3, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot_q8_1(words, xq, xscales, base, blk, pos), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part, pos)[row, part] + contrib, end=pos)
    return acc.end(row, part, blk_part).sink(arg=_kernel_info(f"q4k_q8_1_gemv_partial_{rows}_{k}_{parts}", schedule, opts))

  return kernel

def q4k_q8_1_gemm_kernel(rows:int, k:int, b:int, parts:int, schedule:str, opts:tuple[Opt, ...], name:str="q4k_q8_1_gemm"):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, xq:UOp, xscales:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(b, 1)
    part = UOp.range(parts, 2)
    blk_part = UOp.range(blocks_per_part, 3, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 4, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot_q8_1_gemm(words, xq, xscales, base, blk, pos, bb, k), UOp.const(dtypes.float32, 0.0))

    acc = partials[row, bb, part].set(0.0)
    acc = partials[row, bb, part].set(acc.after(blk_part, pos)[row, bb, part] + contrib, end=pos)
    return acc.end(row, bb, part, blk_part).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}_{parts}", schedule, opts))

  return kernel

def q4k_q8_1_sdot4_gemm_kernel(rows:int, k:int, b:int, parts:int, schedule:str, opts:tuple[Opt, ...],
                               name:str="q4k_q8_1_sdot4_gemm"):
  if schedule != "none" or opts:
    raise ValueError("q4k_q8_1_sdot4_gemm_kernel is a generated-UOp dot4 candidate; schedule opts unsupported")
  if parts != 1: raise ValueError("q4k_q8_1_sdot4_gemm_kernel currently supports parts=1 only")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, q8packed:UOp, xscales:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(b, 1)
    part = UOp.range(parts, 2)
    blk_part = UOp.range(blocks_per_part, 3, axis_type=AxisType.REDUCE)
    lane4 = UOp.range(8, 4, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    d_blk = _f16_word(words[base], False); dmin_blk = _f16_word(words[base], True)
    psd = UOp.const(dtypes.float32, 0.0); psm = UOp.const(dtypes.float32, 0.0)
    ones = UOp.const(dtypes.int32, 0x01010101); z = UOp.const(dtypes.int32, 0)
    q8_base = bb * (k // 4) + blk * 64
    scale_base = bb * (k // Q8_1_BLOCK_ELEMS) + blk * 8
    for grp in range(8):
      _, _, sc, mn = _q4k_group_params(words, base, grp)
      qword = words[base + 4 + (grp // 2) * 8 + lane4]
      q4 = qword.rshift((grp % 2) * 4).bitwise_and(0x0F0F0F0F).cast(dtypes.int32)
      q8 = q8packed[q8_base + grp * 8 + lane4].cast(dtypes.int32)
      d8 = xscales[scale_base + grp].cast(dtypes.float32)
      psd = psd + d8 * _sdot4_op(q8, q4, z).cast(dtypes.float32) * sc.cast(dtypes.float32)
      psm = psm + d8 * _sdot4_op(q8, ones, z).cast(dtypes.float32) * mn.cast(dtypes.float32)
    contrib = in_range.where(d_blk * psd - dmin_blk * psm, UOp.const(dtypes.float32, 0.0))

    acc = partials[row, bb, part].set(0.0)
    acc = partials[row, bb, part].set(acc.after(blk_part, lane4)[row, bb, part] + contrib, end=lane4)
    return acc.end(row, bb, part, blk_part).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}_{parts}", schedule, opts))

  return kernel

def q4k_q8_1_sdot4_coop_gemm_kernel(rows:int, k:int, b:int, row_tile:int=1, token_tile:int=1,
                                    name:str="q4k_q8_1_sdot4_coop_gemm"):
  # Generated-UOp MMQ-shaped first step: 8 local lanes split the Q4_K word columns for one output element.
  # Output partials shape is [rows, b, 8]; the caller reduces axis=2.
  if row_tile < 1 or token_tile < 1: raise ValueError("row_tile/token_tile must be >= 1")
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(partials:UOp, words:UOp, q8packed:UOp, xscales:UOp) -> UOp:
    row_o = UOp.range(cdiv(rows, row_tile), 0)
    bb_o = UOp.range(cdiv(b, token_tile), 1)
    row_i = UOp.range(row_tile, 2, axis_type=AxisType.LOCAL)
    bb_i = UOp.range(token_tile, 3, axis_type=AxisType.LOCAL)
    lane4 = UOp.range(8, 4, axis_type=AxisType.LOCAL)
    blk = UOp.range(k_blocks, 5, axis_type=AxisType.REDUCE)
    row = row_o * row_tile + row_i
    bb = bb_o * token_tile + bb_i
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    d_blk = _f16_word(words[base], False); dmin_blk = _f16_word(words[base], True)
    psd = UOp.const(dtypes.float32, 0.0); psm = UOp.const(dtypes.float32, 0.0)
    ones = UOp.const(dtypes.int32, 0x01010101); z = UOp.const(dtypes.int32, 0)
    q8_base = bb * (k // 4) + blk * 64
    scale_base = bb * (k // Q8_1_BLOCK_ELEMS) + blk * 8
    for grp in range(8):
      _, _, sc, mn = _q4k_group_params(words, base, grp)
      qword = words[base + 4 + (grp // 2) * 8 + lane4]
      q4 = qword.rshift((grp % 2) * 4).bitwise_and(0x0F0F0F0F).cast(dtypes.int32)
      q8 = q8packed[q8_base + grp * 8 + lane4].cast(dtypes.int32)
      d8 = xscales[scale_base + grp].cast(dtypes.float32)
      psd = psd + d8 * _sdot4_op(q8, q4, z).cast(dtypes.float32) * sc.cast(dtypes.float32)
      psm = psm + d8 * _sdot4_op(q8, ones, z).cast(dtypes.float32) * mn.cast(dtypes.float32)
    contrib = d_blk * psd - dmin_blk * psm

    acc = partials[row, bb, lane4].set(0.0)
    acc = partials[row, bb, lane4].set(acc.after(blk)[row, bb, lane4] + contrib, end=blk)
    return acc.end(row_o, bb_o, row_i, bb_i, lane4).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}_8", "", ()))

  return kernel

def q4k_q8_1_sdot4_coop_direct_out_kernel(rows:int, k:int, b:int, row_tile:int=1, token_tile:int=4,
                                          name:str="q4k_q8_1_sdot4_coop_direct_out_gemm"):
  # Same generated-UOp Q4_K/Q8_1 dot4 algebra as q4k_q8_1_sdot4_coop_gemm_kernel, but the 8 lane partials are reduced
  # inside the wave and written directly to [b, rows]. This removes the full-model [rows,b,8] partial tensor.
  from extra.qk.amd_warp_reduce import warp_reduce_sum
  if row_tile < 1 or token_tile < 1 or row_tile * token_tile != 4:
    raise ValueError("direct-out coop requires row_tile*token_tile == 4 for one wave of four 8-lane outputs")
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(out:UOp, words:UOp, q8packed:UOp, xscales:UOp) -> UOp:
    row_o = UOp.special(cdiv(rows, row_tile), "gidx0")
    bb_o = UOp.special(cdiv(b, token_tile), "gidx1")
    lane = UOp.special(32, "lidx0")
    out_lane = lane // 8
    lane4 = lane % 8
    row_i = out_lane // token_tile
    bb_i = out_lane % token_tile
    blk = UOp.range(k_blocks, 0, axis_type=AxisType.REDUCE)
    row = row_o * row_tile + row_i
    bb = bb_o * token_tile + bb_i
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    d_blk = _f16_word(words[base], False); dmin_blk = _f16_word(words[base], True)
    psd = UOp.const(dtypes.float32, 0.0); psm = UOp.const(dtypes.float32, 0.0)
    ones = UOp.const(dtypes.int32, 0x01010101); z = UOp.const(dtypes.int32, 0)
    q8_base = bb * (k // 4) + blk * 64
    scale_base = bb * (k // Q8_1_BLOCK_ELEMS) + blk * 8
    for grp in range(8):
      _, _, sc, mn = _q4k_group_params(words, base, grp)
      qword = words[base + 4 + (grp // 2) * 8 + lane4]
      q4 = qword.rshift((grp % 2) * 4).bitwise_and(0x0F0F0F0F).cast(dtypes.int32)
      q8 = q8packed[q8_base + grp * 8 + lane4].cast(dtypes.int32)
      d8 = xscales[scale_base + grp].cast(dtypes.float32)
      psd = psd + d8 * _sdot4_op(q8, q4, z).cast(dtypes.float32) * sc.cast(dtypes.float32)
      psm = psm + d8 * _sdot4_op(q8, ones, z).cast(dtypes.float32) * mn.cast(dtypes.float32)
    contrib = d_blk * psd - dmin_blk * psm
    acc = UOp.placeholder((1,), dtypes.float32, 212, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
    total = warp_reduce_sum(acc[0], lane, 8)
    return out[bb, row].store(total).sink(arg=_kernel_info(f"{name}_{rows}_{k}_{b}", "", ()))

  return kernel

def q4k_q8_1_intdot_partial_kernel(rows:int, k:int, parts:int, schedule:str, opts:tuple[Opt, ...]):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, xq:UOp, xscales:UOp) -> UOp:
    row = UOp.range(rows, 0)
    part = UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot_q8_1_intdot(words, xq, xscales, base, blk, (row, part, blk_part)),
                             UOp.const(dtypes.float32, 0.0))

    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part)[row, part] + contrib, end=blk_part)
    return acc.end(row, part).sink(arg=_kernel_info(f"q4k_q8_1_intdot_partial_{rows}_{k}_{parts}", schedule, opts))

  return kernel

def q4k_q8_1_vdot_parallel_partial_kernel(rows:int, k:int, parts:int, schedule:str, opts:tuple[Opt, ...]):
  k_blocks = k // Q4_K_BLOCK_ELEMS
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, xq_bias_words:UOp, xscales:UOp) -> UOp:
    row = UOp.range(rows, 0)
    part = UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot_q8_1_vdot_parallel(words, xq_bias_words, xscales, base, blk, (row, part, blk_part)),
                             UOp.const(dtypes.float32, 0.0))

    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part)[row, part] + contrib, end=blk_part)
    return acc.end(row, part).sink(arg=_kernel_info(f"q4k_q8_1_vdot_parallel_partial_{rows}_{k}_{parts}", schedule, opts))

  return kernel

def q4k_q8_1_vdot_partial_kernel(rows:int, k:int, parts:int, schedule:str, opts:tuple[Opt, ...]):
  if schedule != "none" or opts:
    raise ValueError("q4k_q8_1_vdot_partial_kernel is a fixed inline-asm smoke candidate; schedule opts are not supported")
  if parts != 1: raise ValueError("q4k_q8_1_vdot_partial_kernel currently supports parts=1 only")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  source = _q4k_q8_1_vdot_source(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, xq_bias_words:UOp, xscales:UOp) -> UOp:
    gid = UOp.special(rows, "gidx0")
    out_ptr = partials.flatten().index(gid, ptr=True)
    row_words = words.index(gid * k_blocks * Q4K_WORDS_PER_BLOCK, ptr=True)
    stmt = UOp(Ops.CUSTOM, dtypes.void, (out_ptr, row_words, xq_bias_words, xscales), arg=source)
    return stmt.sink(arg=_kernel_info(f"q4k_q8_1_vdot_partial_{rows}_{k}_{parts}", schedule, opts))

  return kernel

def q4k_q8_1_vdot_builtin_partial_kernel(rows:int, k:int, parts:int, schedule:str, opts:tuple[Opt, ...]):
  # D1: the v_dot4 GEMV via the SCHEDULABLE __builtin_amdgcn_udot4 (through the _dp4a device helper the
  # HIPRenderer emits) instead of asm volatile. Same structure as q4k_q8_1_vdot_partial_kernel.
  if schedule != "none" or opts:
    raise ValueError("q4k_q8_1_vdot_builtin_partial_kernel is a fixed builtin-dp4a candidate; opts unsupported")
  if parts != 1: raise ValueError("q4k_q8_1_vdot_builtin_partial_kernel currently supports parts=1 only")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  source = _q4k_q8_1_vdot_source(k_blocks, parts, builtin=True)

  LOCAL = 64  # rows per workgroup -> full-occupancy launch (the fp kernel gets this via LOCAL:0:64)
  def kernel(partials:UOp, words:UOp, xq_bias_words:UOp, xscales:UOp) -> UOp:
    if rows % LOCAL == 0:
      gid = UOp.special(rows // LOCAL, "gidx0") * LOCAL + UOp.special(LOCAL, "lidx0")
    else:
      gid = UOp.special(rows, "gidx0")
    out_ptr = partials.flatten().index(gid, ptr=True)
    row_words = words.index(gid * k_blocks * Q4K_WORDS_PER_BLOCK, ptr=True)
    stmt = UOp(Ops.CUSTOM, dtypes.void, (out_ptr, row_words, xq_bias_words, xscales), arg=source)
    return stmt.sink(arg=_kernel_info(f"q4k_q8_1_vdot_builtin_partial_{rows}_{k}_{parts}", schedule, opts))

  return kernel

def q4k_gemv_tile_custom_partial_kernel(rows:int, k:int, parts:int, schedule:str, opts:tuple[Opt, ...]):
  if schedule != "none" or opts:
    raise ValueError("q4k_gemv_tile_custom_partial_kernel is a fixed semantic packed-tile lowering; schedule opts are not supported")
  if parts < 1: raise ValueError("parts must be >= 1")
  k_blocks = k // Q4_K_BLOCK_ELEMS
  source = _q4k_tile_custom_partial_source(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.special(rows, "gidx0", dtype=dtypes.int)
    row_words = words.index(row * k_blocks * Q4K_WORDS_PER_BLOCK, ptr=True)
    half_marker = x[0]
    if parts == 1:
      out_ptr = partials.flatten().index(row, ptr=True)
      srcs = (out_ptr, row_words, x, half_marker)
    else:
      part = UOp.special(parts, "gidx1", dtype=dtypes.int)
      out_ptr = partials.flatten().index(row * parts + part, ptr=True)
      srcs = (out_ptr, row_words, x, part, half_marker)
    stmt = UOp(Ops.CUSTOM, dtypes.void, srcs, arg=source)
    return stmt.sink(arg=_kernel_info(f"q4k_gemv_tile_custom_partial_{rows}_{k}_{parts}", schedule, opts))

  return kernel

def q8_1_bias_pack_u32_kernel(k:int):
  if k % 4 != 0: raise ValueError(f"K={k} is not divisible by 4")

  def kernel(out:UOp, q:UOp) -> UOp:
    idx = UOp.range(k//4, 0)
    base = idx * 4
    word = UOp.const(dtypes.uint32, 0)
    for lane in range(4):
      biased = (q[base+lane].cast(dtypes.int32) + 128).cast(dtypes.uint32).bitwise_and(255)
      word = word.bitwise_or(biased.lshift(8*lane))
    return out[idx].store(word).end(idx).sink(arg=_kernel_info(f"q8_1_bias_pack_u32_{k}", "none", ()))

  return kernel

def q4k_unpack_kernel(rows:int, k:int):
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(out:UOp, words:UOp) -> UOp:
    row = UOp.range(rows, 0)
    blk = UOp.range(k_blocks, 1)
    pos = UOp.range(32, 2)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    stores = []
    for grp in range(8):
      stores.append(out[row, blk*Q4_K_BLOCK_ELEMS + grp*32 + pos].store(_q4k_weight(words, base, grp, pos)))
    return UOp.group(*stores).end(row, blk, pos).sink(arg=_kernel_info(f"q4k_unpack_{rows}_{k}", "none", ()))

  return kernel

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
  parser = argparse.ArgumentParser(description="Correctness-first custom Q4_K GEMV primitive probe")
  parser.add_argument("gguf", type=pathlib.Path)
  parser.add_argument("--tensor", default="blk.0.ffn_gate.weight")
  parser.add_argument("--device", default=None)
  parser.add_argument("--rows", type=int, default=2)
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--mode", choices=("serial", "partial", "packed_load", "vector_load", "grouped", "tile_custom"), default="serial")
  parser.add_argument("--parts", type=int, default=16, help="number of K-block partitions for --mode partial")
  parser.add_argument("--row-group", type=int, default=1, help="output rows per group for --mode grouped")
  parser.add_argument("--schedule", choices=("none", "auto"), default="none",
                      help="schedule opts for the custom primitive")
  parser.add_argument("--opt", action="append", default=[], help="explicit primitive opt OP:AXIS:ARG, e.g. LOCAL:0:32")
  parser.add_argument("--unpack-check-rows", type=int, default=2, help="rows to use for direct decoded-weight correctness gate")
  parser.add_argument("--seed", type=int, default=1337, help="seed for random activation correctness gate")
  args = parser.parse_args()
  if args.schedule == "auto":
    assert_q4k_risky_search_allowed(args.device, "Q4_K primitive --schedule auto")

  meta = read_metadata(args.gguf)
  info = pick_tensor(meta.infos, args.tensor)
  if info.typ != GGML_Q4_K: raise ValueError(f"{info.name} is ggml_type={info.typ}, expected Q4_K")
  n, shape = prod(info.dims), tensor_shape(info)
  if len(shape) != 2: raise ValueError(f"{info.name} is not a matrix: shape={shape}")
  rows, k = min(args.rows, shape[0]), shape[1]
  if k % Q4_K_BLOCK_ELEMS != 0: raise ValueError(f"K={k} is not Q4_K block aligned")
  byte_start = meta.data_start + info.off
  if byte_start % 4 != 0: raise ValueError(f"Q4_K tensor byte offset is not uint32 aligned: {byte_start}")

  row_bytes = k // Q4_K_BLOCK_ELEMS * Q4_K_BLOCK_BYTES
  q4_bytes = rows * row_bytes
  nwords = q4_bytes // 4
  if args.parts < 1: raise ValueError("--parts must be >= 1")
  parts = min(args.parts, k // Q4_K_BLOCK_ELEMS)
  opts = tuple(parse_opt(x) for x in args.opt)
  print(f"tensor={info.name} full_shape={shape} primitive_shape=({rows},{k}) q4_bytes={q4_bytes} nwords={nwords} "
        f"mode={args.mode} parts={parts} row_group={args.row_group} schedule={args.schedule} "
        f"opts={[str(x) for x in opts]} device={args.device or 'default'}")

  raw_words = Tensor(args.gguf, dtype=dtypes.uint32)
  words = raw_words[byte_start//4:byte_start//4+nwords].to(args.device).contiguous().realize()
  Tensor.manual_seed(args.seed)
  x = Tensor.randn(k, dtype=dtypes.float16, device=args.device).realize()
  out = Tensor.empty(rows, dtype=dtypes.float32, device=args.device)
  partials = Tensor.empty(rows, parts, dtype=dtypes.float32, device=args.device)
  vector_partials = Tensor.empty(rows, parts, dtype=dtypes.float32, device=args.device)

  raw_u8 = Tensor(args.gguf)[byte_start:byte_start+q4_bytes].to(args.device)
  decoded = q4_k_reference(raw_u8, rows*k).reshape(rows, k).cast(dtypes.float16).realize()
  ref = (decoded.cast(dtypes.float32) * x.reshape(1, k).cast(dtypes.float32)).sum(axis=1).realize()

  unpack_rows = min(args.unpack_check_rows, rows)
  if unpack_rows > 0:
    unpack_words = raw_words[byte_start//4:byte_start//4+(unpack_rows*row_bytes)//4].to(args.device).contiguous().realize()
    unpack_out = Tensor.empty(unpack_rows, k, dtype=dtypes.float32, device=args.device)
    unpack_got = unpack_out.custom_kernel(unpack_words, fxn=q4k_unpack_kernel(unpack_rows, k))[0].realize()
    unpack_ref = q4_k_reference(Tensor(args.gguf)[byte_start:byte_start+unpack_rows*row_bytes].to(args.device), unpack_rows*k).reshape(unpack_rows, k).realize()
    unpack_max_abs = (unpack_got - unpack_ref).abs().max().item()
    print(f"unpack_correctness: rows={unpack_rows} max_abs={unpack_max_abs:.6g}")
    if unpack_max_abs != 0:
      raise AssertionError("Q4_K unpack primitive correctness failed")

  def primitive():
    if args.mode == "serial":
      return out.custom_kernel(words, x, fxn=q4k_gemv_kernel(rows, k, args.schedule, opts))[0]
    if args.mode == "packed_load":
      partial = partials.custom_kernel(words, x, fxn=q4k_gemv_packed_load_partial_kernel(rows, k, parts, args.schedule, opts))[0]
      return partial.sum(axis=1)
    if args.mode == "vector_load":
      partial = vector_partials.custom_kernel(words, x, fxn=q4k_gemv_vector_load_partial_kernel(rows, k, parts, args.schedule, opts))[0]
      return partial.sum(axis=1)
    if args.mode == "grouped":
      partial = partials.custom_kernel(words, x, fxn=q4k_gemv_grouped_partial_kernel(rows, k, parts, args.row_group, args.schedule, opts))[0]
    elif args.mode == "tile_custom":
      partial = partials.custom_kernel(words, x, fxn=q4k_gemv_tile_custom_partial_kernel(rows, k, parts, args.schedule, opts))[0]
    else:
      partial = partials.custom_kernel(words, x, fxn=q4k_gemv_partial_kernel(rows, k, parts, args.schedule, opts))[0]
    return partial.sum(axis=1)

  got = primitive().realize()
  max_abs = (got - ref).abs().max().item()
  print(f"correctness: max_abs={max_abs:.6g}")
  if max_abs > 1e-2:
    print("got", got.numpy())
    print("ref", ref.numpy())
    raise AssertionError("Q4_K GEMV primitive correctness failed")
  bench(f"q4k_gemv_primitive_{args.mode}", args.iters, q4_bytes, primitive)
