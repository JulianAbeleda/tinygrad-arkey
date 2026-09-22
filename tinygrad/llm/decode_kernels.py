"""Production Q4_K G3 and Q6_K decode kernel lowerings.

These are statically promoted results: search and qualification live outside the
runtime, while this module owns the selected data descriptions and generic UOp
lowerings used for inference.  Keep this module independent of ``extra``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from tinygrad import dtypes
from tinygrad.codegen.late.warp_reduce import WARP, _staged_shfl, _warp_reduce_sum_staged
from tinygrad.dtype import AddrSpace, DType
from tinygrad.helpers import cdiv
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK, Q6_K, Q6_K_BLOCK_ELEMS, Q6K_HALFWORDS_PER_BLOCK, QuantFormat
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp


# Q4_K G3 selected lane-map lowering.
@dataclass(frozen=True)
class Q4KGateUpLaneMap:
  k: int
  n: int
  qk_k: int = Q4_K_BLOCK_ELEMS
  lane_extent: int = WARP
  block_groups: int = 4
  words_per_group: int = 8
  q4k_words_per_block: int = Q4K_WORDS_PER_BLOCK

  @property
  def k_blocks(self) -> int: return self.k // self.qk_k

  @property
  def blocks_per_group(self) -> int: return self.k_blocks // self.block_groups

  def validate(self) -> None:
    if self.k % self.qk_k != 0: raise ValueError(f"k={self.k} must divide qk_k={self.qk_k}")
    if self.k_blocks % self.block_groups != 0:
      raise ValueError(f"k_blocks={self.k_blocks} must divide block_groups={self.block_groups}")
    if self.lane_extent != self.block_groups * self.words_per_group:
      raise ValueError(f"lane extent mismatch: {self.lane_extent} != {self.block_groups} * {self.words_per_group}")
    if self.words_per_group != 8: raise ValueError("Q4_K G3 lane map requires eight packed words per group")
    if self.q4k_words_per_block != 36: raise ValueError("Q4_K block layout must be 36 uint32 words")


@dataclass(frozen=True)
class LanePartition:
  lane: UOp
  lane_extent: int = WARP
  words_per_group: int = 8

  def validate(self) -> None:
    if self.lane_extent != WARP: raise ValueError(f"only wave{WARP} is supported, got {self.lane_extent}")
    if self.words_per_group <= 0 or self.lane_extent % self.words_per_group != 0:
      raise ValueError(f"words_per_group must divide lane_extent, got {self.words_per_group} / {self.lane_extent}")
    if self.lane.dtype.scalar() is not dtypes.weakint: raise ValueError(f"lane must be weakint index dtype, got {self.lane.dtype}")

  @property
  def word_col(self) -> UOp: return self.lane % self.words_per_group

  @property
  def block_group(self) -> UOp: return self.lane // self.words_per_group


def _lane_partition_reduce_sum(partial:UOp, part:LanePartition) -> UOp:
  part.validate()
  if partial.dtype.scalar() not in (dtypes.float32, dtypes.float):
    raise ValueError(f"lane partition sum supports float partials, got {partial.dtype}")
  if any(u.op is Ops.RANGE and u.arg[-1].name in ("UPCAST", "UNROLL") for u in partial.toposort()):
    raise ValueError("vectorized lane-partition partials are not supported")
  return _warp_reduce_sum_staged(partial, part.lane, part.lane_extent)


def _f16_word(word:UOp, high:bool) -> UOp:
  bits = (word.rshift(16) if high else word).bitwise_and(0xffff)
  return bits.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)


def _q4k_group_params(words:UOp, base:UOp, grp:int) -> tuple[UOp, UOp, UOp, UOp]:
  d, dmin = _f16_word(words[base], False), _f16_word(words[base], True)
  def scale_byte(idx:int) -> UOp: return words[base + 1 + idx//4].rshift((idx%4)*8).bitwise_and(0xff)
  if grp < 4:
    sc, mn = scale_byte(grp).bitwise_and(63), scale_byte(4+grp).bitwise_and(63)
  else:
    high = scale_byte(8+grp-4)
    sc = high.bitwise_and(0xf).bitwise_or(scale_byte(grp-4).rshift(6).lshift(4))
    mn = high.rshift(4).bitwise_or(scale_byte(4+grp-4).rshift(6).lshift(4))
  return d, dmin, sc, mn


def _q4k_group_dot_packed_load(words:UOp, x:UOp, base:UOp, x_block:UOp, grp:int, lane4:UOp) -> UOp:
  d, dmin, sc, mn = _q4k_group_params(words, base, grp)
  qpack = words[base + 4 + (grp//2)*8 + lane4].rshift((grp%2)*4).bitwise_and(0x0F0F0F0F)
  contrib = UOp.const(dtypes.float32, 0.0)
  for nib in range(4):
    pos = lane4 * 4 + nib
    q = qpack.rshift(nib*8).bitwise_and(0xf)
    weight = d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)
    contrib = contrib + weight * x[x_block*Q4_K_BLOCK_ELEMS + grp*32 + pos].cast(dtypes.float32)
  return contrib


def _q4k_block_dot_packed_load(words:UOp, x:UOp, base:UOp, x_block:UOp, lane4:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(8): contrib = contrib + _q4k_group_dot_packed_load(words, x, base, x_block, grp, lane4)
  return contrib

def _half4_lane(v:UOp, nib:int) -> UOp:
  """Extract one fp16 lane of a global half4 vector load as fp32. GEP on a float vector load
  is spec-illegal, so the extraction rides CUSTOMI (the same family as fdot2/exp2f). The scalar
  anchor keeps the CUSTOMI shape scalar while the real operand (a half4 LOAD) is src[1], which
  also keeps the half4 type prefix emitted."""
  return UOp(Ops.CUSTOMI, dtypes.float32, (UOp.const(dtypes.float32, 0.0), v),
             arg=f"float({{1}}.{'xyzw'[nib]})")

def _q4k_block_dot_packed_load_vec(words:UOp, x:UOp, base:UOp, x_block:UOp, lane4:UOp) -> UOp:
  """Vectorized-load Q4_K block dot, bit-identical to `_q4k_block_dot_packed_load`: the four
  header words load as one uint4, each qpack word loads once for its two groups, and the four
  per-group fp16 activations load as one half4. The per-lane accumulation order (nib 0..3,
  group 0..7) and every shift/mask are unchanged, so the fp32 result is identical."""
  hdr = words.index(base).load(dtype=dtypes.uint32.vec(4))
  w0, w1, w2, w3 = hdr.gep(0), hdr.gep(1), hdr.gep(2), hdr.gep(3)
  contrib = UOp.const(dtypes.float32, 0.0)
  for g2 in range(4):
    qw = words[base + 4 + g2 * 8 + lane4]
    for gp in range(2):
      grp = 2 * g2 + gp
      d, dmin, sc, mn = _q4k_group_params_from_words(w0, w1, w2, w3, grp)
      qpack = qw.rshift(gp * 4).bitwise_and(0x0F0F0F0F)
      xv = x.index(x_block * Q4_K_BLOCK_ELEMS + grp * 32 + lane4 * 4).load(dtype=dtypes.float16.vec(4))
      for nib in range(4):
        q = qpack.rshift(nib * 8).bitwise_and(0xf)
        weight = d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)
        contrib = contrib + weight * _half4_lane(xv, nib)
  return contrib

def _q4k_block_dot_rms_affine(words:UOp, x:UOp, norm_weight:UOp, scale:UOp, base:UOp, x_block:UOp, lane4:UOp) -> UOp:
  """Packed Q4 dot with the ordinary ffn-norm epilogue applied per packed-Q4 load.
  Control contract (E_32_32_4_f14a5cc0): the norm epilogue is `(half)((x*s)*w)` with
  the weight upcast fp16->fp32 and ONE fp16 RNE round at the very end.  The fused
  kernel reproduces that exact value per x element (fp32 multiply chain, then the
  same single round), so the stored half is bitwise-identical to the standalone
  epilogue's output.  The older double-round spelling (fp16(x*s), then fp16(*w))
  is NOT the control round point and fails the exact-logits gate."""
  contrib=UOp.const(dtypes.float32,0.0)
  for grp in range(8):
    d,dmin,sc,mn=_q4k_group_params(words,base,grp)
    qpack=words[base+4+(grp//2)*8+lane4].rshift((grp%2)*4).bitwise_and(0x0F0F0F0F)
    for nib in range(4):
      pos=lane4*4+nib; idx=x_block*Q4_K_BLOCK_ELEMS+grp*32+pos
      q=qpack.rshift(nib*8).bitwise_and(0xf)
      qw=d*sc.cast(dtypes.float32)*q.cast(dtypes.float32)-dmin*mn.cast(dtypes.float32)
      # Ordinary ffn-norm epilogue: (half)((x*s)*w) -- ONE fp16 round at the end.
      xv=((x[idx].cast(dtypes.float32)*scale[0])*norm_weight[idx].cast(dtypes.float32)).cast(dtypes.float16).cast(dtypes.float32)
      contrib=contrib+qw*xv
  return contrib

def q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(rows:int,k:int,store_fp16:bool=False):
  """Research-only raw-x RMS scale/affine fused Q4 gate/up consumer."""
  # This is deliberately not a general RMSNorm lowering.  Its only admitted
  # lease has the exact NVIDIA decode FFN shape; widening it would silently
  # turn a measured experiment into a route selector.
  if (rows,k) != (12288,4096): raise ValueError(f"rms-affine gate/up requires (12288,4096), got ({rows},{k})")
  lm=Q4KGateUpLaneMap(k=k,n=rows); lm.validate()
  # The fp16 store spelling mirrors the landed fused16 variant
  # (q4k_g3_lanemap_gemv_w1w3fused16_*): the fused z is cast to fp16 in-kernel so
  # the graph's fp32->fp16 ffn-activation cast folds away (the control graph's
  # ffn_down consumes fp16 z).  The legacy fp32 name is unchanged.
  name=f"q4k_g3_lanemap_w1w3_rms_affine16_{rows}_{k}" if store_fp16 else f"q4k_g3_lanemap_w1w3_rms_affine_{rows}_{k}"
  def kernel(out:UOp,gate_words:UOp,up_words:UOp,x:UOp,norm_weight:UOp,scale:UOp) -> UOp:
    row,lane=UOp.special(rows,"gidx0"),UOp.special(WARP,"lidx0")
    part=LanePartition(lane,lane_extent=lm.lane_extent,words_per_group=lm.words_per_group)
    lblk=UOp.range(lm.blocks_per_group,0,axis_type=AxisType.REDUCE); blk=part.block_group*lm.blocks_per_group+lblk
    bg=(row*lm.k_blocks+blk)*Q4K_WORDS_PER_BLOCK
    cg=_q4k_block_dot_rms_affine(gate_words,x,norm_weight,scale,bg,blk,part.word_col)
    cu=_q4k_block_dot_rms_affine(up_words,x,norm_weight,scale,bg,blk,part.word_col)
    ag=UOp.placeholder((1,),dtypes.float32,20,addrspace=AddrSpace.REG); au=UOp.placeholder((1,),dtypes.float32,21,addrspace=AddrSpace.REG)
    init=ag[0].store(0.0); init=au.after(init)[0].store(0.0); ag,au=ag.after(init),au.after(init)
    ug=ag[0].store(ag.after(lblk)[0]+cg); uu=au.after(ug)[0].store(au.after(lblk)[0]+cu).end(lblk)
    total=(_silu_uop(_warp_reduce_sum_staged(ag.after(uu)[0],part.lane,part.lane_extent,90))*
           _warp_reduce_sum_staged(au.after(uu)[0],part.lane,part.lane_extent,95))
    result=total.cast(dtypes.float16) if store_fp16 else total
    return out[row].store(result).sink(arg=KernelInfo(name=name,opts_to_apply=()))
  return kernel


def _silu_uop(val):
  """SiLU in the exact lowering Tensor.silu uses (tinygrad/mixin/elementwise.py:786):
  self * (1 + (self * (-1/math.log(2))).exp2()).reciprocal(). Mirroring the expression
  op-for-op (including Ops.RECIPROCAL) keeps the fused prelude bit-identical to the
  legacy graph's silu kernels, so the decode sha256 cannot move from rounding deltas."""
  one = UOp.const(dtypes.float32, 1.0)
  log2e_neg = UOp.const(dtypes.float32, -1.0 / math.log(2))
  return val * (one + (val * log2e_neg).exp2()).reciprocal()


@dataclass(frozen=True)
class Q4KGEMVEpilogue:
  """Optional epilogue fused into a Q4_K G3 GEMV kernel (l1-decode-plumbing-fusion-design-20260802.md
  section 2.1). Default kind="" means the legacy kernel (byte-identical UOps). Fused variants get NEW
  kernel names so legacy hashes are untouched. The three variants match the design doc's census
  absorbable classes: (a) o-proj residual add, (b) ffn_down silu(gate)*up prelude + h+ffn_out
  residual epilogue, (c) k/v fp16 cast write, and the M2b ffn_down residual add alone
  (``ffn_down_resadd``: total + h[row] stored fp32, absorbing the standalone h+ffn_out add)."""
  kind: str = ""  # "", "residual_add", "ffn_down_fused", "ffn_down_resadd", "fp16_cast"

  @property
  def kernel_suffix(self) -> str:
    if self.kind == "": return ""
    if self.kind == "residual_add": return "_epi_resadd"
    if self.kind == "ffn_down_fused": return "_epi_ffndown"
    if self.kind == "ffn_down_resadd": return "_epi_ffnresadd"
    if self.kind == "fp16_cast": return "_epi_f16cast"
    raise ValueError(f"unknown Q4K GEMV epilogue kind {self.kind!r}")

  def validate(self, rows: int, k: int) -> None:
    if self.kind == "": return
    if self.kind not in ("residual_add", "ffn_down_fused", "ffn_down_resadd", "fp16_cast"):
      raise ValueError(f"unsupported Q4K GEMV epilogue kind {self.kind!r}")
    if self.kind in ("ffn_down_fused", "ffn_down_resadd") and rows != 4096:
      raise ValueError(f"{self.kind} epilogue requires rows=4096, got rows={rows}")


def q4k_g3_lanemap_gemv_kernel(rows:int, k:int, lanes:int=WARP, epilogue:Q4KGEMVEpilogue|None=None,
                               load_style:str="scalar"):
  """Lower the search-selected G3 lane map. When epilogue is None, the emitted UOps are byte-identical
  to the legacy kernel. Fused variants get NEW kernel names (e.g. q4k_g3_lanemap_gemv_epi_resadd_...)
  so legacy hashes are untouched (pg3 guarantee).

  `load_style="quad"` is a research-only quad-u128-smem spelling for the single-projection FFN-down
  emitter: 16 rows/block x 8 lanes/row, weights loaded as pure uint4, x staged to shared memory once
  per launch and read in-loop as uint4, 3-step XOR ladder 4/2/1 cross-lane reduce over the 8 row lanes
  (the w1w3 quad geometry, single weight buffer). It renders under its own `q4k_g3_lanemap_gemv_quad_*`
  name so legacy hashes are untouched, and is admitted only by an explicit harness lease in
  decode_routes, never by production (which stays on the byte-identical `scalar` default).

  `load_style="vector"` keeps the installed 32-lane/1-row-per-block geometry but widens the global
  loads: the four header words load as one uint4, each qpack word loads once for its two groups, and
  the four per-group fp16 activations load as one half4 (the same `_q4k_block_dot_packed_load_vec`
  spelling as the w1w3 kernel). The per-lane accumulation order is unchanged, so the result is
  bit-identical to `scalar`; it renders under its own `q4k_g3_lanemap_gemv_vec_*` name. The
  `ffn_down_fused` epilogue reads activations instead of x and keeps the scalar inner loop."""
  epi = epilogue or Q4KGEMVEpilogue()
  epi.validate(rows, k)
  lm = Q4KGateUpLaneMap(k=k, n=rows, lane_extent=lanes)
  lm.validate()
  name = f"q4k_g3_lanemap_gemv{epi.kernel_suffix}_{rows}_{k}"

  if load_style == "quad":
    if epi.kind == "ffn_down_fused":
      raise ValueError("quad Q4 GEMV style does not support the ffn_down_fused epilogue")
    rows_per_block, lanes_per_row = 16, 8
    if rows % rows_per_block != 0:
      raise ValueError(f"quad Q4 GEMV style requires rows % {rows_per_block} == 0, got rows={rows}")
    name = f"q4k_g3_lanemap_gemv_quad{epi.kernel_suffix}_{rows}_{k}"
    blocks, threads = rows // rows_per_block, rows_per_block * lanes_per_row

    def kernel(out:UOp, words:UOp, x:UOp, *extra:UOp) -> UOp:
      block = UOp.special(blocks, "gidx0")
      lane = UOp.special(threads, "lidx0")
      lane8 = lane.bitwise_and(UOp.const(dtypes.weakint, 7))
      row_local = lane.rshift(UOp.const(dtypes.weakint, 3))
      row = block.mul(UOp.const(dtypes.weakint, rows_per_block)) + row_local
      bg = lane8.rshift(UOp.const(dtypes.weakint, 1))
      # wc-quad offset in words; MUL form stays a 16B-aligned uint4 load.
      wc0 = lane8.bitwise_and(UOp.const(dtypes.weakint, 1)).mul(UOp.const(dtypes.weakint, 4))
      # x staged once per launch (24 KB fp16 halves at k=12288), read in-loop as uint4.
      xsh = UOp.placeholder((k,), dtypes.float16, 22, addrspace=AddrSpace.LOCAL)
      stage = UOp.range(k // (threads * 4), 0, axis_type=AxisType.REDUCE)
      xoff = (stage.mul(UOp.const(dtypes.weakint, threads)) + lane).mul(UOp.const(dtypes.weakint, 4))
      xvec = x.index(xoff).load(dtype=dtypes.float16.vec(4))
      xstore = xsh.index(xoff).store(xvec)
      barrier = UOp.barrier(UOp.group(xstore.end(stage)))

      acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
      acc = acc.after(acc[0].store(0.0))
      b0 = UOp.range(lm.blocks_per_group, 1, axis_type=AxisType.REDUCE)
      blk = bg.mul(UOp.const(dtypes.weakint, lm.blocks_per_group)) + b0
      base = (row.mul(UOp.const(dtypes.weakint, lm.k_blocks)) + blk).mul(UOp.const(dtypes.weakint, Q4K_WORDS_PER_BLOCK))
      hdr = words.index(base).load(dtype=dtypes.uint32.vec(4))
      contrib = UOp.const(dtypes.float32, 0.0)
      for g2 in range(4):
        qw = words.index(base + UOp.const(dtypes.weakint, 4 + g2 * 8) + wc0).load(dtype=dtypes.uint32.vec(4))
        for gp in range(2):
          grp = 2 * g2 + gp
          d, dmin, sc, mn = _q4k_group_params_from_words(hdr.gep(0), hdr.gep(1), hdr.gep(2), hdr.gep(3), grp)
          xbase = blk.mul(UOp.const(dtypes.weakint, Q4_K_BLOCK_ELEMS)) + UOp.const(dtypes.weakint, grp * 32) + \
            lane8.bitwise_and(UOp.const(dtypes.weakint, 1)).mul(UOp.const(dtypes.weakint, 16))
          for wc in range(4):
            qpack = qw.gep(wc).rshift((grp % 2) * 4).bitwise_and(0x0F0F0F0F)
            for nib in range(4):
              qv = qpack.rshift(nib * 8).bitwise_and(0xf)
              weight = d * sc.cast(dtypes.float32) * qv.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)
              xvv = _f16x4_lane(xsh.after(barrier).index(xbase + UOp.const(dtypes.weakint, wc * 4), ptr=True), nib)
              contrib = contrib + weight * xvv
      upd = acc[0].store(acc.after(b0)[0] + contrib).end(b0)
      total = _warp_reduce_sum_staged(acc.after(upd)[0], lane8, lanes_per_row, 90)

      if epi.kind in ("residual_add", "ffn_down_resadd"):
        result = total + extra[0][row].cast(dtypes.float32)
      elif epi.kind == "fp16_cast":
        result = total.cast(dtypes.float16)
      else:
        result = total

      return out[row].store(result, lane8.eq(0)).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    return kernel

  elif load_style == "vector":
    name = f"q4k_g3_lanemap_gemv_vec{epi.kernel_suffix}_{rows}_{k}"
  elif load_style != "scalar":
    raise ValueError(f"unknown Q4 GEMV load style {load_style!r}")

  if epi.kind == "ffn_down_fused":
    def kernel(out:UOp, words:UOp, gate_out:UOp, up_out:UOp, normed_h:UOp) -> UOp:
      row, lane = UOp.special(rows, "gidx0"), UOp.special(lanes, "lidx0")
      part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
      lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
      blk = part.block_group * lm.blocks_per_group + lblk
      base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
      contrib = UOp.const(dtypes.float32, 0.0)
      for grp in range(8):
        d, dmin, sc, mn = _q4k_group_params(words, base, grp)
        qpack = words[base + 4 + (grp//2)*8 + part.word_col].rshift((grp%2)*4).bitwise_and(0x0F0F0F0F)
        for nib in range(4):
          pos = part.word_col * 4 + nib
          q = qpack.rshift(nib*8).bitwise_and(0xf)
          weight = d * sc.cast(dtypes.float32) * q.cast(dtypes.float32) - dmin * mn.cast(dtypes.float32)
          idx = blk*Q4_K_BLOCK_ELEMS + grp*32 + pos
          g = gate_out[idx].cast(dtypes.float32)
          u = up_out[idx].cast(dtypes.float32)
          activation = _silu_uop(g) * u
          contrib = contrib + weight * activation
      acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
      acc = acc.after(acc[0].store(0.0))
      acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
      total = _lane_partition_reduce_sum(acc[0], part)
      return out[row].store(total + normed_h[row].cast(dtypes.float32)).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    return kernel

  if load_style == "vector":
    def kernel(out:UOp, words:UOp, x:UOp, *extra:UOp) -> UOp:
      row, lane = UOp.special(rows, "gidx0"), UOp.special(lanes, "lidx0")
      part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
      lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
      blk = part.block_group * lm.blocks_per_group + lblk
      base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
      contrib = _q4k_block_dot_packed_load_vec(words, x, base, blk, part.word_col)
      acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
      acc = acc.after(acc[0].store(0.0))
      acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
      total = _lane_partition_reduce_sum(acc[0], part)

      if epi.kind in ("residual_add", "ffn_down_resadd"):
        result = total + extra[0][row].cast(dtypes.float32)
      elif epi.kind == "fp16_cast":
        result = total.cast(dtypes.float16)
      else:
        result = total

      return out[row].store(result).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    return kernel

  def kernel(out:UOp, words:UOp, x:UOp, *extra:UOp) -> UOp:
    row, lane = UOp.special(rows, "gidx0"), UOp.special(lanes, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load(words, x, base, blk, part.word_col)
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    acc = acc.after(acc[0].store(acc.after(lblk)[0] + contrib).end(lblk))
    total = _lane_partition_reduce_sum(acc[0], part)

    if epi.kind in ("residual_add", "ffn_down_resadd"):
      result = total + extra[0][row].cast(dtypes.float32)
    elif epi.kind == "fp16_cast":
      result = total.cast(dtypes.float16)
    else:
      result = total

    return out[row].store(result).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def _q4k_group_params_from_words(w0:UOp, w1:UOp, w2:UOp, w3:UOp, grp:int) -> tuple[UOp, UOp, UOp, UOp]:
  """Header-words variant of `_q4k_group_params`: the four header words are already materialized
  (e.g. lanes of one uint4 header load), so no re-load of the block header is needed per group."""
  d, dmin = _f16_word(w0, False), _f16_word(w0, True)
  def scale_byte(word:UOp, idx:int) -> UOp: return word.rshift(idx % 4 * 8).bitwise_and(0xff)
  if grp < 4:
    sc, mn = scale_byte(w1, grp).bitwise_and(63), scale_byte(w2, grp).bitwise_and(63)
  else:
    high = scale_byte(w3, grp - 4)
    sc = high.bitwise_and(0xf).bitwise_or(scale_byte(w1, grp - 4).rshift(6).lshift(4))
    mn = high.rshift(4).bitwise_or(scale_byte(w2, grp - 4).rshift(6).lshift(4))
  return d, dmin, sc, mn


def _f16x4_lane(xidx:UOp, nib:int) -> UOp:
  """Project one fp32 lane of a half.vec(4) shared-memory read. GEP on float vector loads is
  spec-illegal (only int-typed vector loads may be lane-extracted), so the projection rides the
  CUSTOMI inline-op mechanism (the same family as the fdot2/exp2f providers): the rendered text is
  a 16-byte `half4` load plus a register sub-access, which nvcc folds into one LDS and costs no
  extra instruction. The source is the pointer INDEX (ptr=True, so the add-loads pass leaves it
  alone); the CUSTOMI itself carries the scalar shape."""
  return UOp(Ops.CUSTOMI, dtypes.float32, (xidx,), arg=f"float((*((half4*){{0}})).{'xyzw'[nib]})")


def q4k_g3_lanemap_gemv_w1w3_kernel(rows:int, k:int, load_style:str = "scalar", store_fp16:bool = False):
  """Fused gate/up (w1+w3) decode GEMV: ONE 12288-row kernel computes
  `out[r] = silu(dot(gate_row_r, x)) * dot(up_row_r, x)` from two weight buffers, replacing the
  gate GEMV + silu elementwise + up GEMV + mul elementwise chain (72 -> 36 kernels/token).
  Silu uses `_silu_uop`, the exact Tensor.silu lowering, so the fused prelude is bit-identical to
  the legacy graph's silu kernels when the per-row dot totals are bit-identical (the `scalar` style
  reproduces the installed per-lane accumulation order; `quad` does not).

  `load_style="scalar"` is the MC3 probe shape (mc3-w1w3-fusion-measurement-record-20260803.md): the
  installed 32-lane/1-row-per-block map with the installed per-lane accumulation order, token-safe by
  construction. This is the LANDED in-loop shape (q4k-w1w3-fused-qv-implementation-record-20260803.md):
  same-session d512 census 39.36-39.4 us in-loop vs the pair's 2 x 20.83 = 41.7 us, +1.7-2% wall tok/s.
  `load_style="quad"` is the MC2 measured load pattern (mc2-load-pattern-measurement-record-20260803.md
  section 7): 768 blocks x 128 threads at 12288x4096 (16 rows/block, 8 lanes/row), each lane owns the
  wc-quad `(lane&1)*4` in group-of-2 `lane>>1`, weights load as pure uint4 (5 LDG.128 per block per
  projection), x is staged to shared memory once per launch (8 KB, outside the reduce loop) and read
  in-loop as uint4, and the cross-lane reduce is the 3-step XOR ladder 4/2/1 over the 8 row lanes.
  The quad style is the standalone optimum (22.2 us vs scalar 23.2 us) but REGRESSES in-loop (49.2 us
  census, -5% wall): measured NO-GO for the real loop, kept only for standalone/occupancy study.
  Both styles share the two-accumulator loop pattern proven in production by the flash decode kernel
  (one init chain, first update `.after()` without `.end`, last update ends the range; distinct REG
  slots 20/21 and distinct shuffle staging bases 90-94/95-99).
  `store_fp16=True` (scalar style only) stores the fused result already cast to fp16 under its own
  `q4k_g3_lanemap_gemv_w1w3fused16_*` name, so a consumer's fp32->fp16 cast of the output (the
  ordinary `E_128_32_3` ffn-activation cast) folds away. The in-kernel cast is the same
  round-to-nearest-even fp32->fp16 conversion the separate cast kernel lowers, so the stored bytes
  are bitwise-identical; the legacy fp32 name and hash are unchanged when `store_fp16=False`."""
  lm = Q4KGateUpLaneMap(k=k, n=rows)
  lm.validate()
  if load_style == "quad":
    if store_fp16:
      raise ValueError("quad w1w3 style does not support store_fp16")
    rows_per_block, lanes_per_row = 16, 8
    if rows % rows_per_block != 0:
      raise ValueError(f"quad w1w3 style requires rows % {rows_per_block} == 0, got rows={rows}")
    if lm.blocks_per_group != 4:
      raise ValueError(f"quad w1w3 style requires blocks_per_group == 4 (k_blocks % 4 == 0), got {lm.blocks_per_group}")
    name = f"q4k_g3_lanemap_gemv_w1w3qv_{rows}_{k}"
  elif load_style == "scalar":
    rows_per_block, lanes_per_row = 1, WARP
    name = f"q4k_g3_lanemap_gemv_w1w3fused16_{rows}_{k}" if store_fp16 else f"q4k_g3_lanemap_gemv_w1w3fused_{rows}_{k}"
  elif load_style == "vector":
    rows_per_block, lanes_per_row = 1, WARP
    name = f"q4k_g3_lanemap_gemv_w1w3vec16_{rows}_{k}" if store_fp16 else f"q4k_g3_lanemap_gemv_w1w3vec_{rows}_{k}"
  else:
    raise ValueError(f"unknown w1w3 load style {load_style!r}")
  blocks, threads = rows // rows_per_block, rows_per_block * lanes_per_row

  if load_style == "quad":
    def kernel(out:UOp, gate_words:UOp, up_words:UOp, x:UOp) -> UOp:
      block = UOp.special(blocks, "gidx0")
      lane = UOp.special(threads, "lidx0")
      lane8 = lane.bitwise_and(UOp.const(dtypes.weakint, 7))
      row_local = lane.rshift(UOp.const(dtypes.weakint, 3))
      row = block.mul(UOp.const(dtypes.weakint, rows_per_block)) + row_local
      bg = lane8.rshift(UOp.const(dtypes.weakint, 1))
      # wc-quad offset in words; MUL form is loadable as one 16B-aligned uint4 (the devectorizer
      # keeps vector loads only when the offset provably divides the fold width).
      wc0 = lane8.bitwise_and(UOp.const(dtypes.weakint, 1)).mul(UOp.const(dtypes.weakint, 4))
      # x staged once per launch: 8 KB of fp16 halves in shared memory (LOCAL half4 loads/stores
      # keep the 8-byte fold; the in-loop reads bitcast to u32 pairs so GEP lane extraction stays
      # spec-legal without ever GEP'ing a float vector load).
      xsh = UOp.placeholder((k,), dtypes.float16, 22, addrspace=AddrSpace.LOCAL)
      stage = UOp.range(k // (threads * 4), 0, axis_type=AxisType.REDUCE)
      xoff = (stage.mul(UOp.const(dtypes.weakint, threads)) + lane).mul(UOp.const(dtypes.weakint, 4))
      xvec = x.index(xoff).load(dtype=dtypes.float16.vec(4))
      xstore = xsh.index(xoff).store(xvec)
      barrier = UOp.barrier(UOp.group(xstore.end(stage)))

      acc_g = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
      acc_u = UOp.placeholder((1,), dtypes.float32, 21, addrspace=AddrSpace.REG)
      init = acc_g[0].store(0.0)
      init = acc_u.after(init)[0].store(0.0)
      acc_g, acc_u = acc_g.after(init), acc_u.after(init)
      b0 = UOp.range(lm.blocks_per_group, 1, axis_type=AxisType.REDUCE)
      blk = bg.mul(UOp.const(dtypes.weakint, 4)) + b0
      base_g = (row.mul(UOp.const(dtypes.weakint, lm.k_blocks)) + blk).mul(UOp.const(dtypes.weakint, Q4K_WORDS_PER_BLOCK))
      base_u = (row.mul(UOp.const(dtypes.weakint, lm.k_blocks)) + blk).mul(UOp.const(dtypes.weakint, Q4K_WORDS_PER_BLOCK))
      hdr_g = gate_words.index(base_g).load(dtype=dtypes.uint32.vec(4))
      hdr_u = up_words.index(base_u).load(dtype=dtypes.uint32.vec(4))
      contrib_g = UOp.const(dtypes.float32, 0.0)
      contrib_u = UOp.const(dtypes.float32, 0.0)
      for g2 in range(4):
        qg = gate_words.index(base_g + UOp.const(dtypes.weakint, 4 + g2 * 8) + wc0).load(dtype=dtypes.uint32.vec(4))
        qu = up_words.index(base_u + UOp.const(dtypes.weakint, 4 + g2 * 8) + wc0).load(dtype=dtypes.uint32.vec(4))
        for gp in range(2):
          grp = 2 * g2 + gp
          dg, dming, scg, mng = _q4k_group_params_from_words(hdr_g.gep(0), hdr_g.gep(1), hdr_g.gep(2), hdr_g.gep(3), grp)
          du, dminu, scu, mnu = _q4k_group_params_from_words(hdr_u.gep(0), hdr_u.gep(1), hdr_u.gep(2), hdr_u.gep(3), grp)
          # 16 halves per group as four half4 smem reads (x read once, shared by both projections).
          xbase = blk.mul(UOp.const(dtypes.weakint, Q4_K_BLOCK_ELEMS)) + UOp.const(dtypes.weakint, grp * 32) + \
            lane8.bitwise_and(UOp.const(dtypes.weakint, 1)).mul(UOp.const(dtypes.weakint, 16))
          for wc in range(4):
            qpack_g = qg.gep(wc).rshift((grp % 2) * 4).bitwise_and(0x0F0F0F0F)
            qpack_u = qu.gep(wc).rshift((grp % 2) * 4).bitwise_and(0x0F0F0F0F)
            for nib in range(4):
              qv_g = qpack_g.rshift(nib * 8).bitwise_and(0xf)
              qv_u = qpack_u.rshift(nib * 8).bitwise_and(0xf)
              weight_g = dg * scg.cast(dtypes.float32) * qv_g.cast(dtypes.float32) - dming * mng.cast(dtypes.float32)
              weight_u = du * scu.cast(dtypes.float32) * qv_u.cast(dtypes.float32) - dminu * mnu.cast(dtypes.float32)
              xvv = _f16x4_lane(xsh.after(barrier).index(xbase + UOp.const(dtypes.weakint, wc * 4), ptr=True), nib)
              contrib_g = contrib_g + weight_g * xvv
              contrib_u = contrib_u + weight_u * xvv
      upd_g = acc_g[0].store(acc_g.after(b0)[0] + contrib_g)
      upd_u = acc_u.after(upd_g)[0].store(acc_u.after(b0)[0] + contrib_u).end(b0)
      total_g = _warp_reduce_sum_staged(acc_g.after(upd_u)[0], lane8, lanes_per_row, 90)
      total_u = _warp_reduce_sum_staged(acc_u.after(upd_u)[0], lane8, lanes_per_row, 95)
      val = _silu_uop(total_g) * total_u
      if store_fp16: val = val.cast(dtypes.float16)
      return out[row].store(val, lane8.eq(0)).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    return kernel

  if load_style == "vector":
    def kernel(out:UOp, gate_words:UOp, up_words:UOp, x:UOp) -> UOp:
      row, lane = UOp.special(rows, "gidx0"), UOp.special(WARP, "lidx0")
      part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
      lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
      blk = part.block_group * lm.blocks_per_group + lblk
      base_g = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
      base_u = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
      contrib_g = _q4k_block_dot_packed_load_vec(gate_words, x, base_g, blk, part.word_col)
      contrib_u = _q4k_block_dot_packed_load_vec(up_words, x, base_u, blk, part.word_col)
      acc_g = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
      acc_u = UOp.placeholder((1,), dtypes.float32, 21, addrspace=AddrSpace.REG)
      init = acc_g[0].store(0.0)
      init = acc_u.after(init)[0].store(0.0)
      acc_g, acc_u = acc_g.after(init), acc_u.after(init)
      upd_g = acc_g[0].store(acc_g.after(lblk)[0] + contrib_g)
      upd_u = acc_u.after(upd_g)[0].store(acc_u.after(lblk)[0] + contrib_u).end(lblk)
      total_g = _warp_reduce_sum_staged(acc_g.after(upd_u)[0], part.lane, part.lane_extent, 90)
      total_u = _warp_reduce_sum_staged(acc_u.after(upd_u)[0], part.lane, part.lane_extent, 95)
      val = _silu_uop(total_g) * total_u
      if store_fp16: val = val.cast(dtypes.float16)
      return out[row].store(val).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    return kernel

  def kernel(out:UOp, gate_words:UOp, up_words:UOp, x:UOp) -> UOp:
    row, lane = UOp.special(rows, "gidx0"), UOp.special(WARP, "lidx0")
    part = LanePartition(lane, lane_extent=lm.lane_extent, words_per_group=lm.words_per_group)
    lblk = UOp.range(lm.blocks_per_group, 0, axis_type=AxisType.REDUCE)
    blk = part.block_group * lm.blocks_per_group + lblk
    base_g = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    base_u = (row * lm.k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib_g = _q4k_block_dot_packed_load(gate_words, x, base_g, blk, part.word_col)
    contrib_u = _q4k_block_dot_packed_load(up_words, x, base_u, blk, part.word_col)
    acc_g = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc_u = UOp.placeholder((1,), dtypes.float32, 21, addrspace=AddrSpace.REG)
    init = acc_g[0].store(0.0)
    init = acc_u.after(init)[0].store(0.0)
    acc_g, acc_u = acc_g.after(init), acc_u.after(init)
    upd_g = acc_g[0].store(acc_g.after(lblk)[0] + contrib_g)
    upd_u = acc_u.after(upd_g)[0].store(acc_u.after(lblk)[0] + contrib_u).end(lblk)
    total_g = _warp_reduce_sum_staged(acc_g.after(upd_u)[0], part.lane, part.lane_extent, 90)
    total_u = _warp_reduce_sum_staged(acc_u.after(upd_u)[0], part.lane, part.lane_extent, 95)
    val = _silu_uop(total_g) * total_u
    if store_fp16: val = val.cast(dtypes.float16)
    return out[row].store(val).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


# Q6_K selected spec-driven lowering and shared decode quant grammar.
Q6K_POS_EXTENT = 16
Q6K_VOCAB_SCALAR_REDUCE_MIN_ROWS = 131072

# Per-target coop row_tile values. This is route-config data, not an emitter branch: the emitter
# below is one shared lowering, and a target without a measured row keeps the safe default (the
# AMD gfx1100 machine-search value). NV:sm_120 measured 2026-08-02 on the RTX 5090 (P1 sweep,
# d512 fixed-depth): vocab coop 397.4 -> 330.1us, down coop 49.7 -> 35.5us, decode tok/s
# 163.2 -> 172.4, token sha unchanged.
Q6K_COOP_ROW_TILE_BY_TARGET: dict[tuple[str, str], int] = {("NV", "sm_120"): 2}
Q6K_COOP_ROW_TILE_DEFAULT = 4


def q6k_coop_row_tile_for_target(backend: str | None, architecture: str | None) -> int:
  """Resolve the coop row_tile for a resolved (backend, architecture) fact pair."""
  return Q6K_COOP_ROW_TILE_BY_TARGET.get((backend, architecture), Q6K_COOP_ROW_TILE_DEFAULT)


def _f16_half(half:UOp) -> UOp: return half.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)


def _q6k_byte(halfs:UOp, base:UOp, byte_idx:UOp|int) -> UOp:
  idx = UOp.const(dtypes.int32, byte_idx) if isinstance(byte_idx, int) else byte_idx
  return halfs[base + idx//2].rshift((idx%2)*8).bitwise_and(0xff)


def _i8(byte:UOp) -> UOp: return byte.cast(dtypes.uint8).bitcast(dtypes.int8).cast(dtypes.float32)


def _q6k_weight(halfs:UOp, base:UOp, grp:int, pos:UOp) -> UOp:
  half, pgrp = grp // 8, grp % 8
  ql_byte_idx, ql_shift = half*64 + (pgrp%4)*16 + pos, 4 if pgrp >= 4 else 0
  qh_byte_idx, qh_shift = 128 + half*32 + (pgrp%2)*16 + pos, (pgrp//2) * 2
  ql = _q6k_byte(halfs, base, ql_byte_idx).rshift(ql_shift).bitwise_and(0xf)
  qh = _q6k_byte(halfs, base, qh_byte_idx).rshift(qh_shift).bitwise_and(0x3).lshift(4)
  q = ql.bitwise_or(qh).cast(dtypes.float32) - UOp.const(dtypes.float32, 32.0)
  return _f16_half(halfs[base + 104]) * q * _i8(_q6k_byte(halfs, base, 192 + grp))


def _q6k_block_dot(halfs:UOp, x:UOp, base:UOp, x_block:UOp, pos:UOp) -> UOp:
  contrib = UOp.const(dtypes.float32, 0.0)
  for grp in range(16):
    contrib = contrib + _q6k_weight(halfs, base, grp, pos) * x[x_block*Q6_K_BLOCK_ELEMS + grp*16 + pos].cast(dtypes.float32)
  return contrib


def _q6k_coop_pos_reduce_sum(val:UOp, lane:UOp, row_tile:int, slot_base:int=90) -> UOp:
  """Cross-lane sum over the coop route's 16 pos lanes. The lane map is row_i-fastest
  (tid = pos*row_tile + row_i), so each ladder step must advance by row_tile lane ids to
  stay within one (row, pos) group; the standard offset ladder (width/2..1) would mix the
  row_i bit on the last step. Requires a single warp: row_tile * Q6K_POS_EXTENT <= 32."""
  off = Q6K_POS_EXTENT >> 1
  while off >= 1:
    val = val + _staged_shfl(val, off * row_tile, lane, slot_base)
    off >>= 1
    slot_base += 1
  return val


def _q6k_coop_tile_key_reduce_max(key:UOp, lane:UOp, slot_base:int=100) -> UOp:
  """Warp-wide MAX of the packed per-row (max, index) key for the vocab_top1 epilogue.

  The pos sum ladder deliberately keeps row_i separate; the top-1 reduce must instead mix
  it, because the tile's ``row_tile`` rows compete for the max.  The standard XOR ladder
  over the full warp covers all 32 lanes (row_tile * pos <= 32 is enforced by the
  in_kernel validation), and every lane ends with the tile's best packed key.
  """
  off = WARP >> 1
  while off >= 1:
    key = key.maximum(_staged_shfl(key, off, lane, slot_base))
    off >>= 1
    slot_base += 1
  return key


@dataclass(frozen=True)
class DecodeRMSNormSpec:
  """One fused decode RMSNorm kernel: per-row sumsq reduce + `x * rsqrt(sumsq/dim + eps) * w`
  epilogue in a single UOp builder (l1-decode-plumbing-fusion-design-20260802.md section 6,
  norm family). One warp per row; the epilogue reuses the legacy graph's exact ops (DIV by dim,
  SQRT, RECIPROCAL, then (x*scale)*w) so only the sumsq summation ORDER differs from the generic
  reduce, and that delta is gated by the fixed-depth token sha like every other fused variant."""
  rows: int
  dim: int
  eps: float
  lane_width: int = 32
  warps_per_row: int = 1
  x_dtype: DType = dtypes.float32
  weight_dtype: DType = dtypes.float32
  out_dtype: DType = dtypes.float32
  x_rank: int = 1
  target: str = "amd_gfx1100"
  # Path 3 semantic lowering names the kernel after its scheduler-owned
  # boundary (`rmsnorm_native_*`); the M3 opaque path keeps `decode_rmsnorm_*`
  # byte-identical (pg3 pins: 2f3b80f7b426 / 9cf696d384ba / 061dd2e554d0).
  native: bool = False
  # Research discriminator: retain each lane's reduction inputs for the
  # epilogue instead of issuing a second activation-buffer read. Default-off
  # until a complete native-NV resource/timing/wall gate admits it.
  retain_input: bool = False

  @property
  def kernel_name(self) -> str:
    return f"rmsnorm_native_{self.rows}_{self.dim}" if self.native else f"decode_rmsnorm_{self.rows}_{self.dim}"

  def validate(self) -> None:
    if self.rows < 1: raise ValueError(f"DecodeRMSNormSpec requires rows>=1, got {self.rows}")
    if self.dim < self.lane_width or self.dim % self.lane_width != 0:
      raise ValueError(f"DecodeRMSNormSpec requires dim >= lane_width and dim % lane_width == 0, "
                       f"got dim={self.dim} lane_width={self.lane_width}")
    if self.warps_per_row < 1 or self.dim % (self.lane_width * self.warps_per_row) != 0:
      raise ValueError(f"DecodeRMSNormSpec requires dim % (lane_width * warps_per_row) == 0, got "
                       f"dim={self.dim} lane_width={self.lane_width} warps_per_row={self.warps_per_row}")
    if not isinstance(self.eps, float) or self.eps <= 0: raise ValueError(f"DecodeRMSNormSpec requires eps>0, got {self.eps!r}")
    if self.x_rank not in (1, 2, 3): raise ValueError(f"DecodeRMSNormSpec requires x_rank in (1, 2, 3), got {self.x_rank}")
    if self.out_dtype not in (dtypes.float16, dtypes.float32):
      raise ValueError(f"DecodeRMSNormSpec requires out_dtype float16/float32, got {self.out_dtype}")


def emit_decode_rmsnorm_kernel(spec:DecodeRMSNormSpec):
  spec.validate()
  rows, dim, lane, warps = spec.rows, spec.dim, spec.lane_width, spec.warps_per_row
  per_lane = dim // (lane * warps)
  dim_f = UOp.const(dtypes.float32, float(dim))
  eps = UOp.const(dtypes.float32, spec.eps)
  def kernel(out:UOp, x:UOp, w:UOp) -> UOp:
    row = UOp.range(rows, 0)
    laneid = UOp.range(lane, 1, axis_type=AxisType.LOCAL)
    warp = UOp.range(warps, 2, axis_type=AxisType.LOCAL)
    red = UOp.range(per_lane, 3, axis_type=AxisType.REDUCE)
    base = row * dim + warp * (per_lane * lane) + laneid + red * lane
    # The input is passed as a flat (numel,) view and the custom-kernel transport
    # contiguous()s it into the buffer this flat base indexes. That per-call
    # materialization is the measured reason the norm-fusion route is closed-default
    # non-landing (m3-fused-norm-measurement-record-20260802.md). Rank 3 inputs
    # (B,T,dim activations) index the two leading extent-1 dims; rank 1 inputs
    # (q/k slices) index the flat base directly.
    # The input param mirrors the producer's logical activation view (rank 1
    # flat, rank 2/3 leading unit axes) and is indexed with the flat base, so
    # the scheduler can bind the producer's buffer through a contiguous view
    # instead of materializing a copy. Rank-3 is the decode (1,1,dim) shape;
    # rank-2 is the (1,dim) shape used by the isolation probes.
    x_sel = x[UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0), base] if spec.x_rank == 3 else \
            x[UOp.const(dtypes.int, 0), base] if spec.x_rank == 2 else x[base]
    acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    acc = acc.after(acc[0].store(0.0))
    retained = None
    if spec.retain_input:
      retained = UOp.placeholder((per_lane,), spec.x_dtype, 21, addrspace=AddrSpace.REG)
      retained_ready = UOp.group(*[retained[i].store(x[row * dim + warp * (per_lane * lane) + laneid + i * lane]) for i in range(per_lane)])
      xv = retained.after(retained_ready)[red].cast(dtypes.float32)
    else:
      xv = x_sel.cast(dtypes.float32)
    acc = acc.after(acc[0].store(acc.after(red)[0] + xv * xv).end(red))
    warp_total = _warp_reduce_sum_staged(acc[0], laneid, lane, slot_base=90)
    if warps > 1:
      smem = UOp.placeholder((warps,), dtypes.float32, 230, addrspace=AddrSpace.LOCAL)
      wstore = smem[warp].store(warp_total, laneid.eq(0))
      barrier = UOp.barrier(UOp.group(wstore))
      total = UOp.const(dtypes.float32, 0.0)
      for wi in range(warps):
        total = total + smem.after(barrier)[wi]
    else:
      total = warp_total
    scale = UOp(Ops.RECIPROCAL, dtypes.float32, (UOp(Ops.SQRT, dtypes.float32, (total / dim_f + eps,)),))
    epi = UOp.range(per_lane, 3)
    obase = row * dim + warp * (per_lane * lane) + laneid + epi * lane
    wv = w[warp * (per_lane * lane) + laneid + epi * lane].cast(dtypes.float32)
    x_epi = retained.after(retained_ready)[epi] if retained is not None else \
            x[UOp.const(dtypes.int, 0), UOp.const(dtypes.int, 0), obase] if spec.x_rank == 3 else x[obase]
    # The Path 3 semantic marker's fallback is the ordinary nn.RMSNorm graph,
    # which rounds `(x * rsqrt) ` through x.dtype BEFORE the fp32 weight
    # multiply. The native epilogue replicates that intermediate cast so the
    # lowering and its fallback share one value definition (isolation parity);
    # the M3 opaque epilogue (native=False) is untouched and byte-identical.
    if spec.native:
      normed = (x_epi.cast(dtypes.float32) * scale).cast(spec.x_dtype)
    else:
      normed = (x_epi.cast(dtypes.float32) * scale)
    return out[obase].store((normed * wv).cast(spec.out_dtype)).end(row, laneid, warp, epi).sink(
      arg=KernelInfo(name=spec.kernel_name, opts_to_apply=()))
  return kernel


@dataclass(frozen=True)
class Q6KGEMVRouteSpec:
  rows: int
  k: int
  role: str = ""
  route_family: str = "q6k_coop"
  target: str = "amd_gfx1100"
  row_tile: int = 4
  lane_extent: int = Q6K_POS_EXTENT
  parts: int = 1
  pos_axis: str = "local"
  block_axis: str = "reduce"
  reduction: str = "external_sum"
  epilogue: str = ""  # "", "ffn_down_resadd" (M2b: total + h[row] in-kernel, fp32 store),
                     # "vocab_top1" (P1: per-tile packed (max, index) key + in-kernel warp reduce)
  storage: str = "packed_u16"
  accumulators: int = 1
  quant: QuantFormat = Q6_K
  opts: tuple = field(default_factory=tuple)

  @property
  def k_blocks(self) -> int: return self.k // Q6_K_BLOCK_ELEMS

  @property
  def partial_axis_extent(self) -> int: return self.lane_extent if self.route_family == "q6k_coop" else self.parts

  @property
  def kernel_name(self) -> str:
    suffix = "_inkernel" if self.reduction == "in_kernel" else ""
    if self.epilogue == "ffn_down_resadd": suffix += "_epi_ffnresadd"
    if self.epilogue == "vocab_top1": suffix += "_epi_vocabtop1"
    acc_suffix = f"_nacc{self.accumulators}" if self.accumulators != 1 else ""
    return ((f"q6k_gen_coop_{self.rows}_{self.k}" if self.route_family == "q6k_coop"
            else f"q6k_gen_partial_{self.rows}_{self.k}_{self.parts}") + suffix + acc_suffix)

  def validate(self) -> None:
    if self.quant is not Q6_K: raise ValueError(f"Q6KGEMVRouteSpec quant must be Q6_K, got {self.quant!r}")
    if self.route_family not in ("q6k_coop", "q6k_partial"): raise ValueError(f"unknown route_family {self.route_family!r}")
    if self.reduction not in ("external_sum", "in_kernel"): raise ValueError(f"unsupported reduction {self.reduction!r}")
    if self.epilogue not in ("", "ffn_down_resadd", "vocab_top1"):
      raise ValueError(f"unsupported epilogue {self.epilogue!r}")
    if self.reduction == "in_kernel" and self.route_family == "q6k_partial":
      raise ValueError("in_kernel reduction is not implemented for the q6k_partial family (M2 non-landing, "
                       "l1-decode-plumbing-fusion-design-20260802.md section 6 class 9); use external_sum")
    if self.epilogue == "ffn_down_resadd":
      if self.reduction != "in_kernel": raise ValueError("ffn_down_resadd epilogue requires in_kernel reduction")
      if self.rows != 4096: raise ValueError(f"ffn_down_resadd epilogue requires rows=4096, got rows={self.rows}")
    if self.epilogue == "vocab_top1":
      if self.reduction != "in_kernel": raise ValueError("vocab_top1 epilogue requires in_kernel reduction")
      if self.route_family != "q6k_coop": raise ValueError("vocab_top1 epilogue requires the coop route family")
    if self.storage != "packed_u16": raise ValueError(f"unsupported storage {self.storage!r}")
    if self.accumulators not in (1, 2, 4): raise ValueError(f"unsupported accumulators={self.accumulators}")
    if self.k_blocks % self.accumulators != 0: raise ValueError("k_blocks must divide evenly across accumulators")
    if self.k % Q6_K_BLOCK_ELEMS != 0: raise ValueError(f"k={self.k} must be a multiple of {Q6_K_BLOCK_ELEMS}")
    if self.lane_extent != Q6K_POS_EXTENT: raise ValueError(f"lane_extent must be {Q6K_POS_EXTENT}, got {self.lane_extent}")
    if self.route_family == "q6k_coop":
      if self.pos_axis != "local": raise ValueError("coop route requires pos_axis=local")
      if self.row_tile < 1 or self.rows % self.row_tile != 0:
        raise ValueError(f"coop route requires rows({self.rows}) % row_tile({self.row_tile}) == 0")
      if self.reduction == "in_kernel" and self.row_tile * self.lane_extent > 32:
        raise ValueError("coop in_kernel reduce requires a single warp: row_tile * lane_extent <= 32, got "
                         f"row_tile={self.row_tile} lane_extent={self.lane_extent}")
    else:
      if self.pos_axis != "reduce": raise ValueError("partial route requires pos_axis=reduce")
      if self.parts < 1: raise ValueError(f"partial route requires parts>=1, got {self.parts}")

  def to_json(self) -> dict[str, Any]:
    return {"quant": self.quant.name, "rows": self.rows, "k": self.k, "role": self.role, "route_family": self.route_family,
            "target": self.target, "row_tile": self.row_tile, "lane_extent": self.lane_extent, "parts": self.parts,
            "pos_axis": self.pos_axis, "block_axis": self.block_axis, "reduction": self.reduction,
            "epilogue": self.epilogue, "storage": self.storage, "accumulators": self.accumulators}


def q6k_spec_for_role(rows:int, k:int, *, role:str="", parts:int=1, row_tile:int=4, use_coop:bool=True,
                      target:str="amd_gfx1100", opts:tuple=(), reduction:str="external_sum",
                      epilogue:str="", accumulators:int=1) -> Q6KGEMVRouteSpec:
  if use_coop and parts == 1:
    return Q6KGEMVRouteSpec(rows=rows, k=k, role=role, route_family="q6k_coop", row_tile=row_tile,
                            pos_axis="local", target=target, reduction=reduction, epilogue=epilogue, accumulators=accumulators)
  return Q6KGEMVRouteSpec(rows=rows, k=k, role=role, route_family="q6k_partial", parts=parts,
                          pos_axis="reduce", target=target, opts=opts, reduction=reduction, epilogue=epilogue, accumulators=accumulators)


def emit_q6k_gemv_kernel(spec:Q6KGEMVRouteSpec):
  spec.validate()
  return _emit_q6k_coop(spec) if spec.route_family == "q6k_coop" else _emit_q6k_partial(spec)


def q6k_vocab_scalar_reduce_eligible(spec:Q6KGEMVRouteSpec) -> bool:
  return spec.route_family == "q6k_coop" and spec.rows >= Q6K_VOCAB_SCALAR_REDUCE_MIN_ROWS and spec.partial_axis_extent == Q6K_POS_EXTENT


def emit_q6k_vocab_scalar_reduce_kernel(spec:Q6KGEMVRouteSpec):
  if not q6k_vocab_scalar_reduce_eligible(spec): raise ValueError(f"Q6_K scalar vocab reduction is not admitted for {spec.to_json()}")
  def kernel(out:UOp, partials:UOp) -> UOp:
    row = UOp.range(spec.rows, 0)
    pos = UOp.range(spec.partial_axis_extent, 1, axis_type=AxisType.REDUCE)
    return out[row].store(partials[row, pos].reduce(pos, arg=Ops.ADD)).end(row).sink(
      arg=KernelInfo(name=f"q6k_vocab_scalar_reduce_{spec.rows}_{spec.k}", opts_to_apply=()))
  return kernel


def emit_q6k_vocab_top1_reduce_kernel(spec:Q6KGEMVRouteSpec):
  """Final packed reduce for the vocab_top1 fused epilogue (P1 aux scatter-chain fusion).

  The vocab GEMV's vocab_top1 epilogue already carried one packed u64 (max, index) key per
  warp tile; this tiny kernel turns the per-tile keys into the first-index token id with one
  u64 MAX over the tile axis (the cross-tile reduce), then unpacks ``rows-1-(key & 0xffffffff)``.
  Tie semantics match today's r_16_8 Tensor.argmax chain and packed_argmax_finite_fp32: the
  inverted row index in the low half breaks equal logits to the FIRST row.  No float cast
  participates in the compare (BITCAST + integer ops only), so no float rounding can reorder ties.
  """
  if spec.epilogue != "vocab_top1": raise ValueError(f"vocab_top1 reduce requires the vocab_top1 epilogue, got {spec.to_json()}")
  def kernel(out:UOp, keys:UOp) -> UOp:
    tile = UOp.range(cdiv(spec.rows, spec.row_tile), 0, axis_type=AxisType.REDUCE)
    best = keys[tile].reduce(tile, arg=Ops.MAX)
    token = (UOp.const(dtypes.uint64, spec.rows - 1) - best.bitwise_and(0xffffffff)).cast(dtypes.int32)
    return out[0].store(token).sink(arg=KernelInfo(name=f"q6k_vocab_top1_reduce_{spec.rows}_{spec.k}", opts_to_apply=()))
  return kernel


def _emit_q6k_coop(spec:Q6KGEMVRouteSpec):
  rows, row_tile, k_blocks, name = spec.rows, spec.row_tile, spec.k_blocks, spec.kernel_name
  in_kernel = spec.reduction == "in_kernel"
  def kernel(partials:UOp, halfs:UOp, x:UOp, h:UOp|None=None) -> UOp:
    row_o = UOp.range(cdiv(rows, row_tile), 0)
    row_i = UOp.range(row_tile, 1, axis_type=AxisType.LOCAL)
    pos = UOp.range(Q6K_POS_EXTENT, 2, axis_type=AxisType.LOCAL)
    blk = UOp.range(k_blocks, 3, axis_type=AxisType.REDUCE)
    row, base = row_o * row_tile + row_i, ((row_o * row_tile + row_i) * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = _q6k_block_dot(halfs, x, base, blk, pos)
    if in_kernel:
      if spec.accumulators == 1:
        acc = UOp.placeholder((1,), dtypes.float32, 20, addrspace=AddrSpace.REG)
        acc = acc.after(acc[0].store(0.0))
        acc = acc.after(acc[0].store(acc.after(blk)[0] + contrib).end(blk))
        lane_total = acc[0]
      else:
        lane_total = UOp.const(dtypes.float32, 0.0)
        for stage in range(spec.accumulators):
          blk_stage = UOp.range(k_blocks // spec.accumulators, 30 + stage, axis_type=AxisType.REDUCE)
          staged_blk = stage + spec.accumulators * blk_stage
          staged_base = (row * k_blocks + staged_blk) * Q6K_HALFWORDS_PER_BLOCK
          staged_contrib = _q6k_block_dot(halfs, x, staged_base, staged_blk, pos)
          staged_acc = UOp.placeholder((1,), dtypes.float32, 20 + stage, addrspace=AddrSpace.REG)
          staged_acc = staged_acc.after(staged_acc[0].store(0.0))
          staged_acc = staged_acc.after(staged_acc[0].store(staged_acc.after(blk_stage)[0] + staged_contrib).end(blk_stage))
          lane_total = lane_total + staged_acc[0]
      total = _q6k_coop_pos_reduce_sum(lane_total, pos, row_tile)
      # P1 vocab-head aux scatter-chain fusion (nv-vocab-aux-chain-fusion-scope-20260812.md):
      # carry the per-tile (max, index) as one u64 key instead of the 151936-row logits
      # scatter.  Each lane owns its row's total (replicated over the 16 pos lanes), so the
      # warp reduce over the packed key picks the largest logit and, on ties, the FIRST row
      # (the inverted row index in the low half is largest for the smallest row) with no
      # float cast anywhere in the compare: the value leaves fp32 only through BITCAST and
      # integer ops (identical ordering to packed_argmax_finite_fp32).  The tiny final
      # packed reduce (q6k_vocab_top1_reduce_*) turns the per-warp keys into the token id.
      if spec.epilogue == "vocab_top1":
        bits = total.bitcast(dtypes.uint32)
        bits = total.eq(0.0).where(UOp.const(dtypes.uint32, 0), bits)
        neg = (bits >> UOp.const(dtypes.uint32, 31)).eq(UOp.const(dtypes.uint32, 1))
        ordered = neg.where(bits.bitwise_not(), bits.bitwise_xor(UOp.const(dtypes.uint32, 0x80000000)))
        inv_index = (UOp.const(dtypes.uint64, rows - 1) - row.cast(dtypes.uint64))
        key = (ordered.cast(dtypes.uint64) << UOp.const(dtypes.uint64, 32)).bitwise_or(inv_index)
        tile_key = _q6k_coop_tile_key_reduce_max(key, pos * row_tile + row_i)
        return partials[row_o].store(tile_key).end(row_o, row_i, pos).sink(
          arg=KernelInfo(name=name, opts_to_apply=()))
      # M2b ffn_down residual add (nv-epilogue-absorption-route-scope-20260810.md): the row's
      # total plus the hidden-state residual h[row], stored fp32. The in-kernel fp32 add is
      # bitwise-identical to the standalone h+ffn_out kernel (same values, fp32 add is
      # commutative), so the separate E_32_32_4_02a9738c add folds away under the lease.
      if spec.epilogue == "ffn_down_resadd":
        total = total + h[row].cast(dtypes.float32)
      return partials[row].store(total).end(row_o, row_i, pos).sink(arg=KernelInfo(name=name, opts_to_apply=()))
    acc = partials[row, pos].set(0.0)
    acc = partials[row, pos].set(acc.after(blk)[row, pos] + contrib, end=blk)
    return acc.end(row_o, row_i, pos).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel


def _emit_q6k_partial(spec:Q6KGEMVRouteSpec):
  rows, parts, k_blocks, name, opts = spec.rows, spec.parts, spec.k_blocks, spec.kernel_name, spec.opts
  blocks_per_part = cdiv(k_blocks, parts)
  def kernel(partials:UOp, halfs:UOp, x:UOp) -> UOp:
    row, part = UOp.range(rows, 0), UOp.range(parts, 1)
    blk_part = UOp.range(blocks_per_part, 2, axis_type=AxisType.REDUCE)
    pos = UOp.range(Q6K_POS_EXTENT, 3, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    base = (row * k_blocks + blk) * Q6K_HALFWORDS_PER_BLOCK
    contrib = (blk < k_blocks).where(_q6k_block_dot(halfs, x, base, blk, pos), UOp.const(dtypes.float32, 0.0))
    acc = partials[row, part].set(0.0)
    acc = partials[row, part].set(acc.after(blk_part, pos)[row, part] + contrib, end=pos)
    return acc.end(row, part, blk_part).sink(arg=KernelInfo(name=name, opts_to_apply=opts))
  return kernel


def decode_kv_rope_store_kernel(Hkv:int, Hd:int, MAXC:int, VPART:int=1):
  """Fused decode kv-store kernel (decode-kv-store-chain-fusion-scope-20260803.md, Option A):
  ONE kernel ropes k in fp32 (the exact `apply_rope` arithmetic, full-head rope), casts k and v
  to the CACHE's own dtype (fp16 when the target can express fp16 and the validated 8B shape holds,
  otherwise the default fp32 -- the dtype is capability-resolved by `kv_cache_fp16_eligible`, never a
  backend/architecture string), and writes both into `cache_kv` at slot `start_pos`. Writing cache.dtype
  reproduces the legacy cache bytes bit-for-bit for BOTH cache dtypes. Replaces the k-rope + k-cast + v-cast +
  `Tensor.stack(k, v)` + cache store chain (5 kernels/layer). Elementwise only: no reductions, no
  shared memory, no cross-lane communication -- target-agnostic by construction (no WMMA / shuffle /
  vendor intrinsic), so it renders identically for NV/AMD/Metal.

  VPART absorbs the q4k decode GEMV's v-parts reduce (NV emits 4 fp32 partials per row; the model's
  v is their axis-1 sum). With VPART>1 the v argument is the RAW parts view, shape (Hkv*Hd, VPART),
  and the kernel sums the partials in-register in the legacy left-to-right fp32 order
  `((p0+p1)+p2)+p3` (verified against the cached legacy store source), so the stored bytes are
  bit-identical for both cache dtypes. VPART=1 keeps the reduced flat v and the exact verified
  single-load path.

  Slot 0 is the cache buffer itself (writable receiver); the returned tensor is the cache AFTER
  the store, which is what the flash route reads. `start_pos` binds from the decode graph's
  same-named variable (identical mechanism to the flash tile's Tc)."""
  if Hkv < 1: raise ValueError(f"decode_kv_rope_store requires Hkv>=1, got {Hkv}")
  if Hd < 2 or Hd % 2 != 0: raise ValueError(f"decode_kv_rope_store requires even Hd>=2, got {Hd}")
  if VPART < 1: raise ValueError(f"decode_kv_rope_store requires VPART>=1, got {VPART}")
  half = Hd // 2
  name = f"decode_kv_rope_store_{Hkv}_{Hd}" + (f"_v{VPART}" if VPART > 1 else "")

  def kernel(cache:UOp, k:UOp, v:UOp, freqs:UOp) -> UOp:
    sp = UOp.variable("start_pos", 0, MAXC - 1)
    kvh = UOp.range(Hkv, 0, axis_type=AxisType.GLOBAL)
    elem = UOp.range(Hd, 1, axis_type=AxisType.GLOBAL)
    low = elem < half
    rot = low.where(elem, elem - half)
    # apply_rope replication: y1 = x1*cos - x2*sin (low half), y2 = x2*cos + x1*sin (high half),
    # with freqs laid out [MAXC, Hd] as cos in [:half] and sin in [half:] (precompute_freqs_cis).
    k1 = k[kvh * Hd + rot].cast(dtypes.float32)
    k2 = k[kvh * Hd + rot + half].cast(dtypes.float32)
    cos = freqs[sp, rot].cast(dtypes.float32)
    sin = freqs[sp, half + rot].cast(dtypes.float32)
    # The store writes the CACHE's own dtype (capability-resolved: fp16 when the target expresses it and
    # the validated shape holds, else default fp32). Writing cache.dtype reproduces the legacy cache bytes
    # bit-for-bit for BOTH cache dtypes (fp32: no rounding, identical fp32 arithmetic; fp16: one single
    # fp32->fp16 round, the same single round the legacy store performed).
    kout = low.where(k1 * cos - k2 * sin, k2 * cos + k1 * sin).cast(cache.dtype)
    if VPART == 1:
      vout = v[kvh * Hd + elem].cast(cache.dtype)
    else:
      # Sum the raw partials left-to-right in fp32, then the single store cast -- the exact legacy
      # `(val8.x+val8.y+val8.z+val8.w)` expression, so the cache bytes match bit-for-bit.
      vsum = v[kvh * Hd + elem, 0].cast(dtypes.float32)
      for _r in range(1, VPART): vsum = vsum + v[kvh * Hd + elem, _r].cast(dtypes.float32)
      vout = vsum.cast(cache.dtype)
    kst = cache[0, 0, kvh, sp, elem].store(kout)
    vst = cache.after(kst)[1, 0, kvh, sp, elem].store(vout)
    return vst.end(kvh, elem).sink(arg=KernelInfo(name=name, opts_to_apply=()))
  return kernel
