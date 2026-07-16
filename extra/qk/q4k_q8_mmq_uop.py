"""Direct-UOp Q4_K x Q8_1 MMQ research emitters.

The scalar owner proves explicit grid/index execution without Tensor/callify.
The aligned candidate expresses grouped integer contractions as generic UOps
and lets tinygrad's TC lowering generate gfx1100 WMMA and lane ownership.
Neither emitter is a production route.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS


@dataclass(frozen=True)
class Q4KQ8MMQUOpSpec:
  m: int
  n: int
  k: int
  lanes: int = 32
  name: str = "q4k_q8_mmq_uop_phase1"

  def validate(self) -> None:
    if min(self.m, self.n, self.k) <= 0: raise ValueError("M/N/K must be positive")
    if self.k % Q4_K_BLOCK_ELEMS: raise ValueError(f"K must be a multiple of {Q4_K_BLOCK_ELEMS}")
    if self.lanes != 32: raise ValueError("phase-1 owner mapping requires 32 lanes")

  @property
  def n_tiles(self) -> int: return (self.n + self.lanes - 1) // self.lanes


def describe_q4k_q8_mmq_uop(m:int, n:int, k:int, *, name:str="q4k_q8_mmq_uop_phase1") -> Q4KQ8MMQUOpSpec:
  spec = Q4KQ8MMQUOpSpec(m, n, k, name=name)
  spec.validate()
  return spec


def _byte(words:UOp, base:UOp, idx:UOp) -> UOp:
  return words[base + idx // 4].rshift((idx % 4) * 8).bitwise_and(0xff)


def emit_q4k_q8_mmq_uop(spec:Q4KQ8MMQUOpSpec):
  """Return the sole custom_kernel callback for packed Q4_K and row-major Q8_1.

  ABI: ``(out[M,N], q4_words[N,K/256,36], xq[M,K], xscale[M,K/32])``.
  Q8 values are signed int8 and scales are float32.  Q4_K metadata/payload is
  consumed directly from uint32 words; no dequantized weight tensor exists.
  """
  spec.validate()
  k_blocks, q8_groups = spec.k // Q4_K_BLOCK_ELEMS, spec.k // Q8_1_BLOCK_ELEMS

  def kernel(out:UOp, words:UOp, xq:UOp, xscale:UOp) -> UOp:
    ntile = UOp.special(spec.n_tiles, "gidx0")
    m = UOp.special(spec.m, "gidx1")
    lane = UOp.special(spec.lanes, "lidx0")
    n = ntile * spec.lanes + lane
    owned = n < spec.n
    # Edge lanes still execute the rolled reduction, so redirect their reads to
    # row zero and predicate only the final side effect.
    read_n = owned.where(n, UOp.const(dtypes.weakint, 0))
    kk = UOp.range(spec.k, 0, axis_type=AxisType.REDUCE)
    blk, inblk = kk // Q4_K_BLOCK_ELEMS, kk % Q4_K_BLOCK_ELEMS
    grp, pos = inblk // Q8_1_BLOCK_ELEMS, inblk % Q8_1_BLOCK_ELEMS
    base = (read_n * k_blocks + blk) * Q4K_WORDS_PER_BLOCK

    # Existing Q4 helpers use a Python group branch.  These are the same GGML
    # bitfields with dynamic group indices, preserving the single rolled K loop.
    packed_scale = base + 1
    upper = grp >= 4
    # Keep every dynamically evaluated address valid.  Upper groups use
    # h=grp-4: u[h]/u[4+h] carry the high two bits and u[8+h] the low nibbles.
    h = upper.where(grp - 4, grp)
    low_sc, low_mn = _byte(words, packed_scale, h), _byte(words, packed_scale, 4 + h)
    high = _byte(words, packed_scale, 8 + h)
    sc_hi = high.bitwise_and(0xf).bitwise_or(low_sc.rshift(6).lshift(4))
    mn_hi = high.rshift(4).bitwise_or(low_mn.rshift(6).lshift(4))
    sc, mn = upper.where(sc_hi, low_sc.bitwise_and(63)), upper.where(mn_hi, low_mn.bitwise_and(63))

    scale_word = words[base]
    d = scale_word.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    dmin = scale_word.rshift(16).bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    qword = words[base + 4 + (grp // 2) * 8 + pos // 4]
    q = qword.rshift((pos % 4) * 8 + (grp % 2) * 4).bitwise_and(0xf).cast(dtypes.float32)
    q8 = xq[m * spec.k + kk].cast(dtypes.float32)
    xs = xscale[m * q8_groups + kk // Q8_1_BLOCK_ELEMS]
    # Grouping this rolled sum by K/32 gives exactly
    # xs * (d*sc*sum(q*q8) - dmin*mn*sum(q8)); no ambiguous sums ABI.
    value = (xs * (d * sc.cast(dtypes.float32) * q * q8 - dmin * mn.cast(dtypes.float32) * q8)).reduce(kk, arg=Ops.ADD)
    return out[m, n].store(value, gate=owned).sink(arg=KernelInfo(name=spec.name, opts_to_apply=()))

  return kernel


@dataclass(frozen=True)
class Q4KQ8MMQWMMASpec:
  """Fail-closed aligned research candidate; tails remain unsupported."""
  m: int = 16
  n: int = 16
  k: int = 256

  def validate(self) -> None:
    if min(self.m, self.n, self.k) <= 0 or self.m % 16 or self.n % 16 or self.k % 256:
      raise ValueError("WMMA candidate requires positive M/N multiples of 16 and K multiple of 256 (no tails)")

  @property
  def name(self) -> str: return f"q4k_q8_mmq_uop_wmma_{self.m}x{self.n}x{self.k}"


def describe_q4k_q8_mmq_wmma(*, m:int=16, n:int=16, k:int=256) -> Q4KQ8MMQWMMASpec:
  spec = Q4KQ8MMQWMMASpec(m, n, k)
  spec.validate()
  return spec


def emit_q4k_q8_mmq_wmma(spec:Q4KQ8MMQWMMASpec):
  """Nested Q4/Q8 contraction expressed only with generic UOps.

  The inner signed-char product and int32 ADD reduction are deliberately left
  matcher-visible.  TC selection belongs solely to KernelInfo; this source
  never constructs WMMA or SHAPED_WMMA operations.
  """
  spec.validate()

  def kernel(out:UOp, words:UOp, xq:UOp, xscale:UOp) -> UOp:
    m, n = UOp.range(spec.m, 0), UOp.range(spec.n, 1)
    grp = UOp.range(spec.k // Q8_1_BLOCK_ELEMS, 2, axis_type=AxisType.REDUCE)
    pos = UOp.range(32, 3, axis_type=AxisType.REDUCE)
    block, ingrp = grp // 8, grp % 8
    base = (n * (spec.k // Q4_K_BLOCK_ELEMS) + block) * Q4K_WORDS_PER_BLOCK
    upper = ingrp >= 4
    h = upper.where(ingrp - 4, ingrp)
    low_sc, low_mn = _byte(words, base + 1, h), _byte(words, base + 1, 4 + h)
    high = _byte(words, base + 1, 8 + h)
    sc = upper.where(high.bitwise_and(0xf).bitwise_or(low_sc.rshift(6).lshift(4)), low_sc.bitwise_and(63))
    mn = upper.where(high.rshift(4).bitwise_or(low_mn.rshift(6).lshift(4)), low_mn.bitwise_and(63))
    scale_word = words[base]
    d = scale_word.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    dmin = scale_word.rshift(16).bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)

    qword = words[base + 4 + (ingrp // 2) * 8 + pos // 4]
    q4 = qword.rshift((pos % 4) * 8 + (ingrp % 2) * 4).bitwise_and(0xf).cast(dtypes.uint8).bitcast(dtypes.int8)
    q8 = xq[m * spec.k + grp * 32 + pos]
    # CAST after MUL preserves char*char at the matcher boundary while the
    # reduction accumulator is explicitly int32.
    dot = (q4 * q8).cast(dtypes.int32).reduce(pos, arg=Ops.ADD)
    qsum = q8.cast(dtypes.int32).reduce(pos, arg=Ops.ADD)
    xs = xscale[m * (spec.k // Q8_1_BLOCK_ELEMS) + grp]
    corrected = xs * (d * sc.cast(dtypes.float32) * dot.cast(dtypes.float32) -
                      dmin * mn.cast(dtypes.float32) * qsum.cast(dtypes.float32))
    value = corrected.reduce(grp, arg=Ops.ADD)
    return out[m, n].store(value).end(m, n).sink(arg=KernelInfo(name=spec.name,
      opts_to_apply=(Opt(OptOps.TC, 0, (-1, 2, 1)),)))

  return kernel


@dataclass(frozen=True)
class Q4KQ8MMQWideWMMASpec:
  """One-workgroup 16x32 experiment; never selected by the default emitter."""
  m: int = 16
  n: int = 32
  k: int = 256

  def validate(self) -> None:
    if self.m != 16 or self.n != 32 or self.k <= 0 or self.k % 256:
      raise ValueError("wide WMMA requires exact M=16, N=32 and positive K multiple of 256")

  @property
  def name(self) -> str: return f"q4k_q8_mmq_uop_wide_wmma_{self.m}x{self.n}x{self.k}"


def describe_q4k_q8_mmq_wide_wmma(*, m:int=16, n:int=32, k:int=256) -> Q4KQ8MMQWideWMMASpec:
  spec = Q4KQ8MMQWideWMMASpec(m, n, k)
  spec.validate()
  return spec


def emit_q4k_q8_mmq_wide_wmma(spec:Q4KQ8MMQWideWMMASpec):
  """Return the explicit one-workgroup N=32 variant of the generic TC graph."""
  spec.validate()
  base = emit_q4k_q8_mmq_wmma(Q4KQ8MMQWMMASpec(spec.m, spec.n, spec.k))

  def kernel(out:UOp, words:UOp, xq:UOp, xscale:UOp) -> UOp:
    sink = base(out, words, xq, xscale)
    return sink.replace(arg=KernelInfo(name=spec.name, opts_to_apply=(
      Opt(OptOps.TC, 0, (-1, 2, 1)), Opt(OptOps.UPCAST, 0, 2))))

  return kernel


LLAMA_Q4K_Q8_1_DS4_SOURCE_ANCHORS = (
  "quantize.cu:quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>:make_half2(d, sum)",
  "mmq.cuh:block_q8_1_mmq::ds4 (scale, original-fp partial sum)",
  "vecdotq.cuh:vec_dot_q4_K_q8_1_impl_mmq:dm4f.x*sumf_d-dm4f.y*sumf_m",
  "mmq.cuh:vec_dot_q4_K_q8_1_dp4a:y_ds",
)


@dataclass(frozen=True)
class Q4KQ8MMQSumOriginalFPWMMASpec:
  """Oracle-derived aligned ABI for llama ``block_q8_1_mmq.ds4[].y``."""
  m: int
  n: int
  k: int
  sum_semantics: str = "llama_ds4_y_original_fp32_group_sum"
  activation_layout: str = "llama_mmq_q8_1_ds4"

  def validate(self) -> None:
    if min(self.m, self.n, self.k) <= 0 or self.m % 16 or self.n % 16 or self.k % 256:
      raise ValueError("llama DS4 WMMA requires positive M/N multiples of 16 and K multiple of 256")
    if self.sum_semantics != "llama_ds4_y_original_fp32_group_sum":
      raise ValueError("sum semantics must be llama ds4.y original-fp32 group sum; dequantized/derived sums are rejected")
    if self.activation_layout != "llama_mmq_q8_1_ds4": raise ValueError("activation layout must be llama_mmq_q8_1_ds4")

  @property
  def name(self) -> str: return f"q4k_q8_mmq_uop_wmma_llama_ds4_y_original_fp_sum_{self.m}x{self.n}x{self.k}"

  @property
  def source_anchors(self) -> tuple[str, ...]: return LLAMA_Q4K_Q8_1_DS4_SOURCE_ANCHORS


def describe_q4k_q8_mmq_sum_original_fp_wmma(m:int, n:int, k:int, *,
                                              sum_semantics:str="llama_ds4_y_original_fp32_group_sum",
                                              activation_layout:str="llama_mmq_q8_1_ds4") -> Q4KQ8MMQSumOriginalFPWMMASpec:
  spec = Q4KQ8MMQSumOriginalFPWMMASpec(m, n, k, sum_semantics, activation_layout)
  spec.validate()
  return spec


def emit_q4k_q8_mmq_sum_original_fp_wmma(spec:Q4KQ8MMQSumOriginalFPWMMASpec):
  """ABI: out, words, xq, xscale, ds4_y (original-fp32 group sum).

  This is the row-major operand view of llama's DS4 metadata, not its physical
  block-major/padded storage geometry and not a route-policy claim.
  """
  spec.validate()
  return _emit_q4k_q8_mmq_original_fp(spec, physical_ds4=False)


def _emit_q4k_q8_mmq_original_fp(spec:Q4KQ8MMQSumOriginalFPWMMASpec, *, physical_ds4:bool):
  """Single semantic implementation of the Q4_K/original-fp-sum contraction."""
  def kernel(out:UOp, words:UOp, xq:UOp, xscale:UOp, ds4_y:UOp) -> UOp:
    m, n = UOp.range(spec.m, 0), UOp.range(spec.n, 1)
    grp = UOp.range(spec.k // Q8_1_BLOCK_ELEMS, 2, axis_type=AxisType.REDUCE)
    pos = UOp.range(Q8_1_BLOCK_ELEMS, 3, axis_type=AxisType.REDUCE)
    block, ingrp = grp // 8, grp % 8
    base = (n * (spec.k // Q4_K_BLOCK_ELEMS) + block) * Q4K_WORDS_PER_BLOCK
    upper = ingrp >= 4
    h = upper.where(ingrp - 4, ingrp)
    low_sc, low_mn = _byte(words, base + 1, h), _byte(words, base + 1, 4 + h)
    high = _byte(words, base + 1, 8 + h)
    sc = upper.where(high.bitwise_and(0xf).bitwise_or(low_sc.rshift(6).lshift(4)), low_sc.bitwise_and(63))
    mn = upper.where(high.rshift(4).bitwise_or(low_mn.rshift(6).lshift(4)), low_mn.bitwise_and(63))
    scale_word = words[base]
    d = scale_word.bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    dmin = scale_word.rshift(16).bitwise_and(0xffff).cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)
    qword = words[base + 4 + (ingrp // 2) * 8 + pos // 4]
    q4 = qword.rshift((pos % 4) * 8 + (ingrp % 2) * 4).bitwise_and(0xf).cast(dtypes.uint8).bitcast(dtypes.int8)
    ds4_block, ds4_group = grp // 4, grp % 4
    meta_idx = (ds4_block * spec.m + m) * 4 + ds4_group
    q8 = xq[meta_idx * Q8_1_BLOCK_ELEMS + pos if physical_ds4 else m * spec.k + grp * Q8_1_BLOCK_ELEMS + pos]
    dot = (q4 * q8).cast(dtypes.int32).reduce(pos, arg=Ops.ADD)
    group_idx = meta_idx if physical_ds4 else m * (spec.k // Q8_1_BLOCK_ELEMS) + grp
    xs, supplied_sum = xscale[group_idx], ds4_y[group_idx]
    corrected = xs * d * sc.cast(dtypes.float32) * dot.cast(dtypes.float32) - \
                dmin * mn.cast(dtypes.float32) * supplied_sum
    value = corrected.reduce(grp, arg=Ops.ADD)
    return out[m, n].store(value).end(m, n).sink(arg=KernelInfo(name=spec.name,
      opts_to_apply=(Opt(OptOps.TC, 0, (-1, 2, 1)),)))

  return kernel


@dataclass(frozen=True)
class Q4KQ8MMQRoleSizedWMMASpec(Q4KQ8MMQSumOriginalFPWMMASpec):
  """Model-independent five-buffer emitter over physical llama MMQ DS4 storage."""
  activation_layout: str = "q8_1_mmq_ds4_transposed_blocks"

  def validate(self) -> None:
    if min(self.m, self.n, self.k) <= 0 or self.m % 16 or self.n % 16 or self.k % 256:
      raise ValueError("role-sized DS4 WMMA requires positive M/N multiples of 16 and K multiple of 256 (no tails)")
    if self.sum_semantics != "llama_ds4_y_original_fp32_group_sum":
      raise ValueError("sum semantics must be llama ds4.y original-fp32 group sum; dequantized/derived sums are rejected")
    if self.activation_layout != "q8_1_mmq_ds4_transposed_blocks":
      raise ValueError("activation layout must be q8_1_mmq_ds4_transposed_blocks")

  @property
  def name(self) -> str: return f"q4k_q8_mmq_uop_role_sized_ds4_{self.m}x{self.n}x{self.k}"


def describe_q4k_q8_mmq_role_sized_wmma(m:int, n:int, k:int, *,
                                         sum_semantics:str="llama_ds4_y_original_fp32_group_sum",
                                         activation_layout:str="q8_1_mmq_ds4_transposed_blocks") -> Q4KQ8MMQRoleSizedWMMASpec:
  spec = Q4KQ8MMQRoleSizedWMMASpec(m, n, k, sum_semantics, activation_layout)
  spec.validate()
  return spec


def emit_q4k_q8_mmq_role_sized_wmma(spec:Q4KQ8MMQRoleSizedWMMASpec):
  """ABI slots: out fp32, Q4 uint32, physical DS4 int8, scales fp32, sums fp32."""
  spec.validate()
  return _emit_q4k_q8_mmq_original_fp(spec, physical_ds4=True)
