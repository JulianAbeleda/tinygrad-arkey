#!/usr/bin/env python3
"""Spec-driven Q4_K/Q8_1 prefill MMQ substrate.

This module intentionally does not define a handwritten kernel. It expresses the group dot as ordinary tinygrad
Tensor matmuls with dtype=int, so RDNA3 iu8 WMMA is selected only by the existing tensor-core matcher/codegen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, dtypes
from tinygrad.helpers import getenv

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS


@dataclass(frozen=True)
class Q4KInt8WMMAPrefillSpec:
  n: int
  k: int
  m: int
  role: str = ""
  group_elems: int = Q8_1_BLOCK_ELEMS
  wmma_m: int = 16
  wmma_n: int = 16
  wmma_k: int = 16
  n_tile: int = 256
  target: str = "amd_gfx1100"
  implementation: str = "group_tensor_matmul_v0"

  @property
  def k_blocks(self) -> int:
    return self.k // Q4_K_BLOCK_ELEMS

  @property
  def groups_per_block(self) -> int:
    return Q4_K_BLOCK_ELEMS // self.group_elems

  @property
  def groups(self) -> int:
    return self.k // self.group_elems

  @property
  def kernel_name(self) -> str:
    role = f"_{self.role}" if self.role else ""
    return f"prefill_q4k_q8_1_wmma_generated_gemm{role}_{self.n}_{self.k}_{self.m}"

  def validate(self) -> None:
    if self.group_elems != Q8_1_BLOCK_ELEMS:
      raise ValueError(f"group_elems must match Q8_1 block elems ({Q8_1_BLOCK_ELEMS}), got {self.group_elems}")
    if self.k % Q4_K_BLOCK_ELEMS:
      raise ValueError(f"k={self.k} must be a multiple of Q4_K block elems ({Q4_K_BLOCK_ELEMS})")
    if self.m <= 0 or self.n <= 0 or self.k <= 0:
      raise ValueError(f"invalid non-positive shape m={self.m} n={self.n} k={self.k}")
    if self.m % self.wmma_m or self.n % self.wmma_n or self.group_elems % self.wmma_k:
      raise ValueError(f"shape must align to WMMA tile ({self.wmma_m},{self.wmma_n},{self.wmma_k}), got m={self.m} n={self.n} group={self.group_elems}")
    if self.n_tile <= 0 or self.n_tile % self.wmma_n:
      raise ValueError(f"n_tile must be a positive multiple of WMMA N tile {self.wmma_n}, got {self.n_tile}")
    if self.implementation != "group_tensor_matmul_v0":
      raise ValueError(f"unsupported implementation={self.implementation!r}")

  def to_json(self) -> dict[str, Any]:
    return {"n": self.n, "k": self.k, "m": self.m, "role": self.role, "group_elems": self.group_elems,
            "wmma_m": self.wmma_m, "wmma_n": self.wmma_n, "wmma_k": self.wmma_k, "n_tile": self.n_tile,
            "target": self.target,
            "implementation": self.implementation, "groups": self.groups, "k_blocks": self.k_blocks,
            "kernel_name": self.kernel_name}


@dataclass(frozen=True)
class Q4KInt8WMMATiledPrefillSpec:
  n: int
  k: int
  m: int
  role: str = ""
  group_elems: int = Q8_1_BLOCK_ELEMS
  wmma_m: int = 16
  wmma_n: int = 16
  wmma_k: int = 16
  m_tile: int = 16
  n_tile: int = 16
  group_tile: int = 1
  output_layout: str = "direct"
  target: str = "amd_gfx1100"
  implementation: str = "direct_tiled_wmma_v0"

  @property
  def k_blocks(self) -> int:
    return self.k // Q4_K_BLOCK_ELEMS

  @property
  def groups_per_block(self) -> int:
    return Q4_K_BLOCK_ELEMS // self.group_elems

  @property
  def groups(self) -> int:
    return self.k // self.group_elems

  @property
  def live_raw_elems(self) -> int:
    return self.m_tile * self.n_tile * self.group_tile

  @property
  def forbidden_full_raw_elems(self) -> int:
    return self.groups * self.m * self.n

  @property
  def kernel_name(self) -> str:
    role = f"_{self.role}" if self.role else ""
    return f"prefill_q4k_q8_1_wmma_tiled_generated_gemm{role}_{self.n}_{self.k}_{self.m}_{self.m_tile}x{self.n_tile}x{self.group_tile}"

  def validate(self) -> None:
    if self.group_elems != Q8_1_BLOCK_ELEMS:
      raise ValueError(f"group_elems must match Q8_1 block elems ({Q8_1_BLOCK_ELEMS}), got {self.group_elems}")
    if self.k % Q4_K_BLOCK_ELEMS:
      raise ValueError(f"k={self.k} must be a multiple of Q4_K block elems ({Q4_K_BLOCK_ELEMS})")
    if self.m <= 0 or self.n <= 0 or self.k <= 0:
      raise ValueError(f"invalid non-positive shape m={self.m} n={self.n} k={self.k}")
    if self.m % self.wmma_m or self.n % self.wmma_n or self.group_elems % self.wmma_k:
      raise ValueError(f"shape must align to WMMA tile ({self.wmma_m},{self.wmma_n},{self.wmma_k}), got m={self.m} n={self.n} group={self.group_elems}")
    if self.m_tile <= 0 or self.n_tile <= 0 or self.group_tile <= 0:
      raise ValueError(f"tile sizes must be positive, got m_tile={self.m_tile} n_tile={self.n_tile} group_tile={self.group_tile}")
    if self.m_tile % self.wmma_m or self.n_tile % self.wmma_n:
      raise ValueError(f"m_tile/n_tile must align to WMMA tile ({self.wmma_m},{self.wmma_n}), got {self.m_tile},{self.n_tile}")
    if self.group_tile > self.groups:
      raise ValueError(f"group_tile={self.group_tile} exceeds groups={self.groups}")
    if self.output_layout != "direct":
      raise ValueError(f"unsupported output_layout={self.output_layout!r}")
    if self.implementation != "direct_tiled_wmma_v0":
      raise ValueError(f"unsupported implementation={self.implementation!r}")

  def to_json(self) -> dict[str, Any]:
    return {"n": self.n, "k": self.k, "m": self.m, "role": self.role, "group_elems": self.group_elems,
            "wmma_m": self.wmma_m, "wmma_n": self.wmma_n, "wmma_k": self.wmma_k, "m_tile": self.m_tile,
            "n_tile": self.n_tile, "group_tile": self.group_tile, "output_layout": self.output_layout,
            "target": self.target, "implementation": self.implementation, "groups": self.groups,
            "k_blocks": self.k_blocks, "live_raw_elems": self.live_raw_elems,
            "forbidden_full_raw_elems": self.forbidden_full_raw_elems, "kernel_name": self.kernel_name}


def describe_q4k_int8_wmma_prefill(n:int, k:int, m:int, *, role:str="", n_tile:int=256) -> Q4KInt8WMMAPrefillSpec:
  spec = Q4KInt8WMMAPrefillSpec(n=n, k=k, m=m, role=role, n_tile=n_tile)
  spec.validate()
  return spec


def describe_q4k_int8_wmma_tiled_prefill(n:int, k:int, m:int, *, role:str="", m_tile:int=16, n_tile:int=16,
                                         group_tile:int=1) -> Q4KInt8WMMATiledPrefillSpec:
  spec = Q4KInt8WMMATiledPrefillSpec(n=n, k=k, m=m, role=role, m_tile=m_tile, n_tile=n_tile,
                                     group_tile=group_tile)
  spec.validate()
  return spec


def _intdot_matmul(a:Tensor, b_t:Tensor) -> Tensor:
  # DEV=PYTHON overflows full-range int8 dot products. Widen only the GPU-free oracle; AMD must keep int8 operands so
  # codegen can select iu8 WMMA.
  if getenv("DEV", "") == "PYTHON":
    return a.cast(dtypes.int32).matmul(b_t.cast(dtypes.int32), dtype=dtypes.int)
  return a.matmul(b_t, dtype=dtypes.int)


def _f16_word_tensor(word:Tensor, high:bool) -> Tensor:
  bits = ((word >> 16) if high else word).bitwise_and(0xffff)
  return bits.cast(dtypes.uint16).bitcast(dtypes.float16).cast(dtypes.float32)


def _scale_byte(words3:Tensor, blk:int, idx:int) -> Tensor:
  return (words3[:, blk, 1 + idx // 4] >> ((idx % 4) * 8)).bitwise_and(0xff)


def _q4k_group_params_tensor(words3:Tensor, blk:int, grp:int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  d = _f16_word_tensor(words3[:, blk, 0], False)
  dmin = _f16_word_tensor(words3[:, blk, 0], True)
  if grp < 4:
    sc = _scale_byte(words3, blk, grp).bitwise_and(63)
    mn = _scale_byte(words3, blk, 4 + grp).bitwise_and(63)
  else:
    high = _scale_byte(words3, blk, 8 + grp - 4)
    sc = high.bitwise_and(0xf).bitwise_or(_scale_byte(words3, blk, grp - 4).rshift(6).lshift(4))
    mn = high.rshift(4).bitwise_or(_scale_byte(words3, blk, 4 + grp - 4).rshift(6).lshift(4))
  return d, dmin, sc, mn


def _q4k_group_codes_tensor(words3:Tensor, blk:int, grp:int) -> Tensor:
  qword_base = 4 + (grp // 2) * 8
  lanes = []
  for lane4 in range(8):
    qword = words3[:, blk, qword_base + lane4]
    for nib in range(4):
      lanes.append((qword >> (nib * 8 + (grp % 2) * 4)).bitwise_and(0xf).cast(dtypes.int8).reshape(words3.shape[0], 1))
  return lanes[0].cat(*lanes[1:], dim=1).contiguous()


def _q4k_all_group_codes_tensor(words3:Tensor, spec:Q4KInt8WMMAPrefillSpec) -> Tensor:
  groups = []
  for grp in range(spec.groups_per_block):
    lanes = []
    qword_base = 4 + (grp // 2) * 8
    for lane4 in range(8):
      qword = words3[:, :, qword_base + lane4]
      for nib in range(4):
        lanes.append((qword >> (nib * 8 + (grp % 2) * 4)).bitwise_and(0xf).cast(dtypes.int8).reshape(spec.n, spec.k_blocks, 1))
    groups.append(lanes[0].cat(*lanes[1:], dim=2).reshape(spec.n, spec.k_blocks, 1, spec.group_elems))
  return groups[0].cat(*groups[1:], dim=2).reshape(spec.n, spec.groups, spec.group_elems).permute(1, 0, 2).contiguous()


def _q4k_all_group_params_tensor(words3:Tensor, spec:Q4KInt8WMMAPrefillSpec) -> tuple[Tensor, Tensor, Tensor, Tensor]:
  d_blk = _f16_word_tensor(words3[:, :, 0], False)
  dmin_blk = _f16_word_tensor(words3[:, :, 0], True)
  d_groups = [d_blk.reshape(spec.n, spec.k_blocks, 1) for _ in range(spec.groups_per_block)]
  dmin_groups = [dmin_blk.reshape(spec.n, spec.k_blocks, 1) for _ in range(spec.groups_per_block)]
  sc_groups, mn_groups = [], []
  for grp in range(spec.groups_per_block):
    if grp < 4:
      sc = _scale_byte(words3, slice(None), grp).bitwise_and(63)
      mn = _scale_byte(words3, slice(None), 4 + grp).bitwise_and(63)
    else:
      high = _scale_byte(words3, slice(None), 8 + grp - 4)
      sc = high.bitwise_and(0xf).bitwise_or(_scale_byte(words3, slice(None), grp - 4).rshift(6).lshift(4))
      mn = high.rshift(4).bitwise_or(_scale_byte(words3, slice(None), 4 + grp - 4).rshift(6).lshift(4))
    sc_groups.append(sc.reshape(spec.n, spec.k_blocks, 1))
    mn_groups.append(mn.reshape(spec.n, spec.k_blocks, 1))
  d = d_groups[0].cat(*d_groups[1:], dim=2).reshape(spec.n, spec.groups)
  dmin = dmin_groups[0].cat(*dmin_groups[1:], dim=2).reshape(spec.n, spec.groups)
  sc = sc_groups[0].cat(*sc_groups[1:], dim=2).reshape(spec.n, spec.groups)
  mn = mn_groups[0].cat(*mn_groups[1:], dim=2).reshape(spec.n, spec.groups)
  return d, dmin, sc, mn


def emit_q4k_int8_wmma_prefill_tensor(words:Tensor, xq:Tensor, xscales:Tensor,
                                      spec:Q4KInt8WMMAPrefillSpec, *, vectorized:bool=True) -> Tensor:
  """Return fp32 [m,n] from Q4_K words and Q8_1 activation.

  The int dot is deliberately expressed as `q8_g.matmul(q4_g.T, dtype=dtypes.int)`. On AMD with TC enabled, that
  is the existing codegen route that tensorizes to iu8 WMMA. On CPU/PYTHON this remains a numeric oracle for the
  same algebra.
  """
  spec.validate()
  if vectorized and spec.n > spec.n_tile:
    words3_all = words.reshape(spec.n, spec.k_blocks, Q4K_WORDS_PER_BLOCK)
    outs = []
    for start in range(0, spec.n, spec.n_tile):
      stop = min(start + spec.n_tile, spec.n)
      sub_n = stop - start
      if sub_n % spec.wmma_n:
        raise ValueError(f"sub-tile n={sub_n} must align to WMMA N tile {spec.wmma_n}")
      sub_spec = Q4KInt8WMMAPrefillSpec(n=sub_n, k=spec.k, m=spec.m, role=spec.role, group_elems=spec.group_elems,
                                        wmma_m=spec.wmma_m, wmma_n=spec.wmma_n, wmma_k=spec.wmma_k,
                                        n_tile=spec.n_tile, target=spec.target, implementation=spec.implementation)
      sub_words = words3_all[start:stop].contiguous().reshape(sub_n * spec.k_blocks * Q4K_WORDS_PER_BLOCK)
      outs.append(emit_q4k_int8_wmma_prefill_tensor(sub_words, xq, xscales, sub_spec, vectorized=True))
    return outs[0].cat(*outs[1:], dim=1).contiguous()

  words3 = words.reshape(spec.n, spec.k_blocks, Q4K_WORDS_PER_BLOCK)
  xq2 = xq.reshape(spec.m, spec.k)
  xsc2 = xscales.reshape(spec.m, spec.groups)
  if vectorized:
    q4_g = _q4k_all_group_codes_tensor(words3, spec)                         # [groups, n, 32]
    q8_g = xq2.reshape(spec.m, spec.groups, spec.group_elems).permute(1, 0, 2).contiguous()  # [groups, m, 32]
    raw = _intdot_matmul(q8_g, q4_g.permute(0, 2, 1).contiguous()).cast(dtypes.float32)  # [groups,m,n]
    qsum = q8_g.cast(dtypes.int32).sum(axis=2).cast(dtypes.float32)          # [groups,m]
    d, dmin, sc, mn = _q4k_all_group_params_tensor(words3, spec)             # [n,groups]
    coeff_raw = (d * sc.cast(dtypes.float32)).permute(1, 0).reshape(spec.groups, 1, spec.n)
    coeff_min = (dmin * mn.cast(dtypes.float32)).permute(1, 0).reshape(spec.groups, 1, spec.n)
    xscale = xsc2.permute(1, 0).reshape(spec.groups, spec.m, 1).cast(dtypes.float32)
    return (xscale * (raw * coeff_raw - qsum.reshape(spec.groups, spec.m, 1) * coeff_min)).sum(axis=0).contiguous()

  out = Tensor.zeros(spec.m, spec.n, dtype=dtypes.float32, device=xq.device)
  for blk in range(spec.k_blocks):
    for grp in range(spec.groups_per_block):
      group_idx = blk * spec.groups_per_block + grp
      start = group_idx * spec.group_elems
      q4_g = _q4k_group_codes_tensor(words3, blk, grp)
      q8_g = xq2[:, start:start + spec.group_elems].contiguous()
      raw = _intdot_matmul(q8_g, q4_g.transpose()).cast(dtypes.float32)
      qsum = q8_g.cast(dtypes.int32).sum(axis=1).cast(dtypes.float32)
      d, dmin, sc, mn = _q4k_group_params_tensor(words3, blk, grp)
      xscale = xsc2[:, group_idx].cast(dtypes.float32)
      scaled_raw = raw * (d * sc.cast(dtypes.float32)).reshape(1, spec.n)
      scaled_min = qsum.reshape(spec.m, 1) * (dmin * mn.cast(dtypes.float32)).reshape(1, spec.n)
      out = out + xscale.reshape(spec.m, 1) * (scaled_raw - scaled_min)
  return out.contiguous()
