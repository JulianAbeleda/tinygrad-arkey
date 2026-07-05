#!/usr/bin/env python3
"""Spec-driven Q4_K/Q8_1 prefill MMQ substrate.

This module intentionally does not define a handwritten kernel. It expresses the group dot as ordinary tinygrad
Tensor matmuls with dtype=int, so RDNA3 iu8 WMMA is selected only by the existing tensor-core matcher/codegen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad import Tensor, dtypes

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
    if self.implementation != "group_tensor_matmul_v0":
      raise ValueError(f"unsupported implementation={self.implementation!r}")

  def to_json(self) -> dict[str, Any]:
    return {"n": self.n, "k": self.k, "m": self.m, "role": self.role, "group_elems": self.group_elems,
            "wmma_m": self.wmma_m, "wmma_n": self.wmma_n, "wmma_k": self.wmma_k, "target": self.target,
            "implementation": self.implementation, "groups": self.groups, "k_blocks": self.k_blocks,
            "kernel_name": self.kernel_name}


def describe_q4k_int8_wmma_prefill(n:int, k:int, m:int, *, role:str="") -> Q4KInt8WMMAPrefillSpec:
  spec = Q4KInt8WMMAPrefillSpec(n=n, k=k, m=m, role=role)
  spec.validate()
  return spec


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


def emit_q4k_int8_wmma_prefill_tensor(words:Tensor, xq:Tensor, xscales:Tensor,
                                      spec:Q4KInt8WMMAPrefillSpec) -> Tensor:
  """Return fp32 [m,n] from Q4_K words and Q8_1 activation.

  The int dot is deliberately expressed as `q8_g.matmul(q4_g.T, dtype=dtypes.int)`. On AMD with TC enabled, that
  is the existing codegen route that tensorizes to iu8 WMMA. On CPU/PYTHON this remains a numeric oracle for the
  same algebra.
  """
  spec.validate()
  words3 = words.reshape(spec.n, spec.k_blocks, Q4K_WORDS_PER_BLOCK)
  xq2 = xq.reshape(spec.m, spec.k)
  xsc2 = xscales.reshape(spec.m, spec.groups)
  out = Tensor.zeros(spec.m, spec.n, dtype=dtypes.float32, device=xq.device)
  for blk in range(spec.k_blocks):
    for grp in range(spec.groups_per_block):
      group_idx = blk * spec.groups_per_block + grp
      start = group_idx * spec.group_elems
      q4_g = _q4k_group_codes_tensor(words3, blk, grp)
      q8_g = xq2[:, start:start + spec.group_elems].contiguous()
      raw = q8_g.matmul(q4_g.transpose(), dtype=dtypes.int).cast(dtypes.float32)
      qsum = q8_g.cast(dtypes.int32).sum(axis=1).cast(dtypes.float32)
      d, dmin, sc, mn = _q4k_group_params_tensor(words3, blk, grp)
      xscale = xsc2[:, group_idx].cast(dtypes.float32)
      scaled_raw = raw * (d * sc.cast(dtypes.float32)).reshape(1, spec.n)
      scaled_min = qsum.reshape(spec.m, 1) * (dmin * mn.cast(dtypes.float32)).reshape(1, spec.n)
      out = out + xscale.reshape(spec.m, 1) * (scaled_raw - scaled_min)
  return out.contiguous()
