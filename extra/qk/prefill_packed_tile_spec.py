#!/usr/bin/env python3
"""Spec-driven generated packed-prefill tile candidates.

The first Q4_K candidate is deliberately default-off and microgate-oriented. It keeps the lossless direct-packed math
path but changes the substrate shape: Q4_K word lanes become LOCAL/cooperative work, and row/token tiles form a
256-thread workgroup by default. It writes eight lane partials and lets the caller reduce them; a promoted route should
replace that with an in-kernel lane combine once the cooperative topology proves faster on the hot rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad import dtypes
from tinygrad.helpers import cdiv
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS
from extra.qk.quant.q4_k_gemv_primitive import _q4k_block_dot_packed_load_gemm


@dataclass(frozen=True)
class PackedPrefillTileSpec:
  quant: str
  rows: int
  k: int
  tokens: int
  row_tile: int = 4
  token_tile: int = 8
  lane_tile: int = 8
  role: str = ""
  output_layout: str = "lane_partials"
  accumulator: str = "fp32"
  target: str = "amd_gfx1100"

  @property
  def threads(self) -> int:
    return self.row_tile * self.token_tile * self.lane_tile

  @property
  def kernel_name(self) -> str:
    role = f"_{self.role}" if self.role else ""
    return f"prefill_{self.quant.lower()}_generated_tile{role}_{self.tokens}_{self.rows}_{self.k}"

  def validate(self) -> None:
    if self.quant != "Q4_K": raise ValueError(f"only Q4_K is implemented, got {self.quant}")
    if self.output_layout != "lane_partials": raise ValueError("first generated tile emits lane_partials only")
    if self.accumulator != "fp32": raise ValueError("lossless generated tile requires fp32 accumulation")
    if self.lane_tile != 8: raise ValueError("Q4_K generated tile requires all 8 packed word lanes")
    if self.threads != 256: raise ValueError(f"first generated tile must be 256 threads, got {self.threads}")
    if self.rows % self.row_tile or self.tokens % self.token_tile:
      raise ValueError(f"row_tile/token_tile must divide rows/tokens, got rows={self.rows}, tokens={self.tokens}")
    if self.k % Q4_K_BLOCK_ELEMS: raise ValueError(f"k={self.k} must be a multiple of Q4_K block elems")

  def to_json(self) -> dict[str, Any]:
    return {"quant": self.quant, "rows": self.rows, "k": self.k, "tokens": self.tokens, "row_tile": self.row_tile,
            "token_tile": self.token_tile, "lane_tile": self.lane_tile, "role": self.role,
            "output_layout": self.output_layout, "accumulator": self.accumulator, "target": self.target,
            "threads": self.threads, "kernel_name": self.kernel_name}


def describe_q4k_packed_prefill_tile(rows:int, k:int, tokens:int, *, role:str="",
                                     row_tile:int=4, token_tile:int=8) -> PackedPrefillTileSpec:
  spec = PackedPrefillTileSpec("Q4_K", rows, k, tokens, row_tile=row_tile, token_tile=token_tile, role=role)
  spec.validate()
  return spec


def emit_q4k_packed_prefill_tile(spec:PackedPrefillTileSpec):
  spec.validate()
  rows, k, b = spec.rows, spec.k, spec.tokens
  row_tile, token_tile = spec.row_tile, spec.token_tile
  k_blocks = k // Q4_K_BLOCK_ELEMS

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row_o = UOp.range(cdiv(rows, row_tile), 0)
    bb_o = UOp.range(cdiv(b, token_tile), 1)
    row_i = UOp.range(row_tile, 2, axis_type=AxisType.LOCAL)
    bb_i = UOp.range(token_tile, 3, axis_type=AxisType.LOCAL)
    lane4 = UOp.range(8, 4, axis_type=AxisType.LOCAL)
    blk = UOp.range(k_blocks, 5, axis_type=AxisType.REDUCE)
    row = row_o * row_tile + row_i
    bb = bb_o * token_tile + bb_i
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load_gemm(words, x, base, blk, lane4, bb, k)

    acc = partials[row, bb, lane4].set(0.0)
    acc = partials[row, bb, lane4].set(acc.after(blk)[row, bb, lane4] + contrib, end=blk)
    return acc.end(row_o, bb_o, row_i, bb_i, lane4).sink(
      arg=KernelInfo(name=spec.kernel_name, opts_to_apply=()))

  return kernel
