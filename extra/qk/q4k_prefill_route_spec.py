#!/usr/bin/env python3
"""Spec-driven Q4_K direct-packed prefill lowering."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tinygrad import dtypes
from tinygrad.codegen.opt import Opt
from tinygrad.helpers import cdiv
from tinygrad.uop.ops import AxisType, KernelInfo, UOp

from extra.qk.layout import Q4K_WORDS_PER_BLOCK, Q4_K_BLOCK_ELEMS
from extra.qk.quant.q4_k_gemv_primitive import _q4k_block_dot_packed_load_gemm, parse_opt


_ALLOWED_OUTPUT_LAYOUTS = ("direct_out", "partials")


def _coerce_opts(opts: tuple[str | Opt, ...]) -> tuple[Opt, ...]:
  parsed = []
  for i, opt in enumerate(opts):
    if isinstance(opt, str):
      parsed.append(parse_opt(opt))
    elif isinstance(opt, Opt):
      parsed.append(opt)
    else:
      raise TypeError(f"invalid opts[{i}] type {type(opt)!r}; expected str or Opt")
  return tuple(parsed)


@dataclass(frozen=True)
class Q4KPrefillRouteSpec:
  rows: int
  k: int
  tokens: int
  parts: int = 1
  output_layout: str = "direct_out"
  role: str = ""
  schedule: str = "prefill"
  opts: tuple[Opt, ...] = field(default_factory=tuple)
  target: str = "amd_gfx1100"
  quant: str = "Q4_K"

  @property
  def k_blocks(self) -> int:
    return self.k // Q4_K_BLOCK_ELEMS

  @property
  def kernel_name(self) -> str:
    if self.output_layout == "direct_out":
      return f"q4k_gen_prefill_direct_out_{self.rows}_{self.k}_{self.tokens}"
    return f"q4k_gen_prefill_partials_{self.rows}_{self.k}_{self.tokens}_{self.parts}"

  def validate(self) -> None:
    if self.quant != "Q4_K":
      raise ValueError(f"Q4KPrefillRouteSpec quant must be Q4_K, got {self.quant!r}")
    if self.output_layout not in _ALLOWED_OUTPUT_LAYOUTS:
      raise ValueError(f"unsupported output_layout={self.output_layout!r}")
    if self.rows <= 0 or self.k <= 0 or self.tokens <= 0:
      raise ValueError(f"rows/k/tokens must be positive, got rows={self.rows} k={self.k} tokens={self.tokens}")
    if self.k % Q4_K_BLOCK_ELEMS != 0:
      raise ValueError(f"k={self.k} must be a multiple of {Q4_K_BLOCK_ELEMS}")
    if self.parts < 1:
      raise ValueError(f"parts must be >= 1, got {self.parts}")
    if self.output_layout == "direct_out" and self.parts != 1:
      raise ValueError("direct_out output_layout requires parts==1")

  def to_json(self) -> dict[str, Any]:
    return {"quant": self.quant, "rows": self.rows, "k": self.k, "tokens": self.tokens, "parts": self.parts,
            "output_layout": self.output_layout, "role": self.role, "schedule": self.schedule, "target": self.target,
            "k_blocks": self.k_blocks, "kernel_name": self.kernel_name}


def describe_q4k_packed_prefill(rows:int, k:int, tokens:int, *, role:str="", parts:int=1,
                                output_layout:str="direct_out", schedule:str="prefill",
                                opts:tuple[str | Opt, ...]=()) -> Q4KPrefillRouteSpec:
  spec = Q4KPrefillRouteSpec(rows=rows, k=k, tokens=tokens, role=role, parts=parts,
                             output_layout=output_layout, schedule=schedule, opts=_coerce_opts(opts))
  spec.validate()
  return spec


def emit_q4k_packed_prefill_kernel(spec:Q4KPrefillRouteSpec):
  spec.validate()
  if spec.output_layout == "direct_out":
    return _emit_direct_out(spec)
  return _emit_partials(spec)


def _emit_direct_out(spec:Q4KPrefillRouteSpec):
  rows, k, tokens, k_blocks, name, opts = spec.rows, spec.k, spec.tokens, spec.k_blocks, spec.kernel_name, spec.opts

  def kernel(out:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(tokens, 1)
    blk = UOp.range(k_blocks, 2, axis_type=AxisType.REDUCE)
    lane4 = UOp.range(8, 3, axis_type=AxisType.REDUCE)
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = _q4k_block_dot_packed_load_gemm(words, x, base, blk, lane4, bb, k)

    acc = out[bb, row].set(0.0)
    acc = out[bb, row].set(acc.after(blk, lane4)[bb, row] + contrib, end=lane4)
    return acc.end(row, bb, blk).sink(arg=KernelInfo(name=name, opts_to_apply=opts))

  return kernel


def _emit_partials(spec:Q4KPrefillRouteSpec):
  rows, k, tokens, parts, k_blocks, name, opts = spec.rows, spec.k, spec.tokens, spec.parts, spec.k_blocks, spec.kernel_name, spec.opts
  blocks_per_part = cdiv(k_blocks, parts)

  def kernel(partials:UOp, words:UOp, x:UOp) -> UOp:
    row = UOp.range(rows, 0)
    bb = UOp.range(tokens, 1)
    part = UOp.range(parts, 2)
    blk_part = UOp.range(blocks_per_part, 3, axis_type=AxisType.REDUCE)
    lane4 = UOp.range(8, 4, axis_type=AxisType.REDUCE)
    blk = part * blocks_per_part + blk_part
    in_range = blk < k_blocks
    base = (row * k_blocks + blk) * Q4K_WORDS_PER_BLOCK
    contrib = in_range.where(_q4k_block_dot_packed_load_gemm(words, x, base, blk, lane4, bb, k),
                            UOp.const(dtypes.float32, 0.0))

    acc = partials[row, bb, part].set(0.0)
    acc = partials[row, bb, part].set(acc.after(blk_part, lane4)[row, bb, part] + contrib, end=lane4)
    return acc.end(row, bb, part, blk_part).sink(arg=KernelInfo(name=name, opts_to_apply=opts))

  return kernel
