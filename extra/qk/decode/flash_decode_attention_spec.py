#!/usr/bin/env python3
"""Descriptor scaffolding for live-split flash decode routes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tinygrad.uop.ops import UOp

from extra.qk.flash_kernels import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel
from extra.qk.live_split_geometry import ceildiv_uop, flash_fused_gmax_combine_kernel


@dataclass(frozen=True)
class LiveSplitGeometrySpec:
  """Runtime geometry parameters for fixed-S live-split attention."""
  split_count: int
  token_block: int = 16

  def validate(self) -> None:
    if self.split_count < 1:
      raise ValueError(f"split_count must be >= 1, got {self.split_count!r}")
    if self.token_block < 1:
      raise ValueError(f"token_block must be >= 1, got {self.token_block!r}")

  def per_split_length(self, Tc: UOp) -> UOp:
    """Ceil-divided per-split token extent (runtime, symbolic when Tc is symbolic)."""
    return ceildiv_uop(Tc, self.split_count)

  def aligned_per_split_length(self, Tc: UOp) -> UOp:
    """Per-split length aligned to `token_block` for LDS staging.

    The existing live-split decode implementation stages the tile with fixed TK chunks, so this alignment
    prevents cross-split overlap and keeps split ownership disjoint.
    """
    return ceildiv_uop(self.per_split_length(Tc), self.token_block) * self.token_block

  def blocks(self, Tc: UOp) -> UOp:
    """Runtime number of inner blocks per split."""
    return ceildiv_uop(self.aligned_per_split_length(Tc), self.token_block)


@dataclass(frozen=True)
class FlashDecodeTileSpec:
  Hq: int
  Hd: int
  Hkv: int
  MAXC: int
  split_count: int
  staging: str = "KV_BOTH"
  quant: bool = False
  rope: bool = False
  token_block: int = 16
  target: str = "amd_gfx1100"

  def validate(self) -> None:
    if self.Hq <= 0: raise ValueError(f"Hq must be positive, got {self.Hq}")
    if self.Hd <= 0: raise ValueError(f"Hd must be positive, got {self.Hd}")
    if self.Hkv <= 0: raise ValueError(f"Hkv must be positive, got {self.Hkv}")
    if self.MAXC <= 0: raise ValueError(f"MAXC must be positive, got {self.MAXC}")
    if self.staging not in {"KV_BOTH", "K_ONLY"}:
      raise ValueError(f"unsupported staging={self.staging!r}; allowed {{'KV_BOTH', 'K_ONLY'}}")
    if self.token_block != 16:
      raise ValueError(f"token_block must currently be 16, got {self.token_block}")
    self.geometry.validate()

  @property
  def geometry(self) -> LiveSplitGeometrySpec:
    return LiveSplitGeometrySpec(split_count=self.split_count, token_block=self.token_block)

  @property
  def kernel_name(self) -> str:
    return f"flash_block_tiled_xlane_score_pv_tile_whole_cache_{self.Hq}_{self.Hd}"

  def emit(self, Tc_u: UOp):
    """Emit the live-split tile kernel for this configuration."""
    self.validate()
    return flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(
      self.Hd, self.Hq, self.Hkv, self.MAXC, self.geometry.aligned_per_split_length(Tc_u), self.split_count,
      Tc_u, staging=self.staging, quant=self.quant, rope=self.rope)

  def to_json(self) -> dict[str, Any]:
    return {"Hq": self.Hq, "Hd": self.Hd, "Hkv": self.Hkv, "MAXC": self.MAXC,
            "split_count": self.split_count, "staging": self.staging, "quant": self.quant, "rope": self.rope,
            "token_block": self.token_block, "target": self.target}


@dataclass(frozen=True)
class FlashCombineSpec:
  Hd: int
  Hq: int
  split_count: int
  stride: int | None = None

  def validate(self) -> None:
    if self.Hd <= 0: raise ValueError(f"Hd must be positive, got {self.Hd}")
    if self.Hq <= 0: raise ValueError(f"Hq must be positive, got {self.Hq}")
    if self.split_count < 1: raise ValueError(f"split_count must be >= 1, got {self.split_count}")
    if self.stride is not None and self.stride < 1:
      raise ValueError(f"stride must be >= 1 when set, got {self.stride}")

  @property
  def kernel_name(self) -> str:
    return f"flash_fused_gmax_combine_{self.Hq}_{self.Hd}"

  def emit(self):
    self.validate()
    return flash_fused_gmax_combine_kernel(self.Hd, self.Hq, self.split_count, stride=self.stride)


@dataclass(frozen=True)
class FlashDecodeAttentionSpec:
  tile: FlashDecodeTileSpec
  combine: FlashCombineSpec | None = None

  @property
  def descriptor_artifact(self) -> str:
    return "FlashDecodeAttentionSpec"

  def validate(self) -> None:
    self.tile.validate()
    if self.combine is not None:
      self.combine.validate()

  def emit_tile(self, Tc_u: UOp):
    self.validate()
    return self.tile.emit(Tc_u)

  def emit_combine(self):
    self.validate()
    if self.combine is None: raise ValueError("combine was not requested")
    return self.combine.emit()

  @property
  def emitted_kernel_names(self) -> tuple[str, ...]:
    if self.combine is None:
      return (self.tile.kernel_name,)
    return (self.tile.kernel_name, self.combine.kernel_name)


def describe_flash_decode_attention(Hq:int, Hd:int, Hkv:int, MAXC:int, S:int, *,
                                  staging:str="KV_BOTH", fused_combine:bool=True,
                                  quant:bool=False, rope:bool=False, combine_stride:int|None=None) -> FlashDecodeAttentionSpec:
  tile = FlashDecodeTileSpec(Hq=Hq, Hd=Hd, Hkv=Hkv, MAXC=MAXC, split_count=S, staging=staging, quant=quant, rope=rope)
  combine = FlashCombineSpec(Hd=Hd, Hq=Hq, split_count=S, stride=combine_stride) if fused_combine else None
  return FlashDecodeAttentionSpec(tile=tile, combine=combine)


def emit_flash_decode_tile(spec: FlashDecodeAttentionSpec, Tc_u: UOp):
  return spec.emit_tile(Tc_u)


def emit_flash_decode_combine(spec: FlashDecodeAttentionSpec):
  return spec.emit_combine()
