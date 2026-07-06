#!/usr/bin/env python3
"""Data-only lowering contract for Q4_K/Q8_1 tiled WMMA prefill.

This module deliberately does not emit a kernel. It describes the full-role tile loop that a generated
scheduler/codegen-owned lowering must implement, reusing Q4KInt8WMMATiledPrefillSpec as the algebra/shape authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from extra.qk.prefill_int8_wmma_spec import Q4KInt8WMMATiledPrefillSpec, describe_q4k_int8_wmma_tiled_prefill


@dataclass(frozen=True)
class Q4KWMMAFullRoleLoweringSpec:
  tiled: Q4KInt8WMMATiledPrefillSpec
  wmma_surface: str = "tc_matcher_tile"
  waves_per_block: int = 1
  output_layout: str = "direct"

  @property
  def m(self) -> int: return self.tiled.m

  @property
  def n(self) -> int: return self.tiled.n

  @property
  def k(self) -> int: return self.tiled.k

  @property
  def role(self) -> str: return self.tiled.role

  @property
  def groups(self) -> int: return self.tiled.groups

  @property
  def grid_m(self) -> int: return self.m // self.tiled.m_tile

  @property
  def grid_n(self) -> int: return self.n // self.tiled.n_tile

  @property
  def output_tiles(self) -> int: return self.grid_m * self.grid_n

  @property
  def raw_tile_steps(self) -> int: return self.output_tiles * self.groups

  @property
  def wmma_fragments_per_raw_tile(self) -> int: return self.tiled.group_elems // self.tiled.wmma_k

  @property
  def wmma_fragment_ops(self) -> int: return self.raw_tile_steps * self.wmma_fragments_per_raw_tile

  @property
  def live_raw_elems(self) -> int: return self.tiled.live_raw_elems

  @property
  def forbidden_full_raw_elems(self) -> int: return self.tiled.forbidden_full_raw_elems

  @property
  def bounded_raw_ok(self) -> bool:
    return self.live_raw_elems == self.tiled.m_tile * self.tiled.n_tile * self.tiled.group_tile

  @property
  def requires_scheduler_owned_loop(self) -> bool:
    # A Tensor wrapper per raw tile would create one graph fragment for each step below; full roles need a generated
    # loop nest that owns tile_m/tile_n/group inside the lowering.
    return self.raw_tile_steps > 1024

  def validate(self) -> None:
    self.tiled.validate()
    if self.m % self.tiled.m_tile or self.n % self.tiled.n_tile:
      raise ValueError(f"full-role lowering requires exact tile coverage, got m={self.m} n={self.n} "
                       f"tile={self.tiled.m_tile}x{self.tiled.n_tile}")
    if self.tiled.group_elems % self.tiled.wmma_k:
      raise ValueError(f"group_elems={self.tiled.group_elems} must be divisible by wmma_k={self.tiled.wmma_k}")
    if self.wmma_surface not in ("tc_matcher_tile", "shaped_wmma_tile"):
      raise ValueError(f"unsupported wmma_surface={self.wmma_surface!r}")
    if self.waves_per_block <= 0:
      raise ValueError(f"waves_per_block must be positive, got {self.waves_per_block}")
    if self.output_layout != "direct":
      raise ValueError(f"unsupported output_layout={self.output_layout!r}")

  def to_json(self) -> dict[str, Any]:
    return {"role": self.role, "m": self.m, "n": self.n, "k": self.k, "groups": self.groups,
            "tile": {"m_tile": self.tiled.m_tile, "n_tile": self.tiled.n_tile,
                     "group_tile": self.tiled.group_tile, "group_elems": self.tiled.group_elems,
                     "wmma_m": self.tiled.wmma_m, "wmma_n": self.tiled.wmma_n,
                     "wmma_k": self.tiled.wmma_k},
            "grid": {"m_tiles": self.grid_m, "n_tiles": self.grid_n,
                     "output_tiles": self.output_tiles, "raw_tile_steps": self.raw_tile_steps,
                     "wmma_fragments_per_raw_tile": self.wmma_fragments_per_raw_tile,
                     "wmma_fragment_ops": self.wmma_fragment_ops},
            "bounds": {"live_raw_elems": self.live_raw_elems,
                       "forbidden_full_raw_elems": self.forbidden_full_raw_elems,
                       "bounded_raw_ok": self.bounded_raw_ok},
            "lowering": {"wmma_surface": self.wmma_surface, "waves_per_block": self.waves_per_block,
                         "output_layout": self.output_layout,
                         "requires_scheduler_owned_loop": self.requires_scheduler_owned_loop,
                         "kernel_name": self.tiled.kernel_name}}


def describe_q4k_wmma_full_role_lowering(n:int, k:int, m:int, *, role:str="", m_tile:int=16, n_tile:int=16,
                                         group_tile:int=1, wmma_surface:str="tc_matcher_tile",
                                         waves_per_block:int=1) -> Q4KWMMAFullRoleLoweringSpec:
  tiled = describe_q4k_int8_wmma_tiled_prefill(n, k, m, role=role, m_tile=m_tile, n_tile=n_tile,
                                               group_tile=group_tile)
  spec = Q4KWMMAFullRoleLoweringSpec(tiled=tiled, wmma_surface=wmma_surface, waves_per_block=waves_per_block)
  spec.validate()
  return spec
