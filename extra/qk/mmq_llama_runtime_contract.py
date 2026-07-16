"""Typed vocabulary for the source-derived llama.cpp MMQ runtime contract.

This module describes arithmetic in the conventional ``mul_mat_q`` path.  It
does not emit a kernel, select a route, or make a performance claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


LLAMA_SOURCE_COMMIT = "ac4cddeb0dbd778f650bf568f6f08344a06abe3a"
LLAMA_MMQ_CUH = "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh"
SOURCE_ANCHORS = {
  "k_epochs": "mmq.cuh:3478-3517",
  "identity_ids": "mmq.cuh:3564-3577",
  "conventional_grid_decode": "mmq.cuh:3583-3587",
  "moe_ids": "mmq.cuh:3596-3618",
  "offsets_and_tails": "mmq.cuh:3622-3633",
  "grid": "mmq.cuh:3958-3961",
  "ratios": "mmq.cuh:3963-3973",
  "need_check": "mmq.cuh:3975-3991",
}


def _positive(name: str, value: int) -> None:
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
    raise ValueError(f"{name} must be a positive integer")


def ceil_div(value: int, divisor: int) -> int:
  _positive("value", value)
  _positive("divisor", divisor)
  return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class MMQTile:
  """Runtime tile and packed-format extents; no model role is implied."""
  x: int
  y: int
  qk: int
  iter_k: int
  q8_pair_elements: int
  q8_record_ints: int

  def __post_init__(self) -> None:
    for name in ("x", "y", "qk", "iter_k", "q8_pair_elements", "q8_record_ints"):
      _positive(name, getattr(self, name))
    if self.iter_k % self.qk: raise ValueError("iter_k must contain a whole number of Q4 blocks")
    if self.iter_k % self.q8_pair_elements: raise ValueError("iter_k must contain whole Q8 pairs")

  @property
  def blocks_per_epoch(self) -> int: return self.iter_k // self.qk


@dataclass(frozen=True)
class MMQExtents:
  ncols_x: int
  nrows_x: int
  ncols_dst: int
  ncols_y: int
  ncols_max: int
  nchannels_x: int
  nchannels_y: int
  nsamples_x: int
  nsamples_y: int

  def __post_init__(self) -> None:
    for name in self.__dataclass_fields__: _positive(name, getattr(self, name))
    if self.nchannels_y % self.nchannels_x: raise ValueError("nchannels_y must be divisible by nchannels_x")
    if self.nsamples_y % self.nsamples_x: raise ValueError("nsamples_y must be divisible by nsamples_x")

  @property
  def channel_ratio(self) -> int: return self.nchannels_y // self.nchannels_x

  @property
  def sample_ratio(self) -> int: return self.nsamples_y // self.nsamples_x


@dataclass(frozen=True)
class MMQStrides:
  row_x: int
  channel_x: int
  channel_y: int
  channel_dst: int
  sample_x: int
  sample_y: int
  sample_dst: int
  col_dst: int

  def __post_init__(self) -> None:
    for name in self.__dataclass_fields__: _positive(name, getattr(self, name))


@dataclass(frozen=True)
class Grid3D:
  x: int
  y: int
  z: int


@dataclass(frozen=True)
class TileIndex:
  it: int
  jt: int
  wt: int
  zt: int


@dataclass(frozen=True)
class TailPredicates:
  i_max: int
  j_max: int
  need_check: bool


@dataclass(frozen=True)
class GlobalAddresses:
  q4_block: int
  q8_first_int: int
  q8_second_int: int


@dataclass(frozen=True)
class PhysicalEpochBinding:
  """Physical source offsets and output predicates for one runtime tile/epoch.

  Q4 offsets are uint32 words and Q8 offsets are bytes.  Keeping those units in
  the type prevents accidentally treating llama's packed carriers as dense
  element arrays.
  """
  addresses: GlobalAddresses
  q4_word_offset: int
  q8_byte_offset: int
  tails: TailPredicates

  def valid_output(self, local_i:int, local_j:int) -> bool:
    return 0 <= local_i <= self.tails.i_max and 0 <= local_j <= self.tails.j_max


@dataclass(frozen=True)
class ConventionalTile:
  index: TileIndex
  col_low: int
  col_diff: int
  offset_x: int
  offset_y: int
  offset_dst: int
  ids: tuple[int, ...]
  tails: TailPredicates

  def addresses(self, kb0: int, tile: MMQTile, extents: MMQExtents) -> GlobalAddresses:
    if kb0 < 0 or kb0 >= extents.ncols_x // tile.qk: raise ValueError("kb0 is outside the Q4 K extent")
    if kb0 % tile.blocks_per_epoch: raise ValueError("kb0 must begin an outer K epoch")
    numerator = kb0 * tile.qk
    if numerator % tile.q8_pair_elements: raise ValueError("Q8 pair address is not integral")
    pair = numerator // tile.q8_pair_elements
    first = self.offset_y + extents.ncols_y * pair * tile.q8_record_ints
    return GlobalAddresses(self.offset_x + kb0, first, first + extents.ncols_y * tile.q8_record_ints)

  def bind_epoch(self, kb0:int, tile:MMQTile, extents:MMQExtents) -> PhysicalEpochBinding:
    addresses = self.addresses(kb0, tile, extents)
    return PhysicalEpochBinding(addresses, addresses.q4_block*36, addresses.q8_first_int*4, self.tails)

  def destination(self, local_i: int, local_j: int, strides: MMQStrides) -> int:
    if not (0 <= local_i <= self.tails.i_max): raise ValueError("local_i is outside the tile tail")
    if not (0 <= local_j < len(self.ids) and local_j <= self.tails.j_max):
      raise ValueError("local_j is outside the tile tail")
    return self.offset_dst + self.ids[local_j] * strides.col_dst + local_i


@dataclass(frozen=True)
class ConventionalRuntimeContract:
  tile: MMQTile
  extents: MMQExtents
  strides: MMQStrides

  def __post_init__(self) -> None:
    if self.extents.ncols_x % self.tile.qk: raise ValueError("ncols_x must contain whole Q4 blocks")

  @property
  def grid(self) -> Grid3D:
    return Grid3D(ceil_div(self.extents.nrows_x, self.tile.y),
                  ceil_div(self.extents.ncols_max, self.tile.x),
                  self.extents.nchannels_y * self.extents.nsamples_y)

  @property
  def k_epoch_starts(self) -> tuple[int, ...]:
    return tuple(range(0, self.extents.ncols_x // self.tile.qk, self.tile.blocks_per_epoch))

  @property
  def need_check(self) -> bool: return self.extents.nrows_x % self.tile.y != 0

  def index(self, block_x: int, block_y: int, block_z: int) -> TileIndex:
    g = self.grid
    if not (0 <= block_x < g.x and 0 <= block_y < g.y and 0 <= block_z < g.z):
      raise ValueError("block index is outside the conventional grid")
    return TileIndex(block_x, block_y, block_z // self.extents.nchannels_y,
                     block_z % self.extents.nchannels_y)

  def conventional_tile(self, block_x: int, block_y: int, block_z: int,
                        *, expert_bounds: Optional[tuple[int, int]] = None,
                        moe_ids: Optional[tuple[int, ...]] = None) -> ConventionalTile:
    idx = self.index(block_x, block_y, block_z)
    if (expert_bounds is None) != (moe_ids is None):
      raise ValueError("expert_bounds and moe_ids must be supplied together")
    col_low, col_diff = 0, self.extents.ncols_dst
    offset_y = idx.wt*self.strides.sample_y + idx.zt*self.strides.channel_y
    offset_dst = idx.wt*self.strides.sample_dst + idx.zt*self.strides.channel_dst + idx.jt*self.tile.x*self.strides.col_dst
    ids = tuple(range(self.tile.x))
    if expert_bounds is not None:
      low, high = expert_bounds
      if low < 0 or high < low: raise ValueError("invalid expert bounds")
      col_low, col_diff, offset_y, offset_dst = low, high-low, 0, 0
      if idx.jt*self.tile.x >= col_diff: raise ValueError("tile is outside the expert column extent")
      assert moe_ids is not None
      start = col_low + idx.jt*self.tile.x
      if start + self.tile.x > len(moe_ids): raise ValueError("MoE ID tile is not fully initialized")
      ids = tuple(moe_ids[start:start+self.tile.x])
    offset_y += (col_low + idx.jt*self.tile.x) * self.tile.q8_record_ints
    offset_dst += idx.it*self.tile.y
    offset_x = (idx.wt//self.extents.sample_ratio)*self.strides.sample_x + \
               (idx.zt//self.extents.channel_ratio)*self.strides.channel_x + idx.it*self.tile.y*self.strides.row_x
    tails = TailPredicates(self.extents.nrows_x-idx.it*self.tile.y-1,
                           col_diff-idx.jt*self.tile.x-1, self.need_check)
    return ConventionalTile(idx, col_low, col_diff, offset_x, offset_y, offset_dst, ids, tails)


__all__ = ["ConventionalRuntimeContract", "ConventionalTile", "GlobalAddresses", "Grid3D", "LLAMA_MMQ_CUH",
           "LLAMA_SOURCE_COMMIT", "MMQExtents", "MMQStrides", "MMQTile", "SOURCE_ANCHORS", "TailPredicates",
           "PhysicalEpochBinding", "TileIndex", "ceil_div"]
