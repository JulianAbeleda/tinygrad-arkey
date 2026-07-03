#!/usr/bin/env python3
"""Lane-partitioned reduction primitive for the P2.1 q4k scheduler bridge.

This captures the owned decode thread map as a first-class semantic object before generic gpudims learns to bind
REDUCE shards to hardware lanes:
  lane = block_group * words_per_group + word_col
  block_group = lane // words_per_group
  word_col = lane % words_per_group

The important distinction is semantic: the address split is not enough.  Once work is sharded across lanes, the
partial values must be combined with an explicit cross-lane reduction.  This module reuses the staged warp-reduce
ladder so the combine runs outside divergent store gates.
"""
from __future__ import annotations
from dataclasses import dataclass

from tinygrad.uop.ops import UOp, Ops
from tinygrad.dtype import dtypes
from extra.qk.amd_warp_reduce import WARP
from extra.qk.warp_reduce_lowering import _warp_reduce_sum_staged

class LanePartitionError(ValueError): pass

@dataclass(frozen=True)
class LanePartition:
  lane: UOp
  lane_extent: int = WARP
  words_per_group: int = 8

  def validate(self) -> None:
    if self.lane_extent != WARP: raise LanePartitionError(f"only wave{WARP} is supported, got {self.lane_extent}")
    if self.words_per_group <= 0 or self.lane_extent % self.words_per_group != 0:
      raise LanePartitionError(f"words_per_group must divide lane_extent, got {self.words_per_group} / {self.lane_extent}")
    if self.lane.dtype.scalar() is not dtypes.weakint:
      raise LanePartitionError(f"lane must be weakint index dtype, got {self.lane.dtype}")

  @property
  def block_groups(self) -> int: return self.lane_extent // self.words_per_group
  @property
  def word_col(self) -> UOp: return self.lane % self.words_per_group
  @property
  def block_group(self) -> UOp: return self.lane // self.words_per_group
  def lane_expr(self) -> UOp: return self.block_group * self.words_per_group + self.word_col


def q4k_packed_word_index(base:UOp, grp:int, part:LanePartition) -> UOp:
  """Packed Q4_K quant word index for one group using the owned lane partition.

  Adjacent lanes inside each 8-lane subgroup have consecutive `word_col` values and therefore consecutive packed-word
  offsets.  This is the address half of P2.1; `lane_partition_reduce_sum` is the semantic combine half.
  """
  part.validate()
  if not 0 <= grp < 8: raise LanePartitionError(f"q4k group must be in [0, 8), got {grp}")
  return base + 4 + (grp//2) * 8 + part.word_col


def lane_partition_reduce_sum(partial:UOp, part:LanePartition) -> UOp:
  """Combine lane-owned partials across the whole wave.

  The result is broadcast to every lane, matching `warp_reduce_sum`: each lane sees the full row sum.  Consumers may
  still gate the final store to one lane separately.
  """
  part.validate()
  if partial.dtype.scalar() not in (dtypes.float32, dtypes.float):
    raise LanePartitionError(f"lane partition sum currently supports float partials, got {partial.dtype}")
  if any(u.op is Ops.RANGE and u.arg[-1].name in ("UPCAST", "UNROLL") for u in partial.toposort()):
    raise LanePartitionError("vectorized lane-partition partials are not supported yet")
  return _warp_reduce_sum_staged(partial, part.lane, part.lane_extent)
