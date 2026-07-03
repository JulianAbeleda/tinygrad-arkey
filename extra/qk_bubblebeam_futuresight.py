#!/usr/bin/env python3
"""Static COALESCE candidate scoring for P3 search generalization.

This is the first bounded P3 slice: make the hand-proven q4k lane partition discoverable by a static layout predicate
before timing.  It intentionally does not mutate storage layout or generic gpudims; it ranks candidate thread maps by
whether the packed-word INDEX is unit-stride in the lane expression.
"""
from __future__ import annotations
from dataclasses import dataclass

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from extra.amd_warp_reduce import WARP
from extra.qk_lane_partition_reduce import LanePartition, q4k_packed_word_index
from extra.qk_layout_coalesce_check import axis_stride, vector_width

@dataclass(frozen=True)
class CoalesceCandidate:
  name: str
  index: UOp
  lane: UOp
  requires_lane_partition: bool = False

@dataclass(frozen=True)
class CoalesceScore:
  candidate: CoalesceCandidate
  stride: int|None
  vector_width: int
  score: int
  reason: str


def score_candidate(c:CoalesceCandidate) -> CoalesceScore:
  stride = axis_stride(c.index, c.lane)
  vw = vector_width(c.index, c.lane)
  score = (1000 if stride == 1 else 0) + vw
  reason = "unit_stride_lane" if stride == 1 else f"non_coalesced_stride_{stride}"
  return CoalesceScore(c, stride, vw, score, reason)


def rank_candidates(cands:list[CoalesceCandidate]) -> list[CoalesceScore]:
  return sorted((score_candidate(c) for c in cands), key=lambda s: (s.score, s.vector_width, s.candidate.name), reverse=True)


def q4k_lane_partition_candidates(lane:UOp, base:UOp|None=None) -> list[CoalesceCandidate]:
  base = UOp.const(dtypes.weakint, 0) if base is None else base
  part = LanePartition(lane)
  # The losing candidate models row-per-lane/default packed word access: adjacent lanes jump by a full block.
  uncoalesced = base + 4 + lane * 36
  return [
    CoalesceCandidate("lane_partition_q4k", q4k_packed_word_index(base, 0, part), lane, True),
    CoalesceCandidate("row_serial_q4k", uncoalesced, lane, False),
  ]


def choose_q4k_candidate(lane:UOp|None=None) -> CoalesceScore:
  lane = UOp.range(WARP, 0) if lane is None else lane
  return rank_candidates(q4k_lane_partition_candidates(lane))[0]

def score_layout_transform(name:str, lane:UOp|None=None) -> CoalesceScore:
  if name != "q4k_lane_partition": raise ValueError(f"unknown layout transform {name!r}")
  return choose_q4k_candidate(lane)

def should_route_q4k_lane_partition(out_features:int, in_features:int) -> bool:
  """Search-owned q4k route selector for tracked Q4_K GEMV roles.

  The original P3.3 selector covered only FFN gate/up. It now covers the promoted G3 LaneMap Q4_K roles:
  gate/up, FFN down, and 4096x4096 projection.
  """
  if not ((out_features in (4096, 12288) and in_features == 4096) or (out_features == 4096 and in_features == 12288)): return False
  return choose_q4k_candidate().candidate.requires_lane_partition
