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
from extra.qk.amd_warp_reduce import WARP
from extra.qk.lane_partition_reduce import LanePartition, q4k_packed_word_index
from extra.qk.layout_coalesce_check import axis_stride, vector_width
from extra.qk.route_manifest import ROUTES

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

def _manifest_q4k_g3_shapes() -> frozenset[tuple[int, int]]:
  return frozenset((int(g["N"]), int(g["K"])) for g in ROUTES["decode_q4k_g3_generated"].get("shape_guards", [])
                   if isinstance(g.get("N"), int) and isinstance(g.get("K"), int))

def q4k_g3_manifest_shape(out_features:int, in_features:int) -> bool:
  return (out_features, in_features) in _manifest_q4k_g3_shapes()

def should_route_q4k_lane_partition(out_features:int, in_features:int) -> bool:
  """Search-owned q4k route selector for manifest-tracked Q4_K GEMV roles.

  The original P3.3 selector covered only FFN gate/up. It now covers the promoted G3 LaneMap Q4_K roles:
  gate/up, FFN down, and projection shapes declared by the route manifest.
  """
  if not q4k_g3_manifest_shape(out_features, in_features): return False
  return choose_q4k_candidate().candidate.requires_lane_partition
