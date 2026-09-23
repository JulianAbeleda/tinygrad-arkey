"""The historical Q4_K lane map: adjacent lanes must read adjacent packed words."""
from extra.llm_research.q4k_lane_map import (q4k_lane_partition_candidates, rank_candidates,
                                             should_route_q4k_lane_partition)
from tinygrad.uop.ops import UOp


def test_the_lane_partition_beats_row_serial_access():
  best = rank_candidates(q4k_lane_partition_candidates(UOp.range(32, 0)))[0]
  assert best.candidate.name == "lane_partition_q4k"
  assert best.reason == "unit_stride_lane"
  assert should_route_q4k_lane_partition(12288, 4096) is True


def test_it_carries_no_search_policy_with_it():
  """The reason this half stays in the fork is that it reasons over UOps, not that it decides anything."""
  import extra.llm_research.q4k_lane_map as lane_map
  assert not hasattr(lane_map, "propose_legal_dimensions")
  assert not hasattr(lane_map, "classify_candidates")
