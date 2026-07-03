#!/usr/bin/env python3
import json, pathlib, unittest

from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import AxisType, UOp
from extra.amd_warp_reduce import WARP
from extra.qk_bubblebeam_futuresight import choose_q4k_candidate, q4k_lane_partition_candidates, rank_candidates, should_route_q4k_lane_partition

ROOT = pathlib.Path(__file__).resolve().parents[2]

class TestQKBubbleBeamFutureSight(unittest.TestCase):
  def test_optops_has_coalesce_marker(self):
    self.assertIs(OptOps.COALESCE, OptOps.COALESCE)
    self.assertEqual(Opt(OptOps.COALESCE, 0, 8).op, OptOps.COALESCE)

  def test_static_cost_selects_lane_partition_without_timing(self):
    lane = UOp.range(WARP, 0, AxisType.WARP)
    ranked = rank_candidates(q4k_lane_partition_candidates(lane))
    self.assertEqual(ranked[0].candidate.name, "lane_partition_q4k")
    self.assertEqual(ranked[0].stride, 1)
    self.assertGreater(ranked[0].score, ranked[1].score)

  def test_choose_q4k_candidate(self):
    best = choose_q4k_candidate(UOp.range(WARP, 0, AxisType.WARP))
    self.assertEqual(best.candidate.name, "lane_partition_q4k")
    self.assertTrue(best.candidate.requires_lane_partition)

  def test_shape_route_selector_matches_promoted_g3_roles(self):
    self.assertTrue(should_route_q4k_lane_partition(12288, 4096))
    self.assertTrue(should_route_q4k_lane_partition(4096, 4096))
    self.assertTrue(should_route_q4k_lane_partition(4096, 12288))
    self.assertFalse(should_route_q4k_lane_partition(1024, 4096))

  def test_static_choice_agrees_with_measured_m_e_artifact(self):
    latest = ROOT / "bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_latest.json"
    data = json.loads(latest.read_text())
    self.assertEqual(data["verdict"], "GEMV_PURE_SEARCH_GENERATED__BUBBLEBEAM_G3_FULL_Q4K_GEMV")
    self.assertTrue(data["bubblebeam_futuresight_generated_route_ok"])
    self.assertTrue(data["tokens_match_all_ctx"])
    self.assertTrue(all(v in ("lane_partition", "bubblebeam_futuresight", "g3_lanemap_codegen") for v in data["best_arm"].values()))
    full = data["bubblebeam_g3_full_q4k_gemv_2026_06_25"]
    self.assertEqual(full["verdict"], data["verdict"])
    self.assertTrue(full["tokens_match_all_ctx"])
    self.assertEqual(choose_q4k_candidate(UOp.range(WARP, 0, AxisType.WARP)).candidate.name, "lane_partition_q4k")

if __name__ == "__main__":
  unittest.main()
