#!/usr/bin/env python3
import json, pathlib, unittest

from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.uop.ops import AxisType, UOp
from extra.amd_warp_reduce import WARP
from extra.qk_coalesce_search import choose_q4k_candidate, q4k_lane_partition_candidates, rank_candidates

ROOT = pathlib.Path(__file__).resolve().parents[2]

class TestQKCoalesceSearch(unittest.TestCase):
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

  def test_static_choice_agrees_with_measured_m_e_artifact(self):
    latest = ROOT / "bench/qk-scheduler-gemv-vs-owned/coalesced_dequant_mE_latest.json"
    data = json.loads(latest.read_text())
    self.assertEqual(data["verdict"], "PROCEED_P3_SEARCH_GENERALIZATION")
    self.assertTrue(all(v == "lane_partition" for v in data["best_arm"].values()))
    self.assertEqual(choose_q4k_candidate(UOp.range(WARP, 0, AxisType.WARP)).candidate.name, "lane_partition_q4k")

if __name__ == "__main__":
  unittest.main()
