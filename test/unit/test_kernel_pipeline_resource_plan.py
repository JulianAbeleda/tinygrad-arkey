import unittest

from extra.qk.kernel_pipeline import SchedulerOutputTileLoop
from tinygrad.codegen.opt.compiler_policies import ResourcePlan
from tinygrad.codegen.opt.kernel_pipeline import PINNED_WMMA_VGPR_BUDGET, resource_plan_for_scheduler_tile_loop


class TestResourcePlanPinnedBudget(unittest.TestCase):
  def test_pinned_budget_requires_a_vgpr_ceiling(self):
    ResourcePlan("pinned_budget", vgpr=192)
    with self.assertRaises(ValueError): ResourcePlan("pinned_budget")

  def test_pinned_budget_does_not_require_sgpr(self):
    plan = ResourcePlan("pinned_budget", vgpr=192)
    self.assertIsNone(plan.sgpr)

  def test_host_estimate_and_final_program_invariants_are_unchanged(self):
    ResourcePlan("host_estimate")
    with self.assertRaises(ValueError): ResourcePlan("host_estimate", vgpr=1)
    ResourcePlan("final_program", vgpr=1, sgpr=1)
    with self.assertRaises(ValueError): ResourcePlan("final_program", vgpr=1)


class TestSchedulerTileLoopResourceCapture(unittest.TestCase):
  def test_resource_plan_matches_the_pinned_wmma_budget(self):
    plan = SchedulerOutputTileLoop(64, resident_accumulator_vgprs=128, resident_fragment_vgprs=64)
    resources = plan.resource_plan
    self.assertIsInstance(resources, ResourcePlan)
    self.assertEqual(resources.stage, "pinned_budget")
    self.assertEqual(resources.vgpr, PINNED_WMMA_VGPR_BUDGET)
    self.assertIsNone(resources.sgpr)

  def test_helper_reuses_the_admission_check_and_fails_closed(self):
    self.assertEqual(resource_plan_for_scheduler_tile_loop(
      resident_accumulator_vgprs=128, resident_fragment_vgprs=64).vgpr, PINNED_WMMA_VGPR_BUDGET)
    with self.assertRaises(ValueError):
      resource_plan_for_scheduler_tile_loop(resident_accumulator_vgprs=129, resident_fragment_vgprs=64)


if __name__ == "__main__": unittest.main()
