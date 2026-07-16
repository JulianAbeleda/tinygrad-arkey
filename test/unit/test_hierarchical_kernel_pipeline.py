import unittest
from dataclasses import FrozenInstanceError, replace

from tinygrad.codegen.opt.kernel_pipeline import (HierarchicalKernelPipelinePlan, HierarchicalLifecycleEvent,
  HierarchicalPipelineRole, hierarchical_lifecycle_events, prove_hierarchical_lifecycle)


class TestHierarchicalKernelPipeline(unittest.TestCase):
  def setUp(self):
    self.plan = HierarchicalKernelPipelinePlan(HierarchicalPipelineRole("weights", "outer_epoch"),
                                               HierarchicalPipelineRole("activations", "inner_phase"))

  def test_exact_asymmetric_lifecycle(self):
    events = hierarchical_lifecycle_events(self.plan)
    proof = prove_hierarchical_lifecycle(self.plan, events)
    self.assertTrue(proof.passed, proof.errors)
    self.assertEqual(proof.produced, (("weights", None), ("activations", 0), ("activations", 1)))
    self.assertEqual(proof.barriers, (("publish", 0), ("release", 0), ("publish", 1), ("release", 1)))
    self.assertEqual([x for x in events if x.op == "consume" and x.role == "weights"],
                     [HierarchicalLifecycleEvent("consume", "weights", 0),
                      HierarchicalLifecycleEvent("consume", "weights", 1)])
    self.assertEqual(events[-1], HierarchicalLifecycleEvent("release", "weights", None))

  def test_vocabulary_is_immutable_and_generic(self):
    with self.assertRaises(FrozenInstanceError): self.plan.phase_count = 3
    with self.assertRaises(FrozenInstanceError): hierarchical_lifecycle_events(self.plan)[0].phase = 0
    alternate = HierarchicalKernelPipelinePlan(HierarchicalPipelineRole("lhs", "outer_epoch"),
                                                HierarchicalPipelineRole("rhs", "inner_phase"))
    self.assertTrue(prove_hierarchical_lifecycle(alternate, hierarchical_lifecycle_events(alternate)).passed)

  def test_plan_rejects_wrong_lifetimes_roles_and_invalid_phase_count(self):
    outer, inner = HierarchicalPipelineRole("outer", "outer_epoch"), HierarchicalPipelineRole("inner", "inner_phase")
    with self.assertRaises(ValueError): HierarchicalKernelPipelinePlan(inner, outer)
    with self.assertRaises(ValueError): HierarchicalKernelPipelinePlan(outer, HierarchicalPipelineRole("outer", "inner_phase"))
    for phase_count in (0, True, 1.5):
      with self.subTest(phase_count=phase_count), self.assertRaises(ValueError):
        HierarchicalKernelPipelinePlan(outer, inner, phase_count)

  def test_three_phases_yield_six_ordered_barriers(self):
    plan = replace(self.plan, phase_count=3)
    events = hierarchical_lifecycle_events(plan)
    proof = prove_hierarchical_lifecycle(plan, events)
    self.assertTrue(proof.passed, proof.errors)
    self.assertEqual(proof.barriers, (("publish", 0), ("release", 0),
                                      ("publish", 1), ("release", 1),
                                      ("publish", 2), ("release", 2)))

  def assertRejected(self, events, text):
    proof = prove_hierarchical_lifecycle(self.plan, tuple(events))
    self.assertFalse(proof.passed)
    self.assertTrue(any(text in error for error in proof.errors), proof.errors)

  def test_fail_closed_mutations(self):
    canonical = hierarchical_lifecycle_events(self.plan)
    cases = {
      "overwrite before release": (canonical[:6] + (canonical[6], canonical[1]) + canonical[7:], "expected publish"),
      "duplicate production": (canonical[:2] + (canonical[1],) + canonical[2:], "expected publish"),
      "missing production": (canonical[:1] + canonical[2:], "expected produce"),
      "missing barrier": (canonical[:2] + canonical[3:], "expected publish"),
      "early persistent release": (canonical[:4] + (canonical[-1],) + canonical[4:-1], "expected consume"),
      "wrong role": ((replace(canonical[0], role="activations"),) + canonical[1:], "role weights"),
      "wrong phase": (canonical[:1] + (replace(canonical[1], phase=1),) + canonical[2:], "phase 0"),
    }
    for name, (events, error) in cases.items():
      with self.subTest(name=name): self.assertRejected(events, error)

  def test_requires_four_uniform_barriers(self):
    events = list(hierarchical_lifecycle_events(self.plan))
    events.pop(6)  # phase-zero release barrier
    self.assertRejected(events, "release")


if __name__ == "__main__": unittest.main()
