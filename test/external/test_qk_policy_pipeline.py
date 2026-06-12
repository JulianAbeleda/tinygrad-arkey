import pathlib, unittest
from types import SimpleNamespace

from extra.qk_policy_pipeline import _profile_specs


class TestQKPolicyPipeline(unittest.TestCase):
  def test_generic_reference_profile_does_not_enable_primitives(self):
    args = SimpleNamespace(reference_mode="generic", device="AMD", model_size="32B")
    specs = _profile_specs(args, pathlib.Path("policy.json"))
    reference = [x for x in specs if x[0].startswith("reference")]
    self.assertEqual([x[1] for x in reference], ["32b-baseline-batched-debug2.log", "32b-baseline-named-debug2.log"])
    for _, _, env in reference:
      self.assertNotIn("Q4K_PRIMITIVE", env)
      self.assertNotIn("Q6K_PRIMITIVE", env)

  def test_explicit_reference_profile_enables_primitives(self):
    args = SimpleNamespace(reference_mode="explicit", device="AMD", model_size="14B")
    specs = _profile_specs(args, pathlib.Path("policy.json"))
    reference = [x for x in specs if x[0].startswith("reference")]
    self.assertEqual([x[1] for x in reference], ["14b-q4q6-primitive-batched-debug2.log", "14b-q4q6-primitive-named-debug2.log"])
    for _, _, env in reference:
      self.assertEqual(env["Q4K_PRIMITIVE"], "1")
      self.assertEqual(env["Q6K_PRIMITIVE"], "1")


if __name__ == "__main__":
  unittest.main()
